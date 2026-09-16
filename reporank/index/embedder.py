"""文本编码器：模型加载、批量编码、磁盘缓存。

W4 的核心矛盾是成本。一个 django 快照切分后有上万个 chunk，
50 条实例跑一轮就是几十万次编码。没有缓存的话迭代循环会从
70 秒变成 40 分钟，实验就没法做了。

三个成本控制手段：
  1. 磁盘缓存    —— 按 (模型, 内容哈希) 缓存向量，重跑近乎免费
  2. 设备自适应  —— M 系列 Mac 走 MPS，有 N 卡走 CUDA，否则 CPU
  3. 可换小模型  —— 先用 MiniLM 把管道跑通，再换大模型看上限

向量统一做 L2 归一化，这样余弦相似度退化成点积，检索时一次矩阵乘法搞定。
"""

from __future__ import annotations

import hashlib
import os
from pathlib import Path

import numpy as np

CACHE_DIR = Path(os.environ.get("REPORANK_CACHE", Path.home() / ".cache" / "reporank"))
EMB_CACHE = CACHE_DIR / "embeddings"

# 默认用小模型。22M 参数、384 维，在 MacBook 上可用。
# 先用它把管道跑通拿到基线，再换 Qwen3-Embedding-0.6B 看上限 ——
# 公开报告显示 0.6B 效果接近 8B 而成本低得多，是性价比拐点。
#DEFAULT_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
DEFAULT_MODEL = "BAAI/bge-small-en-v1.5"

# 代码检索的候选模型，按大小排列。换模型只需改 --emb-model 参数。
KNOWN_MODELS = {
    "minilm": "sentence-transformers/all-MiniLM-L6-v2",       # 22M, 384d
    "bge-small": "BAAI/bge-small-en-v1.5",                     # 33M, 384d
    "qwen3-0.6b": "Qwen/Qwen3-Embedding-0.6B",                 # 600M, 1024d
}


def resolve_model(name: str) -> str:
    return KNOWN_MODELS.get(name, name)


def pick_device() -> str:
    import torch

    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _hash(model: str, text: str) -> str:
    h = hashlib.sha1()
    h.update(model.encode("utf-8"))
    h.update(b"\x00")
    h.update(text.encode("utf-8", errors="ignore"))
    return h.hexdigest()


class Embedder:
    """句向量编码器。

    encode_fn 可注入，方便测试时替换成假模型 —— 否则单元测试要下载
    几百 MB 权重，CI 里跑不动。
    """

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL,
        *,
        device: str | None = None,
        batch_size: int = 64,
        max_seq_length: int = 512,
        use_cache: bool = True,
        encode_fn=None,
    ) -> None:
        self.model_name = resolve_model(model_name)
        self.device = device
        self.batch_size = batch_size
        self.max_seq_length = max_seq_length
        self.use_cache = use_cache
        self._encode_fn = encode_fn
        self._model = None
        self._mem: dict[str, np.ndarray] = {}
        self._stats = {"hit": 0, "miss": 0}

    # ---- 模型 ----

    def _load(self):
        if self._model is not None:
            return self._model
        from sentence_transformers import SentenceTransformer

        dev = self.device or pick_device()
        self._model = SentenceTransformer(self.model_name, device=dev)
        self._model.max_seq_length = self.max_seq_length
        return self._model

    @property
    def dim(self) -> int:
        if self._encode_fn is not None:
            return int(self._encode_fn(["probe"]).shape[1])
        return int(self._load().get_sentence_embedding_dimension())

    def _encode_raw(self, texts: list[str]) -> np.ndarray:
        if self._encode_fn is not None:
            vecs = np.asarray(self._encode_fn(texts), dtype=np.float32)
        else:
            vecs = self._load().encode(
                texts,
                batch_size=self.batch_size,
                convert_to_numpy=True,
                normalize_embeddings=False,
                show_progress_bar=False,
            ).astype(np.float32)
        # L2 归一化 -> 余弦相似度 == 点积
        norms = np.linalg.norm(vecs, axis=1, keepdims=True)
        return vecs / np.maximum(norms, 1e-12)

    # ---- 缓存 ----

    def _cache_path(self, key: str) -> Path:
        # 两级目录，避免单目录塞几十万个文件
        return EMB_CACHE / key[:2] / f"{key}.npy"

    def _load_cached(self, key: str) -> np.ndarray | None:
        if key in self._mem:
            return self._mem[key]
        if not self.use_cache:
            return None
        p = self._cache_path(key)
        if p.exists():
            try:
                v = np.load(p)
                self._mem[key] = v
                return v
            except (OSError, ValueError):
                return None
        return None

    def _store(self, key: str, vec: np.ndarray) -> None:
        self._mem[key] = vec
        if not self.use_cache:
            return
        p = self._cache_path(key)
        p.parent.mkdir(parents=True, exist_ok=True)
        tmp = p.with_suffix(".tmp.npy")
        np.save(tmp, vec)
        tmp.replace(p)  # 原子写，防止中断留下半个文件

    # ---- 对外接口 ----

    def encode(self, texts: list[str]) -> np.ndarray:
        """编码一批文本，命中缓存的跳过。"""
        if not texts:
            return np.zeros((0, 1), dtype=np.float32)

        keys = [_hash(self.model_name, t) for t in texts]
        out: list[np.ndarray | None] = []
        todo_idx: list[int] = []
        todo_txt: list[str] = []

        for i, (k, t) in enumerate(zip(keys, texts)):
            v = self._load_cached(k)
            if v is None:
                out.append(None)
                todo_idx.append(i)
                todo_txt.append(t)
                self._stats["miss"] += 1
            else:
                out.append(v)
                self._stats["hit"] += 1

        if todo_txt:
            fresh = self._encode_raw(todo_txt)
            for j, i in enumerate(todo_idx):
                self._store(keys[i], fresh[j])
                out[i] = fresh[j]

        return np.vstack(out)

    @property
    def cache_stats(self) -> dict:
        total = self._stats["hit"] + self._stats["miss"]
        return {
            **self._stats,
            "hit_rate": self._stats["hit"] / total if total else 0.0,
        }