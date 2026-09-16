"""稠密检索。

W4 要解决的是 W2/W3 反复确认的那 18% 硬失败：无论怎么调参、怎么切分，
R@20 恒定在 0.820。这些实例的 gold 文件与 issue 没有共同词项 ——
issue 用用户视角描述现象，代码用实现术语命名，BM25 基于词项匹配，
原理上救不了。稠密检索匹配的是语义，是对症的解法。

两个变体：

  DenseRetriever            全语料编码。诚实但昂贵，是方法上限的度量
  PrefilteredDenseRetriever BM25 先粗筛 N 个文件，只对这些文件的 chunk
                            编码。便宜，但召回上限被 BM25@N 卡死 ——
                            这个上限必须如实报出来，否则数字有水分
"""

from __future__ import annotations

import numpy as np

from ..index.embedder import DEFAULT_MODEL, Embedder
from .base import Document, Hit, Retriever
from .bm25 import BM25Retriever


class DenseRetriever(Retriever):
    """全语料稠密检索。

    语料规模在几万 chunk 以内时，暴力矩阵乘法足够快且无近似误差，
    不需要 ANN 索引。W5 接入 Milvus 后再换成 HNSW，届时可以对比
    「精确检索 vs 近似检索」的召回损失与延迟收益。
    """

    def __init__(
        self,
        *,
        model_name: str = DEFAULT_MODEL,
        embedder: Embedder | None = None,
        query_prefix: str = "",
        doc_prefix: str = "",
        max_query_chars: int = 4000,
        agg: str = "max",
        agg_topk: int = 3,
    ) -> None:
        self.embedder = embedder or Embedder(model_name)
        self.query_prefix = query_prefix
        self.doc_prefix = doc_prefix
        self.max_query_chars = max_query_chars
        self.agg = agg
        self.agg_topk = agg_topk
        self.name = f"dense[{self.embedder.model_name.split('/')[-1]}]"
        if agg != "max":
            self.name += f"-{agg}"
        self._docs: list[Document] = []
        self._mat: np.ndarray | None = None

    def index(self, docs: list[Document]) -> None:
        self._docs = docs
        if not docs:
            self._mat = None
            return
        texts = [self.doc_prefix + d.content for d in docs]
        self._mat = self.embedder.encode(texts)

    def search(self, query: str, top_k: int = 30) -> list[Hit]:
        if self._mat is None or not self._docs:
            return []
        q = self.embedder.encode([self.query_prefix + query[: self.max_query_chars]])[0]
        # 向量已 L2 归一化，点积即余弦相似度
        scores = self._mat @ q
        k = min(top_k, len(scores))
        # argpartition 取 top-k 再排序，比全排序快得多
        idx = np.argpartition(-scores, k - 1)[:k]
        idx = idx[np.argsort(-scores[idx])]
        return [
            Hit(doc_id=self._docs[i].doc_id, path=self._docs[i].path, score=float(scores[i]))
            for i in idx
        ]


class PrefilteredDenseRetriever(Retriever):
    """BM25 粗筛 + 稠密精排。

    成本从「编码全语料」降到「编码 N 个文件的 chunk」，通常是
    一到两个数量级的差别。代价是召回上限被 BM25@N 卡死。

    这个上限不是隐患而是必须报告的事实：如果 BM25 在 Top-200 文件里
    都没召回 gold，稠密层再强也无力回天。评测时要同时报告
    prefilter_recall，让读者知道天花板在哪。
    """

    def __init__(
        self,
        *,
        prefilter_files: int = 200,
        bm25_kwargs: dict | None = None,
        model_name: str = DEFAULT_MODEL,
        embedder: Embedder | None = None,
        query_prefix: str = "",
        doc_prefix: str = "",
        agg: str = "max",
        agg_topk: int = 3,
    ) -> None:
        self.prefilter_files = prefilter_files
        self.bm25_kwargs = bm25_kwargs or {"k1": 1.2, "b": 0.75, "path_boost": 10}
        self.embedder = embedder or Embedder(model_name)
        self.query_prefix = query_prefix
        self.doc_prefix = doc_prefix
        self.agg = agg
        self.agg_topk = agg_topk
        self.name = f"prefilter{prefilter_files}+dense[{self.embedder.model_name.split('/')[-1]}]"
        self._docs: list[Document] = []
        self._bm25: BM25Retriever | None = None

    def index(self, docs: list[Document]) -> None:
        """只建 BM25 索引。稠密编码推迟到 search 时按需进行 ——
        因为要先知道查询是什么，才知道该编码哪些文件。"""
        self._docs = docs
        # BM25 侧始终按文件粒度粗筛：粗筛阶段整文件信号更完整，
        # 这正是 W3 验证过的结论
        by_path: dict[str, list[str]] = {}
        for d in docs:
            by_path.setdefault(d.path, []).append(d.content)
        file_docs = [
            Document(doc_id=p, path=p, content="\n".join(cs)) for p, cs in by_path.items()
        ]
        self._bm25 = BM25Retriever(**self.bm25_kwargs)
        self._bm25.index(file_docs)

    def search(self, query: str, top_k: int = 30) -> list[Hit]:
        if self._bm25 is None or not self._docs:
            return []

        keep = set(self._bm25.search_files(query, top_k=self.prefilter_files))
        cand = [d for d in self._docs if d.path in keep]
        if not cand:
            return []

        mat = self.embedder.encode([self.doc_prefix + d.content for d in cand])
        q = self.embedder.encode([self.query_prefix + query[:4000]])[0]
        scores = mat @ q

        k = min(top_k, len(scores))
        idx = np.argpartition(-scores, k - 1)[:k]
        idx = idx[np.argsort(-scores[idx])]
        return [
            Hit(doc_id=cand[i].doc_id, path=cand[i].path, score=float(scores[i])) for i in idx
        ]

    def prefilter_recall(self, query: str, gold: list[str]) -> float:
        """粗筛阶段的召回率 —— 稠密层的性能天花板。"""
        if self._bm25 is None:
            return 0.0
        keep = set(self._bm25.search_files(query, top_k=self.prefilter_files))
        g = set(gold)
        return len(keep & g) / len(g) if g else float("nan")