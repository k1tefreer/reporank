"""检索结果融合。

W4 验证了互补性：BM25 与稠密检索在 50 条上共同失败仅 4 例，
Oracle 融合上限 R@20 = 0.920，比最佳单一方案（0.880）高 4pt。
W5 的任务是把这 4pt 尽可能兑现。

两种融合方式，做对照：

  RRF (Reciprocal Rank Fusion)
      只用排名，score = Σ wᵢ / (k + rankᵢ)
      优点：无需分数归一化，对异构检索器天然鲁棒
      缺点：丢弃分数大小。排名 1 和排名 2 之间是断崖还是平缓，RRF 看不见

  Score fusion (CombSUM)
      归一化分数后加权求和
      优点：保留了置信度信息
      缺点：BM25 分数无界且可能为负，余弦相似度在 [-1,1]，
            两者量纲完全不同，归一化方式的选择本身就是个变量

本项目的两路检索粒度不同（BM25 文件级、稠密 chunk 级），
这正是 RRF 的典型适用场景——它只需要排序列表，不关心文档怎么来的。
"""

from __future__ import annotations

import numpy as np

from .base import Document, Hit, Retriever
from .bm25 import BM25Retriever
from .dense import PrefilteredDenseRetriever

# RRF 的一个反直觉性质：1/(k+rank) 是凸函数，所以「一路第 1、一路第 3」
# 的总分高于「两路都第 2」。RRF 不奖励稳定的中游表现，它奖励
# 至少在某一路里冲进头部。这直接影响加权策略——把权重给
# 「整体更准」的那一路未必最优，给「偶尔极准」的那一路可能更好。
#
# RRF 的平滑常数。原论文建议 60，但这是在 TREC 数据上定的，
# 未必适用于代码检索——W5 会扫这个值。
# k 越大，各名次的权重越平均；k 越小，头部名次越占优。
DEFAULT_RRF_K = 60


def rrf_fuse(
    rankings: list[list[str]],
    weights: list[float] | None = None,
    k: int = DEFAULT_RRF_K,
) -> list[tuple[str, float]]:
    """倒数排名融合。

    rankings: 每个检索器的有序结果（文件路径列表，越靠前越相关）
    返回融合后的 (path, score) 列表，按分数降序。
    """
    if weights is None:
        weights = [1.0] * len(rankings)
    if len(weights) != len(rankings):
        raise ValueError("weights 与 rankings 长度必须一致")

    scores: dict[str, float] = {}
    for ranking, w in zip(rankings, weights):
        for rank, path in enumerate(ranking, start=1):
            scores[path] = scores.get(path, 0.0) + w / (k + rank)
    return sorted(scores.items(), key=lambda kv: -kv[1])


def _normalize(vals: np.ndarray, method: str) -> np.ndarray:
    if len(vals) == 0:
        return vals
    if method == "minmax":
        lo, hi = float(vals.min()), float(vals.max())
        return (vals - lo) / (hi - lo) if hi > lo else np.zeros_like(vals)
    if method == "zscore":
        mu, sd = float(vals.mean()), float(vals.std())
        return (vals - mu) / sd if sd > 1e-9 else np.zeros_like(vals)
    raise ValueError(f"unknown normalization: {method}")


def score_fuse(
    scored: list[list[tuple[str, float]]],
    weights: list[float] | None = None,
    norm: str = "minmax",
) -> list[tuple[str, float]]:
    """分数融合（CombSUM）。

    必须先归一化：BM25 分数无界且可能为负（高频词 IDF 为负），
    余弦相似度在 [-1, 1]，直接相加等于让量纲大的那一路独裁。

    未出现在某一路结果中的文档，该路贡献记为 0 —— 这相当于
    假设「没召回 = 最低分」，是个偏乐观的假设，因为真实情况可能是
    排在第 201 名而不是完全无关。RRF 没有这个问题。
    """
    if weights is None:
        weights = [1.0] * len(scored)

    total: dict[str, float] = {}
    for lst, w in zip(scored, weights):
        if not lst:
            continue
        paths = [p for p, _ in lst]
        vals = _normalize(np.array([s for _, s in lst], dtype=np.float64), norm)
        for p, v in zip(paths, vals):
            total[p] = total.get(p, 0.0) + w * float(v)
    return sorted(total.items(), key=lambda kv: -kv[1])


class HybridRetriever(Retriever):
    """BM25 + 稠密检索的混合检索器。

    索引时做一次粒度分流：
      BM25  拿按文件合并后的整文件文档（W3 验证过文件级对 BM25 最优）
      Dense 拿原始 chunk（512 token 限制下必须切分）

    这是「同一技术在不同检索范式下价值相反」的直接体现——
    切分对 BM25 有害、对稠密必需，所以两路各用各的粒度。
    """

    def __init__(
        self,
        *,
        method: str = "rrf",
        rrf_k: int = DEFAULT_RRF_K,
        norm: str = "minmax",
        bm25_weight: float = 1.0,
        dense_weight: float = 1.0,
        pool: int = 100,
        bm25_kwargs: dict | None = None,
        embedder=None,
        prefilter_files: int = 200,
    ) -> None:
        self.method = method
        self.rrf_k = rrf_k
        self.norm = norm
        self.weights = [bm25_weight, dense_weight]
        self.pool = pool  # 每路取多少候选参与融合
        self.bm25_kwargs = bm25_kwargs or {"k1": 1.2, "b": 0.75, "path_boost": 10}

        self._bm25 = BM25Retriever(**self.bm25_kwargs)
        self._dense = PrefilteredDenseRetriever(
            prefilter_files=prefilter_files,
            embedder=embedder,
            bm25_kwargs=self.bm25_kwargs,
        )

        tag = method if method == "rrf" else f"{method}-{norm}"
        wtag = ""
        if bm25_weight != 1.0 or dense_weight != 1.0:
            wtag = f"-w{bm25_weight:g}:{dense_weight:g}"
        ktag = f"-k{rrf_k}" if method == "rrf" and rrf_k != DEFAULT_RRF_K else ""
        self.name = f"hybrid-{tag}{ktag}{wtag}"

    def index(self, docs: list[Document]) -> None:
        # 稠密侧直接吃 chunk
        self._dense.index(docs)

        # BM25 侧按 path 合并回整文件
        by_path: dict[str, list[str]] = {}
        for d in docs:
            by_path.setdefault(d.path, []).append(d.content)
        self._bm25.index(
            [Document(doc_id=p, path=p, content="\n".join(cs)) for p, cs in by_path.items()]
        )

    def _arm_results(self, query: str) -> tuple[list[str], list[str], list, list]:
        bm_files = self._bm25.search_files(query, top_k=self.pool)
        dn_files = self._dense.search_files(query, top_k=self.pool)

        # 分数融合需要文件级分数：按 chunk 最高分折叠
        def file_scores(r, files):
            best: dict[str, float] = {}
            for h in r.search(query, top_k=self.pool * 20):
                if h.score > best.get(h.path, -1e18):
                    best[h.path] = h.score
            return [(f, best.get(f, 0.0)) for f in files]

        return bm_files, dn_files, file_scores(self._bm25, bm_files), file_scores(self._dense, dn_files)

    def search(self, query: str, top_k: int = 30) -> list[Hit]:
        bm_files, dn_files, bm_scored, dn_scored = self._arm_results(query)

        if self.method == "rrf":
            fused = rrf_fuse([bm_files, dn_files], self.weights, k=self.rrf_k)
        elif self.method == "score":
            fused = score_fuse([bm_scored, dn_scored], self.weights, norm=self.norm)
        elif self.method == "bm25_only":  # 健全性检查用
            fused = [(p, -i) for i, p in enumerate(bm_files)]
        elif self.method == "dense_only":
            fused = [(p, -i) for i, p in enumerate(dn_files)]
        else:
            raise ValueError(f"unknown fusion method: {self.method}")

        return [Hit(doc_id=p, path=p, score=float(s)) for p, s in fused[:top_k]]

    def search_files(self, query: str, top_k: int = 30) -> list[str]:
        # 融合结果已是文件级，无需再折叠
        return [h.path for h in self.search(query, top_k=top_k)]