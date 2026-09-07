"""检索器统一接口。

后续每一种方案 —— Dense (W4)、RRF 融合 (W5)、Cross-Encoder 重排 (W6)、
调用图 PageRank (W7) —— 都实现同一个接口，这样评测脚本一行不用改，
消融实验就是换一个 `--retriever` 参数。

这个抽象是 W1 最重要的设计决策：先把接口钉死，后面才能快速做对比实验。
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass(slots=True)
class Document:
    """索引单元。W1 是整个文件，W3 起会变成 AST 切出来的 chunk。"""

    doc_id: str  # W1 = 文件路径；W3 后 = "path::func_name"
    path: str  # 归属文件路径（评测始终按文件粒度对齐）
    content: str
    meta: dict | None = None


@dataclass(slots=True)
class Hit:
    doc_id: str
    path: str
    score: float


class Retriever(ABC):
    """所有检索器的基类。"""

    name: str = "base"

    @abstractmethod
    def index(self, docs: list[Document]) -> None:
        """建索引。每个 repo 一次。"""

    @abstractmethod
    def search(self, query: str, top_k: int = 30) -> list[Hit]:
        """检索，返回按分数降序的 Hit。"""

    # chunk -> 文件的分数聚合方式。切分之后这是个关键超参：
    #   max      取该文件所有 chunk 的最高分。不受文件长度影响，是安全默认值
    #   sum      所有 chunk 分数求和。会偏向 chunk 多的大文件，通常更差
    #   topk_sum 取前 N 个 chunk 求和。折中方案：多处相关能加分，但不会
    #            让大文件靠数量取胜
    agg: str = "max"
    agg_topk: int = 3

    def search_files(self, query: str, top_k: int = 30) -> list[str]:
        """把 chunk 级结果折叠成文件级排序。

        W1 时一个文件就是一个 doc，折叠是恒等操作。
        W3 切分之后一个文件会有几十个 chunk，折叠方式直接影响排序。
        """
        # 多取一些再折叠：切分后同一文件会占掉大量名额
        pool = self.search(query, top_k=max(top_k * 20, 200))

        scores: dict[str, float] = {}
        buckets: dict[str, list[float]] = {}
        for hit in pool:
            buckets.setdefault(hit.path, []).append(hit.score)

        for path, vals in buckets.items():
            if self.agg == "sum":
                scores[path] = sum(vals)
            elif self.agg == "topk_sum":
                scores[path] = sum(sorted(vals, reverse=True)[: self.agg_topk])
            else:
                scores[path] = max(vals)

        ranked = sorted(scores, key=lambda p: -scores[p])
        return ranked[:top_k]

    