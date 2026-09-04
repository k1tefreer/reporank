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

    def search_files(self, query: str, top_k: int = 30) -> list[str]:
        """返回去重后的文件路径列表。

        chunk 级检索时一个文件可能出现多次，这里按最高分折叠到文件级，
        因为评测和喂给 LLM 的都是文件粒度。
        """
        seen: set[str] = set()
        out: list[str] = []
        # 多取一些再折叠，避免去重后不够 top_k
        for hit in self.search(query, top_k=top_k * 4):
            if hit.path not in seen:
                seen.add(hit.path)
                out.append(hit.path)
                if len(out) >= top_k:
                    break
        return out
