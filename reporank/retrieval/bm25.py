"""BM25 稀疏检索 —— W1 基线。

公开参照系（SWE-bench Lite，文件级定位）：
  BM25-Lucene           约 33.67%
  Agentless 1.5 + GPT-4o 约 69.67%

我们的目标是从朴素 BM25 出发，逐步逼近并尝试超过后者，
且每一步提升都能归因到具体机制。

两个可切换开关（用于 W2 消融）：
  - naive_tokenizer: 用最朴素的 split() 分词，作为「代码分词器有没有用」的对照组
  - path_boost     : 路径 token 重复次数，0 表示不加权
"""

from __future__ import annotations

from rank_bm25 import BM25Okapi

from ..index.tokenizer import tokenize_document, tokenize_query
from .base import Document, Hit, Retriever


def _naive_tokenize(text: str) -> list[str]:
    """对照组：不做驼峰拆分、不做停用词过滤。"""
    return [t.lower() for t in text.split() if t]


class BM25Retriever(Retriever):
    name = "bm25"

    def __init__(
        self,
        *,
        k1: float = 1.5,
        b: float = 0.75,
        path_boost: int = 3,
        naive_tokenizer: bool = False,
        max_doc_tokens: int = 20_000,
        agg: str = "max",
        agg_topk: int = 3,
    ) -> None:
        self.k1 = k1
        self.b = b
        self.path_boost = path_boost
        self.naive_tokenizer = naive_tokenizer
        self.max_doc_tokens = max_doc_tokens
        self.agg = agg
        self.agg_topk = agg_topk
        self._bm25: BM25Okapi | None = None
        self._docs: list[Document] = []

        suffix = []
        if naive_tokenizer:
            suffix.append("naive")
        if path_boost != 3:
            suffix.append(f"pb{path_boost}")
        if agg != "max":
            suffix.append(agg)
        if suffix:
            self.name = "bm25-" + "-".join(suffix)

    def _tok_doc(self, doc: Document) -> list[str]:
        if self.naive_tokenizer:
            toks = _naive_tokenize(doc.path + " " + doc.content)
        else:
            toks = tokenize_document(doc.path, doc.content, path_boost=self.path_boost)
        # 超大文件截断，防止个别文件拖垮建索引速度
        return toks[: self.max_doc_tokens]

    def index(self, docs: list[Document]) -> None:
        self._docs = docs
        corpus = [self._tok_doc(d) for d in docs]
        # rank_bm25 对空文档会算出 nan，塞一个占位 token
        corpus = [c if c else ["__empty__"] for c in corpus]
        self._bm25 = BM25Okapi(corpus, k1=self.k1, b=self.b)

    def search(self, query: str, top_k: int = 30) -> list[Hit]:
        if self._bm25 is None:
            raise RuntimeError("index() must be called before search()")
        q = _naive_tokenize(query) if self.naive_tokenizer else tokenize_query(query)
        if not q:
            return []
        scores = self._bm25.get_scores(q)
        order = sorted(range(len(scores)), key=lambda i: -scores[i])[:top_k]
        return [
            Hit(doc_id=self._docs[i].doc_id, path=self._docs[i].path, score=float(scores[i]))
            for i in order
        ]

    