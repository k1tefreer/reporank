import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest

from reporank.data.swebench import parse_patch_files
from reporank.eval.metrics import ndcg_at_k, precision_at_k, recall_at_k, reciprocal_rank
from reporank.index.tokenizer import split_identifier, tokenize, tokenize_path


class TestTokenizer:
    def test_camel_case(self):
        assert split_identifier("getUserName") == ["getusername", "get", "user", "name"]

    def test_acronym(self):
        assert split_identifier("HTTPResponse") == ["httpresponse", "http", "response"]

    def test_single_word_not_duplicated(self):
        assert split_identifier("token") == ["token"]

    def test_snake_case(self):
        toks = tokenize("user_id = 1")
        assert "user" in toks and "id" in toks

    def test_stopwords_dropped(self):
        assert "if" not in tokenize("if x else y")

    def test_path_boost_repeats(self):
        once = tokenize_path("src/auth/token.py", repeat=1)
        thrice = tokenize_path("src/auth/token.py", repeat=3)
        assert len(thrice) == 3 * len(once)


class TestPatchParsing:
    def test_diff_git(self):
        patch = (
            "diff --git a/src/a.py b/src/a.py\n"
            "--- a/src/a.py\n+++ b/src/a.py\n@@ -1 +1 @@\n-x\n+y\n"
        )
        assert parse_patch_files(patch) == ["src/a.py"]

    def test_multi_file(self):
        patch = (
            "diff --git a/x.py b/x.py\n@@\n"
            "diff --git a/y.py b/y.py\n@@\n"
        )
        assert parse_patch_files(patch) == ["x.py", "y.py"]

    def test_new_file_dev_null_filtered(self):
        patch = "--- /dev/null\n+++ b/new.py\n@@ -0,0 +1 @@\n+x\n"
        assert parse_patch_files(patch) == ["new.py"]

    def test_empty(self):
        assert parse_patch_files("") == []


class TestMetrics:
    ranked = ["a.py", "b.py", "c.py", "d.py"]

    def test_recall(self):
        assert recall_at_k(self.ranked, ["b.py"], 3) == 1.0
        assert recall_at_k(self.ranked, ["d.py"], 3) == 0.0
        assert recall_at_k(self.ranked, ["a.py", "d.py"], 4) == 1.0
        assert recall_at_k(self.ranked, ["a.py", "d.py"], 2) == 0.5

    def test_precision(self):
        assert precision_at_k(self.ranked, ["a.py"], 2) == 0.5

    def test_mrr(self):
        assert reciprocal_rank(self.ranked, ["b.py"]) == 0.5
        assert reciprocal_rank(self.ranked, ["zzz.py"]) == 0.0

    def test_ndcg_perfect(self):
        assert ndcg_at_k(["a.py"], ["a.py"], 5) == pytest.approx(1.0)

    def test_ndcg_decreases_with_rank(self):
        hi = ndcg_at_k(self.ranked, ["a.py"], 4)
        lo = ndcg_at_k(self.ranked, ["d.py"], 4)
        assert hi > lo

    def test_path_normalisation(self):
        assert recall_at_k(["./a.py"], ["a.py"], 1) == 1.0


class TestChunker:
    SRC = '''"""Module docstring."""
import os

CONST = 1


def top_level(a, b):
    """Adds."""
    return a + b


class Greeter:
    """A greeter."""

    prefix = "hi"

    @property
    def name(self):
        return self._name

    async def greet(self, who):
        return f"{self.prefix} {who}"
'''

    def test_ast_produces_expected_symbols(self):
        from reporank.index.chunker import chunk_by_ast
        docs = chunk_by_ast("src/g.py", self.SRC)
        quals = {d.meta["qualname"] for d in docs}
        assert "top_level" in quals
        assert "Greeter" in quals              # 类头
        assert "Greeter.name" in quals         # 带装饰器的方法
        assert "Greeter.greet" in quals        # async 方法
        assert "<module>" in quals             # import / 常量残余

    def test_all_chunks_keep_file_path(self):
        from reporank.index.chunker import chunk_by_ast
        docs = chunk_by_ast("src/g.py", self.SRC)
        assert all(d.path == "src/g.py" for d in docs)
        assert len({d.doc_id for d in docs}) == len(docs)  # doc_id 唯一

    def test_chunk_carries_context_header(self):
        from reporank.index.chunker import chunk_by_ast
        docs = chunk_by_ast("src/g.py", self.SRC)
        d = next(d for d in docs if d.meta["qualname"] == "Greeter.greet")
        assert "src/g.py" in d.content and "Greeter.greet" in d.content

    def test_module_leftovers_keep_imports(self):
        from reporank.index.chunker import chunk_by_ast
        docs = chunk_by_ast("src/g.py", self.SRC)
        mod = next(d for d in docs if d.meta["qualname"] == "<module>")
        assert "import os" in mod.content and "CONST" in mod.content

    def test_syntax_error_falls_back_to_whole_file(self):
        from reporank.index.chunker import chunk_by_ast
        docs = chunk_by_ast("bad.py", "def broken(:\n  pass")
        assert len(docs) == 1 and docs[0].doc_id == "bad.py"

    def test_window_chunker_overlaps_and_covers(self):
        from reporank.index.chunker import chunk_by_window
        src = "\n".join(f"line{i}" for i in range(200))
        docs = chunk_by_window("a.py", src, size=60, stride=45)
        assert len(docs) > 1
        assert all(d.path == "a.py" for d in docs)

    def test_registry(self):
        from reporank.index.chunker import get_chunker
        assert get_chunker("ast") is not None
        with pytest.raises(ValueError):
            get_chunker("nope")


class TestAggregation:
    """chunk -> file 聚合方式的行为对照。

    注意：查询词必须只出现在部分文档里。BM25Okapi 对出现在几乎所有
    文档中的词会给出负 IDF，此时 sum 聚合反而惩罚 chunk 多的文件 ——
    这是真实语料里也会遇到的陷阱，测试用例要避开它才能测到本意。
    """

    def _make(self, agg):
        from reporank.retrieval.base import Document
        from reporank.retrieval.bm25 import BM25Retriever
        docs = [
            Document("big.py::a", "big.py", "widget handler"),
            Document("big.py::b", "big.py", "widget parser"),
            Document("big.py::c", "big.py", "widget writer"),
            Document("small.py::x", "small.py", "widget"),
            Document("noise1.py::x", "noise1.py", "unrelated content here"),
            Document("noise2.py::x", "noise2.py", "totally different stuff"),
            Document("noise3.py::x", "noise3.py", "nothing to see"),
        ]
        r = BM25Retriever(agg=agg, path_boost=0)
        r.index(docs)
        return r

    def test_sum_favours_files_with_many_matching_chunks(self):
        # sum 奖励 chunk 多的文件，max 不受文件长度影响 —— 正是要做对照的原因
        assert self._make("sum").search_files("widget", 2)[0] == "big.py"

    def test_max_is_length_neutral(self):
        r = self._make("max")
        top2 = set(r.search_files("widget", 2))
        assert top2 == {"big.py", "small.py"}

    def test_topk_sum_between_the_two(self):
        assert self._make("topk_sum").search_files("widget", 1)[0] == "big.py"