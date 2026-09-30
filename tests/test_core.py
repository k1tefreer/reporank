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


class _FakeEmbedder:
    """确定性假嵌入，让稠密检索的逻辑可以脱离模型下载来测试。

    用词袋哈希投影成向量：内容相似 -> 向量相似。够用来验证
    索引/检索/聚合的正确性，不验证真实模型质量。
    """

    model_name = "fake/test-model"

    def __init__(self, dim=32):
        self._dim = dim
        self.calls = 0

    def encode(self, texts):
        import numpy as np
        self.calls += len(texts)
        out = np.zeros((len(texts), self._dim), dtype=np.float32)
        for i, t in enumerate(texts):
            for tok in t.lower().split():
                out[i, hash(tok) % self._dim] += 1.0
        n = np.linalg.norm(out, axis=1, keepdims=True)
        return out / np.maximum(n, 1e-12)


class TestDenseRetriever:
    def _docs(self):
        from reporank.retrieval.base import Document
        return [
            Document("a.py", "a.py", "token expiry validation ttl"),
            Document("b.py", "b.py", "http header parsing case"),
            Document("c.py", "c.py", "database query builder sql"),
        ]

    def test_finds_semantically_closest(self):
        from reporank.retrieval.dense import DenseRetriever
        r = DenseRetriever(embedder=_FakeEmbedder())
        r.index(self._docs())
        assert r.search("token expiry ttl", top_k=1)[0].path == "a.py"

    def test_empty_corpus_is_safe(self):
        from reporank.retrieval.dense import DenseRetriever
        r = DenseRetriever(embedder=_FakeEmbedder())
        r.index([])
        assert r.search("anything") == []

    def test_scores_are_cosine_bounded(self):
        from reporank.retrieval.dense import DenseRetriever
        r = DenseRetriever(embedder=_FakeEmbedder())
        r.index(self._docs())
        assert all(-1.01 <= h.score <= 1.01 for h in r.search("sql query", top_k=3))

    def test_top_k_respects_limit(self):
        from reporank.retrieval.dense import DenseRetriever
        r = DenseRetriever(embedder=_FakeEmbedder())
        r.index(self._docs())
        assert len(r.search("token", top_k=2)) == 2


class TestPrefilteredDense:
    def _docs(self):
        from reporank.retrieval.base import Document
        return [
            Document(f"mod{i}.py::c", f"mod{i}.py", f"unrelated filler content number {i}")
            for i in range(30)
        ] + [
            Document("auth/token.py::c", "auth/token.py", "token expiry validation ttl"),
        ]

    def test_prefilter_limits_encoding_cost(self):
        from reporank.retrieval.dense import PrefilteredDenseRetriever
        emb = _FakeEmbedder()
        r = PrefilteredDenseRetriever(prefilter_files=5, embedder=emb)
        r.index(self._docs())
        r.search("token expiry ttl", top_k=3)
        # 只编码粗筛出的 5 个文件 + 1 次查询，远少于 31 个文档
        assert emb.calls <= 6

    def test_prefilter_recall_reports_ceiling(self):
        from reporank.retrieval.dense import PrefilteredDenseRetriever
        r = PrefilteredDenseRetriever(prefilter_files=5, embedder=_FakeEmbedder())
        r.index(self._docs())
        rec = r.prefilter_recall("token expiry validation ttl", ["auth/token.py"])
        assert 0.0 <= rec <= 1.0


class TestEmbedderCache:
    def test_cache_avoids_recompute(self, tmp_path, monkeypatch):
        import reporank.index.embedder as E
        monkeypatch.setattr(E, "EMB_CACHE", tmp_path / "emb")
        calls = {"n": 0}

        def fake(texts):
            import numpy as np
            calls["n"] += len(texts)
            return np.ones((len(texts), 8), dtype=np.float32)

        e1 = E.Embedder("m", encode_fn=fake)
        e1.encode(["alpha", "beta"])
        assert calls["n"] == 2

        e2 = E.Embedder("m", encode_fn=fake)   # 新实例，只能命中磁盘缓存
        e2.encode(["alpha", "beta"])
        assert calls["n"] == 2                  # 没有新增编码
        assert e2.cache_stats["hit"] == 2

    def test_vectors_are_normalised(self, tmp_path, monkeypatch):
        import numpy as np
        import reporank.index.embedder as E
        monkeypatch.setattr(E, "EMB_CACHE", tmp_path / "emb")
        e = E.Embedder("m", encode_fn=lambda t: np.full((len(t), 4), 3.0, dtype=np.float32))
        v = e.encode(["x"])
        assert abs(float(np.linalg.norm(v[0])) - 1.0) < 1e-5


class TestFusion:
    def test_rrf_rewards_appearing_in_both_lists(self):
        from reporank.retrieval.fusion import rrf_fuse
        # z.py 两路都出现，a.py / b.py 各只出现一次
        fused = dict(rrf_fuse([["a.py", "z.py"], ["b.py", "z.py"]], k=10))
        assert fused["z.py"] > fused["a.py"]
        assert fused["z.py"] > fused["b.py"]

    def test_rrf_is_convex_extremes_beat_middle(self):
        """RRF 的一个反直觉性质：1/x 是凸函数，所以「一路第1一路第3」
        的总分高于「两路都第2」。RRF 并不奖励稳定的中游表现，
        它奖励至少在一路里冲进头部。这会影响加权策略的设计。"""
        from reporank.retrieval.fusion import rrf_fuse
        fused = dict(rrf_fuse([["a.py", "b.py", "c.py"], ["c.py", "b.py", "a.py"]], k=10))
        assert fused["a.py"] > fused["b.py"]   # 1/11 + 1/13 > 2/12
        assert fused["c.py"] > fused["b.py"]

    def test_rrf_k_controls_head_emphasis(self):
        from reporank.retrieval.fusion import rrf_fuse
        r = [["a.py", "b.py"]]
        small_k = dict(rrf_fuse(r, k=1))
        large_k = dict(rrf_fuse(r, k=1000))
        # k 小 -> 头部名次权重差距大；k 大 -> 各名次趋于平均
        assert (small_k["a.py"] / small_k["b.py"]) > (large_k["a.py"] / large_k["b.py"])

    def test_rrf_weights_shift_ranking(self):
        from reporank.retrieval.fusion import rrf_fuse
        rankings = [["a.py"], ["b.py"]]   # 两路各自独有一个文档
        assert dict(rrf_fuse(rankings, [3.0, 1.0], k=10))["a.py"] > \
               dict(rrf_fuse(rankings, [3.0, 1.0], k=10))["b.py"]
        assert dict(rrf_fuse(rankings, [1.0, 3.0], k=10))["b.py"] > \
               dict(rrf_fuse(rankings, [1.0, 3.0], k=10))["a.py"]

    def test_rrf_rejects_mismatched_weights(self):
        from reporank.retrieval.fusion import rrf_fuse
        with pytest.raises(ValueError):
            rrf_fuse([["a"], ["b"]], [1.0])

    def test_score_fuse_normalizes_across_scales(self):
        from reporank.retrieval.fusion import score_fuse
        # BM25 量纲 ~40，余弦 ~0.8。不归一化的话 BM25 会独裁
        bm = [("a.py", 40.0), ("b.py", 39.0)]
        dn = [("b.py", 0.9), ("a.py", 0.1)]
        fused = dict(score_fuse([bm, dn], norm="minmax"))
        assert abs(fused["a.py"] - fused["b.py"]) < 0.01   # 两路互相抵消，接近平局

    def test_score_fuse_handles_constant_scores(self):
        from reporank.retrieval.fusion import score_fuse
        out = dict(score_fuse([[("a.py", 5.0), ("b.py", 5.0)]], norm="minmax"))
        assert out["a.py"] == out["b.py"] == 0.0

    def test_zscore_normalisation(self):
        from reporank.retrieval.fusion import score_fuse
        out = score_fuse([[("a.py", 10.0), ("b.py", 0.0)]], norm="zscore")
        assert out[0][0] == "a.py"

    def test_unknown_norm_raises(self):
        from reporank.retrieval.fusion import score_fuse
        with pytest.raises(ValueError):
            score_fuse([[("a.py", 1.0)]], norm="nope")


class TestHybridRetriever:
    def _docs(self):
        from reporank.retrieval.base import Document
        return [
            Document("auth/token.py::A", "auth/token.py", "token expiry ttl validation"),
            Document("auth/token.py::B", "auth/token.py", "refresh rotate secret"),
            Document("http/req.py::A", "http/req.py", "header parsing case sensitive"),
            Document("db/query.py::A", "db/query.py", "sql builder where clause"),
        ]

    def test_bm25_arm_sees_file_level_docs(self):
        from reporank.retrieval.fusion import HybridRetriever
        r = HybridRetriever(embedder=_FakeEmbedder())
        r.index(self._docs())
        # 4 个 chunk 合并成 3 个文件
        assert len(r._bm25._docs) == 3

    def test_sanity_single_arm_modes(self):
        from reporank.retrieval.fusion import HybridRetriever
        for m in ("bm25_only", "dense_only"):
            r = HybridRetriever(method=m, embedder=_FakeEmbedder())
            r.index(self._docs())
            assert len(r.search_files("token expiry", top_k=3)) > 0

    def test_fusion_returns_file_paths_not_chunk_ids(self):
        from reporank.retrieval.fusion import HybridRetriever
        r = HybridRetriever(embedder=_FakeEmbedder())
        r.index(self._docs())
        assert all("::" not in p for p in r.search_files("token expiry", top_k=3))

    def test_no_duplicate_files_in_output(self):
        from reporank.retrieval.fusion import HybridRetriever
        r = HybridRetriever(embedder=_FakeEmbedder())
        r.index(self._docs())
        files = r.search_files("token", top_k=5)
        assert len(files) == len(set(files))