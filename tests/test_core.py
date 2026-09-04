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
