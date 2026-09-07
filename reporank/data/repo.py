"""仓库获取与语料构建。

SWE-bench 只给 (repo, base_commit)，源码要自己拉。策略：
  1. 每个 repo 只 clone 一次（裸仓库缓存在 ~/.cache/reporank/repos）
  2. 每个 instance 用 `git archive <commit>` 导出快照，不做 checkout，
     这样同一个 repo 的不同 commit 可以并行处理，也不会互相污染工作区

一个容易踩的坑：SWE-bench 的仓库都很大（django 几万个文件），
全量建索引很慢。必须先过滤掉测试、文档、二进制、迁移脚本等。
过滤规则本身是要做消融的 —— 过滤太狠会把 gold 文件误杀，
所以评测脚本会统计「gold 文件不在语料里」的比例作为健全性检查。
"""

from __future__ import annotations

import os
import subprocess
import tarfile
import tempfile
from pathlib import Path

from ..retrieval.base import Document

CACHE_DIR = Path(os.environ.get("REPORANK_CACHE", Path.home() / ".cache" / "reporank"))

SOURCE_EXTS = {".py"}  # W1 只做 Python（SWE-bench Lite 全是 Python）

EXCLUDE_DIR_PARTS = {
    ".git", "__pycache__", ".tox", ".venv", "venv", "node_modules",
    "build", "dist", ".eggs", "site-packages",
}

# 注意：不排除 tests/ —— 有些 gold patch 确实改测试辅助文件。
# 是否排除由调用方通过 exclude_tests 控制，方便做消融。
TEST_DIR_PARTS = {"tests", "test", "testing", "_pytest"}

MAX_FILE_BYTES = 400_000


def _run(cmd: list[str], cwd: Path | None = None) -> str:
    r = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(f"{' '.join(cmd)} failed:\n{r.stderr[:2000]}")
    return r.stdout


def ensure_repo(repo: str) -> Path:
    """确保 repo 的裸仓库已缓存，返回路径。repo 形如 'django/django'。"""
    dest = CACHE_DIR / "repos" / (repo.replace("/", "__") + ".git")
    if dest.exists():
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    url = f"https://github.com/{repo}.git"
    _run(["git", "clone", "--bare", "--filter=blob:none", url, str(dest)])
    return dest


def export_snapshot(repo: str, commit: str) -> Path:
    """把指定 commit 导出成一个目录，返回目录路径（带缓存）。"""
    out = CACHE_DIR / "snapshots" / f"{repo.replace('/', '__')}__{commit[:12]}"
    if out.exists():
        return out
    bare = ensure_repo(repo)
    # 部分 clone 下 blob 可能没拉全，按需 fetch
    try:
        _run(["git", "cat-file", "-e", f"{commit}^{{commit}}"], cwd=bare)
    except RuntimeError:
        _run(["git", "fetch", "origin", commit], cwd=bare)

    out.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(suffix=".tar", delete=False) as tmp:
        tar_path = Path(tmp.name)
    try:
        with open(tar_path, "wb") as fh:
            r = subprocess.run(
                ["git", "archive", "--format=tar", commit], cwd=bare, stdout=fh, stderr=subprocess.PIPE
            )
        if r.returncode != 0:
            raise RuntimeError(r.stderr.decode()[:2000])
        with tarfile.open(tar_path) as tf:
            tf.extractall(out)
    finally:
        tar_path.unlink(missing_ok=True)
    return out


def is_source_file(rel_path: str, *, exclude_tests: bool = False) -> bool:
    parts = Path(rel_path).parts
    if any(p in EXCLUDE_DIR_PARTS for p in parts):
        return False
    if exclude_tests and any(p in TEST_DIR_PARTS for p in parts):
        return False
    return Path(rel_path).suffix in SOURCE_EXTS


def build_corpus(
    root: Path,
    *,
    exclude_tests: bool = False,
    chunking: str = "file",
) -> list[Document]:
    """遍历快照目录，产出 Document 列表。

    chunking 决定切分粒度："file" 是 W2 基线，"ast" / "window" 见 index/chunker.py。
    无论哪种策略，Document.path 始终是文件路径，评测层完全不用改。
    """
    from ..index.chunker import get_chunker

    chunker = get_chunker(chunking)
    docs: list[Document] = []
    root = Path(root)
    for p in root.rglob("*"):
        if not p.is_file():
            continue
        rel = p.relative_to(root).as_posix()
        if not is_source_file(rel, exclude_tests=exclude_tests):
            continue
        try:
            if p.stat().st_size > MAX_FILE_BYTES:
                continue
            content = p.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            continue
        docs.extend(chunker(rel, content))
    return docs