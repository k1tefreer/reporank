"""SWE-bench 数据加载与 ground truth 抽取。

核心思路：SWE-bench 每条实例都带一个 `patch`（gold patch，即人类真实
提交的修复）。从这个 unified diff 里解析出被修改的文件路径，就是
「文件定位」任务的标准答案 —— 免费的 ground truth，不需要跑测试、
不需要 Docker。这是整个 W1-W7 能在笔记本上跑完的前提。

数据来源（二选一）：
  - HuggingFace: princeton-nlp/SWE-bench_Lite  (300 条)
  - 本地 JSONL（离线/受限网络环境，或用 fixtures 里的样例先跑通）
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator

# `diff --git a/path/to/x.py b/path/to/x.py`
_DIFF_GIT = re.compile(r"^diff --git a/(?P<a>.+?) b/(?P<b>.+?)$", re.MULTILINE)
# `--- a/path` / `+++ b/path`
_MINUS = re.compile(r"^--- (?:a/)?(?P<p>.+?)(?:\t.*)?$", re.MULTILINE)
_PLUS = re.compile(r"^\+\+\+ (?:b/)?(?P<p>.+?)(?:\t.*)?$", re.MULTILINE)

_DEV_NULL = {"/dev/null", "dev/null"}


def parse_patch_files(patch: str) -> list[str]:
    """从 unified diff 中解析出被修改的文件路径。

    优先用 `diff --git` 行（最可靠）；缺失时回退到 ---/+++ 行。
    新增文件的 `---` 是 /dev/null，删除文件的 `+++` 是 /dev/null，都要过滤。
    """
    if not patch:
        return []

    paths: list[str] = []
    for m in _DIFF_GIT.finditer(patch):
        # a/ 和 b/ 通常相同；重命名时不同，两个都收
        for key in ("a", "b"):
            p = m.group(key)
            if p and p not in _DEV_NULL:
                paths.append(p)

    if not paths:
        for rx in (_MINUS, _PLUS):
            for m in rx.finditer(patch):
                p = m.group("p")
                if p and p not in _DEV_NULL:
                    paths.append(p)

    # 去重保序
    seen: set[str] = set()
    out: list[str] = []
    for p in paths:
        if p not in seen:
            seen.add(p)
            out.append(p)
    return out


@dataclass(slots=True)
class Instance:
    """一条 SWE-bench 实例。"""

    instance_id: str
    repo: str  # "django/django"
    base_commit: str
    problem_statement: str  # issue 正文 —— 这就是我们的 query
    patch: str  # gold patch —— ground truth 来源
    test_patch: str = ""
    hints_text: str = ""
    _gold_files: list[str] | None = field(default=None, repr=False)

    @property
    def gold_files(self) -> list[str]:
        """需要被修改的源码文件（标准答案）。

        注意：只用 `patch`，不用 `test_patch`。test_patch 是测试文件的改动，
        属于另一个任务（test localization），指标要分开算。
        """
        if self._gold_files is None:
            self._gold_files = parse_patch_files(self.patch)
        return self._gold_files

    @property
    def query(self) -> str:
        """检索查询。W1 直接用 issue 原文；后续可实验查询扩展/改写。"""
        return self.problem_statement

    @classmethod
    def from_dict(cls, d: dict) -> "Instance":
        return cls(
            instance_id=d["instance_id"],
            repo=d["repo"],
            base_commit=d["base_commit"],
            problem_statement=d.get("problem_statement", ""),
            patch=d.get("patch", ""),
            test_patch=d.get("test_patch", ""),
            hints_text=d.get("hints_text", ""),
        )


def load_jsonl(path: str | Path) -> list[Instance]:
    """从本地 JSONL 加载。"""
    out: list[Instance] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(Instance.from_dict(json.loads(line)))
    return out


def load_hf(
    dataset: str = "princeton-nlp/SWE-bench_Lite",
    split: str = "test",
) -> list[Instance]:
    """从 HuggingFace 加载。需要 `pip install datasets` 且能访问 HF。

    国内访问建议先设 HF_ENDPOINT=https://hf-mirror.com
    """
    from datasets import load_dataset  # 延迟导入，避免离线环境报错

    ds = load_dataset(dataset, split=split)
    return [Instance.from_dict(dict(row)) for row in ds]


def dump_jsonl(instances: list[Instance], path: str | Path) -> None:
    """缓存到本地，避免每次重新下载。"""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for ins in instances:
            f.write(
                json.dumps(
                    {
                        "instance_id": ins.instance_id,
                        "repo": ins.repo,
                        "base_commit": ins.base_commit,
                        "problem_statement": ins.problem_statement,
                        "patch": ins.patch,
                        "test_patch": ins.test_patch,
                        "hints_text": ins.hints_text,
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )


def iter_instances(source: str | Path) -> Iterator[Instance]:
    """统一入口：路径存在就读本地，否则当作 HF dataset 名。"""
    p = Path(source)
    if p.exists():
        yield from load_jsonl(p)
    else:
        yield from load_hf(str(source))
