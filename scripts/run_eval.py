"""评测入口。

用法：
  # 1. 先在离线 fixture 上跑通管道
  python scripts/make_fixture.py
  python scripts/run_eval.py --data fixtures/sample_instances.jsonl \
      --local-corpus fixtures/mini_repo --retriever bm25

  # 2. 消融：关掉代码分词器，看差多少
  python scripts/run_eval.py --data fixtures/sample_instances.jsonl \
      --local-corpus fixtures/mini_repo --retriever bm25-naive

  # 3. 换真实数据（需要能访问 HF 和 GitHub）
  python scripts/prepare_data.py --out data/swebench_lite.jsonl
  python scripts/run_eval.py --data data/swebench_lite.jsonl --retriever bm25 --limit 50
"""

from __future__ import annotations

import argparse
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from reporank.data import repo as repo_mod
from reporank.data.swebench import Instance, load_jsonl
from reporank.eval.metrics import format_report
from reporank.eval.runner import run_eval, save_results
from reporank.retrieval.bm25 import BM25Retriever

# 检索器注册表。W3-W7 每加一种方法，在这里加一行即可，
# 评测脚本其他部分完全不用动。
RETRIEVERS = {
    "bm25": lambda: BM25Retriever(),
    "bm25-naive": lambda: BM25Retriever(naive_tokenizer=True),
    "bm25-nopath": lambda: BM25Retriever(path_boost=0),
    "bm25-pathx6": lambda: BM25Retriever(path_boost=6),
}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="instances JSONL")
    ap.add_argument("--retriever", default="bm25", choices=sorted(RETRIEVERS))
    ap.add_argument("--local-corpus", help="用本地目录当语料（跳过 git clone）")
    ap.add_argument("--top-k", type=int, default=30)
    ap.add_argument("--limit", type=int, default=0, help="只跑前 N 条")
    ap.add_argument("--exclude-tests", action="store_true")
    ap.add_argument("--out", default="results")
    ap.add_argument("--run-name", default="")
    args = ap.parse_args()

    instances = load_jsonl(args.data)
    if args.limit:
        instances = instances[: args.limit]

    if args.local_corpus:
        root = Path(args.local_corpus)
        cached = repo_mod.build_corpus(root, exclude_tests=args.exclude_tests)

        def corpus_fn(_: Instance):
            return cached
    else:
        def corpus_fn(ins: Instance):
            snap = repo_mod.export_snapshot(ins.repo, ins.base_commit)
            return repo_mod.build_corpus(snap, exclude_tests=args.exclude_tests)

    print(f"running {args.retriever} on {len(instances)} instances ...")
    agg, scores = run_eval(
        instances,
        corpus_fn,
        RETRIEVERS[args.retriever],
        top_k=args.top_k,
    )

    print()
    print(format_report(agg))
    print()
    print(f"skipped: {agg['skipped']}   gold_not_in_corpus: {agg['gold_not_in_corpus']}")

    run_name = args.run_name or f"{args.retriever}_{datetime.now():%m%d_%H%M}"
    out = save_results(
        args.out,
        run_name,
        config=vars(args),
        agg=agg,
        scores=scores,
    )
    print(f"\nsaved -> {out}")


if __name__ == "__main__":
    main()
