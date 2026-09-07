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
    # W3: chunk -> file 的分数聚合方式对照
    "bm25-sum": lambda: BM25Retriever(agg="sum"),
    "bm25-topksum": lambda: BM25Retriever(agg="topk_sum"),

    #### 调参 尝试########
    # W3.5: BM25 超参扫描，为基线定标（在前 50 条开发集上做）
    "bm25-b0.3": lambda: BM25Retriever(b=0.3),
    "bm25-b0.5": lambda: BM25Retriever(b=0.5),
    "bm25-b0.9": lambda: BM25Retriever(b=0.9),
    "bm25-k1.2": lambda: BM25Retriever(k1=1.2),
    "bm25-k2.0": lambda: BM25Retriever(k1=2.0),
    # 路径加权强度
    "bm25-pathx1": lambda: BM25Retriever(path_boost=1),
    "bm25-pathx6": lambda: BM25Retriever(path_boost=6),
    "bm25-pathx10": lambda: BM25Retriever(path_boost=10),
    # run_eval.py 注册表里加
    #"bm25-tuned": lambda: BM25Retriever(k1=1.2, b=0.5, path_boost=10),
    #"bm25-tuned-b075": lambda: BM25Retriever(k1=1.2, b=0.75, path_boost=10)
    # W3.5 最终基线
    "bm25-tuned": lambda: BM25Retriever(k1=1.2, b=0.75, path_boost=10),
    # 对照：b=0.5 单独扫描时最优，但组合下反而更差
    "bm25-tuned-b05": lambda: BM25Retriever(k1=1.2, b=0.5, path_boost=10),
}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="instances JSONL")
    ap.add_argument("--retriever", default="bm25", choices=sorted(RETRIEVERS))
    ap.add_argument("--local-corpus", help="用本地目录当语料（跳过 git clone）")
    ap.add_argument(
        "--chunking",
        default="file",
        choices=["file", "window", "ast"],
        help="切分策略: file=整文件(W2基线) window=固定行窗口(对照) ast=函数级",
    )
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
        cached = repo_mod.build_corpus(
            root, exclude_tests=args.exclude_tests, chunking=args.chunking
        )

        def corpus_fn(_: Instance):
            return cached
    else:
        def corpus_fn(ins: Instance):
            snap = repo_mod.export_snapshot(ins.repo, ins.base_commit)
            return repo_mod.build_corpus(
                snap, exclude_tests=args.exclude_tests, chunking=args.chunking
            )

    print(f"running {args.retriever} [chunking={args.chunking}] on {len(instances)} instances ...")
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

    run_name = args.run_name or f"{args.retriever}_{args.chunking}_{datetime.now():%m%d_%H%M}"
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