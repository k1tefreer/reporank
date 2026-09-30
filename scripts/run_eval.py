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

  # 4. W5 融合
  python scripts/run_eval.py --data data/swebench_lite.jsonl \
      --retriever rrf --chunking ast --emb-model bge-small --limit 50
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
from reporank.retrieval.dense import DenseRetriever, PrefilteredDenseRetriever
from reporank.retrieval.fusion import HybridRetriever

EMB_MODEL = "minilm"  # 由 --emb-model 覆盖

# 模型加载很贵。retriever_fn() 每条实例调用一次，若每次都 new 一个
# Embedder，模型会被重复加载 N 次（实测 50 条实例多花 10 分钟以上）。
# 这里全局复用一个实例。
_EMBEDDER = None


def _emb():
    global _EMBEDDER
    if _EMBEDDER is None:
        from reporank.index.embedder import Embedder

        _EMBEDDER = Embedder(EMB_MODEL)
    return _EMBEDDER


# 检索器注册表。W3-W7 每加一种方法，在这里加一行即可，
# 评测脚本其他部分完全不用动。
RETRIEVERS = {
    "bm25": lambda: BM25Retriever(),
    "bm25-naive": lambda: BM25Retriever(naive_tokenizer=True),
    "bm25-nopath": lambda: BM25Retriever(path_boost=0),
    "bm25-pathx1": lambda: BM25Retriever(path_boost=1),
    "bm25-pathx6": lambda: BM25Retriever(path_boost=6),
    "bm25-pathx10": lambda: BM25Retriever(path_boost=10),
    "bm25-b0.3": lambda: BM25Retriever(b=0.3),
    "bm25-b0.5": lambda: BM25Retriever(b=0.5),
    "bm25-b0.9": lambda: BM25Retriever(b=0.9),
    "bm25-k1.2": lambda: BM25Retriever(k1=1.2),
    "bm25-k2.0": lambda: BM25Retriever(k1=2.0),
    # W3.5 最终基线（后续所有实验以此为准）
    "bm25-tuned": lambda: BM25Retriever(k1=1.2, b=0.75, path_boost=10),
    "bm25-tuned-b05": lambda: BM25Retriever(k1=1.2, b=0.5, path_boost=10),
    # W3: chunk -> file 的分数聚合方式对照
    "bm25-sum": lambda: BM25Retriever(agg="sum"),
    "bm25-topksum": lambda: BM25Retriever(agg="topk_sum"),
    # W4: 稠密检索
    "dense": lambda: DenseRetriever(embedder=_emb()),
    "dense-topksum": lambda: DenseRetriever(embedder=_emb(), agg="topk_sum"),
    "prefilter100": lambda: PrefilteredDenseRetriever(prefilter_files=100, embedder=_emb()),
    "prefilter200": lambda: PrefilteredDenseRetriever(prefilter_files=200, embedder=_emb()),
    "prefilter500": lambda: PrefilteredDenseRetriever(prefilter_files=500, embedder=_emb()),
    # W5: 融合。单路模式是健全性检查，应复现各自单独跑的数字
    "hybrid-bm25only": lambda: HybridRetriever(method="bm25_only", embedder=_emb()),
    "hybrid-denseonly": lambda: HybridRetriever(method="dense_only", embedder=_emb()),
    # RRF：k 控制头部名次的权重差距
    "rrf": lambda: HybridRetriever(method="rrf", embedder=_emb()),
    "rrf-k10": lambda: HybridRetriever(method="rrf", rrf_k=10, embedder=_emb()),
    "rrf-k30": lambda: HybridRetriever(method="rrf", rrf_k=30, embedder=_emb()),
    "rrf-k100": lambda: HybridRetriever(method="rrf", rrf_k=100, embedder=_emb()),
    # 加权 RRF：BM25 的 Top-1 更准，稠密的整体召回更强
    "rrf-w2:1": lambda: HybridRetriever(method="rrf", bm25_weight=2.0, embedder=_emb()),
    "rrf-w1:2": lambda: HybridRetriever(method="rrf", dense_weight=2.0, embedder=_emb()),
    "rrf-w1:3": lambda: HybridRetriever(method="rrf", dense_weight=3.0, embedder=_emb()),


        # W5: k 与权重的组合（两个维度都单独有效，验证能否叠加）
    "rrf-k10-w1:2": lambda: HybridRetriever(method="rrf", rrf_k=10, dense_weight=2.0, embedder=_emb()),
    "rrf-k10-w1:3": lambda: HybridRetriever(method="rrf", rrf_k=10, dense_weight=3.0, embedder=_emb()),
    "rrf-k5-w1:2": lambda: HybridRetriever(method="rrf", rrf_k=5, dense_weight=2.0, embedder=_emb()),
    "rrf-k30-w1:2": lambda: HybridRetriever(method="rrf", rrf_k=30, dense_weight=2.0, embedder=_emb()),


    # 分数融合对照：RRF 丢弃分数大小，这组检验那个取舍值不值
    "score-minmax": lambda: HybridRetriever(method="score", norm="minmax", embedder=_emb()),
    "score-zscore": lambda: HybridRetriever(method="score", norm="zscore", embedder=_emb()),
}

# 需要嵌入模型的检索器前缀，用于日志标注
_NEEDS_EMB = ("dense", "prefilter", "rrf", "score", "hybrid")


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
    ap.add_argument(
        "--emb-model",
        default="minilm",
        help="嵌入模型: minilm(22M) / bge-small(33M) / qwen3-0.6b(600M) 或任意 HF 模型名",
    )
    ap.add_argument("--out", default="results")
    ap.add_argument("--run-name", default="")
    args = ap.parse_args()

    global EMB_MODEL
    EMB_MODEL = args.emb_model

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

    tag = f"{args.retriever} [chunking={args.chunking}]"
    if args.retriever.startswith(_NEEDS_EMB):
        tag += f" [emb={args.emb_model}]"
    print(f"running {tag} on {len(instances)} instances ...")

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