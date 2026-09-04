"""下载 SWE-bench 并缓存成本地 JSONL。

国内网络建议先设置镜像：
    export HF_ENDPOINT=https://hf-mirror.com

同时会打印数据集的基本统计 —— 这些数字要写进 README 的
「数据集说明」一节，也是判断任务难度的依据：
多文件修改的比例越高，文件定位任务越难。
"""

from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from reporank.data.swebench import dump_jsonl, load_hf


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="princeton-nlp/SWE-bench_Lite")
    ap.add_argument("--split", default="test")
    ap.add_argument("--out", default="data/swebench_lite.jsonl")
    args = ap.parse_args()

    print(f"downloading {args.dataset}:{args.split} ...")
    instances = load_hf(args.dataset, args.split)
    dump_jsonl(instances, args.out)

    n_gold = Counter(len(i.gold_files) for i in instances)
    repos = Counter(i.repo for i in instances)
    multi = sum(v for k, v in n_gold.items() if k > 1)

    print(f"\n{len(instances)} instances -> {args.out}")
    print(f"多文件修改比例: {multi / len(instances):.1%}")
    print("\ngold 文件数分布:")
    for k in sorted(n_gold):
        print(f"  {k} 个文件: {n_gold[k]}")
    print("\nTop repos:")
    for r, c in repos.most_common(10):
        print(f"  {c:>4}  {r}")


if __name__ == "__main__":
    main()
