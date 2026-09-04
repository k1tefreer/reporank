"""评测主循环。

设计原则（对应课程里的实验设计规范）：
  1. 每次实验的完整配置写进结果文件，保证可复现
  2. 保留每条实例的明细，不只是平均值 —— 错误分析全靠这个
  3. 内置健全性检查：如果 gold 文件根本不在语料里，说明是语料构建的 bug，
     不是检索器的锅。这个数字必须单独报出来，否则会误判优化效果。
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict
from pathlib import Path
from typing import Callable, Sequence

from ..data.swebench import Instance
from ..retrieval.base import Document, Retriever
from .metrics import DEFAULT_KS, InstanceScore, aggregate, format_report, score_instance

CorpusFn = Callable[[Instance], list[Document]]
RetrieverFn = Callable[[], Retriever]


def run_eval(
    instances: Sequence[Instance],
    corpus_fn: CorpusFn,
    retriever_fn: RetrieverFn,
    *,
    top_k: int = 30,
    ks: Sequence[int] = DEFAULT_KS,
    verbose: bool = True,
) -> tuple[dict, list[InstanceScore]]:
    scores: list[InstanceScore] = []
    skipped: list[str] = []
    oov_gold = 0  # gold 文件不在语料中的实例数
    t0 = time.time()

    for i, ins in enumerate(instances, 1):
        gold = ins.gold_files
        if not gold:
            skipped.append(ins.instance_id)
            continue
        try:
            docs = corpus_fn(ins)
        except Exception as e:  # 单条失败不能中断整轮实验
            if verbose:
                print(f"  [skip] {ins.instance_id}: {e}")
            skipped.append(ins.instance_id)
            continue
        if not docs:
            skipped.append(ins.instance_id)
            continue

        corpus_paths = {d.path for d in docs}
        if not (set(gold) & corpus_paths):
            oov_gold += 1

        r = retriever_fn()
        r.index(docs)
        ranked = r.search_files(ins.query, top_k=top_k)
        scores.append(score_instance(ins.instance_id, ranked, gold, ks=ks))

        if verbose and i % 10 == 0:
            print(f"  {i}/{len(instances)}  ({time.time() - t0:.0f}s)")

    agg = aggregate(scores, ks=ks)
    agg["skipped"] = len(skipped)
    agg["gold_not_in_corpus"] = oov_gold
    agg["elapsed_sec"] = round(time.time() - t0, 1)
    return agg, scores


def save_results(
    out_dir: str | Path,
    run_name: str,
    config: dict,
    agg: dict,
    scores: list[InstanceScore],
) -> Path:
    out = Path(out_dir) / run_name
    out.mkdir(parents=True, exist_ok=True)
    (out / "config.json").write_text(
        json.dumps(config, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    (out / "summary.json").write_text(
        json.dumps(agg, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    with open(out / "per_instance.jsonl", "w", encoding="utf-8") as f:
        for s in scores:
            f.write(json.dumps(asdict(s), ensure_ascii=False) + "\n")
    (out / "report.txt").write_text(format_report(agg), encoding="utf-8")
    return out
