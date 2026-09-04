"""检索评测指标。

全部按 binary relevance 处理（一个文件要么在 gold patch 里，要么不在）。
指标选择理由：
  - Recall@k  : 最关键。召回不到，后面 LLM 再强也没用（上限被卡死）
  - Hit@k     : Top-k 里至少命中一个 gold 文件的比例。单文件任务下等价于 Recall@k
  - Precision@k: 上下文预算的代价。K 越大 Recall 越高但 Precision 崩塌
  - MRR       : 第一个正确结果的位置，反映排序质量
  - NDCG@k    : 综合排序质量，对多 gold 文件的实例更敏感

注意 Precision@k 和 Recall@k 的张力就是本项目的核心命题：
放大 K 能把 Recall 推上去，但塞进 LLM 的无关文件会拖垮端到端成功率。
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from statistics import mean
from typing import Iterable, Sequence


def _norm(p: str) -> str:
    """路径归一化，避免 './x.py' 和 'x.py' 被判成不同文件。"""
    return p.strip().lstrip("./").replace("\\", "/")


def recall_at_k(ranked: Sequence[str], gold: Iterable[str], k: int) -> float:
    g = {_norm(x) for x in gold}
    if not g:
        return float("nan")
    top = {_norm(x) for x in ranked[:k]}
    return len(top & g) / len(g)


def precision_at_k(ranked: Sequence[str], gold: Iterable[str], k: int) -> float:
    if k <= 0:
        return 0.0
    g = {_norm(x) for x in gold}
    top = [_norm(x) for x in ranked[:k]]
    if not top:
        return 0.0
    return sum(1 for x in top if x in g) / len(top)


def hit_at_k(ranked: Sequence[str], gold: Iterable[str], k: int) -> float:
    g = {_norm(x) for x in gold}
    return 1.0 if any(_norm(x) in g for x in ranked[:k]) else 0.0


def reciprocal_rank(ranked: Sequence[str], gold: Iterable[str]) -> float:
    g = {_norm(x) for x in gold}
    for i, x in enumerate(ranked, start=1):
        if _norm(x) in g:
            return 1.0 / i
    return 0.0


def ndcg_at_k(ranked: Sequence[str], gold: Iterable[str], k: int) -> float:
    """binary relevance 下的 NDCG@k，log2 折扣。"""
    g = {_norm(x) for x in gold}
    if not g:
        return float("nan")
    dcg = sum(
        1.0 / math.log2(i + 1)
        for i, x in enumerate(ranked[:k], start=1)
        if _norm(x) in g
    )
    ideal_n = min(len(g), k)
    idcg = sum(1.0 / math.log2(i + 1) for i in range(1, ideal_n + 1))
    return dcg / idcg if idcg > 0 else 0.0


@dataclass(slots=True)
class InstanceScore:
    """单条实例的评测结果，保留下来用于错误分析。"""

    instance_id: str
    n_gold: int
    n_retrieved: int
    recall: dict[int, float]
    precision: dict[int, float]
    hit: dict[int, float]
    ndcg: dict[int, float]
    rr: float
    missed: list[str]  # 完全没被召回的 gold 文件 —— 错误分析的入口


DEFAULT_KS = (1, 3, 5, 10, 20, 30)


def score_instance(
    instance_id: str,
    ranked: Sequence[str],
    gold: Iterable[str],
    ks: Sequence[int] = DEFAULT_KS,
) -> InstanceScore:
    gold_list = list(gold)
    g = {_norm(x) for x in gold_list}
    retrieved_all = {_norm(x) for x in ranked}
    return InstanceScore(
        instance_id=instance_id,
        n_gold=len(g),
        n_retrieved=len(ranked),
        recall={k: recall_at_k(ranked, gold_list, k) for k in ks},
        precision={k: precision_at_k(ranked, gold_list, k) for k in ks},
        hit={k: hit_at_k(ranked, gold_list, k) for k in ks},
        ndcg={k: ndcg_at_k(ranked, gold_list, k) for k in ks},
        rr=reciprocal_rank(ranked, gold_list),
        missed=sorted(g - retrieved_all),
    )


def aggregate(scores: Sequence[InstanceScore], ks: Sequence[int] = DEFAULT_KS) -> dict:
    """汇总成一行报告。"""
    #if not scores:
     #   return {}
    if not scores:
        return {"n_instances": 0, "mrr": 0.0}
    out: dict = {"n_instances": len(scores), "mrr": mean(s.rr for s in scores)}
    for k in ks:
        out[f"recall@{k}"] = mean(s.recall[k] for s in scores)
        out[f"precision@{k}"] = mean(s.precision[k] for s in scores)
        out[f"hit@{k}"] = mean(s.hit[k] for s in scores)
        out[f"ndcg@{k}"] = mean(s.ndcg[k] for s in scores)
    return out


def format_report(agg: dict, ks: Sequence[int] = DEFAULT_KS) -> str:
    """打印成对齐表格，方便直接贴进 README。"""
    #if not agg:
    #    return "(no results)"
    if not agg or not agg.get("n_instances"):
        return "(no results — 所有实例都被跳过，先看上面的 [skip] 原因)"
    
    lines = [
        f"instances: {agg['n_instances']}    MRR: {agg['mrr']:.4f}",
        "",
        f"{'k':>4} | {'Recall':>8} | {'Precision':>9} | {'Hit':>8} | {'NDCG':>8}",
        "-" * 50,
    ]
    for k in ks:
        lines.append(
            f"{k:>4} | {agg[f'recall@{k}']:>8.4f} | {agg[f'precision@{k}']:>9.4f} "
            f"| {agg[f'hit@{k}']:>8.4f} | {agg[f'ndcg@{k}']:>8.4f}"
        )
    return "\n".join(lines)
