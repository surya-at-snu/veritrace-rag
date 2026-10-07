from __future__ import annotations

import math

import numpy as np


def precision_at(ranked: list[str], rel: dict[str, int], k: int) -> float:
    return sum(1 for d in ranked[:k] if rel.get(d, 0) > 0) / k


def recall_at(ranked: list[str], rel: dict[str, int], k: int) -> float:
    n = sum(1 for v in rel.values() if v > 0)
    return sum(1 for d in ranked[:k] if rel.get(d, 0) > 0) / n if n else 0.0


def average_precision(ranked: list[str], rel: dict[str, int], k: int = 1000) -> float:
    n = sum(1 for v in rel.values() if v > 0)
    if not n:
        return 0.0
    hits, s = 0, 0.0
    for i, d in enumerate(ranked[:k], 1):
        if rel.get(d, 0) > 0:
            hits += 1
            s += hits / i
    return s / n


def reciprocal_rank(ranked: list[str], rel: dict[str, int], k: int = 10) -> float:
    for i, d in enumerate(ranked[:k], 1):
        if rel.get(d, 0) > 0:
            return 1.0 / i
    return 0.0


# graded relevance, gain = 2^rel - 1
def ndcg_at(ranked: list[str], rel: dict[str, int], k: int = 10) -> float:
    dcg = sum((2 ** rel.get(d, 0) - 1) / math.log2(i + 1) for i, d in enumerate(ranked[:k], 1))
    ideal = sorted(rel.values(), reverse=True)[:k]
    idcg = sum((2 ** r - 1) / math.log2(i + 1) for i, r in enumerate(ideal, 1))
    return dcg / idcg if idcg > 0 else 0.0


METRICS = {
    "nDCG@10": lambda r, q: ndcg_at(r, q, 10),
    "P@1": lambda r, q: precision_at(r, q, 1),
    "P@5": lambda r, q: precision_at(r, q, 5),
    "P@10": lambda r, q: precision_at(r, q, 10),
    "R@10": lambda r, q: recall_at(r, q, 10),
    "R@100": lambda r, q: recall_at(r, q, 100),
    "MAP@100": lambda r, q: average_precision(r, q, 100),
    "MRR@10": lambda r, q: reciprocal_rank(r, q, 10),
}


def evaluate_run(run: dict[str, list[str]], qrels: dict[str, dict[str, int]], metrics=None) -> dict:
    metrics = metrics or list(METRICS)
    per_q = {m: [] for m in metrics}
    for qid, rel in qrels.items():
        ranked = run.get(qid, [])
        for m in metrics:
            per_q[m].append(METRICS[m](ranked, rel))
    means = {m: float(np.mean(v)) for m, v in per_q.items()}
    return {"mean": means, "per_query": per_q, "qids": list(qrels)}


# paired t-test over the per query scores
def paired_t_pvalue(a: list[float], b: list[float]) -> float:
    from scipy import stats
    a, b = np.asarray(a), np.asarray(b)
    if np.allclose(a, b):
        return 1.0
    return float(stats.ttest_rel(a, b).pvalue)
