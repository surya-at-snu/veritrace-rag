from __future__ import annotations

import math

import numpy as np

from .fusion import Pool, ranks, to_docs
from .index import InvertedIndex
from .text import analyze

PRE_FEATURES = ["q_len", "q_oov", "idf_avg", "idf_max", "idf_min", "idf_std", "scq_avg", "scq_max",
                "ictf_avg", "scope"]
POST_FEATURES = ["sp_top1", "sp_top1_rel", "sp_gap12", "sp_nqc", "de_top1", "de_gap12", "de_std10",
                 "de_mean10", "agree_jacc10", "agree_top1", "sp1_rank_in_de", "de1_rank_in_sp"]
QPP_FEATURES = PRE_FEATURES + POST_FEATURES


# pre retrieval predictors only need dictionary stats from the index
def pre_retrieval(index: InvertedIndex, query: str, sparse_full: np.ndarray | None = None) -> dict:
    z = index.body
    terms = analyze(query)
    uniq = list(dict.fromkeys(terms))
    known = [t for t in uniq if z.tid(t) >= 0]
    total_tokens = float(z.cf.sum())
    idfs = [z.idf(t) for t in known] or [0.0]
    # scq: how similar each query term is to the whole collection
    scq = [(1 + math.log(z.cf[z.tid(t)])) * math.log(1 + z.n_units / z.df[z.tid(t)]) for t in known] or [0.0]
    ictf = [math.log2(total_tokens / z.cf[z.tid(t)]) for t in known] or [0.0]
    scope = float((sparse_full > 0).mean()) if sparse_full is not None else 0.0
    return {"q_len": len(uniq), "q_oov": len(uniq) - len(known), "idf_avg": float(np.mean(idfs)),
            "idf_max": float(np.max(idfs)), "idf_min": float(np.min(idfs)), "idf_std": float(np.std(idfs)),
            "scq_avg": float(np.mean(scq)), "scq_max": float(np.max(scq)), "ictf_avg": float(np.mean(ictf)),
            "scope": scope}


def post_retrieval(index: InvertedIndex, pool: Pool, idf_sum: float) -> dict:
    sd, ss, _ = to_docs(pool.sparse, pool.chunks, index.chunk_doc)
    dd, ds, _ = to_docs(pool.dense, pool.chunks, index.chunk_doc)
    s10, d10 = ss[:10], ds[:10]
    s1 = float(s10[0]) if len(s10) else 0.0
    s2 = float(s10[1]) if len(s10) > 1 else 0.0
    d1 = float(d10[0]) if len(d10) else 0.0
    d2 = float(d10[1]) if len(d10) > 1 else 0.0
    top_s, top_d = set(sd[:10].tolist()), set(dd[:10].tolist())
    s_rank = {int(d): i for i, d in enumerate(sd)}
    d_rank = {int(d): i for i, d in enumerate(dd)}
    big = len(sd) + 1
    return {
        "sp_top1": s1,
        "sp_top1_rel": s1 / (idf_sum + 1e-9),
        "sp_gap12": (s1 - s2) / (s1 + 1e-9),
        "sp_nqc": float(np.std(ss[:100]) / (np.mean(ss[:100]) + 1e-9)) if len(ss) else 0.0,
        "de_top1": d1,
        "de_gap12": d1 - d2,
        "de_std10": float(np.std(d10)) if len(d10) else 0.0,
        "de_mean10": float(np.mean(d10)) if len(d10) else 0.0,
        # if bm25 and dense agree on the top 10 the query is usually easy
        "agree_jacc10": len(top_s & top_d) / max(1, len(top_s | top_d)),
        "agree_top1": float(len(sd) > 0 and len(dd) > 0 and sd[0] == dd[0]),
        "sp1_rank_in_de": math.log1p(d_rank.get(int(sd[0]), big)) if len(sd) else math.log1p(big),
        "de1_rank_in_sp": math.log1p(s_rank.get(int(dd[0]), big)) if len(dd) else math.log1p(big),
    }


def qpp_features(index: InvertedIndex, query: str, pool: Pool, sparse_full: np.ndarray | None = None) -> dict:
    pre = pre_retrieval(index, query, sparse_full)
    idf_sum = sum(index.body.bm25_idf(t) for t in set(analyze(query)))
    return {**pre, **post_retrieval(index, pool, idf_sum)}


def as_vector(feats: dict, names=QPP_FEATURES) -> np.ndarray:
    return np.asarray([feats[n] for n in names], dtype=np.float32)


__all__ = ["QPP_FEATURES", "qpp_features", "as_vector", "ranks"]
