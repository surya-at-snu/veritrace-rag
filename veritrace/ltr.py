from __future__ import annotations

import math
import pickle

import numpy as np

from .fusion import Pool, convex, minmax, ranks, rrf
from .index import InvertedIndex
from .sparse import min_cover_window, phrase_matches, query_positions
from .text import analyze

DOC_FEATURES = ["bm25_zone", "bm25_body", "bm25_title", "tfidf_body", "tfidf_title", "dense",
                "sp_rank", "de_rank", "rrf", "fused", "coverage", "idf_coverage", "phrase", "window",
                "chunk_len", "ce", "ce_rank", "ce_minus_max"]
QUERY_FEATURES = ["q_len", "idf_avg", "agree_jacc10", "sp_gap12", "de_gap12", "de_top1", "sp_top1_rel"]
LTR_FEATURES = DOC_FEATURES + QUERY_FEATURES


def doc_candidates(index: InvertedIndex, pool: Pool, alpha: float, n: int = 50):
    fused = convex(pool, alpha)
    rr = rrf(pool)
    docs = index.chunk_doc[pool.chunks]
    feats = {}
    order = np.lexsort((-fused, docs))
    d_sorted = docs[order]
    first = np.ones(len(order), dtype=bool)
    first[1:] = d_sorted[1:] != d_sorted[:-1]
    group_start = np.flatnonzero(first)
    best = order[first]
    # doc level feature = max over its chunks
    def dmax(x):
        return np.maximum.reduceat(x[order], group_start)
    feats["bm25_zone"] = dmax(pool.sparse)
    feats["bm25_body"] = dmax(pool.extra["bm25_body"])
    feats["bm25_title"] = dmax(pool.extra["bm25_title"])
    feats["tfidf_body"] = dmax(pool.extra["tfidf_body"])
    feats["tfidf_title"] = dmax(pool.extra["tfidf_title"])
    feats["dense"] = dmax(pool.dense)
    feats["rrf"] = dmax(rr)
    feats["fused"] = fused[best]
    feats["sp_rank"] = np.log1p(ranks(feats["bm25_zone"]))
    feats["de_rank"] = np.log1p(ranks(feats["dense"]))
    top = np.argsort(-feats["fused"], kind="stable")[:n]
    doc_idx = docs[best][top]
    best_chunk = pool.chunks[best][top]
    return doc_idx, best_chunk, {k: v[top] for k, v in feats.items()}


def lexical_features(index: InvertedIndex, query: str, best_chunks: np.ndarray, doc_idx: np.ndarray) -> dict:
    terms = list(dict.fromkeys(analyze(query)))
    qpos = query_positions(query)
    idf = {t: index.body.idf(t) for t in terms}
    idf_tot = sum(idf.values()) or 1.0
    cov, icov, phr, win, clen = [], [], [], [], []
    for c, d in zip(best_chunks.tolist(), doc_idx.tolist()):
        present = [t for t in terms if len(index.body.positions(t, c)) or index.title.positions(t, d).size]
        cov.append(len(present) / max(1, len(terms)))
        icov.append(sum(idf[t] for t in present) / idf_tot)
        phr.append(phrase_matches(index, qpos, c))
        # proximity feature straight from the positional index
        w, n = min_cover_window(index, terms, c)
        win.append(n / w if w > 0 else 0.0)
        clen.append(index.body.unit_len[c])
    return {"coverage": np.array(cov), "idf_coverage": np.array(icov), "phrase": np.array(phr, dtype=float),
            "window": np.array(win), "chunk_len": np.array(clen, dtype=float)}


def feature_matrix(index, query, pool, qpp: dict, alpha: float, ce_fn, n: int = 50):
    doc_idx, best_chunks, f = doc_candidates(index, pool, alpha, n)
    f.update(lexical_features(index, query, best_chunks, doc_idx))
    passages = [f"{index.chunks[c].title}. {index.chunks[c].text}" for c in best_chunks.tolist()]
    # the cross encoder score is just one feature next to the ir ones
    ce = ce_fn([(query, p) for p in passages])
    f["ce"] = ce
    f["ce_rank"] = np.log1p(ranks(ce))
    f["ce_minus_max"] = ce - ce.max() if len(ce) else ce
    X = np.column_stack([f[k] for k in DOC_FEATURES] +
                        [np.full(len(doc_idx), qpp[k], dtype=float) for k in QUERY_FEATURES])
    return X.astype(np.float32), doc_idx, best_chunks


class LTRReranker:
    def __init__(self, **params):
        # lambdamart via lightgbm
        base = dict(objective="lambdarank", n_estimators=400, learning_rate=0.03, num_leaves=15,
                    min_child_samples=20, subsample=0.8, subsample_freq=1, colsample_bytree=0.8,
                    reg_lambda=1.0, verbose=-1, random_state=0)
        base.update(params)
        self.params = base
        self.model = None
        self.feature_idx = None

    def fit(self, Xs: list[np.ndarray], ys: list[np.ndarray], feature_idx=None):
        import lightgbm as lgb
        self.feature_idx = feature_idx
        X = np.vstack([x if feature_idx is None else x[:, feature_idx] for x in Xs])
        y = np.concatenate(ys).astype(int)
        self.model = lgb.LGBMRanker(**self.params)
        self.model.fit(X, y, group=[len(x) for x in Xs])
        return self

    def score(self, X: np.ndarray) -> np.ndarray:
        if self.feature_idx is not None:
            X = X[:, self.feature_idx]
        return self.model.predict(X)

    def save(self, path):
        with open(path, "wb") as f:
            pickle.dump(self, f)

    @staticmethod
    def load(path) -> "LTRReranker":
        with open(path, "rb") as f:
            return pickle.load(f)


__all__ = ["LTR_FEATURES", "feature_matrix", "LTRReranker", "minmax", "math"]
