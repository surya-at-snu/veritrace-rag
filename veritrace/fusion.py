from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class Pool:
    chunks: np.ndarray
    sparse: np.ndarray
    dense: np.ndarray
    extra: dict


def heap_select(scores: np.ndarray, n: int, candidates: np.ndarray | None = None) -> np.ndarray:
    """top-n indices with a size-n min-heap (IIR 7.1): one pass over the candidates, O(N log n).
    for bm25 only the chunks with a non-zero accumulator are candidates (term-at-a-time scoring
    never touches the others)."""
    import heapq
    cand = np.arange(len(scores)) if candidates is None else candidates
    vals = scores[cand].tolist()
    heap: list = []
    for v, i in zip(vals, cand.tolist()):
        if len(heap) < n:
            heapq.heappush(heap, (v, -i))
        elif v > heap[0][0]:
            heapq.heapreplace(heap, (v, -i))
    return np.array(sorted((-i for _, i in heap)), dtype=np.int64)


def build_pool(sparse_full: np.ndarray, dense_full: np.ndarray, n: int = 200, extra_full: dict | None = None) -> Pool:
    # heap-based top-n of each list; bm25 only looks at chunks that got a non-zero score
    top_s = heap_select(sparse_full, n, np.flatnonzero(sparse_full > 0))
    top_d = heap_select(dense_full, n)
    # candidate pool = top n from bm25 plus top n from dense
    u = np.union1d(top_s, top_d)
    extra = {name: arr[u] for name, arr in (extra_full or {}).items()}
    return Pool(u.astype(np.int32), sparse_full[u].astype(np.float32), dense_full[u].astype(np.float32), extra)


def minmax(x: np.ndarray) -> np.ndarray:
    lo, hi = float(x.min()), float(x.max())
    return np.zeros_like(x) if hi - lo < 1e-9 else (x - lo) / (hi - lo)


def ranks(x: np.ndarray) -> np.ndarray:
    order = np.argsort(-x, kind="stable")
    r = np.empty(len(x), dtype=np.int64)
    r[order] = np.arange(1, len(x) + 1)
    return r


# reciprocal rank fusion with the usual k = 60
def rrf(p: Pool, k: int = 60) -> np.ndarray:
    return 1.0 / (k + ranks(p.sparse)) + 1.0 / (k + ranks(p.dense))


# alpha is the weight on bm25, both lists are min max normalised first
def convex(p: Pool, alpha: float) -> np.ndarray:
    return alpha * minmax(p.sparse) + (1 - alpha) * minmax(p.dense)


def to_docs(pool_scores: np.ndarray, pool_chunks: np.ndarray, chunk_doc: np.ndarray):
    docs = chunk_doc[pool_chunks]
    # group chunks by document and keep the best one (maxp)
    order = np.lexsort((-pool_scores, docs))
    d_sorted = docs[order]
    first = np.ones(len(order), dtype=bool)
    first[1:] = d_sorted[1:] != d_sorted[:-1]
    best = order[first]
    dscore = pool_scores[best]
    o = np.argsort(-dscore, kind="stable")
    return docs[best][o], dscore[o], pool_chunks[best][o]


def doc_run(pool_scores, pool_chunks, chunk_doc, doc_ids: list[str], k: int = 100) -> list[str]:
    d, _, _ = to_docs(pool_scores, pool_chunks, chunk_doc)
    return [doc_ids[i] for i in d[:k]]


def full_doc_run(chunk_scores_full: np.ndarray, index, k: int = 1000) -> list[str]:
    ds = index.doc_max(chunk_scores_full)
    k = min(k, len(ds))
    top = np.argpartition(-ds, k - 1)[:k]
    top = top[np.argsort(-ds[top], kind="stable")]
    top = top[ds[top] > -1e9]
    return [index.docs[i].doc_id for i in top]
