from __future__ import annotations

import heapq
import math
from collections import Counter

import numpy as np

from .index import InvertedIndex, ZoneIndex
from .text import analyze, analyze_with_positions


# min heap of size k, only chunks that got a non zero score are visited
def heap_topk(scores: np.ndarray, k: int, candidates: np.ndarray | None = None) -> list[tuple[int, float]]:
    cand = np.flatnonzero(scores) if candidates is None else candidates
    heap: list[tuple[float, int]] = []
    for i in cand.tolist():
        s = float(scores[i])
        if len(heap) < k:
            heapq.heappush(heap, (s, i))
        elif s > heap[0][0]:
            heapq.heapreplace(heap, (s, i))
    return [(i, s) for s, i in sorted(heap, reverse=True)]


def fast_topk(scores: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
    k = min(k, len(scores))
    idx = np.argpartition(-scores, k - 1)[:k]
    idx = idx[np.argsort(-scores[idx], kind="stable")]
    return idx, scores[idx]


class SparseRetriever:
    def __init__(self, index: InvertedIndex, k1: float = 0.9, b: float = 0.4, title_weight: float = 0.2,
                 analyzer=None):
        self.index = index
        self.k1, self.b, self.g = k1, b, title_weight
        # the query must go through the same analysis chain as the indexed text
        self.analyzer = analyzer or analyze
        self._norm_cache: dict = {}
        self._smart_cache: dict = {}
        # champion lists are the first r postings in bm25-impact order; make sure that order uses the
        # same (tuned) k1 and b that we score with, otherwise the "best" postings are the wrong ones
        for z in (index.body, index.title):
            if getattr(z, "impact_params", None) != (float(k1), float(b)):
                z._build_impact(k1, b)

    def query_terms(self, text: str) -> Counter:
        return Counter(self.analyzer(text))

    def _len_norm(self, z: ZoneIndex) -> np.ndarray:
        key = (z.name, self.k1, self.b)
        if key not in self._norm_cache:
            self._norm_cache[key] = (self.k1 * (1 - self.b + self.b * z.unit_len / max(z.avg_len, 1e-9))).astype(np.float32)
        return self._norm_cache[key]

    def bm25_zone(self, q: Counter, z: ZoneIndex, min_idf: float = 0.0) -> np.ndarray:
        acc = np.zeros(z.n_units, dtype=np.float32)
        norm = self._len_norm(z)
        for term, qtf in q.items():
            t = z.tid(term)
            if t < 0:
                continue
            idf = z.bm25_idf(term)
            # index elimination: skip low idf terms
            if idf < min_idf:
                continue
            u, tf = z.units[t], z.tf[t]
            # bm25 term weight, added term at a time into the accumulator
            acc[u] += qtf * idf * tf * (self.k1 + 1) / (tf + norm[u])
        return acc

    def tfidf_zone(self, q: Counter, z: ZoneIndex) -> np.ndarray:
        acc = np.zeros(z.n_units, dtype=np.float32)
        wq = {}
        for term, qtf in q.items():
            t = z.tid(term)
            if t >= 0:
                wq[term] = (1 + math.log10(qtf)) * z.idf(term)
        qnorm = math.sqrt(sum(w * w for w in wq.values())) or 1.0
        for term, w in wq.items():
            t = z.tid(term)
            u, tf = z.units[t], z.tf[t]
            # lnc.ltc: log tf on the doc side, log tf * idf on the query side, cosine normalised
            acc[u] += (w / qnorm) * (1 + np.log10(tf))
        return acc / z.lnc_norm

    def chunk_scores(self, text_or_terms, model: str = "bm25", g: float | None = None,
                     min_idf: float = 0.0) -> np.ndarray:
        q = text_or_terms if isinstance(text_or_terms, Counter) else self.query_terms(text_or_terms)
        g = self.g if g is None else g
        if model == "bm25":
            body = self.bm25_zone(q, self.index.body, min_idf)
            title = self.bm25_zone(q, self.index.title, min_idf) if g > 0 else None
        elif model == "tfidf":
            body = self.tfidf_zone(q, self.index.body)
            title = self.tfidf_zone(q, self.index.title) if g > 0 else None
        else:
            raise ValueError(model)
        if title is None:
            return body
        # weighted zone scoring, g is learned on the training queries
        return (1 - g) * body + g * title[self.index.chunk_doc]

    def zone_scores(self, text: str, model: str = "bm25"):
        q = self.query_terms(text)
        if model == "bm25":
            body, title = self.bm25_zone(q, self.index.body), self.bm25_zone(q, self.index.title)
        else:
            body, title = self.tfidf_zone(q, self.index.body), self.tfidf_zone(q, self.index.title)
        return title[self.index.chunk_doc], body

    def search(self, text: str, k: int = 10, model: str = "bm25", level: str = "chunk") -> list[tuple[int, float]]:
        s = self.chunk_scores(text, model)
        if level == "doc":
            s = self.index.doc_max(s)
        return heap_topk(s, k)

    def search_champions(self, text: str, r: int, k: int = 10, min_idf: float = 0.0) -> list[tuple[int, float]]:
        q = self.query_terms(text)
        body, title = self.index.body, self.index.title
        lists = [body.champion_list(t, r) for t in q if body.tid(t) >= 0 and body.bm25_idf(t) >= min_idf]
        if not lists:
            return []
        # only chunks that sit on some query term's champion list get scored
        cand = np.unique(np.concatenate(lists))
        norm = self._len_norm(body)
        sc = np.zeros(len(cand), dtype=np.float32)
        for term, qtf in q.items():
            t = body.tid(term)
            if t < 0 or body.bm25_idf(term) < min_idf:
                continue
            u, tf = body.units[t], body.tf[t]
            pos = np.searchsorted(u, cand)
            pos = np.minimum(pos, len(u) - 1)
            hit = u[pos] == cand
            tfc = np.where(hit, tf[pos], 0.0)
            sc += qtf * body.bm25_idf(term) * tfc * (self.k1 + 1) / (tfc + norm[cand])
        if self.g > 0:
            tsc = self.bm25_zone(q, title)[self.index.chunk_doc[cand]]
            sc = (1 - self.g) * sc + self.g * tsc
        order = heap_topk(sc, k, candidates=np.arange(len(cand)))
        return [(int(cand[i]), s) for i, s in order]


    # per-term breakdown of one chunk's score: shows exactly where a BM25 / lnc.ltc score comes from
    def explain(self, text: str, chunk: int) -> list[dict]:
        q = self.query_terms(text)
        body, title = self.index.body, self.index.title
        d = int(self.index.chunk_doc[chunk])
        nb, nt = self._len_norm(body), self._len_norm(title)
        wq = {t: (1 + math.log10(c)) * body.idf(t) for t, c in q.items() if body.tid(t) >= 0}
        qn = math.sqrt(sum(w * w for w in wq.values())) or 1.0
        rows = []
        for t, qtf in q.items():
            row = {"term": t, "qtf": qtf, "df": 0, "idf_log10": 0.0, "bm25_idf": 0.0, "tf_body": 0,
                   "bm25_body": 0.0, "tf_title": 0, "bm25_title": 0.0, "lnc_ltc": 0.0}
            tid = body.tid(t)
            if tid >= 0:
                u = body.units[tid]
                i = np.searchsorted(u, chunk)
                tf = float(body.tf[tid][i]) if i < len(u) and u[i] == chunk else 0.0
                idf = body.bm25_idf(t)
                row.update(df=int(body.df[tid]), idf_log10=round(body.idf(t), 3), bm25_idf=round(idf, 3),
                           tf_body=int(tf),
                           bm25_body=float(qtf * idf * tf * (self.k1 + 1) / (tf + nb[chunk])) if tf else 0.0)
                if tf:
                    row["lnc_ltc"] = float((wq[t] / qn) * (1 + math.log10(tf)) / body.lnc_norm[chunk])
            ttid = title.tid(t)
            if ttid >= 0:
                u = title.units[ttid]
                i = np.searchsorted(u, d)
                tf = float(title.tf[ttid][i]) if i < len(u) and u[i] == d else 0.0
                if tf:
                    row.update(tf_title=int(tf),
                               bm25_title=float(qtf * title.bm25_idf(t) * tf * (self.k1 + 1) / (tf + nt[d])))
            row["weighted"] = float((1 - self.g) * row["bm25_body"] + self.g * row["bm25_title"])
            rows.append(row)
        return rows

    # ------------------------------------------------------------------------------------------
    # SMART notation ddd.qqq (IIR fig 6.15): tf  n=raw, l=1+log10(tf), b=boolean, a=0.5+0.5*tf/max_tf
    #                                      df  n=none, t=log10(N/df)
    #                                      norm n=none, c=cosine
    # lnc.ltc (our baseline) is tfidf_zone above; this general version is used for the comparison
    # of weighting schemes in scripts/exp_ir_extras.py.
    @staticmethod
    def _tf_weight(letter: str, tf: np.ndarray, max_tf: np.ndarray | float = 1.0) -> np.ndarray:
        if letter == "n":
            return tf.astype(np.float64)
        if letter == "l":
            return 1.0 + np.log10(tf)
        if letter == "b":
            return np.ones_like(tf, dtype=np.float64)
        if letter == "a":
            return 0.5 + 0.5 * tf / max_tf
        raise ValueError(letter)

    def _smart_doc_stats(self, z: ZoneIndex, scheme: str):
        key = (z.name, scheme)
        if key not in self._smart_cache:
            max_tf = np.ones(z.n_units)
            for u, tf in zip(z.units, z.tf):
                np.maximum.at(max_tf, u, tf)
            sq = np.zeros(z.n_units)
            for t, (u, tf) in enumerate(zip(z.units, z.tf)):
                w = self._tf_weight(scheme[0], tf, max_tf[u])
                if scheme[1] == "t":
                    w = w * math.log10(z.n_units / z.df[t])
                np.add.at(sq, u, w * w)
            norm = np.sqrt(np.maximum(sq, 1e-12)) if scheme[2] == "c" else np.ones(z.n_units)
            self._smart_cache[key] = (max_tf, norm)
        return self._smart_cache[key]

    def smart_zone(self, q: Counter, z: ZoneIndex, scheme: str = "lnc.ltc") -> np.ndarray:
        dsch, qsch = scheme.split(".")
        max_tf, dnorm = self._smart_doc_stats(z, dsch)
        known = {t: c for t, c in q.items() if z.tid(t) >= 0}
        if not known:
            return np.zeros(z.n_units, dtype=np.float32)
        qmax = max(known.values())
        wq = {}
        for t, c in known.items():
            w = float(self._tf_weight(qsch[0], np.array([c], dtype=np.float32), qmax)[0])
            if qsch[1] == "t":
                w *= z.idf(t)
            wq[t] = w
        qn = math.sqrt(sum(w * w for w in wq.values())) if qsch[2] == "c" else 1.0
        acc = np.zeros(z.n_units)
        for t, w in wq.items():
            tid = z.tid(t)
            u, tf = z.units[tid], z.tf[tid]
            wd = self._tf_weight(dsch[0], tf, max_tf[u])
            if dsch[1] == "t":
                wd = wd * z.idf(t)
            acc[u] += (w / (qn or 1.0)) * wd
        return (acc / dnorm).astype(np.float32)

    # jaccard coefficient between the query term set and each chunk's term set (lecture 6 baseline)
    def jaccard_zone(self, q: Counter, z: ZoneIndex) -> np.ndarray:
        inter = np.zeros(z.n_units)
        known = [t for t in q if z.tid(t) >= 0]
        for t in known:
            inter[z.units[z.tid(t)]] += 1
        if not hasattr(z, "_n_distinct"):
            nd = np.zeros(z.n_units)
            for u in z.units:
                nd[u] += 1
            z._n_distinct = nd
        union = z._n_distinct + len(q) - inter
        return (inter / np.maximum(union, 1)).astype(np.float32)

    # ------------------------------------------------------------------------------------------
    # many-term matching (IIR 7.1.2): only chunks that contain at least m of the query terms are scored
    def bm25_min_match(self, text: str, m: int) -> np.ndarray:
        q = self.query_terms(text)
        body = self.index.body
        count = np.zeros(body.n_units, dtype=np.int16)
        for t in q:
            tid = body.tid(t)
            if tid >= 0:
                count[body.units[tid]] += 1
        m = min(m, int(count.max()) if len(count) else 0)
        s = self.chunk_scores(q)
        return np.where(count >= max(m, 1), s, 0.0).astype(np.float32)

    # tiered index (IIR 7.2.1): tier 1 = the champion part of every postings list (top r by impact),
    # tier 2 = the rest. we answer from tier 1 and only go down to tier 2 when tier 1 cannot fill k
    # documents. returns (hits, tier_used, chunks_scored)
    def search_tiered(self, text: str, r: int = 100, k: int = 10, k_docs: int | None = None):
        k_docs = k_docs or k
        q = self.query_terms(text)
        lists = [self.index.body.champion_list(t, r) for t in q if self.index.body.tid(t) >= 0]
        cand = np.unique(np.concatenate(lists)) if lists else np.empty(0, np.int32)
        # tier 1 is enough when its postings already cover at least k_docs different documents
        n_docs = len(np.unique(self.index.chunk_doc[cand])) if len(cand) else 0
        if n_docs >= k_docs:
            return self.search_champions(text, r, k), 1, int(len(cand))
        full = self.chunk_scores(q)
        return heap_topk(full, k), 2, int((full > 0).sum())


def query_positions(text: str) -> list[tuple[str, int]]:
    return analyze_with_positions(text)


# counts query bigrams that show up in the chunk with the same gap as in the query
def phrase_matches(index: InvertedIndex, qpos: list[tuple[str, int]], chunk: int) -> int:
    hits = 0
    for (t1, p1), (t2, p2) in zip(qpos, qpos[1:]):
        if t1 == t2:
            continue
        a = index.body.positions(t1, chunk)
        if not len(a):
            continue
        b = index.body.positions(t2, chunk)
        if not len(b):
            continue
        hits += int(np.isin(a + (p2 - p1), b).any())
    return hits


# smallest window covering all the query terms present, used as a proximity feature
def min_cover_window(index: InvertedIndex, terms: list[str], chunk: int) -> tuple[int, int]:
    lists = []
    for t in dict.fromkeys(terms):
        p = index.body.positions(t, chunk)
        if len(p):
            lists.append(p)
    if len(lists) < 2:
        return 0, len(lists)
    events = sorted((int(p), i) for i, ps in enumerate(lists) for p in ps)
    need, have, cnt = len(lists), {}, 0
    best, lo = 10 ** 9, 0
    for hi, (p, i) in enumerate(events):
        have[i] = have.get(i, 0) + 1
        if have[i] == 1:
            cnt += 1
        while cnt == need:
            best = min(best, p - events[lo][0] + 1)
            j = events[lo][1]
            have[j] -= 1
            if have[j] == 0:
                cnt -= 1
            lo += 1
    return best, need
