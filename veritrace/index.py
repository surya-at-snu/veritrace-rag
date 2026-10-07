from __future__ import annotations

import math
import pickle
from collections import defaultdict
from dataclasses import dataclass, field

import numpy as np

from .chunking import Chunk
from .corpus import Document
from .text import analyze_with_positions

BM25_K1 = 0.9
BM25_B = 0.4


@dataclass
class ZoneIndex:
    name: str
    n_units: int
    vocab: dict = field(default_factory=dict)
    terms: list = field(default_factory=list)
    df: np.ndarray = None
    cf: np.ndarray = None
    units: list = field(default_factory=list)
    tf: list = field(default_factory=list)
    pos: list = field(default_factory=list)
    pos_off: list = field(default_factory=list)
    impact: list = field(default_factory=list)
    unit_len: np.ndarray = None
    lnc_norm: np.ndarray = None
    avg_len: float = 0.0

    @classmethod
    def build(cls, name: str, texts: list[str], analyzer=None) -> "ZoneIndex":
        z = cls(name=name, n_units=len(texts))
        # analyzer: default chain (stop words + porter) unless an ablation passes another one
        analyze_pos = analyzer.with_positions if analyzer is not None else analyze_with_positions
        # term -> list of (unit id, positions), built in one pass over the texts
        acc: dict[str, list] = defaultdict(list)
        unit_len = np.zeros(len(texts), dtype=np.int32)
        for u, text in enumerate(texts):
            tp = analyze_pos(text)
            unit_len[u] = len(tp)
            local: dict[str, list[int]] = defaultdict(list)
            for term, p in tp:
                local[term].append(p)
            for term, ps in local.items():
                acc[term].append((u, ps))
        z.unit_len = unit_len
        z.avg_len = float(unit_len.mean()) if len(texts) else 0.0
        lnc_sq = np.zeros(len(texts), dtype=np.float64)
        df, cf = [], []
        for tid, term in enumerate(sorted(acc)):
            plist = acc[term]
            z.vocab[term] = tid
            z.terms.append(term)
            u = np.fromiter((p[0] for p in plist), dtype=np.int32, count=len(plist))
            tf = np.fromiter((len(p[1]) for p in plist), dtype=np.float32, count=len(plist))
            off = np.zeros(len(plist) + 1, dtype=np.int64)
            off[1:] = np.cumsum(tf.astype(np.int64))
            # positions are stored flat with offsets, much lighter than one list per posting
            pos = np.fromiter((x for p in plist for x in p[1]), dtype=np.int32, count=int(off[-1]))
            z.units.append(u)
            z.tf.append(tf)
            z.pos.append(pos)
            z.pos_off.append(off)
            df.append(len(plist))
            cf.append(int(off[-1]))
            # precompute the lnc length of every unit for cosine scoring
            np.add.at(lnc_sq, u, (1.0 + np.log10(tf)) ** 2)
        z.df = np.asarray(df, dtype=np.int32)
        z.cf = np.asarray(cf, dtype=np.int64)
        z.lnc_norm = np.sqrt(np.maximum(lnc_sq, 1e-12)).astype(np.float32)
        z._build_impact()
        return z

    # impact ordered postings: each list sorted by bm25 weight, the first r entries are the champion list.
    # the ordering depends on k1 and b, so it is rebuilt with the tuned values (see SparseRetriever)
    def _build_impact(self, k1: float = BM25_K1, b: float = BM25_B):
        self.impact_params = (float(k1), float(b))
        norm = k1 * (1 - b + b * self.unit_len / max(self.avg_len, 1e-9))
        self.impact = []
        for u, tf in zip(self.units, self.tf):
            w = tf * (k1 + 1) / (tf + norm[u])
            self.impact.append(np.argsort(-w, kind="stable").astype(np.int32))

    def tid(self, term: str) -> int:
        return self.vocab.get(term, -1)

    def postings(self, term: str):
        t = self.tid(term)
        if t < 0:
            return np.empty(0, np.int32), np.empty(0, np.float32)
        return self.units[t], self.tf[t]

    def positions(self, term: str, unit: int) -> np.ndarray:
        t = self.tid(term)
        if t < 0:
            return np.empty(0, np.int32)
        u = self.units[t]
        i = np.searchsorted(u, unit)
        if i >= len(u) or u[i] != unit:
            return np.empty(0, np.int32)
        return self.pos[t][self.pos_off[t][i]:self.pos_off[t][i + 1]]

    def champion_list(self, term: str, r: int) -> np.ndarray:
        t = self.tid(term)
        if t < 0:
            return np.empty(0, np.int32)
        return self.units[t][self.impact[t][:r]]

    # log10(n/df) like in the lectures
    def idf(self, term: str) -> float:
        t = self.tid(term)
        return math.log10(self.n_units / self.df[t]) if t >= 0 else 0.0

    # lucene style bm25 idf, never goes negative
    def bm25_idf(self, term: str) -> float:
        t = self.tid(term)
        if t < 0:
            return 0.0
        n = self.df[t]
        return math.log(1.0 + (self.n_units - n + 0.5) / (n + 0.5))


@dataclass
class InvertedIndex:
    docs: list[Document]
    chunks: list[Chunk]
    title: ZoneIndex
    body: ZoneIndex
    chunk_doc: np.ndarray
    doc_chunk_start: np.ndarray
    config: dict = field(default_factory=dict)

    @classmethod
    def build(cls, docs: list[Document], chunks: list[Chunk], config: dict | None = None,
              analyzer=None) -> "InvertedIndex":
        # two zones: title is indexed per document, body per chunk
        title = ZoneIndex.build("title", [d.title for d in docs], analyzer)
        body = ZoneIndex.build("body", [c.text for c in chunks], analyzer)
        chunk_doc = np.asarray([c.doc_idx for c in chunks], dtype=np.int32)
        starts = np.searchsorted(chunk_doc, np.arange(len(docs)))
        return cls(docs, chunks, title, body, chunk_doc, starts.astype(np.int64), config or {})

    @property
    def n_docs(self) -> int:
        return len(self.docs)

    @property
    def n_chunks(self) -> int:
        return len(self.chunks)

    # maxp: a document scores as its best chunk
    def doc_max(self, chunk_scores: np.ndarray) -> np.ndarray:
        return np.maximum.reduceat(chunk_scores, self.doc_chunk_start)

    def describe_term(self, term: str, max_postings: int = 8) -> dict:
        z = self.body
        t = z.tid(term)
        if t < 0:
            return {"term": term, "df": 0}
        out = []
        for i in range(min(max_postings, z.df[t])):
            c = int(z.units[t][i])
            out.append({"chunk": c, "doc": self.chunks[c].doc_id, "tf": int(z.tf[t][i]),
                        "positions": z.pos[t][z.pos_off[t][i]:z.pos_off[t][i + 1]].tolist()})
        champions = [int(c) for c in z.champion_list(term, 5)]
        return {"term": term, "df": int(z.df[t]), "cf": int(z.cf[t]), "idf": round(z.idf(term), 4),
                "title_df": int(self.title.df[self.title.tid(term)]) if self.title.tid(term) >= 0 else 0,
                "postings": out, "champions": champions}

    def save(self, path):
        with open(path, "wb") as f:
            pickle.dump(self, f, protocol=pickle.HIGHEST_PROTOCOL)

    @staticmethod
    def load(path) -> "InvertedIndex":
        with open(path, "rb") as f:
            return pickle.load(f)
