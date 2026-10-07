from __future__ import annotations

import json
import time
from dataclasses import dataclass, field

import numpy as np

from .boolean import BooleanEngine
from .config import art
from .dense import DenseRetriever, encode_queries, load_encoder
from .fusion import build_pool, convex, to_docs
from .index import InvertedIndex
from .ltr import LTRReranker, feature_matrix
from .qpp import qpp_features
from .rerank import ce_scores
from .router import FusionRouter, ce_arbitration, router_vector
from .sparse import SparseRetriever
from .text import analyze, tokenize, STOPWORDS
from .verify import CitationAuditor, ClaimVerifier, Evidence, Featurizer, trust_score

N_LTR = 30


@dataclass
class Retrieval:
    query: str
    terms: list
    term_stats: list
    sparse_top: list
    dense_top: list
    qpp: dict
    ce_arb: dict
    alpha: float
    p_lexical: float
    ranked: list
    contexts: list
    confidence: float
    boolean_filter: str | None = None
    n_filtered: int | None = None
    timings: dict = field(default_factory=dict)
    out_of_scope: bool = False
    best_ce: float = 0.0
    scope_tau: float | None = None
    original_query: str | None = None
    corrections: list = field(default_factory=list)


OUT_OF_SCOPE_MSG = ("NO RELEVANT DOCUMENTS: this question is outside the scope of the corpus "
                    "(the best-matching chunk is judged irrelevant), so 0 documents are returned.")


@dataclass
class Answer:
    retrieval: Retrieval
    raw_answer: str
    verdicts: list
    trust: float
    abstained: bool
    generator: str
    timings: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        r = self.retrieval
        return {"query": r.query, "alpha": r.alpha, "confidence": r.confidence, "abstained": self.abstained,
                "answer": self.raw_answer, "trust": self.trust,
                "sources": [{"n": i + 1, "doc_id": x["doc_id"], "title": x["title"], "chunk": x["chunk"]}
                            for i, x in enumerate(r.ranked[:len(r.contexts)])],
                "claims": [v.__dict__ for v in self.verdicts], "timings": {**r.timings, **self.timings}}


class VeriTrace:
    def __init__(self, dataset: str = "scifact", index_name: str = "window120", generator: str | None = "extractive",
                 use_ltr: bool = True, load_verifier: bool = True):
        out = art(dataset)
        self.dataset = dataset
        self.index = InvertedIndex.load(out / f"index_{index_name}.pkl")
        p = json.load(open(out / "sparse_params.json"))
        self.sparse = SparseRetriever(self.index, p["k1"], p["b"], p["g"])
        self.alpha_fixed = p["alpha_fixed"]
        self.dense = DenseRetriever(np.load(out / f"emb_{index_name}_bge.npy"))
        self.encoder = load_encoder()
        self.router = FusionRouter.load(out / "router.pkl")
        self.ltr = LTRReranker.load(out / "ltr.pkl") if use_ltr and (out / "ltr.pkl").exists() else None
        from .abstain import RetrievalConfidence
        self.conf = RetrievalConfidence.load(out / "abstain.pkl") if (out / "abstain.pkl").exists() else None
        gate = out / "scope_gate.json"
        self.scope_tau = json.load(open(gate))["tau"] if gate.exists() else None
        self.boolean = BooleanEngine(self.index)
        self._speller = None
        self.auditor = None
        if load_verifier and (out / "verifier.pkl").exists():
            verifier = ClaimVerifier.load(out / "verifier.pkl")
            self.auditor = CitationAuditor(Featurizer(self.index, self.encoder), verifier,
                                           retrieve_fn=self._claim_retrieve)
        self.generator = None
        self.generator_kind = generator
        if generator:
            from .generate import make_generator
            self.generator = make_generator(generator, self.index)

    # k-gram + edit-distance spelling corrector over the corpus vocabulary, built on first use (~1 s)
    @property
    def speller(self):
        if self._speller is None:
            from .spell import SpellCorrector
            self._speller = SpellCorrector.from_index(self.index)
        return self._speller

    def suggest(self, query: str):
        return self.speller.correct_query(query)

    def evidence(self, chunk_id: int) -> Evidence:
        c = self.index.chunks[int(chunk_id)]
        return Evidence(c.title, c.sentences, int(chunk_id))

    # used by the auditor: the claim itself becomes a new query
    def _claim_retrieve(self, claim: str, k: int = 5):
        s = self.sparse.chunk_scores(claim)
        d = self.dense.emb @ encode_queries([claim], self.encoder)[0]
        pool = build_pool(s, d, 100)
        f = convex(pool, self.alpha_fixed)
        top = pool.chunks[np.argsort(-f)[:k]]
        return [(int(c), self.evidence(c)) for c in top]

    def retrieve(self, query: str, k: int = 5, boolean_filter: str | None = None, spell: bool = False) -> Retrieval:
        T = {}
        t0 = time.time()
        original, fixes = None, []
        if spell:
            # 0. spelling correction of words that are not in the vocabulary (did-you-mean)
            corrected, fixes = self.suggest(query)
            if fixes:
                original, query = query, corrected
            T["spell_ms"] = (time.time() - t0) * 1000
        terms = analyze(query)
        stats = []
        for t in dict.fromkeys(terms):
            z = self.index.body
            tid = z.tid(t)
            stats.append((t, int(z.df[tid]) if tid >= 0 else 0, round(z.idf(t), 3),
                          int(self.index.title.df[self.index.title.tid(t)]) if self.index.title.tid(t) >= 0 else 0))
        # 1. bm25 with learned zone weights over the inverted index
        sparse_full = self.sparse.chunk_scores(query)
        title_c, body = self.sparse.zone_scores(query, "bm25")
        t_title, t_body = self.sparse.zone_scores(query, "tfidf")
        T["sparse_ms"] = (time.time() - t0) * 1000
        t1 = time.time()
        # 2. dense cosine over the whole collection
        dense_full = self.dense.emb @ encode_queries([query], self.encoder)[0]
        T["dense_ms"] = (time.time() - t1) * 1000

        n_filtered = None
        if boolean_filter:
            # optional boolean filter, only these chunks can come back
            allowed = np.asarray(self.boolean.query(boolean_filter), dtype=np.int64)
            n_filtered = int(len(allowed))
            mask = np.zeros(self.index.n_chunks, dtype=bool)
            mask[allowed] = True
            sparse_full = np.where(mask, sparse_full, 0.0).astype(np.float32)
            dense_full = np.where(mask, dense_full, -1.0).astype(np.float32)

        pool = build_pool(sparse_full, dense_full, 200, {"bm25_title": title_c, "bm25_body": body,
                                                          "tfidf_title": t_title, "tfidf_body": t_body})
        if boolean_filter:
            keep = mask[pool.chunks]
            pool.chunks, pool.sparse, pool.dense = pool.chunks[keep], pool.sparse[keep], pool.dense[keep]
            pool.extra = {kk: v[keep] for kk, v in pool.extra.items()}
            if len(pool.chunks) == 0:
                raise ValueError(f"Boolean filter '{boolean_filter}' matches no chunk")
        qpp = qpp_features(self.index, query, pool, sparse_full)
        sd, ss, sc = to_docs(pool.sparse, pool.chunks, self.index.chunk_doc)
        dd, dsc, dc = to_docs(pool.dense, pool.chunks, self.index.chunk_doc)
        t2 = time.time()
        top_s = [int(c) for c in sc[:3]]
        top_d = [int(c) for c in dc[:3]]
        while len(top_s) < 3:
            top_s.append(top_s[-1] if top_s else top_d[0])
        # 3. cross encoder on the top 3 of each list, used by the router and the scope gate
        ce6 = ce_scores([(query, f"{self.index.chunks[c].title}. {self.index.chunks[c].text}") for c in top_s + top_d])
        ce_arb = ce_arbitration(ce6[:3], ce6[3:])
        entry = {"qpp": qpp, "ce_arb": ce_arb}
        xr = router_vector(entry, self.router.feature_names)
        # 4. the router picks the fusion weight for this query
        alpha = float(self.router.predict_alpha(xr)[0])
        p_lex = float(self.router.predict_proba(xr)[0])
        T["router_ms"] = (time.time() - t2) * 1000
        best_ce = float(ce6.max())
        # 5. even the best chunk looks irrelevant -> return zero documents
        out_of_scope = self.scope_tau is not None and best_ce < self.scope_tau

        t3 = time.time()
        if out_of_scope:
            ranked = []
        elif self.ltr is not None:
            # 6. lambdamart re-ranks the top 30 documents
            X, doc_idx, best_chunks = feature_matrix(self.index, query, pool, qpp, alpha, ce_scores, N_LTR)
            ltr_s = self.ltr.score(X)
            order = np.argsort(-ltr_s, kind="stable")
            from .ltr import LTR_FEATURES
            col = {n: LTR_FEATURES.index(n) for n in ("fused", "bm25_zone", "dense", "ce")}
            ranked = [{"doc_id": self.index.docs[int(doc_idx[i])].doc_id, "title": self.index.docs[int(doc_idx[i])].title,
                       "chunk": int(best_chunks[i]), "ltr": float(ltr_s[i]), "fused": float(X[i, col["fused"]]),
                       "bm25": float(X[i, col["bm25_zone"]]), "dense": float(X[i, col["dense"]]),
                       "ce": float(X[i, col["ce"]])} for i in order]
        else:
            d, s, c = to_docs(convex(pool, alpha), pool.chunks, self.index.chunk_doc)
            ranked = [{"doc_id": self.index.docs[int(di)].doc_id, "title": self.index.docs[int(di)].title,
                       "chunk": int(ci), "ltr": None, "fused": float(si), "bm25": None, "dense": None, "ce": None}
                      for di, si, ci in zip(d[:N_LTR], s[:N_LTR], c[:N_LTR])]
        T["ltr_ms"] = (time.time() - t3) * 1000
        # 7. how likely is a relevant source in the top 5
        confidence = float(self.conf.predict(xr)[0]) if self.conf is not None else 1.0
        T["retrieval_total_ms"] = (time.time() - t0) * 1000
        return Retrieval(query, terms, stats,
                         [(self.index.docs[int(x)].doc_id, float(y)) for x, y in zip(sd[:10], ss[:10])],
                         [(self.index.docs[int(x)].doc_id, float(y)) for x, y in zip(dd[:10], dsc[:10])],
                         qpp, ce_arb, alpha, p_lex, ranked, [self.evidence(r["chunk"]) for r in ranked[:k]],
                         confidence, boolean_filter, n_filtered, T, out_of_scope, best_ce, self.scope_tau,
                         original, [f.__dict__ for f in fixes])

    def answer(self, query: str, k: int = 5, boolean_filter: str | None = None, strict: bool = False,
               verify: bool = True, spell: bool = False) -> Answer:
        r = self.retrieve(query, k, boolean_filter, spell)
        query = r.query
        T = {}
        if r.out_of_scope:
            return Answer(r, OUT_OF_SCOPE_MSG, [], 0.0, True, "none", T)
        abstain = self.conf is not None and r.confidence < self.conf.tau
        if strict and abstain:
            return Answer(r, "INSUFFICIENT EVIDENCE (retrieval confidence too low)", [], 0.0, True, "none", T)
        t0 = time.time()
        text = self.generator(query, r.contexts) if self.generator else ""
        T["generate_s"] = time.time() - t0
        verdicts = []
        if verify and self.auditor is not None and text and "INSUFFICIENT EVIDENCE" not in text:
            t1 = time.time()
            # check every sentence of the answer against the sources
            verdicts = self.auditor.audit(text, r.contexts)
            T["verify_s"] = time.time() - t1
        return Answer(r, text, verdicts, trust_score(verdicts), abstain,
                      getattr(self.generator, "name", "none"), T)


    def audit_text(self, retrieval: Retrieval, text: str) -> Answer:
        t0 = time.time()
        verdicts = self.auditor.audit(text, retrieval.contexts) if self.auditor else []
        return Answer(retrieval, text, verdicts, trust_score(verdicts),
                      self.conf is not None and retrieval.confidence < self.conf.tau, "user-edited",
                      {"verify_s": time.time() - t0})


def explain_query_processing(query: str) -> list[tuple[str, str]]:
    from .text import stem
    return [(t, "<stop>" if t in STOPWORDS else stem(t)) for t in tokenize(query)]
