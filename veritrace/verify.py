from __future__ import annotations

import math
import pickle
import re
from collections import Counter
from dataclasses import dataclass, field

import numpy as np

from .corrupt import DIRECTION_STEMS
from .text import analyze, has_negation, numbers, stem, tokenize

FEATURES = ["cos_chunk", "cos_best", "cos_best2", "idf_cov_chunk", "idf_cov_best", "miss_max_idf",
            "biword_cov", "num_claim", "num_missing", "neg_claim", "neg_best", "neg_mismatch",
            "dir_conflict", "dense_chunk", "dense_best", "nli_e_best", "nli_c_best", "nli_e_max",
            "nli_c_max", "nli_n_min", "nli_e_chunk", "nli_c_chunk", "claim_len"]
IR_FEATURES = [f for f in FEATURES if not f.startswith(("nli", "dense"))]
NO_NLI_FEATURES = [f for f in FEATURES if not f.startswith("nli")]


# small lnc.ltc helper for short texts, uses the corpus idf
class _VSM:

    def __init__(self, index):
        self.index = index
        self.N = index.body.n_units

    def idf(self, t: str) -> float:
        z = self.index.body
        tid = z.tid(t)
        return math.log10(self.N / z.df[tid]) if tid >= 0 else math.log10(self.N)

    def qvec(self, terms: list[str]) -> dict:
        tf = Counter(terms)
        w = {t: (1 + math.log10(c)) * self.idf(t) for t, c in tf.items()}
        n = math.sqrt(sum(v * v for v in w.values())) or 1.0
        return {t: v / n for t, v in w.items()}

    @staticmethod
    def dvec(terms: list[str]) -> dict:
        tf = Counter(terms)
        w = {t: 1 + math.log10(c) for t, c in tf.items()}
        n = math.sqrt(sum(v * v for v in w.values())) or 1.0
        return {t: v / n for t, v in w.items()}

    @staticmethod
    def cos(q: dict, d: dict) -> float:
        return sum(v * d.get(t, 0.0) for t, v in q.items())


def _biwords(terms: list[str]) -> set:
    return set(zip(terms, terms[1:]))


def _dir_conflict(claim_terms: set, sent_terms: set) -> float:
    for t in claim_terms:
        opp = DIRECTION_STEMS.get(t)
        if opp and (opp & sent_terms) and t not in sent_terms:
            return 1.0
    return 0.0


@dataclass
class Evidence:
    title: str
    sentences: list[str]
    ref: object = None

    @property
    def text(self) -> str:
        return " ".join(self.sentences)


class Featurizer:
    def __init__(self, index, encoder=None, nli=None, use_nli: bool = True):
        from .dense import load_encoder
        from .nli import get_nli
        self.vsm = _VSM(index)
        self.encoder = encoder or load_encoder()
        self.nli = nli if nli is not None else (get_nli() if use_nli else None)
        self._emb_cache: dict = {}

    def _embed(self, texts: list[str]) -> dict:
        new = [t for t in dict.fromkeys(texts) if t not in self._emb_cache]
        if new:
            vecs = self.encoder.encode(new, batch_size=64, normalize_embeddings=True, show_progress_bar=False)
            for t, v in zip(new, vecs):
                self._emb_cache[t] = v
        return self._emb_cache

    def featurize(self, pairs: list[tuple[str, Evidence]]) -> tuple[np.ndarray, list[dict]]:
        texts = []
        for claim, ev in pairs:
            texts.append(claim)
            texts.extend(ev.sentences)
            texts.append(f"{ev.title}. {ev.text}")
        emb = self._embed(texts)
        rows, info, nli_jobs = [], [], []
        for claim, ev in pairs:
            ct = analyze(claim)
            cset = set(ct)
            q = self.vsm.qvec(ct)
            idf = {t: self.vsm.idf(t) for t in cset}
            idf_tot = sum(idf.values()) or 1.0
            sent_terms = [analyze(s) for s in ev.sentences]
            chunk_terms = analyze(ev.title) + [t for st in sent_terms for t in st]
            chunk_set = set(chunk_terms)
            cos_s = np.array([self.vsm.cos(q, self.vsm.dvec(st)) for st in sent_terms]) if sent_terms else np.zeros(1)
            cv = emb[claim]
            den_s = np.array([float(emb[s] @ cv) for s in ev.sentences]) if ev.sentences else np.zeros(1)
            # rank the chunk's sentences against the claim (lexical + dense) to find the evidence sentence
            score = cos_s / (cos_s.max() + 1e-9) + den_s / (den_s.max() + 1e-9)
            order = np.argsort(-score)
            b1 = int(order[0])
            b2 = int(order[1]) if len(order) > 1 else b1
            best_set = set(sent_terms[b1]) if sent_terms else set()
            missing = [idf[t] for t in cset if t not in chunk_set]
            cn = numbers(claim)
            en = numbers(ev.title + " " + ev.text)
            neg_c = float(has_negation(claim))
            best_sent = ev.sentences[b1] if ev.sentences else ev.title
            neg_b = float(has_negation(best_sent))
            row = {
                "cos_chunk": self.vsm.cos(q, self.vsm.dvec(chunk_terms)),
                "cos_best": float(cos_s[b1]),
                "cos_best2": float(cos_s[b2]),
                "idf_cov_chunk": sum(idf[t] for t in cset if t in chunk_set) / idf_tot,
                "idf_cov_best": sum(idf[t] for t in cset if t in best_set) / idf_tot,
                # a rare claim word missing from the source is a typical sign of a made up entity
                "miss_max_idf": max(missing) if missing else 0.0,
                "biword_cov": len(_biwords(ct) & _biwords(chunk_terms)) / max(1, len(_biwords(ct))),
                "num_claim": float(len(cn)),
                # numbers in the claim that never appear in the source
                "num_missing": float(len(cn - en)),
                "neg_claim": neg_c, "neg_best": neg_b, "neg_mismatch": float(neg_c != neg_b),
                # claim says increase but the evidence says decrease (or the other way round)
                "dir_conflict": _dir_conflict(cset, best_set),
                "dense_chunk": float(emb[f"{ev.title}. {ev.text}"] @ cv),
                "dense_best": float(den_s[b1]),
                "claim_len": float(len(ct)),
            }
            rows.append(row)
            info.append({"best": b1, "best2": b2})
            if self.nli is not None:
                s2 = ev.sentences[b2] if ev.sentences else ev.title
                # nli on the two best sentences and on the whole chunk
                nli_jobs.append([(best_sent, claim), (s2, claim), (f"{ev.title}. {ev.text}", claim)])
        if self.nli is not None:
            flat = [p for job in nli_jobs for p in job]
            probs = self.nli(flat).reshape(len(pairs), 3, 3)
            for row, p in zip(rows, probs):
                row.update({"nli_e_best": p[0, 0], "nli_c_best": p[0, 2], "nli_e_max": p[:, 0].max(),
                            "nli_c_max": p[:, 2].max(), "nli_n_min": p[:, 1].min(),
                            "nli_e_chunk": p[2, 0], "nli_c_chunk": p[2, 2]})
        else:
            for row in rows:
                row.update({k: 0.0 for k in FEATURES if k.startswith("nli")})
        X = np.array([[r[f] for f in FEATURES] for r in rows], dtype=np.float32)
        return X, info


class ClaimVerifier:

    def __init__(self, features=FEATURES, **params):
        # 3 classes: supported / contradicted / unsupported
        base = dict(objective="multiclass", num_class=3, n_estimators=300, learning_rate=0.04,
                    num_leaves=15, min_child_samples=15, subsample=0.8, subsample_freq=1,
                    colsample_bytree=0.8, reg_lambda=1.0, verbose=-1, random_state=0,
                    class_weight="balanced")
        base.update(params)
        self.params = base
        self.features = list(features)
        self.cols = [FEATURES.index(f) for f in self.features]
        self.model = None
        self.tau_support = 0.5

    def fit(self, X, y):
        import lightgbm as lgb
        self.model = lgb.LGBMClassifier(**self.params)
        self.model.fit(X[:, self.cols], y)
        return self

    def predict_proba(self, X) -> np.ndarray:
        return self.model.predict_proba(np.atleast_2d(X)[:, self.cols])

    def save(self, path):
        with open(path, "wb") as f:
            pickle.dump(self, f)

    @staticmethod
    def load(path) -> "ClaimVerifier":
        with open(path, "rb") as f:
            return pickle.load(f)


_CITE = re.compile(r"\[(\d+(?:\s*[,;]\s*\d+)*)\]")


# strip wrappers like 'source [1] states that x' so we verify x itself
_ATTRIBUTION = re.compile(
    r"^[^.;:]{0,90}?\b(?:states?|stated|shows?|showed|indicates?|indicated|suggests?|suggested|reports?|reported|"
    r"finds?|found|demonstrates?|demonstrated|notes?|noted|mentions?|mentioned|confirms?|confirmed|reveals?|"
    r"revealed|concludes?|concluded|supports? the claim|support the claim|agrees?|observes?|observed) that\s+", re.I)
# sentences about the sources themselves ('the sources do not mention ...') are not factual claims
_META = re.compile(
    r"\b(?:do|does|did) not (?:directly |explicitly |specifically |clearly )?(?:support|address|mention|discuss|provide|"
    r"contradict|report|state|describe|include|cover)\b|\bcannot be (?:supported|confirmed|determined|verified|"
    r"contradicted)\b|\bno (?:direct |specific |explicit )?(?:information|evidence|mention|data)\b|"
    r"\b(?:insufficient|not enough) (?:evidence|information)\b", re.I)
_META_SUBJECT = re.compile(r"\b(?:sources?|claim|provided|passages?|documents?|context|it|they|this)\b", re.I)


def normalize_claim(text: str) -> tuple[str, bool]:
    if _META.search(text) and _META_SUBJECT.search(text):
        return text, True
    m = _ATTRIBUTION.match(text)
    if m:
        rest = text[m.end():].strip()
        if len(tokenize(rest)) >= 4:
            text = rest[0].upper() + rest[1:]
    return text, False


def split_claims(answer: str) -> list[dict]:
    from .text import split_sentences
    out = []
    pending: list[int] = []
    answer = answer.replace("\n", " ").strip()
    # some answers put [n] before the sentence and some after, handle both
    leading = bool(re.match(r"^\[\d+", answer))
    if not leading:
        answer = re.sub(r"([.!?])((?:\s*\[\d+(?:\s*[,;]\s*\d+)*\])+)", lambda m: m.group(2) + m.group(1), answer)
    for s in split_sentences(answer):
        cites = [int(x) for grp in _CITE.findall(s) for x in re.split(r"[,;]\s*", grp) if x.strip()]
        text = _CITE.sub("", s).strip()
        text = re.sub(r"\s+([.,;:])", r"\1", re.sub(r"\s{2,}", " ", text))
        if len(tokenize(text)) < 3:
            if out and cites and not leading:
                out[-1]["cites"] = sorted(set(out[-1]["cites"]) | set(cites))
            else:
                pending.extend(cites)
            continue
        norm, meta = normalize_claim(text)
        out.append({"text": norm, "raw": text, "meta": meta, "cites": sorted(set(cites + pending))})
        pending = []
    return out


@dataclass
class ClaimVerdict:
    text: str
    cited: list[int]
    status: str
    support_ref: int | None = None
    support_sentence: str = ""
    p_support: float = 0.0
    p_contradict: float = 0.0
    new_chunk: int | None = None
    probs: dict = field(default_factory=dict)


class CitationAuditor:
    def __init__(self, featurizer: Featurizer, verifier: ClaimVerifier, tau_support: float | None = None,
                 tau_contradict: float = 0.5, retrieve_fn=None, max_external: int = 3):
        self.f = featurizer
        self.v = verifier
        self.tau_s = verifier.tau_support if tau_support is None else tau_support
        self.tau_c = tau_contradict
        self.retrieve_fn = retrieve_fn
        self.max_external = max_external

    def _score(self, claim: str, evs: list[Evidence]):
        X, info = self.f.featurize([(claim, e) for e in evs])
        return self.v.predict_proba(X), info

    def audit(self, answer: str, context: list[Evidence]) -> list[ClaimVerdict]:
        claims = split_claims(answer)
        verdicts = []
        if not context:
            return verdicts
        for c in claims:
            if c["meta"]:
                verdicts.append(ClaimVerdict(c["text"], c["cites"], "NOT_A_CLAIM"))
                continue
            P, info = self._score(c["text"], context)
            cited = [i - 1 for i in c["cites"] if 1 <= i <= len(context)]
            v = ClaimVerdict(c["text"], c["cites"], "UNSUPPORTED")
            v.probs = {i + 1: P[i].round(3).tolist() for i in range(len(context))}
            best_cited = max(cited, key=lambda i: P[i, 0]) if cited else None
            best_any = int(np.argmax(P[:, 0]))
            jc = int(np.argmax(P[:, 1]))
            # 1. a cited source supports it -> verified
            if best_cited is not None and P[best_cited, 0] >= self.tau_s:
                v.status, k = "VERIFIED", best_cited
            # 2. another retrieved source supports it -> move the citation there
            elif P[best_any, 0] >= self.tau_s:
                v.status, k = "REPAIRED", best_any
            # 3. the evidence says the opposite -> contradicted
            elif P[jc, 1] >= self.tau_c and P[jc, 1] > P[jc, 0]:
                v.status, k = "CONTRADICTED", jc
            else:
                k = None
                if self.retrieve_fn is not None:
                    # 4. verify by retrieval: search the whole index with the claim as the query
                    ext = [(cid, e) for cid, e in self.retrieve_fn(c["text"])
                           if all(e.text != x.text for x in context)][: self.max_external]
                    if ext:
                        Pe, info_e = self._score(c["text"], [e for _, e in ext])
                        j = int(np.argmax(Pe[:, 0]))
                        if Pe[j, 0] >= self.tau_s:
                            ev = ext[j][1]
                            v.status, v.new_chunk = "REPAIRED_NEW", int(ext[j][0])
                            v.p_support, v.p_contradict = float(Pe[j, 0]), float(Pe[j, 1])
                            v.support_sentence = ev.sentences[info_e[j]["best"]] if ev.sentences else ev.title
            if k is not None:
                ev = context[k]
                v.support_ref = k + 1
                v.p_support, v.p_contradict = float(P[k, 0]), float(P[k, 1])
                v.support_sentence = ev.sentences[info[k]["best"]] if ev.sentences else ev.title
            elif v.status == "UNSUPPORTED":
                v.p_support, v.p_contradict = float(P[:, 0].max()), float(P[:, 1].max())
            verdicts.append(v)
        return verdicts


# share of real claims that ended up grounded, meta sentences dont count
def trust_score(verdicts: list[ClaimVerdict]) -> float:
    factual = [v for v in verdicts if v.status != "NOT_A_CLAIM"]
    if not factual:
        return 0.0
    return sum(v.status in ("VERIFIED", "REPAIRED", "REPAIRED_NEW") for v in factual) / len(factual)


__all__ = ["FEATURES", "IR_FEATURES", "Featurizer", "ClaimVerifier", "CitationAuditor", "Evidence",
           "split_claims", "trust_score", "stem"]
