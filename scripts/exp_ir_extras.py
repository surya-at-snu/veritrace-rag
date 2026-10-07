"""extra IR-principle experiments on SciFact (all sparse except 'cluster'):

  smart     SMART weighting schemes (ddd.qqq) + Jaccard baseline, body zone only
  analysis  analysis-chain ablation: stemming (none / S-stemmer / Porter) and stop words
  spell     query spelling correction (k-gram + edit distance + soundex) under injected typos
  cluster   cluster pruning (leaders/followers) for the dense index

every learned or tuned value (BM25 k1/b per chain) uses only the training claims; numbers are on the
300 test claims. writes results/scifact_ir_extras.json (merged, one key per experiment)."""
import argparse
import json
import random
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from veritrace.config import RESULTS, art
from veritrace.corpus import split_queries
from veritrace.fusion import full_doc_run
from veritrace.index import InvertedIndex
from veritrace.metrics import evaluate_run, paired_t_pvalue
from veritrace.sparse import SparseRetriever

OUT = RESULTS / "scifact_ir_extras.json"
DS = "scifact"


def _eval(run, qrels):
    e = evaluate_run(run, qrels)
    return e["mean"], e["per_query"]["nDCG@10"]


def _save(key, value):
    data = json.load(open(OUT)) if OUT.exists() else {}
    data[key] = value
    json.dump(data, open(OUT, "w"), indent=1)


def load_default():
    out = art(DS)
    index = InvertedIndex.load(out / "index_window120.pkl")
    p = json.load(open(out / "sparse_params.json"))
    return index, p


# ---------------------------------------------------------------------------------------------
def exp_smart():
    index, p = load_default()
    sp = SparseRetriever(index, p["k1"], p["b"], 0.0)
    queries, qrels = split_queries(DS, "test")
    schemes = ["nnn.nnn", "nnc.nnc", "ntc.ntc", "ltc.ltc", "lnc.ltc", "lnc.lnc", "anc.ltc", "bnc.btc", "ltn.ltn"]
    rows, per_q = [], {}
    for sch in schemes:
        run = {q: full_doc_run(sp.smart_zone(sp.query_terms(queries[q]), index.body, sch), index, 100) for q in qrels}
        m, pq = _eval(run, qrels)
        rows.append({"scheme": sch, **m})
        per_q[sch] = pq
        print(f"{sch:10s} nDCG@10={m['nDCG@10']:.3f} P@1={m['P@1']:.3f} R@10={m['R@10']:.3f} MAP={m['MAP@100']:.3f}")
    run = {q: full_doc_run(sp.jaccard_zone(sp.query_terms(queries[q]), index.body), index, 100) for q in qrels}
    m, pq = _eval(run, qrels)
    rows.append({"scheme": "jaccard", **m})
    per_q["jaccard"] = pq
    print(f"{'jaccard':10s} nDCG@10={m['nDCG@10']:.3f}")
    run = {q: full_doc_run(sp.chunk_scores(queries[q], "bm25", g=0.0), index, 100) for q in qrels}
    m, pq = _eval(run, qrels)
    rows.append({"scheme": "bm25 (k1=%.1f, b=%.1f)" % (p["k1"], p["b"]), **m})
    per_q["bm25"] = pq
    # sanity check: the general SMART code must reproduce the dedicated lnc.ltc scorer
    q0 = next(iter(qrels))
    a = sp.smart_zone(sp.query_terms(queries[q0]), index.body, "lnc.ltc")
    b = sp.tfidf_zone(sp.query_terms(queries[q0]), index.body)
    assert np.allclose(a, b, atol=1e-5), "SMART lnc.ltc != tfidf_zone"
    pv = {s: paired_t_pvalue(per_q["lnc.ltc"], per_q[s]) for s in per_q if s != "lnc.ltc"}
    _save("smart", {"rows": rows, "p_vs_lnc_ltc": pv})


# ---------------------------------------------------------------------------------------------
def exp_analysis():
    from veritrace.chunking import chunk_corpus
    from veritrace.corpus import load_corpus
    from veritrace.text import Analyzer
    docs = load_corpus(DS)
    chunks = chunk_corpus(docs, "window", 120, 1)
    tq, tqrels = split_queries(DS, "train", limit=400)
    queries, qrels = split_queries(DS, "test")
    chains = [("porter", True), ("s", True), ("none", True), ("porter", False), ("none", False)]
    rows = []
    for stemmer, stop in chains:
        an = Analyzer(stemmer, stop)
        t0 = time.time()
        index = InvertedIndex.build(docs, chunks, {"analyzer": an.name}, analyzer=an)
        build_s = time.time() - t0
        # tune k1, b on training claims for this chain (title weight fixed at 0.2)
        best = None
        for k1 in (0.6, 0.9, 1.2):
            for b in (0.3, 0.5, 0.75):
                sp = SparseRetriever(index, k1, b, 0.2, analyzer=an)
                run = {q: full_doc_run(sp.chunk_scores(tq[q]), index, 10) for q in tqrels}
                n = evaluate_run(run, tqrels, ["nDCG@10"])["mean"]["nDCG@10"]
                if best is None or n > best[0]:
                    best = (n, k1, b)
        _, k1, b = best
        sp = SparseRetriever(index, k1, b, 0.2, analyzer=an)
        run = {q: full_doc_run(sp.chunk_scores(queries[q]), index, 100) for q in qrels}
        m, pq = _eval(run, qrels)
        sp0 = SparseRetriever(index, k1, b, 0.0, analyzer=an)
        run = {q: full_doc_run(sp0.tfidf_zone(sp0.query_terms(queries[q]), index.body), index, 100) for q in qrels}
        mt, pqt = _eval(run, qrels)
        row = {"chain": an.name, "stemmer": stemmer, "stopwords_removed": stop,
               "vocab": int(len(index.body.vocab)), "postings": int(index.body.df.sum()),
               "positions": int(index.body.cf.sum()), "build_s": build_s, "k1": k1, "b": b,
               "bm25_zones": m, "tfidf_body": mt, "_pq_bm25": pq, "_pq_tfidf": pqt}
        rows.append(row)
        print(f"{an.name:18s} vocab={row['vocab']:6d} postings={row['postings']:7d} k1={k1} b={b} "
              f"BM25 nDCG@10={m['nDCG@10']:.3f} R@100={m['R@100']:.3f} | lnc.ltc nDCG@10={mt['nDCG@10']:.3f}")
    base = rows[0]
    for r in rows:
        r["p_bm25_vs_porter"] = paired_t_pvalue(base["_pq_bm25"], r["_pq_bm25"])
        r["p_tfidf_vs_porter"] = paired_t_pvalue(base["_pq_tfidf"], r["_pq_tfidf"])
    for r in rows:
        r.pop("_pq_bm25"); r.pop("_pq_tfidf")
    _save("analysis", rows)


# ---------------------------------------------------------------------------------------------
def make_typo(word: str, rnd: random.Random) -> str:
    i = rnd.randrange(1, len(word) - 1)
    op = rnd.choice(["delete", "insert", "substitute", "transpose"])
    letters = "abcdefghijklmnopqrstuvwxyz"
    if op == "delete":
        return word[:i] + word[i + 1:]
    if op == "insert":
        return word[:i] + rnd.choice(letters) + word[i:]
    if op == "substitute":
        return word[:i] + rnd.choice([c for c in letters if c != word[i]]) + word[i + 1:]
    return word[:i] + word[i + 1] + word[i] + word[i + 2:] if i + 1 < len(word) else word[:i - 1] + word[i] + word[i - 1]


def inject(query: str, corr, rnd: random.Random, n_typos: int = 1):
    import re
    # whole alphabetic words only (not parts of RUNX1 or up-regulation), no acronyms
    toks = [m for m in re.finditer(r"(?<![A-Za-z0-9-])[A-Za-z]+(?![A-Za-z0-9-])", query)
            if len(m.group(0)) >= 5 and not m.group(0).isupper() and not corr.needs_correction(m.group(0).lower())
            and corr.cf.get(m.group(0).lower(), 0) > 0]
    if not toks:
        return query, []
    picks = rnd.sample(toks, min(n_typos, len(toks)))
    out, changes = query, []
    for m in sorted(picks, key=lambda m: -m.start()):
        w = m.group(0)
        for _ in range(10):
            t = make_typo(w.lower(), rnd)
            if t != w.lower() and corr.cf.get(t, 0) == 0:
                break
        out = out[:m.start()] + t + out[m.end():]
        changes.append((w.lower(), t))
    return out, changes


def exp_spell():
    from veritrace.spell import SpellCorrector
    index, p = load_default()
    sp = SparseRetriever(index, p["k1"], p["b"], p["g"])
    t0 = time.time()
    corr = SpellCorrector.from_index(index)
    build_s = time.time() - t0
    queries, qrels = split_queries(DS, "test")
    tq, _ = split_queries(DS, "train")
    # 1) false corrections: how many correctly spelled claims does the corrector change?
    train_changes = []
    for q in tq.values():
        fixes = corr.correct_query(q)[1]
        if fixes:
            train_changes.append([(f.original, f.corrected, f.distance) for f in fixes])
    changed_train = len(train_changes)
    res = {"vocab_words": len(corr.words), "kgram_postings": int(sum(len(v) for v in corr.kgram_index.values())),
           "build_s": build_s, "train_claims_changed": changed_train, "n_train": len(tq), "train_changes": train_changes, "settings": {}}
    for n_typos in (1, 2):
        rnd = random.Random(7 + n_typos)
        typo_q, fixed_q, n_fixed_right, n_typos_total, examples, methods = {}, {}, 0, 0, [], {}
        clean_fixed = {}
        for q in qrels:
            tq_, changes = inject(queries[q], corr, rnd, n_typos)
            typo_q[q] = tq_
            fq, fixes = corr.correct_query(tq_)
            fixed_q[q] = fq
            clean_fixed[q] = corr.correct_query(queries[q])[0]
            got = {f.original: f for f in fixes}
            for orig, bad in changes:
                n_typos_total += 1
                if bad in got and got[bad].corrected == orig:
                    n_fixed_right += 1
                    methods[got[bad].method] = methods.get(got[bad].method, 0) + 1
            if len(examples) < 8 and changes:
                examples.append({"clean": queries[q], "typo": tq_, "corrected": fq,
                                 "fixes": [f.__dict__ for f in fixes]})
        runs = {}
        for name, qs in (("clean", queries), ("clean+corrector", clean_fixed), ("typo", typo_q), ("typo+corrector", fixed_q)):
            runs[name] = {q: full_doc_run(sp.chunk_scores(qs[q]), index, 100) for q in qrels}
        ms = {}
        pqs = {}
        for name, run in runs.items():
            m, pq = _eval(run, qrels)
            ms[name], pqs[name] = m, pq
            print(f"typos={n_typos} {name:16s} nDCG@10={m['nDCG@10']:.3f} P@1={m['P@1']:.3f} R@10={m['R@10']:.3f}")
        res["settings"][str(n_typos)] = {
            "metrics": ms, "typos": n_typos_total, "restored_exactly": n_fixed_right, "methods": methods,
            "p_typo_vs_fixed": paired_t_pvalue(pqs["typo"], pqs["typo+corrector"]),
            "p_clean_vs_fixed": paired_t_pvalue(pqs["clean"], pqs["typo+corrector"]), "examples": examples}
        print(f"  restored {n_fixed_right}/{n_typos_total} typos exactly; methods {methods}")
    print(f"clean training claims changed by the corrector: {changed_train}/{len(tq)}")
    _save("spell", res)


# ---------------------------------------------------------------------------------------------
def exp_cluster(encoder_path: str, emb_name: str):
    from veritrace.cluster import ClusterPrunedIndex
    from veritrace.dense import encode_queries, load_encoder
    out = art(DS)
    index = InvertedIndex.load(out / "index_window120.pkl")
    emb = np.load(out / emb_name)
    queries, qrels = split_queries(DS, "test")
    qids = list(qrels)
    qv = encode_queries([queries[q] for q in qids], load_encoder(encoder_path))
    t0 = time.perf_counter()
    exact = {q: full_doc_run(emb @ v, index, 100) for q, v in zip(qids, qv)}
    ms_exact = (time.perf_counter() - t0) * 1000 / len(qids)
    m, pq_exact = _eval(exact, qrels)
    rows = [{"config": "exact (all chunks)", "b1": None, "b2": None, "vectors_scored": len(emb),
             "ms_per_query": ms_exact, **m}]
    print(f"exact          nDCG@10={m['nDCG@10']:.3f} R@100={m['R@100']:.3f} scored={len(emb)}")
    for b1 in (1, 2):
        cp = ClusterPrunedIndex(emb, b1=b1, seed=0)
        for b2 in (1, 2, 5, 10, 20):
            run, scored = {}, []
            t0 = time.perf_counter()
            for q, v in zip(qids, qv):
                s, n = cp.scores(v, b2)
                run[q] = full_doc_run(s, index, 100)
                scored.append(n)
            ms = (time.perf_counter() - t0) * 1000 / len(qids)
            m, pq = _eval(run, qrels)
            # how often is the exact top-1 document still found
            top1 = float(np.mean([run[q][:1] == exact[q][:1] for q in qids]))
            rows.append({"config": f"cluster pruning b1={b1} b2={b2}", "b1": b1, "b2": b2,
                         "vectors_scored": float(np.mean(scored)), "ms_per_query": ms, "top1_agreement": top1,
                         "p_vs_exact": paired_t_pvalue(pq_exact, pq), **m})
            print(f"b1={b1} b2={b2:<3} nDCG@10={m['nDCG@10']:.3f} R@100={m['R@100']:.3f} "
                  f"scored={np.mean(scored):.0f} top1-agree={top1:.2f}")
    _save("cluster", {"encoder": Path(encoder_path).name, "n_leaders": int(round(np.sqrt(len(emb)))),
                      "rows": rows})


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("which", nargs="+", choices=["smart", "analysis", "spell", "cluster", "all"])
    ap.add_argument("--encoder", default=str(art(DS) / "bge-ft"),
                    help="encoder for the cluster-pruning queries (must match --emb)")
    ap.add_argument("--emb", default="emb_window120_bgeft.npy")
    a = ap.parse_args()
    w = set(a.which)
    if "all" in w:
        w = {"smart", "analysis", "spell", "cluster"}
    if "smart" in w:
        exp_smart()
    if "analysis" in w:
        exp_analysis()
    if "spell" in w:
        exp_spell()
    if "cluster" in w:
        exp_cluster(a.encoder, a.emb)
