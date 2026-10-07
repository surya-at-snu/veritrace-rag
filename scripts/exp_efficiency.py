import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from veritrace.config import RESULTS, art
from veritrace.corpus import split_queries
from veritrace.index import InvertedIndex
from veritrace.metrics import evaluate_run
from veritrace.sparse import SparseRetriever, heap_topk


def to_doc_run(index, hits, k=100):
    seen, out = set(), []
    for c, _ in hits:
        d = index.docs[index.chunk_doc[c]].doc_id
        if d not in seen:
            seen.add(d)
            out.append(d)
        if len(out) >= k:
            break
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="scifact")
    ap.add_argument("--index", default="window120")
    args = ap.parse_args()
    out = art(args.dataset)
    index = InvertedIndex.load(out / f"index_{args.index}.pkl")
    p = json.load(open(out / "sparse_params.json"))
    sp = SparseRetriever(index, p["k1"], p["b"], p["g"])
    queries, qrels = split_queries(args.dataset, "test")
    qids = list(qrels)
    rows = []

    def run_config(name, fn, scored_fn, extra=None):
        t0 = time.perf_counter()
        run, scored = {}, []
        for q in qids:
            hits = fn(queries[q])
            run[q] = to_doc_run(index, hits)
        ms = (time.perf_counter() - t0) * 1000 / len(qids)
        for q in qids:
            scored.append(scored_fn(queries[q]))
        ev = evaluate_run(run, qrels)["mean"]
        rows.append({"config": name, "nDCG@10": ev["nDCG@10"], "R@100": ev["R@100"], "ms_per_query": ms,
                     "chunks_scored": float(np.mean(scored)), **(extra or {})})
        print(f"{name:28s} nDCG@10={ev['nDCG@10']:.4f} R@100={ev['R@100']:.4f}  {ms:6.2f} ms/q  scored={np.mean(scored):.0f}")

    def n_nonzero(text):
        return int((sp.chunk_scores(text) > 0).sum())

    run_config("exhaustive (heap top-K)", lambda t: heap_topk(sp.chunk_scores(t), 300), n_nonzero)
    # champion lists of growing size vs exhaustive scoring
    for r in (5, 10, 20, 50, 100, 200, 400):
        def cand(t, r=r):
            q = sp.query_terms(t)
            lists = [index.body.champion_list(x, r) for x in q if index.body.tid(x) >= 0]
            return len(np.unique(np.concatenate(lists))) if lists else 0
        run_config(f"champion lists r={r}", lambda t, r=r: sp.search_champions(t, r, 300), cand)
    # index elimination: drop the low idf query terms
    for thr in (0.5, 1.0, 1.5, 2.0, 3.0):
        def nz(t, thr=thr):
            return int((sp.chunk_scores(t, min_idf=thr) > 0).sum())
        run_config(f"index elimination idf>={thr}", lambda t, thr=thr: heap_topk(sp.chunk_scores(t, min_idf=thr), 300), nz)
    # many-term matching: only chunks containing at least m of the query terms are kept
    for m in (2, 3):
        def nm(t, m=m):
            return int((sp.bm25_min_match(t, m) > 0).sum())
        run_config(f"many-term matching m>={m}", lambda t, m=m: heap_topk(sp.bm25_min_match(t, m), 300), nm)
    # tiered index: tier 1 = champion lists, fall back to the full postings (tier 2) only when tier 1
    # gives fewer than 100 documents
    for r in (20, 50, 100):
        tiers = {}

        def tiered(t, r=r):
            hits, tier, n = sp.search_tiered(t, r, k=300, k_docs=100)
            tiers[t] = (tier, n)
            return hits
        run_config(f"tiered index r={r}", tiered, lambda t: tiers[t][1])
        rows[-1]["tier2_fallback_rate"] = float(np.mean([v[0] == 2 for v in tiers.values()]))
    rows.append({"config": "_impact_params", "k1": sp.k1, "b": sp.b})
    json.dump(rows, open(RESULTS / f"{args.dataset}_efficiency.json", "w"), indent=1)


if __name__ == "__main__":
    main()
