import argparse
import csv
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from veritrace.config import DATA, RESULTS
from veritrace.metrics import ndcg_at, precision_at, reciprocal_rank

QFILE = DATA / "custom_queries.tsv"
JFILE = DATA / "custom_judgments.csv"
RUNFILE = RESULTS / "custom_runs.json"
DEPTH = 5


def read_queries():
    with open(QFILE, encoding="utf-8") as f:
        return {r["qid"]: r["query"] for r in csv.DictReader(f, delimiter="\t")}


def make():
    from veritrace.fusion import full_doc_run
    from veritrace.pipeline import VeriTrace
    vt = VeriTrace("scifact", generator=None, load_verifier=False)
    queries = read_queries()
    runs = {"tfidf_lnc_ltc": {}, "bm25_zones": {}, "veritrace": {}}
    for qid, q in queries.items():
        runs["tfidf_lnc_ltc"][qid] = full_doc_run(vt.sparse.chunk_scores(q, "tfidf", g=0.0), vt.index, 20)
        runs["bm25_zones"][qid] = full_doc_run(vt.sparse.chunk_scores(q), vt.index, 20)
        runs["veritrace"][qid] = [x["doc_id"] for x in vt.retrieve(q).ranked[:20]]
    json.dump(runs, open(RUNFILE, "w"), indent=1)
    docs = {d.doc_id: d for d in vt.index.docs}
    old = {}
    if JFILE.exists():
        with open(JFILE, encoding="utf-8") as f:
            old = {(r["qid"], r["doc_id"]): r["relevant"] for r in csv.DictReader(f)}
    with open(JFILE, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["qid", "query", "doc_id", "title", "abstract_start", "relevant"])
        for qid, q in queries.items():
            # pooling: union of the top 5 from all three systems
            pool = list(dict.fromkeys(d for r in runs.values() for d in r[qid][:DEPTH]))
            for d in sorted(pool):
                w.writerow([qid, q, d, docs[d].title, docs[d].text[:600], old.get((qid, d), "")])
    print(f"wrote {JFILE}; fill the 'relevant' column with 1/0, then run --eval")


def evaluate():
    runs = json.load(open(RUNFILE))
    qrels = {}
    with open(JFILE, encoding="utf-8") as f:
        for r in csv.DictReader(f):
            if r["relevant"].strip() == "":
                raise SystemExit(f"unjudged row: {r['qid']} {r['doc_id']}")
            qrels.setdefault(r["qid"], {})[r["doc_id"]] = int(r["relevant"])
    out = {}
    for name, run in runs.items():
        m = {"P@1": [], "P@5": [], "R@5 (pool)": [], "MRR@5": [], "nDCG@5": []}
        for qid, rel in qrels.items():
            ranked = run[qid]
            n_rel = sum(rel.values())
            m["P@1"].append(precision_at(ranked, rel, 1))
            m["P@5"].append(precision_at(ranked, rel, 5))
            # recall is measured against the judged pool
            m["R@5 (pool)"].append(sum(rel.get(d, 0) for d in ranked[:5]) / n_rel if n_rel else 0.0)
            m["MRR@5"].append(reciprocal_rank(ranked, rel, 5))
            m["nDCG@5"].append(ndcg_at(ranked, rel, 5))
        out[name] = {k: float(np.mean(v)) for k, v in m.items()}
        out[name]["per_query_P@5"] = dict(zip(qrels, m["P@5"]))
        print(f"{name:14s} " + "  ".join(f"{k}={v:.3f}" for k, v in out[name].items() if k != "per_query_P@5"))
    out["n_queries"] = len(qrels)
    out["n_judged"] = sum(len(v) for v in qrels.values())
    out["n_relevant"] = sum(sum(v.values()) for v in qrels.values())
    json.dump(out, open(RESULTS / "custom_eval.json", "w"), indent=1)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--make", action="store_true")
    ap.add_argument("--eval", action="store_true")
    a = ap.parse_args()
    if a.make:
        make()
    if a.eval:
        evaluate()
