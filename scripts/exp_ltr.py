import argparse
import json
import pickle
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from veritrace.config import RESULTS, art
from veritrace.corpus import split_queries
from veritrace.fusion import convex, doc_run
from veritrace.index import InvertedIndex
from veritrace.ltr import DOC_FEATURES, LTR_FEATURES, LTRReranker, feature_matrix
from veritrace.metrics import evaluate_run, paired_t_pvalue
from veritrace.rerank import ce_scores
from veritrace.router import FusionRouter, router_vector

N_CAND = 30


def build(index, pools, queries, qrels, alpha_of, cache):
    if cache.exists():
        with open(cache, "rb") as f:
            return pickle.load(f)
    data = {}
    t0 = time.time()
    for i, q in enumerate(qrels):
        P = pools[q]
        # ltr features for the top 30 docs, cached because the cross encoder is slow
        X, d, c = feature_matrix(index, queries[q], P["pool"], P["qpp"], alpha_of(q), ce_scores, N_CAND)
        y = np.array([qrels[q].get(index.docs[j].doc_id, 0) for j in d])
        data[q] = (X, y, d, c)
        if (i + 1) % 200 == 0:
            print(f"  {i + 1}/{len(qrels)} queries  {time.time() - t0:.0f}s", flush=True)
    with open(cache, "wb") as f:
        pickle.dump(data, f)
    return data


def rerank_run(index, data, pools, scorer, alpha_of, qids):
    doc_ids = [d.doc_id for d in index.docs]
    run = {}
    for q in qids:
        X, _, d, _ = data[q]
        s = scorer(X)
        top = [doc_ids[j] for j in d[np.argsort(-s, kind="stable")]]
        P = pools[q]["pool"]
        rest = [x for x in doc_run(convex(P, alpha_of(q)), P.chunks, index.chunk_doc, doc_ids, 200) if x not in set(top)]
        run[q] = top + rest[:100 - len(top)]
    return run


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="scifact")
    ap.add_argument("--index", default="window120")
    ap.add_argument("--train-limit", type=int, default=1500)
    ap.add_argument("--test-split", default="test")
    args = ap.parse_args()
    ds, out = args.dataset, art(args.dataset)
    index = InvertedIndex.load(out / f"index_{args.index}.pkl")
    router = FusionRouter.load(out / "router.pkl")
    a_fixed = router.best_fixed_alpha
    te_q, te_rel = split_queries(ds, args.test_split)
    pools = {}
    for split in ("train", "test"):
        with open(out / f"pools_{split}.pkl", "rb") as f:
            pools[split] = pickle.load(f)
    tr_q, tr_rel = split_queries(ds, "train")
    # only use training queries that were pooled earlier
    keep = sorted(q for q in tr_rel if q in pools["train"])
    if args.train_limit and len(keep) > args.train_limit:
        import random
        keep = sorted(random.Random(5).sample(keep, args.train_limit))
    tr_rel = {q: tr_rel[q] for q in keep}

    alpha = {}
    for split in ("train", "test"):
        qs = list(pools[split])
        X = np.stack([router_vector(pools[split][q], router.feature_names) for q in qs])
        alpha.update(dict(zip(qs, router.predict_alpha(X))))

    t0 = time.time()
    tr = build(index, pools["train"], tr_q, tr_rel, lambda q: alpha[q], out / "ltr_train.pkl")
    te = build(index, pools["test"], te_q, te_rel, lambda q: alpha[q], out / "ltr_test.pkl")
    print(f"features built in {time.time() - t0:.0f}s;  recall of relevant docs in top-{N_CAND} candidates (test): "
          f"{np.mean([te[q][1].sum() > 0 for q in te_rel]):.3f}")

    tr_ids, te_ids = list(tr_rel), list(te_rel)
    Xs, ys = [tr[q][0] for q in tr_ids], [tr[q][1] for q in tr_ids]
    variants = {
        "ltr_full": None,
        "ltr_no_ce": [i for i, n in enumerate(LTR_FEATURES) if not n.startswith("ce")],
        "ltr_ir_only": [i for i, n in enumerate(LTR_FEATURES) if n not in ("ce", "ce_rank", "ce_minus_max", "dense", "de_rank")],
    }
    ce_col = LTR_FEATURES.index("ce")
    runs = {
        "router_fusion": rerank_run(index, te, pools["test"], lambda X: X[:, LTR_FEATURES.index("fused")], lambda q: alpha[q], te_ids),
        "ce_rerank": rerank_run(index, te, pools["test"], lambda X: X[:, ce_col], lambda q: alpha[q], te_ids),
    }
    models = {}
    for name, fidx in variants.items():
        # full model plus two ablations
        m = LTRReranker().fit(Xs, ys, fidx)
        models[name] = m
        runs[name] = rerank_run(index, te, pools["test"], m.score, lambda q: alpha[q], te_ids)
    models["ltr_full"].save(out / "ltr.pkl")

    results = {}
    for name, run in runs.items():
        ev = evaluate_run(run, te_rel)
        results[name] = ev
        print(f"{name:14s} " + "  ".join(f"{m}={v:.4f}" for m, v in ev["mean"].items()))
    for name in ("ce_rerank", "ltr_full"):
        p = paired_t_pvalue(results[name]["per_query"]["nDCG@10"], results["router_fusion"]["per_query"]["nDCG@10"])
        print(f"paired t-test nDCG@10 {name} vs router_fusion: p={p:.4g}")

    imp = models["ltr_full"].model.booster_.feature_importance("gain")
    summary = {"dataset": ds, "n_candidates": N_CAND,
               "metrics": {n: r["mean"] for n, r in results.items()},
               "per_query_ndcg": {n: r["per_query"]["nDCG@10"] for n, r in results.items()},
               "test_qids": te_ids,
               "feature_gain": dict(zip(LTR_FEATURES, (imp / imp.sum()).round(4).tolist())),
               "doc_features": DOC_FEATURES}
    with open(RESULTS / f"{ds}_ltr.json", "w") as f:
        json.dump(summary, f, indent=1)


if __name__ == "__main__":
    main()
