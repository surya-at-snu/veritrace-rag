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
from veritrace.dense import encode_queries, load_encoder
from veritrace.fusion import build_pool, convex, doc_run, full_doc_run, rrf, to_docs
from veritrace.index import InvertedIndex
from veritrace.metrics import evaluate_run, ndcg_at, paired_t_pvalue
from veritrace.qpp import QPP_FEATURES, as_vector, qpp_features
from veritrace.rerank import ce_scores
from veritrace.router import ALPHAS, CE_ARB_FEATURES, FusionRouter, ce_arbitration
from veritrace.sparse import SparseRetriever


# grid search k1, b and the title weight g on the training queries
def tune_bm25(index, queries, qrels, max_q=400):
    qids = list(queries)[:max_q]
    best = (-1, None)
    grid = []
    sp = SparseRetriever(index)
    for k1 in (0.6, 0.9, 1.2, 1.5):
        for b in (0.3, 0.5, 0.75, 0.9):
            sp.k1, sp.b = k1, b
            sp._norm_cache.clear()
            comps = []
            for q in qids:
                qt = sp.query_terms(queries[q])
                comps.append((sp.bm25_zone(qt, index.title)[index.chunk_doc], sp.bm25_zone(qt, index.body)))
            for g in (0.0, 0.1, 0.2, 0.3, 0.4, 0.5):
                vals = []
                for q, (t, bd) in zip(qids, comps):
                    s = index.doc_max((1 - g) * bd + g * t)
                    top = np.argpartition(-s, 10)[:10]
                    top = top[np.argsort(-s[top])]
                    vals.append(ndcg_at([index.docs[i].doc_id for i in top], qrels[q], 10))
                m = float(np.mean(vals))
                grid.append({"k1": k1, "b": b, "g": g, "ndcg10": m})
                if m > best[0]:
                    best = (m, (k1, b, g))
    return best, grid


def tune_tfidf_g(index, queries, qrels, max_q=400):
    sp = SparseRetriever(index)
    qids = list(queries)[:max_q]
    comps = [sp.zone_scores(queries[q], "tfidf") for q in qids]
    res = {}
    for g in (0.0, 0.1, 0.2, 0.3, 0.4, 0.5):
        vals = []
        for q, (t, bd) in zip(qids, comps):
            s = index.doc_max((1 - g) * bd + g * t)
            top = np.argpartition(-s, 10)[:10]
            top = top[np.argsort(-s[top])]
            vals.append(ndcg_at([index.docs[i].doc_id for i in top], qrels[q], 10))
        res[g] = float(np.mean(vals))
    return max(res, key=res.get), res


def make_pools(index, sp, emb, queries, enc, pool_n=200):
    qids = list(queries)
    qv = encode_queries([queries[q] for q in qids], enc)
    out = {}
    for qid, v in zip(qids, qv):
        text = queries[qid]
        title_c, body = sp.zone_scores(text, "bm25")
        sparse_full = (1 - sp.g) * body + sp.g * title_c
        t_title, t_body = sp.zone_scores(text, "tfidf")
        dense_full = emb @ v
        # candidate pool + qpp features for every query
        pool = build_pool(sparse_full, dense_full, pool_n, {"bm25_title": title_c, "bm25_body": body,
                                                             "tfidf_title": t_title, "tfidf_body": t_body})
        feats = qpp_features(index, text, pool, sparse_full)
        runs = {
            "tfidf_body": full_doc_run(t_body, index, 1000),
            "tfidf_zone": full_doc_run((1 - sp.g_tfidf) * t_body + sp.g_tfidf * t_title, index, 1000),
            "bm25_body": full_doc_run(body, index, 1000),
            "bm25_zone": full_doc_run(sparse_full, index, 1000),
            "dense": full_doc_run(dense_full, index, 1000),
        }
        out[qid] = {"pool": pool, "qpp": feats, "runs": runs, "qvec": v}
    return out


# cross encoder on the top 3 of each list, batched over all queries
def add_ce_arbitration(index, pools, queries):
    pairs, owners = [], []
    for q, P in pools.items():
        p = P["pool"]
        _, _, sc = to_docs(p.sparse, p.chunks, index.chunk_doc)
        _, _, dc = to_docs(p.dense, p.chunks, index.chunk_doc)
        for c in list(sc[:3]) + list(dc[:3]):
            ch = index.chunks[int(c)]
            pairs.append((queries[q], f"{ch.title}. {ch.text}"))
        owners.append(q)
    ce = ce_scores(pairs).reshape(len(owners), 6)
    for q, row in zip(owners, ce):
        pools[q]["ce_arb"] = ce_arbitration(row[:3], row[3:])


ROUTER_FEATURES = QPP_FEATURES + CE_ARB_FEATURES


def router_X(pools, qids, names=None):
    names = names or ROUTER_FEATURES
    return np.stack([np.array([{**pools[q]["qpp"], **pools[q].get("ce_arb", {})}[n] for n in names], dtype=np.float32)
                     for q in qids])


def alpha_curves(index, pools, qrels):
    doc_ids = [d.doc_id for d in index.docs]
    qids = list(qrels)
    C = np.zeros((len(qids), len(ALPHAS)))
    for i, q in enumerate(qids):
        p = pools[q]["pool"]
        for j, a in enumerate(ALPHAS):
            C[i, j] = ndcg_at(doc_run(convex(p, a), p.chunks, index.chunk_doc, doc_ids, 10), qrels[q], 10)
    return qids, C


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="scifact")
    ap.add_argument("--index", default="window120")
    ap.add_argument("--emb", default="bge")
    ap.add_argument("--train-limit", type=int, default=1500)
    ap.add_argument("--test-split", default="test")
    args = ap.parse_args()
    ds, out = args.dataset, art(args.dataset)
    t_start = time.time()

    index = InvertedIndex.load(out / f"index_{args.index}.pkl")
    emb = np.load(out / f"emb_{args.index}_{args.emb}.npy")
    tr_q, tr_rel = split_queries(ds, "train", limit=args.train_limit)
    te_q, te_rel = split_queries(ds, args.test_split)
    print(f"[{ds}] chunks={index.n_chunks} docs={index.n_docs} train_q={len(tr_q)} test_q={len(te_q)}")

    (best_ndcg, (k1, b, g)), grid = tune_bm25(index, tr_q, tr_rel)
    g_tfidf, tfidf_grid = tune_tfidf_g(index, tr_q, tr_rel)
    print(f"BM25 tuned on train: k1={k1} b={b} g_title={g} (train nDCG@10={best_ndcg:.4f}); tf-idf g={g_tfidf}")
    sp = SparseRetriever(index, k1=k1, b=b, title_weight=g)
    sp.g_tfidf = g_tfidf

    enc = load_encoder()
    pools = {}
    for split, qs in (("train", tr_q), ("test", te_q)):
        t0 = time.time()
        pools[split] = make_pools(index, sp, emb, qs, enc)
        add_ce_arbitration(index, pools[split], qs)
        with open(out / f"pools_{split}.pkl", "wb") as f:
            pickle.dump(pools[split], f)
        print(f"pools {split}: {len(qs)} queries in {time.time() - t0:.1f}s")

    tr_ids, tr_C = alpha_curves(index, pools["train"], tr_rel)
    te_ids, te_C = alpha_curves(index, pools["test"], te_rel)
    X_tr, X_te = router_X(pools["train"], tr_ids), router_X(pools["test"], te_ids)
    # the router only ever sees training queries
    router = FusionRouter().fit(X_tr, tr_C, ROUTER_FEATURES)
    router.save(out / "router.pkl")
    router_qpp = FusionRouter().fit(router_X(pools["train"], tr_ids, QPP_FEATURES), tr_C, QPP_FEATURES)
    a_fixed = router.best_fixed_alpha
    a_pred = router.predict_alpha(X_te)
    a_pred_qpp = router_qpp.predict_alpha(router_X(pools["test"], te_ids, QPP_FEATURES))
    print(f"best fixed alpha (train) = {a_fixed}; router a0={router.a0} beta={router.beta} "
          f"cv gain={router.cv_gain:+.4f};  predicted alpha mean={a_pred.mean():.2f} std={a_pred.std():.2f}")
    print(f"QPP-only router a0={router_qpp.a0} beta={router_qpp.beta} cv gain={router_qpp.cv_gain:+.4f}")

    doc_ids = [d.doc_id for d in index.docs]
    P = pools["test"]
    runs = {name: {q: P[q]["runs"][name] for q in te_ids} for name in
            ("tfidf_body", "tfidf_zone", "bm25_body", "bm25_zone", "dense")}
    runs["rrf"] = {q: doc_run(rrf(P[q]["pool"]), P[q]["pool"].chunks, index.chunk_doc, doc_ids, 100) for q in te_ids}
    runs["convex_fixed"] = {q: doc_run(convex(P[q]["pool"], a_fixed), P[q]["pool"].chunks, index.chunk_doc, doc_ids, 100)
                            for q in te_ids}
    runs["router"] = {q: doc_run(convex(P[q]["pool"], a), P[q]["pool"].chunks, index.chunk_doc, doc_ids, 100)
                      for q, a in zip(te_ids, a_pred)}
    runs["router_qpp_only"] = {q: doc_run(convex(P[q]["pool"], a), P[q]["pool"].chunks, index.chunk_doc, doc_ids, 100)
                               for q, a in zip(te_ids, a_pred_qpp)}
    # oracle = best weight per test query, only an upper bound
    a_oracle = ALPHAS[np.argmax(te_C, axis=1)]
    runs["oracle_alpha"] = {q: doc_run(convex(P[q]["pool"], a), P[q]["pool"].chunks, index.chunk_doc, doc_ids, 100)
                            for q, a in zip(te_ids, a_oracle)}

    results = {}
    for name, run in runs.items():
        ev = evaluate_run(run, te_rel)
        results[name] = ev
        print(f"{name:14s} " + "  ".join(f"{m}={v:.4f}" for m, v in ev["mean"].items()))
    base = results["convex_fixed"]["per_query"]["nDCG@10"]
    for name in ("rrf", "router", "router_qpp_only", "bm25_zone", "dense"):
        p = paired_t_pvalue(results[name]["per_query"]["nDCG@10"], base)
        print(f"paired t-test nDCG@10 {name} vs convex_fixed: p={p:.4f}")

    summary = {
        "dataset": ds, "n_train": len(tr_ids), "n_test": len(te_ids),
        "bm25": {"k1": k1, "b": b, "g_title": g, "train_ndcg10": best_ndcg}, "bm25_grid": grid,
        "tfidf_g": g_tfidf, "tfidf_grid": tfidf_grid,
        "alpha_fixed": a_fixed, "alpha_pred": a_pred.tolist(), "alpha_oracle": a_oracle.tolist(),
        "train_mean_curve": tr_C.mean(0).tolist(), "test_mean_curve": te_C.mean(0).tolist(),
        "router_cv_gain": router.cv_gain, "router_a0": router.a0, "router_beta": router.beta,
        "router_qpp_cv_gain": router_qpp.cv_gain, "router_coef": router.coefficients(),
        "router_qpp_coef": router_qpp.coefficients(), "router_p_test": router.predict_proba(X_te).tolist(),
        "alphas": ALPHAS.tolist(),
        "ce_arb_test": {q: P[q].get("ce_arb") for q in te_ids},
        "metrics": {n: r["mean"] for n, r in results.items()},
        "per_query_ndcg": {n: r["per_query"]["nDCG@10"] for n, r in results.items()},
        "test_qids": te_ids,
        "qpp_test": {q: P[q]["qpp"] for q in te_ids},
    }
    with open(RESULTS / f"{ds}_retrieval.json", "w") as f:
        json.dump(summary, f, indent=1)
    with open(out / "sparse_params.json", "w") as f:
        json.dump({"k1": k1, "b": b, "g": g, "g_tfidf": g_tfidf, "alpha_fixed": a_fixed}, f)
    print(f"done in {time.time() - t_start:.0f}s")


if __name__ == "__main__":
    main()
