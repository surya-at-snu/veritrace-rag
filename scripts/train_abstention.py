import argparse
import json
import pickle
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
from sklearn.metrics import roc_auc_score

from veritrace.abstain import RetrievalConfidence, risk_coverage
from veritrace.config import RESULTS, art
from veritrace.corpus import split_queries
from veritrace.fusion import convex, doc_run
from veritrace.index import InvertedIndex
from veritrace.router import FusionRouter, router_vector

K = 5


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="scifact")
    ap.add_argument("--index", default="window120")
    ap.add_argument("--train-limit", type=int, default=1500)
    args = ap.parse_args()
    ds, out = args.dataset, art(args.dataset)
    index = InvertedIndex.load(out / f"index_{args.index}.pkl")
    doc_ids = [d.doc_id for d in index.docs]
    router = FusionRouter.load(out / "router.pkl")
    names = router.feature_names
    data = {}
    for split in ("train", "test"):
        _, rel = split_queries(ds, split, limit=args.train_limit if split == "train" else None)
        with open(out / f"pools_{split}.pkl", "rb") as f:
            P = pickle.load(f)
        qids = list(rel)
        X = np.stack([router_vector(P[q], names) for q in qids])
        alpha = router.predict_alpha(X)
        # label: was any relevant doc in the top 5
        y = np.array([any(d in rel[q] for d in doc_run(convex(P[q]["pool"], a), P[q]["pool"].chunks, index.chunk_doc, doc_ids, K))
                      for q, a in zip(qids, alpha)], dtype=int)
        data[split] = (X, y)
    (Xtr, ytr), (Xte, yte) = data["train"], data["test"]
    model = RetrievalConfidence().fit(Xtr, ytr, names)
    model.save(out / "abstain.pkl")
    conf = model.predict(Xte)

    curves, res = {}, {}
    baselines = {"VeriTrace confidence (LR)": conf,
                 "BM25 top-1 score": Xte[:, names.index("sp_top1")],
                 "Dense top-1 cosine": Xte[:, names.index("de_top1")],
                 "Cross-encoder top-1": np.maximum(Xte[:, names.index("ce_sp1")], Xte[:, names.index("ce_de1")])}
    for name, c in baselines.items():
        cov, risk, aurc = risk_coverage(c, yte)
        res[name] = {"auroc": float(roc_auc_score(yte, c)), "aurc": aurc,
                     "risk_at_80cov": float(risk[int(0.8 * len(risk)) - 1]), "risk_at_100cov": float(risk[-1])}
        curves[name] = {"coverage": cov[::3].tolist(), "risk": risk[::3].tolist()}
        print(f"{name:28s} AUROC={res[name]['auroc']:.3f}  AURC={aurc:.4f}  risk@80%cov={res[name]['risk_at_80cov']:.3f}  risk@100%={risk[-1]:.3f}")
    answered = conf >= model.tau
    summary = {"dataset": ds, "k": K, "base_success_rate_test": float(yte.mean()), "tau": model.tau,
               "test_abstain_rate": float(1 - answered.mean()),
               "test_success_when_answered": float(yte[answered].mean()),
               "test_success_when_abstained": float(yte[~answered].mean()) if (~answered).any() else None,
               "results": res, "curves": curves,
               "coef": dict(zip(names, model.clf.coef_[0].round(4).tolist()))}
    print(f"tau={model.tau:.3f}: abstain on {summary['test_abstain_rate']:.1%} of test queries; success@{K} "
          f"answered={summary['test_success_when_answered']:.3f} vs abstained={summary['test_success_when_abstained']}")
    with open(RESULTS / f"{ds}_abstention.json", "w") as f:
        json.dump(summary, f, indent=1)


if __name__ == "__main__":
    main()
