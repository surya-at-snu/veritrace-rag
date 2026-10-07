import json
import pickle
import random
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
from sklearn.metrics import f1_score, precision_recall_fscore_support, roc_auc_score
from sklearn.model_selection import StratifiedKFold

from veritrace.config import RESULTS, art
from veritrace.corpus import load_scifact_claims
from veritrace.corrupt import CONTRADICTED, LABELS, SUPPORTED, UNSUPPORTED, corrupt
from veritrace.index import InvertedIndex
from veritrace.sparse import SparseRetriever
from veritrace.verify import FEATURES, IR_FEATURES, NO_NLI_FEATURES, ClaimVerifier, Evidence, Featurizer

OUT = art("scifact")


def chunk_ev(index, c):
    ch = index.chunks[c]
    return Evidence(ch.title, ch.sentences, c)


def doc_chunks(index, d):
    s = int(index.doc_chunk_start[d])
    e = int(index.doc_chunk_start[d + 1]) if d + 1 < index.n_docs else index.n_chunks
    return list(range(s, e))


# (claim, chunk, label) pairs from scifact rationales plus hard negatives
def build_pairs(index, sp, claims, rnd, synthetic: str = "none"):
    doc_pos = {d.doc_id: i for i, d in enumerate(index.docs)}
    pairs = []
    for cl in claims:
        text, cid = cl["claim"], cl["id"]
        ev_docs = cl["evidence"]
        involved = set(ev_docs) | {str(x) for x in cl["cited_doc_ids"]}
        for did, evs in ev_docs.items():
            if did not in doc_pos:
                continue
            label = SUPPORTED if evs[0]["label"] == "SUPPORT" else CONTRADICTED
            rat = {s for e in evs for s in e["sentences"]}
            chs = doc_chunks(index, doc_pos[did])
            overlap = [len(rat & set(range(index.chunks[c].sent_start, index.chunks[c].sent_end))) for c in chs]
            best = chs[int(np.argmax(overlap))]
            pairs.append((text, chunk_ev(index, best), label, {"type": "gold", "claim": cid}))
            for c, o in zip(chs, overlap):
                if o == 0:
                    pairs.append((text, chunk_ev(index, c), UNSUPPORTED, {"type": "same_doc", "claim": cid}))
            if label == SUPPORTED and synthetic != "none":
                kinds = ["negation", "direction", "number", "entity"]
                todo = kinds if synthetic == "all" else [None]
                for k in todo:
                    # n3: corrupt supported claims to get free hallucination examples
                    r = corrupt(text, index.chunks[best].text, index, rnd, k)
                    if r:
                        pairs.append((r[0], chunk_ev(index, best), r[2], {"type": f"syn_{r[1]}", "claim": cid,
                                                                          "orig": text}))
        if not ev_docs:
            for did in cl["cited_doc_ids"]:
                if str(did) in doc_pos:
                    for c in doc_chunks(index, doc_pos[str(did)])[:2]:
                        pairs.append((text, chunk_ev(index, c), UNSUPPORTED, {"type": "nei_cited", "claim": cid}))
        s = sp.chunk_scores(text)
        for c in np.argsort(-s)[:30]:
            if index.docs[index.chunk_doc[c]].doc_id not in involved:
                pairs.append((text, chunk_ev(index, int(c)), UNSUPPORTED, {"type": "bm25_neg", "claim": cid}))
                break
        if rnd.random() < 0.34:
            c = rnd.randrange(index.n_chunks)
            if index.docs[index.chunk_doc[c]].doc_id not in involved:
                pairs.append((text, chunk_ev(index, c), UNSUPPORTED, {"type": "random", "claim": cid}))
    return pairs


def featurize_cached(fz, pairs, name):
    path = OUT / f"verifier_feats_{name}.pkl"
    if path.exists():
        with open(path, "rb") as f:
            return pickle.load(f)
    t0 = time.time()
    X, info = fz.featurize([(c, e) for c, e, _, _ in pairs])
    y = np.array([lab for _, _, lab, _ in pairs])
    meta = [m for _, _, _, m in pairs]
    with open(path, "wb") as f:
        pickle.dump((X, y, meta, info), f)
    print(f"featurized {name}: {len(pairs)} pairs in {time.time() - t0:.0f}s", flush=True)
    return X, y, meta, info


def best_threshold(scores, is_pos):
    grid = np.unique(np.quantile(scores, np.linspace(0.01, 0.99, 197)))
    f1s = [f1_score(is_pos, scores >= t) for t in grid]
    return float(grid[int(np.argmax(f1s))])


def binary_report(scores, is_pos, tau):
    pred = scores >= tau
    p, r, f, _ = precision_recall_fscore_support(is_pos, pred, average="binary", zero_division=0)
    hp, hr, hf, _ = precision_recall_fscore_support(~is_pos, ~pred, average="binary", zero_division=0)
    return {"auroc": float(roc_auc_score(is_pos, scores)), "tau": tau, "support_P": float(p), "support_R": float(r),
            "support_F1": float(f), "halluc_P": float(hp), "halluc_R": float(hr), "halluc_F1": float(hf),
            "acc": float((pred == is_pos).mean())}


def main():
    rnd = random.Random(7)
    index = InvertedIndex.load(OUT / "index_window120.pkl")
    params = json.load(open(OUT / "sparse_params.json"))
    sp = SparseRetriever(index, params["k1"], params["b"], params["g"])
    tr_claims, dev_claims = load_scifact_claims("train"), load_scifact_claims("dev")

    tr_pairs = build_pairs(index, sp, tr_claims, rnd, synthetic="one")
    dev_pairs = build_pairs(index, sp, dev_claims, random.Random(11), synthetic="all")
    print("train pairs by type:", {t: sum(m["type"] == t for *_, m in tr_pairs) for t in sorted({m["type"] for *_, m in tr_pairs})})
    print("dev pairs by type:", {t: sum(m["type"] == t for *_, m in dev_pairs) for t in sorted({m["type"] for *_, m in dev_pairs})})

    fz = Featurizer(index)
    Xtr, ytr, mtr, _ = featurize_cached(fz, tr_pairs, "train")
    Xdv, ydv, mdv, idv = featurize_cached(fz, dev_pairs, "dev")
    syn_tr = np.array([m["type"].startswith("syn") for m in mtr])
    syn_dv = np.array([m["type"].startswith("syn") for m in mdv])
    human = ~syn_dv

    variants = {
        "full": (FEATURES, np.ones(len(ytr), bool)),
        "full_no_synthetic": (FEATURES, ~syn_tr),
        "no_nli": (NO_NLI_FEATURES, np.ones(len(ytr), bool)),
        "ir_only": (IR_FEATURES, np.ones(len(ytr), bool)),
    }
    results, models = {}, {}
    for name, (feats, mask) in variants.items():
        oof = np.zeros(mask.sum())
        Xm, ym = Xtr[mask], ytr[mask]
        for a, b in StratifiedKFold(5, shuffle=True, random_state=0).split(Xm, ym):
            oof[b] = ClaimVerifier(feats).fit(Xm[a], ym[a]).predict_proba(Xm[b])[:, 0]
        # threshold picked on out of fold training predictions, never on dev
        tau = best_threshold(oof, ym == SUPPORTED)
        m = ClaimVerifier(feats).fit(Xm, ym)
        m.tau_support = tau
        models[name] = m
        P = m.predict_proba(Xdv)
        rep = binary_report(P[human, 0], ydv[human] == SUPPORTED, tau)
        pred3 = np.argmax(P[human], 1)
        rep["macro_F1_3class"] = float(f1_score(ydv[human], pred3, average="macro"))
        rep["per_class_F1"] = dict(zip(LABELS, f1_score(ydv[human], pred3, average=None).round(4).tolist()))
        rep["synthetic_detection"] = {k: float((P[syn_dv & np.array([mm["type"] == k for mm in mdv]), 0] < tau).mean())
                                      for k in ("syn_negation", "syn_direction", "syn_number", "syn_entity")}
        results[name] = rep

    # single score baselines, including the cosine checker from the track description
    for name, col in (("cosine_chunk", "cos_chunk"), ("cosine_best_sentence", "cos_best"),
                      ("dense_cosine", "dense_best"), ("nli_entailment", "nli_e_max")):
        j = FEATURES.index(col)
        tau = best_threshold(Xtr[~syn_tr, j], ytr[~syn_tr] == SUPPORTED)
        rep = binary_report(Xdv[human, j], ydv[human] == SUPPORTED, tau)
        rep["synthetic_detection"] = {k: float((Xdv[syn_dv & np.array([mm["type"] == k for mm in mdv]), j] < tau).mean())
                                      for k in ("syn_negation", "syn_direction", "syn_number", "syn_entity")}
        results[name] = rep
    e, c = Xdv[human, FEATURES.index("nli_e_max")], Xdv[human, FEATURES.index("nli_c_max")]
    pred = np.where(e >= 0.5, SUPPORTED, np.where(c >= 0.5, CONTRADICTED, UNSUPPORTED))
    results["nli_entailment"]["macro_F1_3class"] = float(f1_score(ydv[human], pred, average="macro"))

    gold_sup = np.array([mm["type"] == "gold" for mm in mdv]) & (ydv == SUPPORTED)
    for name, m in models.items():
        results[name]["false_alarm_on_supported"] = float((m.predict_proba(Xdv[gold_sup])[:, 0] < m.tau_support).mean())

    for name, r in results.items():
        print(f"{name:22s} AUROC={r['auroc']:.3f} supF1={r['support_F1']:.3f} hallucF1={r['halluc_F1']:.3f} "
              f"acc={r['acc']:.3f} macroF1-3={r.get('macro_F1_3class', float('nan')):.3f}  syn={ {k[4:]: round(v, 2) for k, v in r['synthetic_detection'].items()} }")

    models["full"].save(OUT / "verifier.pkl")
    imp = models["full"].model.booster_.feature_importance("gain")
    out = {"results": results, "n_train": int(len(ytr)), "n_train_synthetic": int(syn_tr.sum()),
           "n_dev_human": int(human.sum()), "n_dev_synthetic": int(syn_dv.sum()),
           "dev_label_counts": {LABELS[k]: int((ydv[human] == k).sum()) for k in range(3)},
           "feature_gain": dict(zip(FEATURES, (imp / imp.sum()).round(4).tolist())),
           "tau_support": models["full"].tau_support}
    ex = []
    Pf = models["full"].predict_proba(Xdv)
    jc = FEATURES.index("cos_chunk")
    for i, mm in enumerate(mdv):
        if mm["type"].startswith("syn") and len(ex) < 12 and mm["type"] not in [e_["type"] for e_ in ex[-3:]]:
            ex.append({"type": mm["type"], "orig": mm["orig"], "corrupted": dev_pairs[i][0],
                       "cosine": round(float(Xdv[i, jc]), 3), "p_support": round(float(Pf[i, 0]), 3),
                       "p_contradict": round(float(Pf[i, 1]), 3)})
    out["examples"] = ex
    with open(RESULTS / "verifier.json", "w") as f:
        json.dump(out, f, indent=1)


if __name__ == "__main__":
    main()
