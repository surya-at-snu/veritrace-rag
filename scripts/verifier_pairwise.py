import json
import pickle
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from veritrace.config import RESULTS, art
from veritrace.verify import FEATURES, ClaimVerifier

OUT = art("scifact")


def main():
    X, y, meta, _ = pickle.load(open(OUT / "verifier_feats_dev.pkl", "rb"))
    model = ClaimVerifier.load(OUT / "verifier.pkl")
    p_full = model.predict_proba(X)[:, 0]
    scores = {"Cosine (best sentence)": X[:, FEATURES.index("cos_best")],
              "Cosine (cited chunk)": X[:, FEATURES.index("cos_chunk")],
              "Dense cosine": X[:, FEATURES.index("dense_best")],
              "NLI entailment": X[:, FEATURES.index("nli_e_max")],
              "VeriTrace verifier": p_full}
    gold = {}
    for i, m in enumerate(meta):
        if m["type"] == "gold" and y[i] == 0:
            gold.setdefault(m["claim"], i)
    out = {}
    for name, s in scores.items():
        res = {}
        for kind in ("syn_negation", "syn_direction", "syn_number", "syn_entity"):
            wins = []
            for i, m in enumerate(meta):
                if m["type"] == kind and m["claim"] in gold:
                    g = gold[m["claim"]]
                    # does the corrupted claim score lower than the original one
                    wins.append(1.0 if s[i] < s[g] else (0.5 if s[i] == s[g] else 0.0))
            res[kind] = float(np.mean(wins)) if wins else None
            res[kind + "_n"] = len(wins)
        out[name] = res
        print(f"{name:24s} " + "  ".join(f"{k[4:]}={res[k]:.2f}" for k in res if not k.endswith("_n")))
    v = json.load(open(RESULTS / "verifier.json"))
    v["pairwise"] = out
    json.dump(v, open(RESULTS / "verifier.json", "w"), indent=1)


if __name__ == "__main__":
    main()
