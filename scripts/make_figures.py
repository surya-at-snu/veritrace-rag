import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from veritrace.config import FIGURES, RESULTS

# fixed colour order, checked for colour blind safety
C = ["#9C5221", "#0E8CA3", "#D39A12", "#8A4593", "#4F8F2C", "#3E5DB8", "#B23A2B", "#6B5B4B"]
INK, INK2, MUTED, GRID = "#2B2118", "#5A4636", "#9A8670", "#EDE3D3"
BASE_GRAY = "#CDBEA8"

plt.rcParams.update({
    "font.family": ["Segoe UI", "DejaVu Sans"], "font.size": 9, "axes.titlesize": 10, "axes.titleweight": "semibold",
    "axes.edgecolor": GRID, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
    "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "grid.color": GRID,
    "grid.linewidth": 0.8, "axes.axisbelow": True, "legend.frameon": False, "figure.dpi": 200,
    "savefig.bbox": "tight", "savefig.pad_inches": 0.05, "text.color": INK,
})


PRETTY = {"ce_sp3": "cross-enc. score, BM25 top-3", "ce_de3": "cross-enc. score, dense top-3",
          "ce_sp1": "cross-enc. score, BM25 top-1", "ce_de1": "cross-enc. score, dense top-1",
          "ce_diff1": "cross-enc. top-1 BM25 minus dense", "ce_diff3": "cross-enc. best BM25 minus dense",
          "de_gap12": "dense top-1 minus top-2 gap", "sp1_rank_in_de": "rank of BM25 #1 in dense list",
          "de1_rank_in_sp": "rank of dense #1 in BM25 list", "scq_avg": "avg SCQ (collection similarity)",
          "scq_max": "max SCQ", "idf_std": "std of query-term idf", "idf_avg": "mean query-term idf",
          "idf_max": "max query-term idf", "idf_min": "min query-term idf", "de_std10": "dense top-10 spread",
          "de_top1": "dense top-1 cosine", "de_mean10": "dense top-10 mean", "agree_jacc10": "top-10 overlap (Jaccard)",
          "agree_top1": "same top-1 document", "sp_top1": "BM25 top-1 score", "sp_top1_rel": "BM25 top-1 / max possible",
          "sp_gap12": "BM25 top-1/2 relative gap", "sp_nqc": "BM25 score spread (NQC)", "q_len": "query length",
          "q_oov": "out-of-vocabulary terms", "ictf_avg": "avg inverse coll. term freq.", "scope": "query scope",
          "fused": "router-fused score", "ce_minus_max": "cross-enc. minus list best", "de_rank": "rank in dense list",
          "rrf": "RRF score", "chunk_len": "chunk length", "idf_coverage": "idf-weighted term coverage",
          "coverage": "query-term coverage", "ce": "cross-encoder score", "ce_rank": "cross-encoder rank",
          "dense": "dense cosine", "window": "proximity (min. window)", "sp_rank": "rank in BM25 list",
          "bm25_zone": "BM25 weighted zones", "bm25_body": "BM25 body zone", "bm25_title": "BM25 title zone",
          "tfidf_body": "lnc.ltc body", "tfidf_title": "lnc.ltc title", "phrase": "phrase (bigram) matches"}


def load(name):
    p = RESULTS / name
    return json.load(open(p)) if p.exists() else None


def save(fig, name):
    fig.savefig(FIGURES / f"{name}.png")
    plt.close(fig)
    print("saved", name)


def hbar(ax, labels, values, colors, fmt="{:.3f}", xlim=None):
    y = np.arange(len(labels))[::-1]
    ax.barh(y, values, height=0.62, color=colors)
    for yi, v in zip(y, values):
        ax.text(v + (xlim[1] - xlim[0]) * 0.008 if xlim else v, yi, fmt.format(v), va="center", fontsize=8, color=INK)
    ax.set_yticks(y, labels)
    ax.grid(axis="y", visible=False)
    if xlim:
        ax.set_xlim(*xlim)


def fig_retrieval():
    r = load("scifact_retrieval.json")
    if not r:
        return
    m = r["metrics"]
    ltr = load("scifact_ltr.json")
    ft = load("scifact_dense_ft.json")
    rows = [("tf-idf lnc.ltc (body only)", m["tfidf_body"]["nDCG@10"], BASE_GRAY),
            ("tf-idf lnc.ltc + zones", m["tfidf_zone"]["nDCG@10"], BASE_GRAY),
            ("BM25 (body only)", m["bm25_body"]["nDCG@10"], BASE_GRAY),
            ("BM25 + learned zones", m["bm25_zone"]["nDCG@10"], BASE_GRAY),
            ("Dense BGE-small (zero-shot)", m["dense"]["nDCG@10"], BASE_GRAY)]
    if ft:
        rows.append(("Dense fine-tuned (IR-mined negatives)", ft["metrics"]["dense_finetuned"]["nDCG@10"], C[2]))
    rows += [("Hybrid RRF", m["rrf"]["nDCG@10"], BASE_GRAY),
             ("Hybrid convex, best fixed weight", m["convex_fixed"]["nDCG@10"], BASE_GRAY),
             ("Adaptive router (ours)", m["router"]["nDCG@10"], C[0])]
    if ltr:
        rows.append(("Router + LambdaMART (ours)", ltr["metrics"]["ltr_full"]["nDCG@10"], C[0]))
    rows.append(("Oracle per-query weight (upper bound)", m["oracle_alpha"]["nDCG@10"], "#E6DCCB"))
    fig, ax = plt.subplots(figsize=(6.4, 0.27 * len(rows) + 0.6))
    hbar(ax, [x[0] for x in rows], [x[1] for x in rows], [x[2] for x in rows], xlim=(0.55, 0.83))
    ax.set_xlabel("nDCG@10 on SciFact test (300 queries)")
    save(fig, "retrieval_scifact")


def fig_router():
    r = load("scifact_retrieval.json")
    if not r:
        return
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(8.2, 2.9), gridspec_kw={"width_ratios": [1, 1.05]}, layout="constrained")
    al = np.array(r["alphas"])
    a1.plot(al, r["train_mean_curve"], color=C[0], lw=2, label="train queries")
    a1.plot(al, r["test_mean_curve"], color=C[1], lw=2, label="test queries")
    a1.axvline(r["alpha_fixed"], color=MUTED, lw=1)
    a1.text(r["alpha_fixed"] + 0.03, min(r["test_mean_curve"]), f"best fixed a = {r['alpha_fixed']}", color=INK2, fontsize=8, va="bottom")
    a1.set_xlabel("fusion weight a on BM25 (0 = dense only, 1 = BM25 only)")
    a1.set_ylabel("mean nDCG@10")
    a1.set_title("Convex fusion: one weight for all queries")
    a1.legend(loc="upper right")
    coef = r["router_coef"]
    top = sorted(coef.items(), key=lambda kv: -abs(kv[1]))[:10]
    names = [PRETTY.get(k, k) for k, _ in top][::-1]
    vals = [v for _, v in top][::-1]
    a2.barh(range(len(vals)), vals, height=0.62, color=[C[0] if v > 0 else C[1] for v in vals])
    a2.set_yticks(range(len(vals)), names)
    a2.axvline(0, color=MUTED, lw=1)
    a2.grid(axis="y", visible=False)
    a2.set_title("Router: what makes BM25 win? (LR coefficients)")
    a2.set_xlabel("<- trust dense        trust BM25 ->")
    save(fig, "router")


def fig_efficiency():
    r = load("scifact_efficiency.json")
    if not r:
        return
    ex = r[0]
    ch = [x for x in r if x["config"].startswith("champion")]
    rs = [int(x["config"].split("=")[1]) for x in ch]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(7.2, 2.4))
    a1.plot(rs, [x["nDCG@10"] for x in ch], color=C[0], lw=2, marker="o", ms=4, mec="white", mew=1.5)
    a1.axhline(ex["nDCG@10"], color=MUTED, lw=1)
    a1.text(rs[0], ex["nDCG@10"] + 0.004, "exhaustive scoring", color=INK2, fontsize=8)
    a1.set_xscale("log")
    a1.set_xticks(rs, [str(x) for x in rs])
    a1.set_xlabel("champion list size r (postings per term)")
    a1.set_ylabel("BM25 nDCG@10")
    a1.set_title("Effectiveness")
    a2.plot(rs, [x["chunks_scored"] for x in ch], color=C[1], lw=2, marker="o", ms=4, mec="white", mew=1.5)
    a2.axhline(ex["chunks_scored"], color=MUTED, lw=1)
    a2.text(rs[0], ex["chunks_scored"] * 0.93, "exhaustive", color=INK2, fontsize=8, va="top")
    a2.set_xscale("log")
    a2.set_xticks(rs, [str(x) for x in rs])
    a2.set_xlabel("champion list size r (postings per term)")
    a2.set_ylabel("chunks scored per query")
    a2.set_title("Work per query")
    save(fig, "efficiency")


def fig_verifier():
    v = load("verifier.json")
    if not v:
        return
    R = v["results"]
    methods = [("cosine_chunk", "Cosine vs cited chunk\n(sample-idea baseline)"), ("cosine_best_sentence", "Cosine vs best sentence"),
               ("dense_cosine", "Dense cosine"), ("nli_entailment", "NLI entailment only"),
               ("ir_only", "Ours: IR features only"), ("no_nli", "Ours: IR + dense"), ("full", "Ours: full verifier")]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(7.4, 2.9), gridspec_kw={"width_ratios": [1.25, 1]})
    y = np.arange(len(methods))[::-1]
    h = 0.36
    a1.barh(y + h / 2, [R[m]["auroc"] for m, _ in methods], height=h, color=C[0], label="AUROC (supported vs not)")
    a1.barh(y - h / 2, [R[m]["halluc_F1"] for m, _ in methods], height=h, color=C[1], label="F1 for flagging unsupported claims")
    for yi, (m, _) in zip(y, methods):
        a1.text(R[m]["auroc"] + 0.01, yi + h / 2, f"{R[m]['auroc']:.2f}", va="center", fontsize=7, color=INK)
        a1.text(R[m]["halluc_F1"] + 0.01, yi - h / 2, f"{R[m]['halluc_F1']:.2f}", va="center", fontsize=7, color=INK)
    a1.set_yticks(y, [n for _, n in methods], fontsize=8)
    a1.set_xlim(0.4, 1.05)
    a1.grid(axis="y", visible=False)
    a1.legend(loc="upper center", bbox_to_anchor=(0.45, -0.13), ncol=1, fontsize=8)
    a1.set_title("SciFact dev claims (human labels)")
    kinds = ["syn_negation", "syn_direction", "syn_number", "syn_entity"]
    pw = v.get("pairwise", {})
    show = [("Cosine (cited chunk)", "Cosine"), ("NLI entailment", "NLI only"), ("VeriTrace verifier", "Ours (full)")]
    x = np.arange(len(kinds))
    w = 0.24
    for i, (m, lab) in enumerate(show):
        vals = [pw[m][k] for k in kinds]
        a2.bar(x + (i - 1) * w, vals, width=w - 0.03, color=C[[1, 2, 0][i]], label=lab)
    a2.axhline(0.5, color=MUTED, lw=1)
    a2.text(3.42, 0.515, "chance", color=INK2, fontsize=7, ha="left")
    a2.set_xlim(-0.5, 3.95)
    a2.set_xticks(x, ["negation", "direction\nflip", "number\nswap", "entity\nswap"])
    a2.set_ylim(0, 1.05)
    a2.set_ylabel("corrupted claim scored below original")
    a2.grid(axis="x", visible=False)
    a2.legend(loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=3, fontsize=8)
    a2.set_title("Injected hallucinations: paired test (dev)")
    save(fig, "verifier")


def fig_abstention():
    a = load("scifact_abstention.json")
    if not a:
        return
    fig, ax = plt.subplots(figsize=(4.4, 2.7))
    names = list(a["curves"])
    colors = [C[0], C[1], C[2], C[3]]
    for n, c in zip(names, colors):
        cv = a["curves"][n]
        ax.plot(cv["coverage"], cv["risk"], color=c, lw=2 if "VeriTrace" in n else 1.5,
                label=f"{n}  (AURC {a['results'][n]['aurc']:.3f})")
    ax.set_xlabel("coverage (share of queries answered)")
    ax.set_ylabel("risk (no relevant source in top-5)")
    ax.set_title("Know when retrieval failed: risk-coverage")
    ax.set_xlim(0.05, 1.0)
    ax.legend(fontsize=7, loc="upper left")
    save(fig, "risk_coverage")


def fig_ltr():
    l = load("scifact_ltr.json")
    if not l:
        return
    g = sorted(l["feature_gain"].items(), key=lambda kv: -kv[1])[:12][::-1]
    fig, ax = plt.subplots(figsize=(4.2, 2.9))
    ax.barh(range(len(g)), [v for _, v in g], height=0.62, color=C[0])
    ax.set_yticks(range(len(g)), [PRETTY.get(k, k) for k, _ in g])
    ax.grid(axis="y", visible=False)
    ax.set_xlabel("share of total split gain")
    ax.set_title("LambdaMART: which signals rank documents")
    save(fig, "ltr_features")


def fig_generalisation():
    sets = [(d, load(f"{d}_retrieval.json"), load(f"{d}_ltr.json")) for d in ("scifact", "nfcorpus", "fiqa")]
    sets = [s for s in sets if s[1]]
    if len(sets) < 2:
        return
    systems = [("bm25_zone", "BM25 + zones"), ("dense", "Dense"), ("rrf", "RRF"), ("convex_fixed", "Convex fixed"),
               ("router", "Adaptive router"), ("ltr_full", "Router + LambdaMART")]
    fig, ax = plt.subplots(figsize=(7.2, 2.6))
    x = np.arange(len(sets))
    w = 0.13
    for i, (k, lab) in enumerate(systems):
        vals = []
        for _, r, l in sets:
            vals.append(l["metrics"][k]["nDCG@10"] if k == "ltr_full" and l else r["metrics"].get(k, {}).get("nDCG@10", np.nan))
        ax.bar(x + (i - 2.5) * w, vals, width=w - 0.02, color=C[i], label=lab)
    ax.set_xticks(x, [f"{d}\n({r['n_test']} test queries)" for d, r, _ in sets])
    ax.set_ylabel("nDCG@10")
    ax.grid(axis="x", visible=False)
    ax.legend(ncol=6, fontsize=7.5, loc="upper center", bbox_to_anchor=(0.5, 1.16))
    save(fig, "generalisation")


def fig_rag():
    names = [("qwen_v1_raw", "Qwen, audit v1"), ("qwen_reaudit", "Qwen, audit v2"), ("qwen_fresh", "Qwen, held-out (v2)"),
             ("extractive", "Extractive (v2)")]
    data = [(lab, json.load(open(RESULTS / f"rag_eval_{k}.json"))["summary"]) for k, lab in names
            if (RESULTS / f"rag_eval_{k}.json").exists()]
    if not data:
        return
    order = ["VERIFIED", "REPAIRED", "REPAIRED_NEW", "CONTRADICTED", "UNSUPPORTED"]
    # status colours: verified green, repaired teal, new source blue, contradicted red, unsupported ochre
    cols = [C[4], C[1], C[5], C[6], C[2]]
    fig, ax = plt.subplots(figsize=(6.4, 0.55 * len(data) + 0.9))
    for i, (name, s) in enumerate(data):
        tot = max(1, sum(s["status_counts"].get(k, 0) for k in order))
        left = 0
        for st, c in zip(order, cols):
            v = s["status_counts"].get(st, 0) / tot
            ax.barh(i, v, left=left, height=0.5, color=c, label=st if i == 0 else None, edgecolor="white", linewidth=1.5)
            if v > 0.07:
                ax.text(left + v / 2, i, f"{v:.0%}", ha="center", va="center", fontsize=7.5,
                        color="white" if st not in ("UNSUPPORTED",) else INK)
            left += v
    ax.set_yticks(range(len(data)), [f"{lab}\n({s['n_claims']} claims)" for lab, s in data])
    ax.set_xlim(0, 1)
    ax.xaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
    ax.grid(axis="y", visible=False)
    ax.legend(ncol=5, fontsize=7.5, loc="upper center", bbox_to_anchor=(0.5, 1.35))
    ax.set_title("Audit verdicts for generated claims (SciFact test)", pad=28)
    save(fig, "rag_audit")


# ---------------------------------------------------------------------------------------------
# figures for the extra IR-principle experiments (scripts/exp_ir_extras.py, exp_efficiency.py)
def fig_tradeoff():
    r = load("scifact_efficiency.json")
    if not r:
        return
    r = [x for x in r if "nDCG@10" in x]
    ex = r[0]
    groups = [("champion lists", "champion", C[0], "o"), ("index elimination (idf)", "index elimination", C[1], "s"),
              ("many-term matching", "many-term", C[2], "D"), ("tiered index", "tiered", C[3], "^")]
    fig, ax = plt.subplots(figsize=(7.2, 2.7))
    for label, key, col, mk in groups:
        xs = [x for x in r if x["config"].startswith(key)]
        ax.plot([x["chunks_scored"] for x in xs], [x["nDCG@10"] for x in xs], color=col, lw=2 if key == "champion" else 0,
                marker=mk, ms=6, mec="white", mew=1.2, label=label)
    ax.scatter([ex["chunks_scored"]], [ex["nDCG@10"]], s=60, color=INK, zorder=5, label="exhaustive (heap top-K)")
    ax.set_xscale("log")
    ax.set_xlabel("chunks scored per query (log scale)")
    ax.set_ylabel("BM25 nDCG@10")
    ax.set_title("Quality vs. work: every inexact top-K method on SciFact test")
    ax.legend(loc="lower right", fontsize=7.5, ncol=1)
    save(fig, "efficiency_tradeoff")


def fig_smart():
    d = load("scifact_ir_extras.json")
    if not d or "smart" not in d:
        return
    rows = sorted(d["smart"]["rows"], key=lambda x: -x["nDCG@10"])
    labels = [x["scheme"] for x in rows]
    vals = [x["nDCG@10"] for x in rows]
    cols = [C[0] if l == "lnc.ltc" else (C[1] if l.startswith("bm25") else BASE_GRAY) for l in labels]
    fig, ax = plt.subplots(figsize=(3.6, 3.0))
    hbar(ax, labels, vals, cols, xlim=(0, 0.75))
    ax.set_xlabel("nDCG@10 (body zone only)")
    ax.set_title("SMART weighting schemes")
    save(fig, "smart_schemes")


def fig_analysis():
    d = load("scifact_ir_extras.json")
    if not d or "analysis" not in d:
        return
    rows = d["analysis"]
    pretty = {"porter": "Porter + stop list", "s": "S-stemmer + stop list", "none": "no stemming + stop list",
              "porter+stopwords": "Porter, keep stop words", "none+stopwords": "no stemming, keep stop words"}
    labels = [pretty.get(x["chain"], x["chain"]) for x in rows]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(7.2, 2.3), gridspec_kw={"width_ratios": [1.15, 1]})
    hbar(a1, labels, [x["bm25_zones"]["nDCG@10"] for x in rows], [C[0]] + [BASE_GRAY] * (len(rows) - 1), xlim=(0.6, 0.7))
    a1.set_xlabel("BM25 + zones nDCG@10")
    a1.set_title("Effectiveness")
    hbar(a2, ["" for _ in labels], [x["postings"] / 1000 for x in rows], [C[1]] + [BASE_GRAY] * (len(rows) - 1),
         fmt="{:.0f}k", xlim=(0, 900))
    a2.set_xlabel("postings in the body index (thousands)")
    a2.set_title("Index size")
    save(fig, "analysis_chain")


def fig_spell():
    d = load("scifact_ir_extras.json")
    if not d or "spell" not in d:
        return
    st = d["spell"]["settings"]
    clean = st["1"]["metrics"]["clean"]["nDCG@10"]
    fig, ax = plt.subplots(figsize=(3.6, 2.4))
    w = 0.34
    for i, (cond, col, lab) in enumerate((("typo", BASE_GRAY, "with typos"), ("typo+corrector", C[0], "typos + corrector"))):
        vals = [st[n]["metrics"][cond]["nDCG@10"] for n in ("1", "2")]
        xs = np.arange(2) + (i - 0.5) * (w + 0.04)
        ax.bar(xs, vals, width=w, color=col, label=lab)
        for x, v in zip(xs, vals):
            ax.text(x, v - 0.004, f"{v:.3f}", ha="center", va="top", fontsize=7.5,
                    color="white" if col == C[0] else INK)
    ax.axhline(clean, color=INK2, lw=1, ls="--")
    ax.text(-0.4, clean + 0.003, f"clean queries {clean:.3f}", ha="left", fontsize=7.5, color=INK2)
    ax.set_xticks(np.arange(2), ["1 typo per claim", "2 typos per claim"])
    ax.grid(axis="x", visible=False)
    ax.set_ylim(0.55, 0.71)
    ax.set_ylabel("BM25 nDCG@10")
    ax.set_title("Spelling correction")
    ax.legend(fontsize=7.5, loc="lower center", ncol=2, bbox_to_anchor=(0.5, -0.36))
    save(fig, "spelling")


def fig_cluster():
    d = load("scifact_ir_extras.json")
    if not d or "cluster" not in d:
        return
    rows = d["cluster"]["rows"]
    ex = rows[0]
    fig, ax = plt.subplots(figsize=(3.6, 2.5))
    for b1, col, mk in ((1, C[0], "o"), (2, C[1], "s")):
        xs = [x for x in rows if x["b1"] == b1]
        ax.plot([x["vectors_scored"] for x in xs], [x["nDCG@10"] for x in xs], color=col, lw=2, marker=mk, ms=5,
                mec="white", mew=1.2, label=f"b1 = {b1} leader{'s' if b1 > 1 else ''} per follower")
        if b1 == 1:
            for x in xs:
                ax.text(x["vectors_scored"] * 1.08, x["nDCG@10"] - 0.012, f"b2={x['b2']}", fontsize=6.5, color=INK2, va="top")
    ax.axhline(ex["nDCG@10"], color=MUTED, lw=1, ls="--")
    ax.text(205, ex["nDCG@10"] + 0.006, f"exact search over all {ex['vectors_scored']} chunks: {ex['nDCG@10']:.3f}",
            fontsize=7, color=INK2, ha="left", va="bottom")
    ax.set_ylim(0.35, 0.79)
    ax.set_xscale("log")
    ticks = [200, 500, 1000, 2000, 5000]
    ax.set_xticks(ticks, [str(t) for t in ticks])
    ax.minorticks_off()
    ax.set_xlabel("vectors scored per query (log scale)")
    ax.set_ylabel("dense nDCG@10")
    ax.set_title("Cluster pruning (leaders / followers)")
    ax.legend(fontsize=7, loc="lower right")
    save(fig, "cluster_pruning")

# side-by-side panel used by the 10-page report
def fig_spell_cluster():
    from PIL import Image
    a, b = Image.open(FIGURES / "spelling.png"), Image.open(FIGURES / "cluster_pruning.png")
    out = Image.new("RGB", (a.size[0] + b.size[0] + 40, max(a.size[1], b.size[1])), "white")
    out.paste(a, (0, 0))
    out.paste(b, (a.size[0] + 40, 0))
    out.save(FIGURES / "spell_cluster.png")
    print("saved spell_cluster")


if __name__ == "__main__":
    for fn in (fig_retrieval, fig_router, fig_efficiency, fig_verifier, fig_abstention, fig_ltr, fig_generalisation, fig_rag,
               fig_tradeoff, fig_smart, fig_analysis, fig_spell, fig_cluster, fig_spell_cluster):
        try:
            fn()
        except Exception as e:
            print("FAILED", fn.__name__, repr(e))
