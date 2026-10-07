import argparse
import json
import random
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from veritrace.config import JUDGE_NLI_MODEL, RESULTS
from veritrace.corpus import split_queries
from veritrace.corrupt import corrupt
from veritrace.nli import NLIScorer
from veritrace.pipeline import VeriTrace
from veritrace.verify import FEATURES

GROUNDED = ("VERIFIED", "REPAIRED", "REPAIRED_NEW")
STATUSES = ("VERIFIED", "REPAIRED", "REPAIRED_NEW", "CONTRADICTED", "UNSUPPORTED", "NOT_A_CLAIM")


# corrupt a verified claim and see if the audit catches it
def inject(vt, rnd, verdicts, contexts):
    ok = [v for v in verdicts if v.status == "VERIFIED"]
    if not ok:
        return None
    v = rnd.choice(ok)
    ev = contexts[v.support_ref - 1]
    c = corrupt(v.text, ev.text, vt.index, rnd)
    if not c:
        return None
    vv = vt.auditor.audit(f"{c[0]} [{v.support_ref}]", contexts)
    X, _ = vt.auditor.f.featurize([(c[0], ev)])
    return {"orig": v.text, "corrupted": c[0], "kind": c[1], "status": vv[0].status if vv else "NONE",
            "cos_best": float(X[0, FEATURES.index("cos_best")])}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--generator", default="qwen")
    ap.add_argument("--n", type=int, default=50)
    ap.add_argument("--seed", type=int, default=3)
    ap.add_argument("--reaudit", default=None, help="re-audit the answers stored in this result file")
    ap.add_argument("--exclude", default=None, help="skip the queries used in this result file")
    ap.add_argument("--tag", default=None)
    args = ap.parse_args()
    rnd = random.Random(args.seed)
    tau_cos = json.load(open(RESULTS / "verifier.json"))["results"]["cosine_best_sentence"]["tau"]

    records = []
    t0 = time.time()
    if args.reaudit:
        vt = VeriTrace("scifact", generator=None)
        old = json.load(open(args.reaudit, encoding="utf-8"))
        gen_name = old["summary"]["generator"]
        for i, rec in enumerate(old["records"]):
            contexts = [vt.evidence(c) for c in rec["context_chunks"]]
            vs = vt.auditor.audit(rec["answer"], contexts) if "INSUFFICIENT" not in rec["answer"] else []
            new = {k: rec[k] for k in ("qid", "query", "answer", "context_chunks", "context_has_relevant", "confidence", "alpha")}
            new["timings"] = rec["timings"]
            new["verdicts"] = [v.__dict__ for v in vs]
            inj = inject(vt, rnd, vs, contexts)
            if inj:
                new["injected"] = inj
            records.append(new)
            print(f"re-audit {i + 1}/{len(old['records'])}  {time.time() - t0:.0f}s  {[v.status[:4] for v in vs]}", flush=True)
    else:
        vt = VeriTrace("scifact", generator=args.generator)
        gen_name = vt.generator.name
        queries, qrels = split_queries("scifact", "test")
        pool = sorted(queries)
        if args.exclude:
            # held out run: skip the queries we already looked at
            used = {r["qid"] for r in json.load(open(args.exclude, encoding="utf-8"))["records"]}
            pool = [q for q in pool if q not in used]
        qids = sorted(rnd.sample(pool, args.n))
        for i, q in enumerate(qids):
            a = vt.answer(queries[q], k=5)
            r = a.retrieval
            rec = {"qid": q, "query": queries[q], "answer": a.raw_answer,
                   "context_chunks": [ev.ref for ev in r.contexts],
                   "context_has_relevant": any(x["doc_id"] in qrels[q] for x in r.ranked[:5]),
                   "confidence": r.confidence, "alpha": r.alpha,
                   "verdicts": [v.__dict__ for v in a.verdicts], "timings": {**r.timings, **a.timings}}
            inj = inject(vt, rnd, a.verdicts, r.contexts)
            if inj:
                rec["injected"] = inj
            records.append(rec)
            print(f"{i + 1}/{len(qids)}  {time.time() - t0:.0f}s  claims={len(a.verdicts)} "
                  f"status={[v.status[:4] for v in a.verdicts]}", flush=True)

    # independent judge, a bigger nli model the system itself never uses
    judge = NLIScorer(JUDGE_NLI_MODEL)

    def ent(chunk_ids, claim):
        if not chunk_ids:
            return np.zeros(0)
        pairs = [(f"{vt.index.chunks[c].title}. {vt.index.chunks[c].text}", claim) for c in chunk_ids]
        return judge(pairs)[:, 0]

    rows = []
    for rec in records:
        ctx = rec["context_chunks"]
        for v in rec["verdicts"]:
            if v["status"] == "NOT_A_CLAIM":
                rows.append({"qid": rec["qid"], "status": v["status"]})
                continue
            e_ctx = ent(ctx, v["text"])
            cited = [j - 1 for j in v["cited"] if 1 <= j <= len(ctx)]
            if v["support_ref"]:
                final = float(e_ctx[v["support_ref"] - 1])
            elif v["new_chunk"] is not None:
                final = float(ent([v["new_chunk"]], v["text"])[0])
            else:
                final = None
            rows.append({"qid": rec["qid"], "status": v["status"],
                         "judge_cited": float(e_ctx[cited].max()) if cited else 0.0,
                         "judge_any": float(e_ctx.max()) if len(e_ctx) else 0.0,
                         "judge_final": final if v["status"] in GROUNDED else None,
                         "n_cited": len(cited)})
    J = 0.5
    fact = [r for r in rows if r["status"] != "NOT_A_CLAIM"]
    st = [r["status"] for r in fact]
    cited_ok = np.array([r["judge_cited"] >= J for r in fact])
    any_ok = np.array([r["judge_any"] >= J for r in fact])
    kept = np.array([s in GROUNDED for s in st])
    final_ok = np.array([(r["judge_final"] or 0) >= J for r in fact])
    from sklearn.metrics import precision_recall_fscore_support
    p, rc, f, _ = precision_recall_fscore_support(any_ok, kept, average="binary", zero_division=0)
    inj = [r["injected"] for r in records if "injected" in r]
    by_kind = {}
    for x in inj:
        d = by_kind.setdefault(x["kind"], {"n": 0, "veritrace": 0, "cosine": 0})
        d["n"] += 1
        d["veritrace"] += x["status"] not in GROUNDED
        d["cosine"] += x["cos_best"] < tau_cos
    summary = {
        "generator": gen_name, "mode": "reaudit" if args.reaudit else "generate", "n_queries": len(records),
        "n_sentences": len(rows), "n_claims": len(fact), "n_meta": len(rows) - len(fact),
        "insufficient_evidence_answers": sum("INSUFFICIENT" in r["answer"] for r in records),
        "context_has_relevant": float(np.mean([r["context_has_relevant"] for r in records])),
        "status_counts": {s: sum(r["status"] == s for r in rows) for s in STATUSES},
        "raw_citation_support": float(cited_ok.mean()) if fact else None,
        "claims_without_citation": int(sum(r["n_cited"] == 0 for r in fact)),
        "judge_supported_by_any_context": float(any_ok.mean()) if fact else None,
        "after_audit_kept_fraction": float(kept.mean()) if fact else None,
        "after_audit_precision_of_kept": float(final_ok[kept].mean()) if kept.any() else None,
        "verifier_vs_judge": {"precision": float(p), "recall": float(rc), "f1": float(f)},
        "injected": {"n": len(inj), "veritrace_detect": float(np.mean([x["status"] not in GROUNDED for x in inj])) if inj else None,
                     "cosine_detect": float(np.mean([x["cos_best"] < tau_cos for x in inj])) if inj else None,
                     "by_kind": by_kind, "examples": inj[:8]},
        "mean_timings": {k: float(np.mean([r["timings"].get(k, 0) for r in records])) for k in records[0]["timings"]},
    }
    print(json.dumps({k: v for k, v in summary.items() if k != "injected"}, indent=1))
    print("injected:", json.dumps({k: v for k, v in summary["injected"].items() if k != "examples"}, indent=1))
    tag = args.tag or args.generator.replace(".", "")
    with open(RESULTS / f"rag_eval_{tag}.json", "w", encoding="utf-8") as f:
        json.dump({"summary": summary, "records": records, "claims": rows}, f, indent=1, default=str)


if __name__ == "__main__":
    main()
