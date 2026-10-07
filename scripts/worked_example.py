"""one query through the classic IR part of the pipeline, with every intermediate number written to
results/worked_example.json (used in the report's worked example and handy for the demo video)."""
import json
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from veritrace.boolean import BooleanEngine
from veritrace.config import RESULTS, art
from veritrace.index import InvertedIndex
from veritrace.pipeline import explain_query_processing
from veritrace.sparse import SparseRetriever, heap_topk
from veritrace.spell import SpellCorrector

QUERY = "Does vitamin D deficiency increase the risk of multiple sclerosis?"
TYPO = "Does vitamin D deficency increse the risk of multiple sclerosis?"
BOOL = '"multiple sclerosis" AND vitamin NOT mice'


def main():
    out = art("scifact")
    index = InvertedIndex.load(out / "index_window120.pkl")
    p = json.load(open(out / "sparse_params.json"))
    sp = SparseRetriever(index, p["k1"], p["b"], p["g"])
    body = index.body
    res = {"query": QUERY, "N_chunks": index.n_chunks, "N_docs": index.n_docs,
           "avg_chunk_len": round(float(body.avg_len), 2), "k1": sp.k1, "b": sp.b, "g": sp.g}
    res["analysis"] = explain_query_processing(QUERY)
    q = sp.query_terms(QUERY)
    terms = []
    for t in q:
        tid = body.tid(t)
        if tid < 0:
            terms.append({"term": t, "df": 0})
            continue
        terms.append({"term": t, "df": int(body.df[tid]), "cf": int(body.cf[tid]),
                      "idf_log10": round(body.idf(t), 3), "bm25_idf": round(body.bm25_idf(t), 3),
                      "first_postings": [{"chunk": int(body.units[tid][i]), "tf": int(body.tf[tid][i]),
                                          "positions": body.pos[tid][body.pos_off[tid][i]:body.pos_off[tid][i + 1]].tolist()}
                                         for i in range(min(3, int(body.df[tid])))],
                      "champions_top5": [int(c) for c in body.champion_list(t, 5)]})
    res["terms"] = terms
    # query vector for lnc.ltc (ltc side)
    wq = {t: (1 + math.log10(c)) * body.idf(t) for t, c in q.items() if body.tid(t) >= 0}
    qn = math.sqrt(sum(w * w for w in wq.values()))
    res["ltc_query_vector"] = {t: round(w / qn, 4) for t, w in wq.items()}
    full = sp.chunk_scores(QUERY)
    top = heap_topk(full, 5)
    res["bm25_top5"] = [{"chunk": c, "doc": index.chunks[c].doc_id, "title": index.chunks[c].title, "score": round(s, 3)}
                        for c, s in top]
    c0 = top[0][0]
    res["top_chunk"] = {"chunk": c0, "len": int(body.unit_len[c0]), "lnc_norm": round(float(body.lnc_norm[c0]), 4),
                        "text": index.chunks[c0].text[:400]}
    res["breakdown"] = [{k: (round(v, 4) if isinstance(v, float) else v) for k, v in r.items()}
                        for r in sp.explain(QUERY, c0)]
    res["n_nonzero"] = int((full > 0).sum())
    hits, tier, n = sp.search_tiered(QUERY, 100, 10)
    res["tiered"] = {"tier": tier, "scored": n, "same_top10": len({c for c, _ in hits} & {c for c, _ in heap_topk(full, 10)})}
    be = BooleanEngine(index)
    r = be.query(BOOL)
    res["boolean"] = {"query": BOOL, "parse": str(be.parse(BOOL)), "hits": len(r),
                      "titles": sorted({index.chunks[c].title for c in r})[:5],
                      "postings_sizes": {w: len(be.term_postings(w)) for w in ("vitamin", "mice")},
                      "phrase_hits": len(be.phrase_postings("multiple sclerosis")),
                      "and_hits": len(be.query("multiple AND sclerosis"))}
    corr = SpellCorrector.from_index(index)
    fixed, fixes = corr.correct_query(TYPO)
    res["spell"] = {"typo": TYPO, "corrected": fixed, "fixes": [f.__dict__ for f in fixes],
                    "typo_top1": index.chunks[heap_topk(sp.chunk_scores(TYPO), 1)[0][0]].title,
                    "fixed_top1": index.chunks[heap_topk(sp.chunk_scores(fixed), 1)[0][0]].title}
    json.dump(res, open(RESULTS / "worked_example.json", "w"), indent=1, default=str)
    print(json.dumps(res, indent=1, default=str)[:6000])


if __name__ == "__main__":
    main()
