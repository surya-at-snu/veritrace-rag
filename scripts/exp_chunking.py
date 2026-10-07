import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from veritrace.config import RESULTS, art
from veritrace.corpus import load_scifact_claims, split_queries
from veritrace.dense import encode_queries, load_encoder
from veritrace.fusion import build_pool, doc_run, full_doc_run, rrf
from veritrace.index import InvertedIndex
from veritrace.metrics import evaluate_run
from veritrace.sparse import SparseRetriever


def main():
    out = art("scifact")
    p = json.load(open(out / "sparse_params.json"))
    queries, qrels = split_queries("scifact", "test")
    claims = {str(c["id"]): c for c in load_scifact_claims("dev")}
    qids = list(qrels)
    enc = load_encoder()
    qv = encode_queries([queries[q] for q in qids], enc)
    rows = []
    # whole abstract vs sentence windows
    for name in ("doc", "window120"):
        index = InvertedIndex.load(out / f"index_{name}.pkl")
        emb = np.load(out / f"emb_{name}_bge.npy")
        sp = SparseRetriever(index, p["k1"], p["b"], p["g"])
        doc_ids = [d.doc_id for d in index.docs]
        runs = {"bm25": {}, "dense": {}, "rrf": {}}
        rat_hits, words = [], []
        for q, v in zip(qids, qv):
            s = sp.chunk_scores(queries[q])
            d = emb @ v
            runs["bm25"][q] = full_doc_run(s, index, 1000)
            runs["dense"][q] = full_doc_run(d, index, 1000)
            pool = build_pool(s, d, 200)
            f = rrf(pool)
            runs["rrf"][q] = doc_run(f, pool.chunks, index.chunk_doc, doc_ids, 100)
            top = pool.chunks[np.argsort(-f)[:5]]
            words.append(sum(len(index.chunks[c].text.split()) for c in top))
            gold = {(did, s_) for did, evs in claims[q]["evidence"].items() for e in evs for s_ in e["sentences"]}
            if gold:
                # did the top 5 context contain a gold rationale sentence
                hit = any((index.chunks[c].doc_id, si) in gold
                          for c in top for si in range(index.chunks[c].sent_start, index.chunks[c].sent_end))
                rat_hits.append(hit)
        row = {"policy": name, "n_chunks": index.n_chunks,
               "avg_chunk_words": float(np.mean([len(c.text.split()) for c in index.chunks]))}
        for r, run in runs.items():
            ev = evaluate_run(run, qrels)["mean"]
            row[f"{r}_nDCG@10"] = ev["nDCG@10"]
            row[f"{r}_R@10"] = ev["R@10"]
        row["rationale_recall@5"] = float(np.mean(rat_hits))
        row["context_words@5"] = float(np.mean(words))
        row["rationale_hits_per_1k_words"] = float(np.mean(rat_hits) / np.mean(words) * 1000)
        rows.append(row)
        print(json.dumps(row, indent=1))
    json.dump(rows, open(RESULTS / "scifact_chunking.json", "w"), indent=1)


if __name__ == "__main__":
    main()
