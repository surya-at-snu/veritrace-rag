import json
import pickle
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from veritrace.config import RESULTS, art
from veritrace.corpus import split_queries
from veritrace.dense import encode_queries, load_encoder
from veritrace.fusion import build_pool, to_docs
from veritrace.index import InvertedIndex
from veritrace.rerank import ce_scores
from veritrace.sparse import SparseRetriever

# off topic questions, only used to test the gate, never to tune it
GENERAL = [
    "Who won the 2018 FIFA World Cup?", "who won the world cup in 2025", "Who won the cricket world cup in 2011?",
    "How do I bake sourdough bread?", "What is the capital of Australia?", "Who is the president of the United States?",
    "Best smartphone to buy in 2024", "How to learn guitar quickly?", "Who wrote Harry Potter?",
    "What is the tallest mountain in the world?", "How do I reset my Gmail password?", "When did World War II end?",
    "Which movie won the Oscar for best picture in 2020?", "How much does a Tesla Model 3 cost?",
    "What time zone is Tokyo in?", "How to make a cup of masala chai?", "Who painted the Mona Lisa?",
    "What is the population of India?", "How do I install Python on Windows?", "Best places to visit in Goa",
    "Who won the IPL in 2023?", "What is the exchange rate of dollar to rupee?", "How to tie a tie?",
    "Lyrics of a famous Bollywood song", "Who is Virat Kohli?", "What is the plot of the movie Inception?",
    "How to start a small business in India?", "What is the speed limit on Indian highways?",
    "Which team has won the most Champions League titles?", "How many players are in a football team?",
    "What is the price of gold today?", "How to write a cover letter for an internship?",
    "When is Diwali celebrated?", "What is the best laptop for gaming?", "How do airplanes fly?",
    "Who invented the telephone?", "What are the rules of chess?", "How do I cook biryani?",
    "Which country hosted the 2016 Olympics?", "What does a stock market index measure?",
]


def best_ce(index, sp, emb, enc, queries):
    qv = encode_queries(queries, enc)
    pairs, owners = [], []
    for i, (q, v) in enumerate(zip(queries, qv)):
        s = sp.chunk_scores(q)
        d = emb @ v
        pool = build_pool(s, d, 200)
        _, _, sc = to_docs(pool.sparse, pool.chunks, index.chunk_doc)
        _, _, dc = to_docs(pool.dense, pool.chunks, index.chunk_doc)
        for c in list(sc[:3]) + list(dc[:3]):
            ch = index.chunks[int(c)]
            pairs.append((q, f"{ch.title}. {ch.text}"))
            owners.append(i)
    ce = ce_scores(pairs)
    best = np.full(len(queries), -1e9)
    for i, s in zip(owners, ce):
        best[i] = max(best[i], s)
    return best


def stored_best(pools):
    return np.array([max(P["ce_arb"]["ce_sp1"], P["ce_arb"]["ce_de1"]) for P in pools.values()])


def main():
    out = art("scifact")
    index = InvertedIndex.load(out / "index_window120.pkl")
    p = json.load(open(out / "sparse_params.json"))
    sp = SparseRetriever(index, p["k1"], p["b"], p["g"])
    emb = np.load(out / "emb_window120_bge.npy")
    enc = load_encoder()
    tr = stored_best(pickle.load(open(out / "pools_train.pkl", "rb")))
    te = stored_best(pickle.load(open(out / "pools_test.pkl", "rb")))
    fq, _ = split_queries("fiqa", "test")
    fiqa = best_ce(index, sp, emb, enc, list(fq.values()))
    gen = best_ce(index, sp, emb, enc, GENERAL)

    print("best cross-encoder score  (percentiles 1/5/50)")
    for name, x in (("SciFact train (in scope)", tr), ("SciFact test (in scope)", te), ("FiQA test (finance)", fiqa),
                    ("general questions", gen)):
        print(f"  {name:26s} {np.percentile(x, 1):7.2f} {np.percentile(x, 5):7.2f} {np.percentile(x, 50):7.2f}")
    rows = []
    for pct in (0.5, 1, 2, 5):
        # threshold = a low percentile of the in domain training scores
        tau = float(np.percentile(tr, pct))
        rows.append({"train_percentile": pct, "tau": tau, "scifact_test_rejected": float((te < tau).mean()),
                     "fiqa_rejected": float((fiqa < tau).mean()), "general_rejected": float((gen < tau).mean())})
        print(f"tau = train p{pct:<4} = {tau:6.2f}:  SciFact test wrongly rejected {rows[-1]['scifact_test_rejected']:.1%}  |  "
              f"FiQA rejected {rows[-1]['fiqa_rejected']:.1%}  |  general rejected {rows[-1]['general_rejected']:.1%}")
    chosen = [r for r in rows if r["train_percentile"] == 1][0]
    missed = [(q, round(float(s), 2)) for q, s in zip(GENERAL, gen) if s >= chosen["tau"]]
    print("general questions NOT rejected at p1:", missed)
    json.dump({"tau": chosen["tau"], "rule": "zero documents if best cross-encoder score < tau (1st percentile of in-domain train)",
               "table": rows, "general_scores": dict(zip(GENERAL, gen.round(2).tolist())), "missed": missed,
               "n": {"scifact_train": len(tr), "scifact_test": len(te), "fiqa": len(fiqa), "general": len(gen)}},
              open(RESULTS / "scope_gate.json", "w"), indent=1)
    json.dump({"tau": chosen["tau"]}, open(out / "scope_gate.json", "w"))


if __name__ == "__main__":
    main()
