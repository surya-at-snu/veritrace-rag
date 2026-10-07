import argparse
import json
import random
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import torch

from veritrace.config import DENSE_QUERY_PREFIX, RESULTS, art
from veritrace.corpus import split_queries
from veritrace.dense import chunk_text_for_encoding, encode_passages, encode_queries, load_encoder
from veritrace.fusion import build_pool, convex, doc_run, full_doc_run, rrf
from veritrace.index import InvertedIndex
from veritrace.metrics import evaluate_run, paired_t_pvalue
from veritrace.sparse import SparseRetriever


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="scifact")
    ap.add_argument("--index", default="window120")
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--train-limit", type=int, default=1500)
    args = ap.parse_args()
    torch.manual_seed(0)
    rnd = random.Random(0)
    ds, out = args.dataset, art(args.dataset)
    index = InvertedIndex.load(out / f"index_{args.index}.pkl")
    emb0 = np.load(out / f"emb_{args.index}_bge.npy")
    p = json.load(open(out / "sparse_params.json"))
    sp = SparseRetriever(index, p["k1"], p["b"], p["g"])
    tr_q, tr_rel = split_queries(ds, "train", limit=args.train_limit)
    te_q, te_rel = split_queries(ds, "test")
    doc_pos = {d.doc_id: i for i, d in enumerate(index.docs)}
    model = load_encoder()

    qv = encode_queries(list(tr_q.values()), model)
    triples = []
    for (q, text), v in zip(tr_q.items(), qv):
        rel_docs = [doc_pos[d] for d in tr_rel[q] if d in doc_pos]
        if not rel_docs:
            continue
        s = sp.chunk_scores(text)
        # hard negatives mined from our own bm25 index
        negs = [int(c) for c in np.argsort(-s)[:20] if index.docs[index.chunk_doc[c]].doc_id not in tr_rel[q]]
        for d in rel_docs:
            chs = np.where(index.chunk_doc == d)[0]
            pos = int(chs[np.argmax(emb0[chs] @ v)])
            triples.append((DENSE_QUERY_PREFIX + text, chunk_text_for_encoding(index.chunks[pos]),
                            chunk_text_for_encoding(index.chunks[negs[0] if negs else rnd.randrange(index.n_chunks)])))
    print(f"{len(triples)} training triples")

    model.train()
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    steps = args.epochs * ((len(triples) + args.batch - 1) // args.batch)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / max(1, int(0.1 * steps))) * max(0.0, (steps - s) / steps))
    dev = model.device

    def embed(texts):
        f = model.tokenize(texts)
        f = {k: v.to(dev) for k, v in f.items() if hasattr(v, "to")}
        return torch.nn.functional.normalize(model(f)["sentence_embedding"], dim=-1)

    step, t0, losses = 0, time.time(), []
    for ep in range(args.epochs):
        rnd.shuffle(triples)
        for i in range(0, len(triples), args.batch):
            b = triples[i:i + args.batch]
            q = embed([x[0] for x in b])
            d = embed([x[1] for x in b] + [x[2] for x in b])
            # infonce: in batch positives + hard negatives, temperature 0.05
            logits = q @ d.T / 0.05
            loss = torch.nn.functional.cross_entropy(logits, torch.arange(len(b), device=dev))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step(); sched.step(); opt.zero_grad()
            losses.append(float(loss))
            step += 1
            if step % 10 == 0:
                print(f"epoch {ep} step {step}/{steps} loss={np.mean(losses[-10:]):.4f}  {time.time() - t0:.0f}s", flush=True)
    model.eval()
    save_dir = out / "bge-ft"
    model.save(str(save_dir))

    emb = encode_passages([chunk_text_for_encoding(c) for c in index.chunks], model)
    np.save(out / f"emb_{args.index}_bgeft.npy", emb)
    te_ids = list(te_rel)
    qv_ft = encode_queries([te_q[q] for q in te_ids], model)
    from sentence_transformers import SentenceTransformer
    base = SentenceTransformer("BAAI/bge-small-en-v1.5", device="cpu")
    qv_0 = encode_queries([te_q[q] for q in te_ids], base)
    doc_ids = [d.doc_id for d in index.docs]
    runs = {"dense_zero_shot": {}, "dense_finetuned": {}, "rrf_bm25_dense_ft": {}, "convex_bm25_dense_ft": {}}
    for q, v0, v1 in zip(te_ids, qv_0, qv_ft):
        s = sp.chunk_scores(te_q[q])
        d0, d1 = emb0 @ v0, emb @ v1
        runs["dense_zero_shot"][q] = full_doc_run(d0, index, 1000)
        runs["dense_finetuned"][q] = full_doc_run(d1, index, 1000)
        pool = build_pool(s, d1, 200)
        runs["rrf_bm25_dense_ft"][q] = doc_run(rrf(pool), pool.chunks, index.chunk_doc, doc_ids, 100)
        runs["convex_bm25_dense_ft"][q] = doc_run(convex(pool, p["alpha_fixed"]), pool.chunks, index.chunk_doc, doc_ids, 100)
    res = {n: evaluate_run(r, te_rel) for n, r in runs.items()}
    for n, r in res.items():
        print(f"{n:22s} " + "  ".join(f"{m}={v:.4f}" for m, v in r["mean"].items()))
    pv = paired_t_pvalue(res["dense_finetuned"]["per_query"]["nDCG@10"], res["dense_zero_shot"]["per_query"]["nDCG@10"])
    print(f"paired t-test fine-tuned vs zero-shot dense: p={pv:.4g}")
    json.dump({"metrics": {n: r["mean"] for n, r in res.items()}, "p_value_ft_vs_zs": pv, "n_triples": len(triples),
               "epochs": args.epochs, "loss_curve": losses}, open(RESULTS / f"{ds}_dense_ft.json", "w"), indent=1)


if __name__ == "__main__":
    main()
