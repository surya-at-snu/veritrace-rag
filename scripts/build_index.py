import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from veritrace.chunking import chunk_corpus
from veritrace.config import DENSE_MODEL, art
from veritrace.corpus import load_corpus
from veritrace.dense import chunk_text_for_encoding, encode_passages, load_encoder
from veritrace.index import InvertedIndex


def index_name(policy: str, max_words: int) -> str:
    return "doc" if policy == "doc" else ("sentence" if policy == "sentence" else f"window{max_words}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="scifact")
    ap.add_argument("--policy", default="window", choices=["doc", "window", "sentence"])
    ap.add_argument("--max-words", type=int, default=120)
    ap.add_argument("--overlap", type=int, default=1)
    ap.add_argument("--no-dense", action="store_true")
    ap.add_argument("--encoder", default=DENSE_MODEL, help="HF name or path of a fine-tuned encoder")
    ap.add_argument("--tag", default="bge", help="name for the embedding file")
    args = ap.parse_args()

    out = art(args.dataset)
    name = index_name(args.policy, args.max_words)
    t0 = time.time()
    docs = load_corpus(args.dataset)
    # cut documents into sentence windows
    chunks = chunk_corpus(docs, args.policy, args.max_words, args.overlap)
    print(f"[{args.dataset}] {len(docs)} documents -> {len(chunks)} chunks ({name})  {time.time() - t0:.1f}s")

    idx_path = out / f"index_{name}.pkl"
    if idx_path.exists():
        print("index exists:", idx_path)
        index = InvertedIndex.load(idx_path)
    else:
        t0 = time.time()
        # build the positional inverted index (title + body zones)
        index = InvertedIndex.build(docs, chunks, {"policy": args.policy, "max_words": args.max_words,
                                                   "overlap": args.overlap})
        index.save(idx_path)
        print(f"index built: body vocab={len(index.body.vocab)} postings={int(index.body.df.sum())} "
              f"positions={int(index.body.cf.sum())} title vocab={len(index.title.vocab)}  {time.time() - t0:.1f}s")

    if not args.no_dense:
        emb_path = out / f"emb_{name}_{args.tag}.npy"
        if emb_path.exists():
            print("embeddings exist:", emb_path)
            return
        t0 = time.time()
        model = load_encoder(args.encoder)
        # dense vectors for every chunk
        emb = encode_passages([chunk_text_for_encoding(c) for c in index.chunks], model)
        np.save(emb_path, emb)
        print(f"dense embeddings {emb.shape} saved  {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
