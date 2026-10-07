from __future__ import annotations

import csv
import json
from dataclasses import dataclass, field

from .config import RAW
from .text import split_sentences


@dataclass
class Document:
    doc_id: str
    title: str
    sentences: list[str]
    meta: dict = field(default_factory=dict)

    @property
    def text(self) -> str:
        return " ".join(self.sentences)


def _jsonl(path):
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def load_corpus(dataset: str) -> list[Document]:
    # for scifact we use the original release because the abstracts are already split into sentences
    if dataset == "scifact":
        docs = []
        for r in _jsonl(RAW / "scifact_orig" / "data" / "corpus.jsonl"):
            sents = [s.strip() for s in r["abstract"] if s.strip()]
            docs.append(Document(str(r["doc_id"]), r["title"].strip(), sents))
        return docs
    docs = []
    for r in _jsonl(RAW / dataset / "corpus.jsonl"):
        docs.append(Document(str(r["_id"]), (r.get("title") or "").strip(), split_sentences(r["text"])))
    return docs


def load_queries(dataset: str) -> dict[str, str]:
    return {str(r["_id"]): r["text"] for r in _jsonl(RAW / dataset / "queries.jsonl")}


def load_qrels(dataset: str, split: str) -> dict[str, dict[str, int]]:
    qrels: dict[str, dict[str, int]] = {}
    with open(RAW / dataset / "qrels" / f"{split}.tsv", encoding="utf-8") as f:
        reader = csv.reader(f, delimiter="\t")
        next(reader)
        for qid, did, score in reader:
            # keep only positive relevance grades
            if int(score) > 0:
                qrels.setdefault(qid, {})[did] = int(score)
    return qrels


def split_queries(dataset: str, split: str, limit: int | None = None, seed: int = 13):
    import random
    queries = load_queries(dataset)
    qrels = load_qrels(dataset, split)
    qids = sorted(q for q in qrels if q in queries)
    if limit and len(qids) > limit:
        rnd = random.Random(seed)
        qids = sorted(rnd.sample(qids, limit))
    return {q: queries[q] for q in qids}, {q: qrels[q] for q in qids}


def load_scifact_claims(split: str) -> list[dict]:
    return list(_jsonl(RAW / "scifact_orig" / "data" / f"claims_{split}.jsonl"))
