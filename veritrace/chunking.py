from __future__ import annotations

from dataclasses import dataclass

from .corpus import Document


@dataclass
class Chunk:
    chunk_id: int
    doc_idx: int
    doc_id: str
    title: str
    sent_start: int
    sentences: list[str]

    @property
    def text(self) -> str:
        return " ".join(self.sentences)

    @property
    def sent_end(self) -> int:
        return self.sent_start + len(self.sentences)


def chunk_document(doc: Document, doc_idx: int, policy: str = "window", max_words: int = 120,
                   overlap: int = 1, start_id: int = 0) -> list[Chunk]:
    sents = doc.sentences or [doc.title]
    if policy == "doc":
        return [Chunk(start_id, doc_idx, doc.doc_id, doc.title, 0, list(sents))]
    if policy == "sentence":
        return [Chunk(start_id + i, doc_idx, doc.doc_id, doc.title, i, [s]) for i, s in enumerate(sents)]

    lengths = [len(s.split()) for s in sents]
    # short abstracts stay as one chunk
    if sum(lengths) <= max_words * 1.5:
        return [Chunk(start_id, doc_idx, doc.doc_id, doc.title, 0, list(sents))]
    chunks, i = [], 0
    while i < len(sents):
        j, words = i, 0
        # pack whole sentences until we hit the word budget
        while j < len(sents) and (words < max_words or j == i):
            words += lengths[j]
            j += 1
        # dont leave a tiny tail chunk, merge it into this one
        if j < len(sents) and sum(lengths[j:]) < max_words * 0.3:
            j = len(sents)
        chunks.append(Chunk(start_id + len(chunks), doc_idx, doc.doc_id, doc.title, i, sents[i:j]))
        if j >= len(sents):
            break
        # next window starts one sentence back so a fact on the boundary isnt cut in half
        i = max(j - overlap, i + 1)
    return chunks


def chunk_corpus(docs: list[Document], policy: str = "window", max_words: int = 120,
                 overlap: int = 1) -> list[Chunk]:
    out: list[Chunk] = []
    for d_idx, d in enumerate(docs):
        out.extend(chunk_document(d, d_idx, policy, max_words, overlap, start_id=len(out)))
    return out
