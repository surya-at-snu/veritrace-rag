from __future__ import annotations

import math
import re

import numpy as np

from .index import InvertedIndex
from .text import STOPWORDS, stem, tokenize


# textbook two pointer merge with skip pointers every sqrt(len) postings
def intersect(p1: list[int], p2: list[int]) -> list[int]:
    out, i, j = [], 0, 0
    s1, s2 = max(1, int(math.sqrt(len(p1)))), max(1, int(math.sqrt(len(p2))))
    while i < len(p1) and j < len(p2):
        if p1[i] == p2[j]:
            out.append(p1[i]); i += 1; j += 1
        elif p1[i] < p2[j]:
            if i % s1 == 0 and i + s1 < len(p1) and p1[i + s1] <= p2[j]:
                while i % s1 == 0 and i + s1 < len(p1) and p1[i + s1] <= p2[j]:
                    i += s1
            else:
                i += 1
        else:
            if j % s2 == 0 and j + s2 < len(p2) and p2[j + s2] <= p1[i]:
                while j % s2 == 0 and j + s2 < len(p2) and p2[j + s2] <= p1[i]:
                    j += s2
            else:
                j += 1
    return out


def union(p1: list[int], p2: list[int]) -> list[int]:
    out, i, j = [], 0, 0
    while i < len(p1) and j < len(p2):
        if p1[i] == p2[j]:
            out.append(p1[i]); i += 1; j += 1
        elif p1[i] < p2[j]:
            out.append(p1[i]); i += 1
        else:
            out.append(p2[j]); j += 1
    out.extend(p1[i:]); out.extend(p2[j:])
    return out


def and_not(p1: list[int], p2: list[int]) -> list[int]:
    out, i, j = [], 0, 0
    while i < len(p1):
        if j >= len(p2) or p1[i] < p2[j]:
            out.append(p1[i]); i += 1
        elif p1[i] == p2[j]:
            i += 1; j += 1
        else:
            j += 1
    return out


class BooleanEngine:
    def __init__(self, index: InvertedIndex):
        self.index = index
        self.all_units = list(range(index.n_chunks))

    def term_postings(self, word: str) -> list[int]:
        term = stem(word.lower())
        body = self.index.body.postings(term)[0]
        tdocs = self.index.title.postings(term)[0]
        if len(tdocs):
            starts = self.index.doc_chunk_start
            ends = np.append(starts[1:], self.index.n_chunks)
            from_title = np.concatenate([np.arange(starts[d], ends[d]) for d in tdocs])
            body = np.union1d(body, from_title)
        return body.tolist()

    # phrase query: intersect the postings, then check the positions keep the same gaps as in the query
    def phrase_postings(self, phrase: str) -> list[int]:
        toks = tokenize(phrase)
        qpos = [(stem(t), i) for i, t in enumerate(toks) if t not in STOPWORDS]
        if not qpos:
            return []
        if len(qpos) == 1:
            return self.index.body.postings(qpos[0][0])[0].tolist()
        body = self.index.body
        lists = sorted((body.postings(t)[0].tolist() for t, _ in qpos), key=len)
        cand = lists[0]
        for p in lists[1:]:
            cand = intersect(cand, p)
        out = []
        for c in cand:
            t0, p0 = qpos[0]
            starts = body.positions(t0, c)
            ok = np.ones(len(starts), dtype=bool)
            for t, p in qpos[1:]:
                ok &= np.isin(starts + (p - p0), body.positions(t, c))
            if ok.any():
                out.append(c)
        return out

    _TOKEN = re.compile(r'"[^"]+"|\(|\)|[^\s()"]+')

    # small recursive descent parser for and / or / not / brackets / "phrases"
    def parse(self, query: str):
        toks = self._TOKEN.findall(query)
        self._toks, self._i = toks, 0
        tree = self._expr()
        return tree

    def _peek(self):
        return self._toks[self._i] if self._i < len(self._toks) else None

    def _next(self):
        t = self._peek(); self._i += 1; return t

    def _expr(self):
        node = self._and()
        while self._peek() == "OR":
            self._next()
            node = ("OR", node, self._and())
        return node

    def _and(self):
        operands, negated = [], []
        while True:
            t = self._peek()
            if t is None or t in (")", "OR"):
                break
            if t == "AND":
                self._next(); continue
            if t == "NOT":
                self._next(); negated.append(self._factor()); continue
            operands.append(self._factor())
        return ("AND", operands, negated)

    def _factor(self):
        t = self._next()
        if t == "(":
            node = self._expr()
            if self._peek() == ")":
                self._next()
            return node
        if t.startswith('"'):
            return ("PHRASE", t.strip('"'))
        return ("TERM", t)

    def evaluate(self, node) -> list[int]:
        kind = node[0]
        if kind == "TERM":
            return self.term_postings(node[1])
        if kind == "PHRASE":
            return self.phrase_postings(node[1])
        if kind == "OR":
            return union(self.evaluate(node[1]), self.evaluate(node[2]))
        _, operands, negated = node
        # process the rarest term first so the intermediate lists stay small
        lists = sorted((self.evaluate(o) for o in operands), key=len)
        result = lists[0] if lists else self.all_units
        for p in lists[1:]:
            if not result:
                break
            result = intersect(result, p)
        for n in negated:
            result = and_not(result, self.evaluate(n))
        return result

    def query(self, q: str) -> list[int]:
        return self.evaluate(self.parse(q))
