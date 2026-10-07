"""query spelling correction (IIR ch. 3): k-gram index + Jaccard for candidate generation,
edit distance for ranking, collection frequency to break ties, Soundex as a phonetic fallback.

only words that are NOT in the corpus vocabulary are touched (isolated-term correction), so a
correctly spelled query word that occurs in the corpus is never changed. guards against "correcting"
real but rare words (gene names, inflections the corpus lacks):
  * a word whose Porter stem is already an index term is left alone (it matches anyway);
  * acronyms (all capitals) and tokens with digits are left alone;
  * at most 1 edit for words shorter than 8 letters, 2 edits otherwise, and bigram Jaccard >= 0.35;
  * a correction that only adds or removes a prefix/suffix (deactivates -> activates) is rejected,
    because that changes the meaning instead of fixing a typo;
  * context check (IIR 3.3.5): the corrected word must co-occur with another query term in at least
    one chunk, found by intersecting their postings lists."""
from __future__ import annotations

import math
from collections import Counter, defaultdict
from dataclasses import dataclass

from .text import STOPWORDS, stem, tokenize

K = 2  # bigrams, with $ marking the word boundaries ("$cancer$" -> $c ca an nc ce er r$)


def kgrams(word: str, k: int = K) -> set[str]:
    w = f"${word}$"
    return {w[i:i + k] for i in range(len(w) - k + 1)}


def edit_distance(a: str, b: str, cap: int = 3) -> int:
    # damerau-levenshtein (adjacent transpositions count as one edit), early exit above cap
    if abs(len(a) - len(b)) > cap:
        return cap + 1
    prev2, prev = None, list(range(len(b) + 1))
    for i in range(1, len(a) + 1):
        cur = [i] + [0] * len(b)
        for j in range(1, len(b) + 1):
            cost = 0 if a[i - 1] == b[j - 1] else 1
            cur[j] = min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + cost)
            if prev2 is not None and i > 1 and j > 1 and a[i - 1] == b[j - 2] and a[i - 2] == b[j - 1]:
                cur[j] = min(cur[j], prev2[j - 2] + 1)
        if min(cur) > cap:
            return cap + 1
        prev2, prev = prev, cur
    return prev[-1]


_SOUNDEX = {**dict.fromkeys("bfpv", "1"), **dict.fromkeys("cgjkqsxz", "2"), **dict.fromkeys("dt", "3"),
            "l": "4", **dict.fromkeys("mn", "5"), "r": "6"}


def soundex(word: str) -> str:
    # IIR 3.4: keep the first letter, map the rest to digit classes, drop vowels/h/w/y, collapse repeats
    word = word.lower()
    if not word:
        return ""
    digits = [_SOUNDEX.get(c, "0") for c in word]
    out, last = [word[0].upper()], digits[0]
    for c, d in zip(word[1:], digits[1:]):
        if d != "0" and d != last:
            out.append(d)
        if c not in "hw":
            last = d
    return ("".join(out) + "000")[:4]


@dataclass
class Correction:
    original: str
    corrected: str
    method: str
    distance: int
    jaccard: float


class SpellCorrector:
    def __init__(self, texts, min_len: int = 4, min_cf: int = 2, max_dist: int = 2, min_jaccard: float = 0.35,
                 index=None):
        self.index = index
        cf = Counter(t for text in texts for t in tokenize(text) if t.isalpha())
        self.cf = cf
        self.words = [w for w, c in cf.items() if c >= min_cf and len(w) >= 3]
        self.min_len, self.max_dist, self.min_jaccard = min_len, max_dist, min_jaccard
        # k-gram index: bigram -> ids of vocabulary words that contain it
        self.kgram_index: dict[str, list[int]] = defaultdict(list)
        self.n_grams = []
        for i, w in enumerate(self.words):
            g = kgrams(w)
            self.n_grams.append(len(g))
            for x in g:
                self.kgram_index[x].append(i)
        self.sdx: dict[str, list[int]] = defaultdict(list)
        for i, w in enumerate(self.words):
            self.sdx[soundex(w)].append(i)

    @classmethod
    def from_index(cls, index, **kw) -> "SpellCorrector":
        return cls([c.text for c in index.chunks] + [d.title for d in index.docs], index=index, **kw)

    def needs_correction(self, token: str) -> bool:
        if not (token.isalpha() and len(token) >= self.min_len and token not in STOPWORDS):
            return False
        if self.cf.get(token, 0) > 0:
            return False
        # the stem is already in the dictionary, so the word will match without any correction
        if self.index is not None and self.index.body.tid(stem(token)) >= 0:
            return False
        return True

    @staticmethod
    def _affix_only(a: str, b: str) -> bool:
        short, long_ = sorted((a, b), key=len)
        return len(long_) - len(short) >= 2 and (long_.startswith(short) or long_.endswith(short))

    def _in_context(self, word: str, context_terms: list[str]) -> bool:
        if self.index is None or not context_terms:
            return True
        from .boolean import intersect
        z = self.index.body
        t = z.tid(stem(word))
        if t < 0:
            return False
        p = z.units[t].tolist()
        for c in context_terms:
            ct = z.tid(c)
            if ct >= 0 and intersect(p, z.units[ct].tolist()):
                return True
        return False

    def candidates(self, token: str) -> list[tuple[int, float]]:
        g = kgrams(token)
        overlap: Counter = Counter()
        for x in g:
            for i in self.kgram_index.get(x, ()):
                overlap[i] += 1
        out = []
        for i, o in overlap.items():
            jac = o / (len(g) + self.n_grams[i] - o)
            if jac >= self.min_jaccard:
                out.append((i, jac))
        return out

    def correct_token(self, token: str, context_terms: list[str] | None = None) -> Correction | None:
        if not self.needs_correction(token):
            return None
        max_d = self.max_dist if len(token) >= 8 else 1
        best = None
        for i, jac in self.candidates(token):
            w = self.words[i]
            d = edit_distance(token, w, max_d)
            if d <= max_d and not self._affix_only(token, w) and self._in_context(w, context_terms or []):
                # rank: fewest edits, then the more frequent word, then higher k-gram overlap
                key = (d, -math.log(self.cf[w]), -jac)
                if best is None or key < best[0]:
                    best = (key, w, d, jac)
        if best is not None:
            return Correction(token, best[1], "k-gram+edit", best[2], round(best[3], 3))
        # phonetic fallback: same soundex code and at most 3 edits
        cands = [(edit_distance(token, self.words[i], 2), -self.cf[self.words[i]], self.words[i])
                 for i in self.sdx.get(soundex(token), ())]
        cands = [c for c in cands if c[0] <= 2 and len(token) >= 6 and not self._affix_only(token, c[2])
                 and self._in_context(c[2], context_terms or [])]
        if cands:
            d, _, w = min(cands)
            return Correction(token, w, "soundex", d, round(len(kgrams(token) & kgrams(w)) /
                                                         len(kgrams(token) | kgrams(w)), 3))
        return None

    def correct_query(self, query: str) -> tuple[str, list[Correction]]:
        import re
        fixes = []
        # the other (known) query terms are the context for the co-occurrence check
        context = [stem(t) for t in tokenize(query) if t not in STOPWORDS and t.isalpha() and self.cf.get(t, 0) > 0]

        def repl(m):
            tok = m.group(0)
            # leave acronyms and gene/protein names (RUNX1, TMEM16A) alone
            if (tok.isupper() and len(tok) <= 6) or any(ch.isdigit() for ch in tok) or "-" in tok:
                return tok
            c = self.correct_token(tok.lower(), [t for t in context if t != stem(tok.lower())])
            if c is None:
                return tok
            fixes.append(c)
            return c.corrected
        return re.sub(r"[A-Za-z0-9][A-Za-z0-9-]*", repl, query), fixes
