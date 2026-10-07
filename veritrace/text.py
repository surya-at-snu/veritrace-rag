from __future__ import annotations

import re
from functools import lru_cache

from nltk.stem import PorterStemmer

# standard english stop list. negation words are stop words for retrieval, the verifier checks them separately
STOPWORDS = frozenset("""
a about above after again against all am an and any are as at be because been before being below
between both but by can could did do does doing down during each few for from further had has have
having he her here hers herself him himself his how i if in into is it its itself just me more most
my myself no nor not of off on once only or other our ours ourselves out over own same she should so
some such than that the their theirs them themselves then there these they this those through to too
under until up very was we were what when where which while who whom why will with would you your
yours yourself yourselves also may might must shall via et al within without upon among whether
""".split())

NEGATION_CUES = frozenset("""no not never none nor neither cannot without lack lacks lacking absence absent
fail fails failed unable doesn't don't didn't isn't aren't wasn't weren't won't wouldn't""".split())

# decimals like 0.05 stay one token, everything else splits on non alphanumerics
_TOKEN_RE = re.compile(r"\d+(?:\.\d+)?|[a-z0-9]+")
_NUMBER_RE = re.compile(r"(?<![a-z])\d+(?:[.,]\d+)?")
_stemmer = PorterStemmer()


@lru_cache(maxsize=500_000)
def stem(token: str) -> str:
    return _stemmer.stem(token)


def tokenize(text: str) -> list[str]:
    return _TOKEN_RE.findall(text.lower())


# the full chain: lowercase -> tokens -> drop stop words -> porter stem
def analyze(text: str, keep_stopwords: bool = False) -> list[str]:
    return [stem(t) for t in tokenize(text) if keep_stopwords or t not in STOPWORDS]


# positions come from the original token stream so phrases still line up after stop words are dropped
def analyze_with_positions(text: str) -> list[tuple[str, int]]:
    return [(stem(t), i) for i, t in enumerate(tokenize(text)) if t not in STOPWORDS]


# ---------------------------------------------------------------------------------------------
# configurable analysis chains, used by the analysis-chain ablation (scripts/exp_ir_extras.py analysis).
# the default chain above (stop words + porter) is what the system uses.
def s_stem(token: str) -> str:
    # harman's s-stemmer: a light stemmer that only conflates plurals
    if len(token) > 3 and token.endswith("ies") and not token.endswith(("eies", "aies")):
        return token[:-3] + "y"
    if len(token) > 3 and token.endswith("es") and not token.endswith(("aes", "ees", "oes")):
        return token[:-1]
    if len(token) > 2 and token.endswith("s") and not token.endswith(("us", "ss")):
        return token[:-1]
    return token


_STEMMERS = {"porter": stem, "s": s_stem, "none": lambda t: t}


class Analyzer:
    """one analysis chain: case folding -> tokens -> (optional) stop words -> (optional) stemmer.
    the same object must be used for the documents and the queries, otherwise terms do not match."""

    def __init__(self, stemmer: str = "porter", stopwords: bool = True):
        self.name = f"{stemmer}{'' if stopwords else '+stopwords'}"
        self.stem = _STEMMERS[stemmer]
        self.stopwords = stopwords

    def __call__(self, text: str) -> list[str]:
        return [self.stem(t) for t in tokenize(text) if not (self.stopwords and t in STOPWORDS)]

    def with_positions(self, text: str) -> list[tuple[str, int]]:
        return [(self.stem(t), i) for i, t in enumerate(tokenize(text)) if not (self.stopwords and t in STOPWORDS)]


def numbers(text: str) -> set[str]:
    out = set()
    for n in _NUMBER_RE.findall(text.lower()):
        n = n.replace(",", "")
        if "." in n:
            n = n.rstrip("0").rstrip(".")
        out.add(n)
    return out


def has_negation(text: str) -> bool:
    toks = set(re.findall(r"[a-z']+", text.lower()))
    return bool(toks & NEGATION_CUES) or "n't" in text.lower()


# protect things like e.g. and et al. so the sentence splitter doesnt cut there
_ABBREV = ("e.g.", "i.e.", "et al.", "vs.", "fig.", "figs.", "no.", "approx.", "ca.", "dr.", "mr.", "ms.",
           "inc.", "ltd.", "u.s.", "u.k.", "cf.", "resp.", "etc.")
_SENT_SPLIT = re.compile(r"(?<=[.!?])[\"')\]]*\s+(?=[A-Z0-9(\[\"'])")


def split_sentences(text: str) -> list[str]:
    text = re.sub(r"\s+", " ", text).strip()
    if not text:
        return []
    protected = text
    for i, ab in enumerate(_ABBREV):
        protected = re.sub(re.escape(ab), f"\x00{i}\x00", protected, flags=re.IGNORECASE)
    parts = _SENT_SPLIT.split(protected)
    out = []
    for p in parts:
        for i, ab in enumerate(_ABBREV):
            p = p.replace(f"\x00{i}\x00", ab)
        p = p.strip()
        if p:
            out.append(p)
    return out
