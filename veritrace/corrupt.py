from __future__ import annotations

import random
import re

from .text import STOPWORDS, numbers, stem, tokenize

SUPPORTED, CONTRADICTED, UNSUPPORTED = 0, 1, 2
LABELS = ["SUPPORTED", "CONTRADICTED", "UNSUPPORTED"]

# word pairs used to flip the direction of a claim (increase <-> decrease and so on)
DIRECTION_PAIRS = [
    ("increases", "decreases"), ("increase", "decrease"), ("increased", "decreased"), ("increasing", "decreasing"),
    ("higher", "lower"), ("more", "less"), ("greater", "smaller"), ("raises", "lowers"), ("raise", "lower"),
    ("promotes", "inhibits"), ("promote", "inhibit"), ("activates", "suppresses"), ("activate", "suppress"),
    ("enhances", "impairs"), ("enhance", "impair"), ("improves", "worsens"), ("improve", "worsen"),
    ("improved", "worsened"), ("upregulates", "downregulates"), ("upregulated", "downregulated"),
    ("positive", "negative"), ("positively", "negatively"), ("high", "low"), ("induces", "prevents"),
    ("induce", "prevent"), ("gain", "loss"), ("above", "below"), ("longer", "shorter"), ("larger", "smaller"),
    ("faster", "slower"), ("better", "worse"), ("benefits", "harms"), ("protects", "damages"),
    ("elevated", "reduced"), ("accelerates", "slows"), ("stimulates", "suppresses"), ("reduces", "increases"),
    ("reduce", "increase"), ("reduced", "increased"), ("lowers", "raises"), ("decreases", "increases"),
    ("prevents", "causes"), ("inhibits", "promotes"), ("suppresses", "enhances"), ("impairs", "enhances"),
    ("blocks", "facilitates"), ("essential", "dispensable"),
]
DIRECTION = {}
for a, b in DIRECTION_PAIRS:
    DIRECTION.setdefault(a, b)
    DIRECTION.setdefault(b, a)
DIRECTION_STEMS = {}
for a, b in DIRECTION_PAIRS:
    DIRECTION_STEMS.setdefault(stem(a), set()).add(stem(b))
    DIRECTION_STEMS.setdefault(stem(b), set()).add(stem(a))

_VERB_BASES = """reduce increase cause induce prevent inhibit promote regulate activate require improve predict affect
contain mediate enhance impair suppress protect correlate lead result play show bind block decrease elevate lower
raise trigger drive control modulate contribute associate depend determine influence limit alter accelerate delay
exhibit express encode interact target confer upregulate downregulate stimulate attenuate ameliorate exacerbate
worsen reverse restore impact facilitate mimic produce generate cross reach respond recruit lack correspond
outperform exceed occur persist decline rise involve trigger form allow enable undergo""".split()


def _third(base: str) -> str:
    if base.endswith(("s", "sh", "ch", "x", "z", "o")):
        return base + "es"
    if base.endswith("y") and base[-2] not in "aeiou":
        return base[:-1] + "ies"
    return base + "s"


_THIRD_TO_BASE = {_third(b): b for b in _VERB_BASES}
_PAST_TO_BASE = {(b + "d" if b.endswith("e") else b + "ed"): b for b in _VERB_BASES}
_PAST_TO_BASE.update({"led": "lead", "showed": "show", "bound": "bind", "rose": "rise", "underwent": "undergo"})

_AUX = ["is", "are", "was", "were", "can", "could", "may", "might", "will", "would", "should", "has", "have", "had", "does", "do", "did"]


def _match_case(src: str, new: str) -> str:
    return new.capitalize() if src[:1].isupper() else new


# add or remove a negation: is -> is not, reduces -> does not reduce
def negate(claim: str) -> str | None:
    for pat, rep in ((r"\bdoes not (\w+)", r"\1s"), (r"\bdo not (\w+)", r"\1"), (r"\b(is|are|was|were|can|has|have) not\b", r"\1"),
                     (r"\bcannot\b", "can"), (r"\bno\b", "some"), (r"\bnot\b ", "")):
        new = re.sub(pat, rep, claim, count=1, flags=re.IGNORECASE)
        if new != claim:
            return new
    for aux in _AUX:
        m = re.search(rf"\b{aux}\b", claim)
        if m:
            neg = "cannot" if aux == "can" else f"{aux} not"
            return claim[:m.start()] + neg + claim[m.end():]
    for m in re.finditer(r"\w+", claim):
        base = _THIRD_TO_BASE.get(m.group(0))
        if base and m.start() > 0:
            return claim[:m.start()] + f"does not {base}" + claim[m.end():]
        base = _PAST_TO_BASE.get(m.group(0))
        if base and m.start() > 0:
            return claim[:m.start()] + f"did not {base}" + claim[m.end():]
    return None


def flip_direction(claim: str) -> str | None:
    words = re.findall(r"[A-Za-z]+", claim)
    for w in words:
        if w.lower() in DIRECTION:
            return re.sub(rf"\b{re.escape(w)}\b", _match_case(w, DIRECTION[w.lower()]), claim, count=1)
    return None


def swap_number(claim: str, rnd: random.Random) -> str | None:
    ms = list(re.finditer(r"(?<![A-Za-z\-])(\d+(?:\.\d+)?)(?![A-Za-z])", claim))
    if not ms:
        return None
    # prefer changing a real quantity over a year
    non_year = [m for m in ms if not (m.group(1).isdigit() and 1900 <= int(m.group(1)) <= 2030)]
    m = rnd.choice(non_year or ms)
    v = float(m.group(1))
    if v in (0.0, 1.0) and len(ms) == 1:
        new = v + rnd.choice([2, 3, 5])
    else:
        new = v * rnd.choice([0.3, 0.5, 2.0, 3.0]) + rnd.choice([0, 1, 7])
    s = f"{new:.{len(m.group(1).split('.')[1])}f}" if "." in m.group(1) else str(int(round(new)))
    if s == m.group(1):
        s = str(int(v) + 11)
    return claim[:m.start(1)] + s + claim[m.end(1):]


def entity_swap(claim: str, evidence: str, index, rnd: random.Random, tries: int = 400) -> str | None:
    ev_stems = {stem(t) for t in tokenize(evidence)}
    cands = []
    for w in set(re.findall(r"[A-Za-z][A-Za-z0-9\-]{3,}", claim)):
        t = w.lower()
        if t in STOPWORDS or t in DIRECTION:
            continue
        s = stem(t)
        if s in ev_stems and index.body.tid(s) >= 0:
            cands.append((index.body.idf(s), w))
    if not cands:
        return None
    # the claim word with the highest idf that the evidence also has carries the meaning
    target_idf, word = max(cands)
    for _ in range(tries):
        c = index.chunks[rnd.randrange(index.n_chunks)]
        toks = [t for t in set(re.findall(r"[a-z][a-z0-9\-]{3,}", c.text.lower())) if t not in STOPWORDS]
        rnd.shuffle(toks)
        for t in toks:
            s = stem(t)
            if s in ev_stems or s == stem(word.lower()) or index.body.tid(s) < 0:
                continue
            # swap it for a word of similar idf from an unrelated chunk so the fake still sounds plausible
            if abs(index.body.idf(s) - target_idf) < 0.25:
                return re.sub(rf"\b{re.escape(word)}\b", _match_case(word, t), claim, count=1)
    return None


def corrupt(claim: str, evidence: str, index, rnd: random.Random, kind: str | None = None):
    kinds = [kind] if kind else rnd.sample(["negation", "direction", "number", "entity"], 4)
    for k in kinds:
        if k == "negation":
            new, lab = negate(claim), CONTRADICTED
        elif k == "direction":
            new, lab = flip_direction(claim), CONTRADICTED
        elif k == "number":
            new, lab = (swap_number(claim, rnd) if numbers(claim) else None), CONTRADICTED
        else:
            new, lab = entity_swap(claim, evidence, index, rnd), UNSUPPORTED
        if new and new.strip() != claim.strip():
            return new, k, lab
    return None
