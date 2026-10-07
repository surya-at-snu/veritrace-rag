from __future__ import annotations

import numpy as np

from .config import NLI_MODEL

_CACHE: dict = {}


class NLIScorer:
    def __init__(self, name: str = NLI_MODEL, max_length: int = 320):
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
        self.torch = torch
        self.tok = AutoTokenizer.from_pretrained(name)
        # force float32, the checkpoint loads in fp16 which is slow and unstable on cpu
        self.model = AutoModelForSequenceClassification.from_pretrained(name, dtype=torch.float32).eval()
        lab = {v.lower(): int(k) for k, v in self.model.config.id2label.items()}
        self.order = [lab["entailment"], lab["neutral"], lab["contradiction"]]
        self.max_length = max_length
        self.memo: dict = {}

    def __call__(self, pairs: list[tuple[str, str]], batch_size: int = 32) -> np.ndarray:
        out = np.zeros((len(pairs), 3), dtype=np.float32)
        # skip pairs we already scored
        todo = [i for i, p in enumerate(pairs) if p not in self.memo]
        todo.sort(key=lambda i: len(pairs[i][0]) + len(pairs[i][1]))
        for s in range(0, len(todo), batch_size):
            idx = todo[s:s + batch_size]
            prem = [pairs[i][0] for i in idx]
            hyp = [pairs[i][1] for i in idx]
            enc = self.tok(prem, hyp, truncation="only_first", max_length=self.max_length,
                           padding=True, return_tensors="pt")
            with self.torch.inference_mode():
                logits = self.model(**enc).logits
            probs = self.torch.softmax(logits, -1).numpy()[:, self.order]
            for i, p in zip(idx, probs):
                self.memo[pairs[i]] = p
        for i, p in enumerate(pairs):
            out[i] = self.memo[p]
        return out


def get_nli(name: str = NLI_MODEL) -> NLIScorer:
    if name not in _CACHE:
        _CACHE[name] = NLIScorer(name)
    return _CACHE[name]
