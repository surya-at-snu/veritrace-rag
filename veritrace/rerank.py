from __future__ import annotations

import numpy as np

from .config import CROSS_ENCODER

_CE = {}


def load_cross_encoder(name: str = CROSS_ENCODER):
    if name not in _CE:
        from sentence_transformers import CrossEncoder
        _CE[name] = CrossEncoder(name, device="cpu", max_length=256)
    return _CE[name]


def ce_scores(pairs: list[tuple[str, str]], batch_size: int = 64) -> np.ndarray:
    if not pairs:
        return np.zeros(0, dtype=np.float32)
    model = load_cross_encoder()
    # length sorted batches, a lot faster on cpu
    order = np.argsort([len(a) + len(b) for a, b in pairs])
    s = model.predict([pairs[i] for i in order], batch_size=batch_size, show_progress_bar=False)
    out = np.empty(len(pairs), dtype=np.float32)
    out[order] = np.asarray(s, dtype=np.float32)
    return out
