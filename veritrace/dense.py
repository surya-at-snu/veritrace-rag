from __future__ import annotations

import numpy as np

from .config import DENSE_MODEL, DENSE_QUERY_PREFIX

_MODELS: dict = {}


def load_encoder(name_or_path: str = DENSE_MODEL):
    if name_or_path not in _MODELS:
        from sentence_transformers import SentenceTransformer
        _MODELS[name_or_path] = SentenceTransformer(str(name_or_path), device="cpu")
        _MODELS[name_or_path].max_seq_length = 256
    return _MODELS[name_or_path]


def encode_passages(texts: list[str], model=None, batch_size: int = 64, show_progress: bool = True) -> np.ndarray:
    model = model or load_encoder()
    # sort by length so batches need less padding
    order = np.argsort([len(t) for t in texts])
    emb = model.encode([texts[i] for i in order], batch_size=batch_size, normalize_embeddings=True,
                       show_progress_bar=show_progress, convert_to_numpy=True)
    out = np.empty_like(emb)
    out[order] = emb
    return out.astype(np.float32)


def encode_queries(queries: list[str], model=None, batch_size: int = 64, prefix: str = DENSE_QUERY_PREFIX) -> np.ndarray:
    model = model or load_encoder()
    return model.encode([prefix + q for q in queries], batch_size=batch_size, normalize_embeddings=True,
                        show_progress_bar=False, convert_to_numpy=True).astype(np.float32)


def chunk_text_for_encoding(chunk) -> str:
    return f"{chunk.title}. {chunk.text}" if chunk.title else chunk.text


class DenseRetriever:
    def __init__(self, embeddings: np.ndarray, model_name: str = DENSE_MODEL):
        self.emb = embeddings
        self.model_name = model_name

    def chunk_scores(self, query: str) -> np.ndarray:
        q = encode_queries([query], load_encoder(self.model_name))[0]
        # vectors are normalised so a dot product is the cosine
        return self.emb @ q

    def chunk_scores_batch(self, qvecs: np.ndarray) -> np.ndarray:
        return qvecs @ self.emb.T
