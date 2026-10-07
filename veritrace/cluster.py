"""cluster pruning for the dense index (IIR 7.1.6, leaders and followers).

preprocessing: pick sqrt(N) random chunks as leaders; attach every chunk (follower) to its b1 nearest
leaders. query time: find the b2 leaders closest to the query and compute the exact cosine only for
their followers. the vectors are L2-normalised, so the dot product is the cosine."""
from __future__ import annotations

import math
import pickle

import numpy as np


class ClusterPrunedIndex:
    def __init__(self, emb: np.ndarray, n_leaders: int | None = None, b1: int = 1, seed: int = 0):
        self.emb = emb
        n = len(emb)
        self.n_leaders = n_leaders or int(round(math.sqrt(n)))
        rng = np.random.default_rng(seed)
        self.leaders = np.sort(rng.choice(n, self.n_leaders, replace=False))
        self.b1 = b1
        sims = emb @ emb[self.leaders].T                       # (n, L)
        nearest = np.argsort(-sims, axis=1)[:, :b1]            # each follower -> its b1 nearest leaders
        self.followers: list[np.ndarray] = [[] for _ in range(self.n_leaders)]
        for f, ls in enumerate(nearest):
            for l in ls:
                self.followers[l].append(f)
        self.followers = [np.asarray(x, dtype=np.int64) for x in self.followers]

    def scores(self, qvec: np.ndarray, b2: int = 5) -> tuple[np.ndarray, int]:
        """chunk scores with -inf for chunks that were never looked at, plus how many were scored"""
        lead_sim = self.emb[self.leaders] @ qvec
        top_leaders = np.argsort(-lead_sim)[:b2]
        cand = np.unique(np.concatenate([self.followers[l] for l in top_leaders]))
        out = np.full(len(self.emb), -np.inf, dtype=np.float32)
        out[cand] = self.emb[cand] @ qvec
        return out, int(len(cand) + self.n_leaders)

    def cluster_sizes(self) -> np.ndarray:
        return np.array([len(f) for f in self.followers])

    def save(self, path):
        with open(path, "wb") as f:
            pickle.dump({"leaders": self.leaders, "followers": self.followers, "b1": self.b1}, f)
