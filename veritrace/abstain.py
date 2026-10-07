from __future__ import annotations

import pickle

import numpy as np


class RetrievalConfidence:
    def __init__(self, C: float = 0.5):
        self.C = C
        self.clf = None
        self.mu = self.sd = None
        self.tau = 0.5
        self.feature_names = None

    def fit(self, X: np.ndarray, y: np.ndarray, feature_names=None, abstain_rate: float = 0.1):
        from sklearn.linear_model import LogisticRegression
        self.feature_names = feature_names
        self.mu, self.sd = X.mean(0), X.std(0) + 1e-9
        self.clf = LogisticRegression(C=self.C, max_iter=5000).fit((X - self.mu) / self.sd, y)
        # threshold = abstain on the least confident 10% of training queries
        self.tau = float(np.quantile(self.predict(X), abstain_rate))
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.clf.predict_proba((np.atleast_2d(X) - self.mu) / self.sd)[:, 1]

    def save(self, path):
        with open(path, "wb") as f:
            pickle.dump(self, f)

    @staticmethod
    def load(path) -> "RetrievalConfidence":
        with open(path, "rb") as f:
            return pickle.load(f)


# answer the most confident queries first and track how often retrieval failed
def risk_coverage(conf: np.ndarray, success: np.ndarray):
    order = np.argsort(-conf, kind="stable")
    s = success[order].astype(float)
    n = np.arange(1, len(s) + 1)
    coverage = n / len(s)
    risk = 1 - np.cumsum(s) / n
    aurc = float(np.mean(risk))
    return coverage, risk, aurc
