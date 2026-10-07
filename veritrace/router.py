from __future__ import annotations

import pickle

import numpy as np

ALPHAS = np.round(np.linspace(0, 1, 21), 2)
BETAS = np.round(np.linspace(0, 0.8, 9), 2)
CE_ARB_FEATURES = ["ce_sp1", "ce_de1", "ce_sp3", "ce_de3", "ce_diff1", "ce_diff3"]


# cross encoder looks at the top 3 of each list and tells the router which list looks better
def ce_arbitration(ce_sparse_top3: np.ndarray, ce_dense_top3: np.ndarray) -> dict:
    return {"ce_sp1": float(ce_sparse_top3[0]), "ce_de1": float(ce_dense_top3[0]),
            "ce_sp3": float(ce_sparse_top3.mean()), "ce_de3": float(ce_dense_top3.mean()),
            "ce_diff1": float(ce_sparse_top3[0] - ce_dense_top3[0]),
            "ce_diff3": float(ce_sparse_top3.max() - ce_dense_top3.max())}


class FusionRouter:
    def __init__(self, C: float = 0.3):
        self.C = C
        self.clf = None
        self.mu = self.sd = None
        self.a0, self.beta = 0.5, 0.0
        self.mean_curve = None
        self.cv_table = None
        self.feature_names = None

    def _z(self, X):
        return (X - self.mu) / self.sd

    def _fit_clf(self, X, curves):
        from sklearn.linear_model import LogisticRegression
        # label: did bm25 alone beat dense alone on this training query
        diff = curves[:, -1] - curves[:, 0]
        m = diff != 0
        clf = LogisticRegression(C=self.C, max_iter=5000)
        clf.fit(self._z(X[m]), (diff[m] > 0).astype(int))
        return clf

    @staticmethod
    def _gain(curves, p, a0, beta):
        # fusion weight a(q) = a0 + beta * (2p - 1), clipped to [0, 1]
        a = np.clip(a0 + beta * (2 * p - 1), 0, 1)
        j = np.abs(ALPHAS[None, :] - a[:, None]).argmin(1)
        return curves[np.arange(len(curves)), j].mean()

    def fit(self, X: np.ndarray, curves: np.ndarray, feature_names=None, folds: int = 5, seed: int = 0):
        from sklearn.model_selection import KFold
        self.feature_names = feature_names
        self.mean_curve = curves.mean(0)
        self.mu, self.sd = X.mean(0), X.std(0) + 1e-9
        oof = np.zeros(len(X))
        for tr, va in KFold(folds, shuffle=True, random_state=seed).split(X):
            clf = self._fit_clf(X[tr], curves[tr])
            oof[va] = clf.predict_proba(self._z(X[va]))[:, 1]
        # pick a0 and beta by cross validation, beta = 0 just means fall back to the fixed weight
        table = np.array([[self._gain(curves, oof, a0, b) for b in BETAS] for a0 in ALPHAS])
        i, j = np.unravel_index(np.argmax(table), table.shape)
        self.a0, self.beta = float(ALPHAS[i]), float(BETAS[j])
        self.cv_table = table
        self.oof_p = oof
        self.clf = self._fit_clf(X, curves)
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return self.clf.predict_proba(self._z(np.atleast_2d(X)))[:, 1]

    def predict_alpha(self, X: np.ndarray) -> np.ndarray:
        return np.clip(self.a0 + self.beta * (2 * self.predict_proba(X) - 1), 0, 1)

    @property
    def best_fixed_alpha(self) -> float:
        return float(ALPHAS[int(np.argmax(self.mean_curve))])

    @property
    def cv_gain(self) -> float:
        return float(self.cv_table.max() - self.cv_table[:, 0].max())

    def coefficients(self) -> dict:
        return dict(zip(self.feature_names or range(len(self.mu)), self.clf.coef_[0].round(4).tolist()))

    def save(self, path):
        with open(path, "wb") as f:
            pickle.dump(self, f)

    @staticmethod
    def load(path) -> "FusionRouter":
        with open(path, "rb") as f:
            return pickle.load(f)


def router_vector(entry: dict, names) -> np.ndarray:
    feats = {**entry["qpp"], **entry.get("ce_arb", {})}
    return np.array([feats[n] for n in names], dtype=np.float32)
