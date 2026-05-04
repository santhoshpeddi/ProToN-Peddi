from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np


def dA(A: np.ndarray, xi: np.ndarray, xj: np.ndarray) -> float:
    """Mahalanobis distance d_A(x_i, x_j) = (x_i - x_j)^T A (x_i - x_j)."""
    v = xi - xj
    return float(v.T @ A @ v)


def _ensure_pd(A: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    A = 0.5 * (A + A.T)
    return A + eps * np.eye(A.shape[0], dtype=A.dtype)


def _eq7_shrink_distance(gamma: float, xiij: float) -> float:
    if gamma <= 0:
        return xiij
    return (gamma * xiij) / (gamma + xiij + 1e-12)


def _project_bregman_like(A: np.ndarray, v: np.ndarray, target: float) -> np.ndarray:
    """One-step projection update A <- A + beta A v v^T A toward a target distance."""
    A = _ensure_pd(A)
    p = float(v.T @ A @ v)
    if p <= 1e-12:
        return A
    beta = (target - p) / (p * p)
    if beta <= (-1.0 / p) + 1e-12:
        beta = (-1.0 / p) + 1e-6
    Av = A @ v
    return _ensure_pd(A + beta * np.outer(Av, Av))


def _mahalanobis_nn_sets(X: np.ndarray, y: np.ndarray, A: np.ndarray) -> Tuple[List[Tuple[int, int]], List[Tuple[int, int]]]:
    """Build nearest-neighbor similar and dissimilar constraints under the current metric."""
    n, _ = X.shape
    A = _ensure_pd(A)
    try:
        L = np.linalg.cholesky(A).T
    except np.linalg.LinAlgError:
        A = _ensure_pd(A, eps=1e-4)
        L = np.linalg.cholesky(A).T

    Z = X @ L.T
    S, D = [], []
    for i in range(n):
        dists = np.sum((Z - Z[i]) ** 2, axis=1)
        dists[i] = np.inf
        same = np.where(y == y[i])[0]
        diff = np.where(y != y[i])[0]
        same = same[same != i]
        if same.size > 0:
            S.append((i, int(same[np.argmin(dists[same])])))
        if diff.size > 0:
            D.append((i, int(diff[np.argmin(dists[diff])])) )
    return S, D


@dataclass
class DEPML:
    u: float = 1.0
    l: float = 4.0
    m_cycles: int = 15
    eta: float = 1e-3
    gamma: float = 1.0
    use_eq7: bool = True
    knn_k: int = 3

    A_: Optional[np.ndarray] = None
    X_: Optional[np.ndarray] = None
    y_: Optional[np.ndarray] = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> "DEPML":
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=int)
        _, d = X.shape
        A = np.eye(d, dtype=np.float64)

        u_t = _eq7_shrink_distance(self.gamma, self.u) if self.use_eq7 else self.u
        l_t = _eq7_shrink_distance(self.gamma, self.l) if self.use_eq7 else self.l
        prev = None

        for cycle in range(1, self.m_cycles + 1):
            S, D = _mahalanobis_nn_sets(X, y, A)
            distances = []

            for i, j in S:
                v = X[i] - X[j]
                dist = float(v.T @ A @ v)
                distances.append(dist)
                if dist > u_t:
                    A = _project_bregman_like(A, v, target=u_t)

            for i, j in D:
                v = X[i] - X[j]
                dist = float(v.T @ A @ v)
                distances.append(dist)
                if dist < l_t:
                    A = _project_bregman_like(A, v, target=l_t)

            distances = np.asarray(distances, dtype=np.float64)
            if prev is not None and len(prev) and len(distances):
                mlen = min(len(prev), len(distances))
                delta = float(np.mean(np.abs(distances[:mlen] - prev[:mlen])))
                print(f"[DEP-ML] cycle {cycle:02d}/{self.m_cycles} mean|delta d|={delta:.6f}")
                if delta < self.eta:
                    break
            else:
                print(f"[DEP-ML] cycle {cycle:02d}/{self.m_cycles}")
            prev = distances

        self.A_ = _ensure_pd(A)
        self.X_ = X
        self.y_ = y
        return self

    def predict(self, Xq: np.ndarray) -> np.ndarray:
        if self.A_ is None or self.X_ is None or self.y_ is None:
            raise RuntimeError("DEPML model has not been fitted.")
        Xq = np.asarray(Xq, dtype=np.float64)
        return np.array([self._predict_one(x) for x in Xq], dtype=int)

    def _predict_one(self, x: np.ndarray) -> int:
        assert self.A_ is not None and self.X_ is not None and self.y_ is not None
        V = self.X_ - x[None, :]
        dists = np.einsum("ni,ij,nj->n", V, self.A_, V)
        k = min(self.knn_k, len(dists))
        nn_idx = np.argpartition(dists, k - 1)[:k]
        votes = self.y_[nn_idx]
        vals, counts = np.unique(votes, return_counts=True)
        return int(vals[np.argmax(counts)])
