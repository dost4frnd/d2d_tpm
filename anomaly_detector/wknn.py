"""Weighted K-Nearest-Neighbours (WKNN) classifier.

Distance-weighted voting: each neighbour contributes ``w = 1 / (d + eps)`` (or a
Gaussian kernel) to its class.  Features are standardised before the neighbour
search.  Implemented on top of ``sklearn.neighbors.NearestNeighbors`` for an
efficient KD-/ball-tree search while keeping the weighting explicit.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

from .features import N_CLASSES


@dataclass
class WKNNConfig:
    k: int = 15
    weighting: Literal["inverse", "gaussian"] = "inverse"
    gamma: float = 1.0          # bandwidth for the gaussian kernel
    eps: float = 1e-6


class WeightedKNN:
    def __init__(self, cfg: WKNNConfig | None = None, n_classes: int = N_CLASSES):
        self.cfg = cfg or WKNNConfig()
        self.n_classes = n_classes
        self.scaler = StandardScaler()
        self.nn = NearestNeighbors(n_neighbors=self.cfg.k)
        self.y_train: np.ndarray | None = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> "WeightedKNN":
        Xs = self.scaler.fit_transform(X)
        self.nn.set_params(n_neighbors=min(self.cfg.k, len(Xs)))
        self.nn.fit(Xs)
        self.y_train = np.asarray(y).astype(int)
        return self

    def _weights(self, dist: np.ndarray) -> np.ndarray:
        if self.cfg.weighting == "gaussian":
            return np.exp(-self.cfg.gamma * dist ** 2)
        return 1.0 / (dist + self.cfg.eps)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if self.y_train is None:
            raise RuntimeError("fit() before predict")
        Xs = self.scaler.transform(X)
        dist, idx = self.nn.kneighbors(Xs)
        w = self._weights(dist)
        proba = np.zeros((Xs.shape[0], self.n_classes), dtype=np.float64)
        neigh_labels = self.y_train[idx]                       # (n, k)
        for c in range(self.n_classes):
            proba[:, c] = np.where(neigh_labels == c, w, 0.0).sum(axis=1)
        proba /= proba.sum(axis=1, keepdims=True) + 1e-12
        return proba

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.predict_proba(X).argmax(axis=1)
