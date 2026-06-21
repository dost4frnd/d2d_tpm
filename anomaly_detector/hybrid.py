"""Hybrid WKNN + FNN ensemble.

The hybrid combines a distance-weighted KNN (strong on local, low-data manifolds
such as rare carrier-disruption clusters) with an FNN (strong on smooth global
boundaries).  Default fusion is confidence-weighted soft voting:

    p = alpha * p_fnn + (1 - alpha) * p_wknn

``alpha`` may be fixed or set to the per-sample FNN confidence (its max softmax),
which defers to the KNN when the network is uncertain.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Tuple

import numpy as np

from .features import N_CLASSES, N_FEATURES
from .fnn import FNNClassifier, FNNConfig
from .wknn import WeightedKNN, WKNNConfig


@dataclass
class HybridConfig:
    alpha: float = 0.6                 # weight on the FNN when fusion == "fixed"
    fusion: str = "fixed"              # "fixed" | "confidence"
    wknn: WKNNConfig = field(default_factory=WKNNConfig)
    fnn: FNNConfig = field(default_factory=FNNConfig)


class HybridWKNNFNN:
    def __init__(self, cfg: Optional[HybridConfig] = None,
                 in_dim: int = N_FEATURES, n_classes: int = N_CLASSES):
        self.cfg = cfg or HybridConfig()
        self.n_classes = n_classes
        self.wknn = WeightedKNN(self.cfg.wknn, n_classes=n_classes)
        self.fnn = FNNClassifier(self.cfg.fnn, in_dim=in_dim, n_classes=n_classes)

    def fit(self, X: np.ndarray, y: np.ndarray,
            X_val: Optional[np.ndarray] = None, y_val: Optional[np.ndarray] = None
            ) -> "HybridWKNNFNN":
        self.wknn.fit(X, y)
        self.fnn.fit(X, y, X_val, y_val)
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        p_knn = self.wknn.predict_proba(X)
        p_fnn = self.fnn.predict_proba(X)
        if self.cfg.fusion == "confidence":
            alpha = p_fnn.max(axis=1, keepdims=True)     # defer to KNN when unsure
        else:
            alpha = self.cfg.alpha
        p = alpha * p_fnn + (1.0 - alpha) * p_knn
        p /= p.sum(axis=1, keepdims=True) + 1e-12
        return p

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.predict_proba(X).argmax(axis=1)
