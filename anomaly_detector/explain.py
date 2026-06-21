"""Explainability for the anomaly detector.

Permutation importance is model-agnostic: it measures the drop in macro-F1 when
each feature column is shuffled, so it works for the WKNN, the FNN and the hybrid
alike.  Per-class feature means give an interpretable profile of how each attack
class manifests in the telemetry.
"""
from __future__ import annotations

from typing import Callable, Dict, List

import numpy as np
from sklearn.metrics import f1_score

from .features import CLASSES, FEATURES


def _macro_f1(model, X: np.ndarray, y: np.ndarray) -> float:
    return f1_score(y, model.predict(X), average="macro")


def permutation_importance(model, X: np.ndarray, y: np.ndarray,
                           n_repeats: int = 10, seed: int = 0,
                           score_fn: Callable | None = None) -> Dict[str, float]:
    """Return {feature_name: mean importance} (higher == more important)."""
    gen = np.random.default_rng(seed)
    score_fn = score_fn or _macro_f1
    baseline = score_fn(model, X, y)
    importances: Dict[str, float] = {}
    for j, name in enumerate(FEATURES):
        drops = []
        for _ in range(n_repeats):
            Xp = X.copy()
            gen.shuffle(Xp[:, j])
            drops.append(baseline - score_fn(model, Xp, y))
        importances[name] = float(np.mean(drops))
    return importances


def per_class_feature_means(X: np.ndarray, y: np.ndarray) -> Dict[str, Dict[str, float]]:
    """{class_name: {feature_name: mean}} - an interpretable class profile."""
    out: Dict[str, Dict[str, float]] = {}
    y = np.asarray(y)
    for ci, cname in enumerate(CLASSES):
        mask = y == ci
        if mask.sum() == 0:
            continue
        means = X[mask].mean(axis=0)
        out[cname] = {f: float(m) for f, m in zip(FEATURES, means)}
    return out
