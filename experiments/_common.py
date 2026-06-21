"""Shared helpers for the experiment scripts: dataset IO, splitting, factories."""
from __future__ import annotations

import os
import sys
from typing import Dict, List, Tuple

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from anomaly_detector.features import FEATURES, N_CLASSES  # noqa: E402
from anomaly_detector.fnn import FNNClassifier, FNNConfig  # noqa: E402
from anomaly_detector.hybrid import HybridConfig, HybridWKNNFNN  # noqa: E402
from anomaly_detector.wknn import WeightedKNN, WKNNConfig  # noqa: E402

DEFAULT_DATASET = os.path.join(_ROOT, "datasets", "telemetry.csv")


def load_dataset(path: str = DEFAULT_DATASET) -> Tuple[np.ndarray, np.ndarray]:
    """Load the telemetry CSV into (X, y). Pure-stdlib parsing (no pandas dep)."""
    import csv

    rows: List[List[float]] = []
    ys: List[int] = []
    with open(path, "r", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            rows.append([float(row[k]) for k in FEATURES])
            ys.append(int(row["label"]))
    return np.asarray(rows, dtype=float), np.asarray(ys, dtype=int)


def stratified_split(X: np.ndarray, y: np.ndarray, test_frac: float = 0.25,
                     seed: int = 0) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Deterministic per-class (stratified) train/test split."""
    rng = np.random.default_rng(seed)
    tr_idx: List[int] = []
    te_idx: List[int] = []
    for c in np.unique(y):
        idx = np.where(y == c)[0]
        rng.shuffle(idx)
        n_te = int(round(test_frac * len(idx)))
        te_idx.extend(idx[:n_te].tolist())
        tr_idx.extend(idx[n_te:].tolist())
    rng.shuffle(tr_idx)
    rng.shuffle(te_idx)
    tr_idx = np.asarray(tr_idx)
    te_idx = np.asarray(te_idx)
    return X[tr_idx], X[te_idx], y[tr_idx], y[te_idx]


def build_detector(name: str, fnn_epochs: int = 120, alpha: float = 0.6,
                   fusion: str = "confidence", k: int = 15):
    """Factory for the three detector variants."""
    name = name.lower()
    if name == "wknn":
        return WeightedKNN(WKNNConfig(k=k))
    if name == "fnn":
        return FNNClassifier(FNNConfig(epochs=fnn_epochs))
    if name == "hybrid":
        return HybridWKNNFNN(HybridConfig(alpha=alpha, fusion=fusion,
                                          wknn=WKNNConfig(k=k),
                                          fnn=FNNConfig(epochs=fnn_epochs)))
    raise ValueError(f"unknown detector '{name}'")
