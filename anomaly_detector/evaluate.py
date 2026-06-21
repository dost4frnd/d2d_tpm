"""Evaluation metrics and publication-quality figures for the detector."""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

import matplotlib
matplotlib.use("Agg")  # headless / reproducible
import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score,
    auc,
    confusion_matrix,
    precision_recall_curve,
    precision_recall_fscore_support,
    roc_auc_score,
    roc_curve,
)
from sklearn.preprocessing import label_binarize

from .features import CLASSES, N_CLASSES

# Consistent, colour-blind-friendly styling for the paper.
plt.rcParams.update({
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "font.size": 10,
    "axes.grid": True,
    "grid.alpha": 0.3,
})


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray,
                    proba: Optional[np.ndarray] = None) -> Dict:
    p, r, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=list(range(N_CLASSES)), zero_division=0
    )
    metrics: Dict = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_precision": float(np.mean(p)),
        "macro_recall": float(np.mean(r)),
        "macro_f1": float(np.mean(f1)),
        "per_class": {
            CLASSES[i]: {"precision": float(p[i]), "recall": float(r[i]),
                          "f1": float(f1[i]), "support": int(support[i])}
            for i in range(N_CLASSES)
        },
    }
    if proba is not None:
        y_bin = label_binarize(y_true, classes=list(range(N_CLASSES)))
        present = y_bin.sum(axis=0) > 0
        try:
            metrics["roc_auc_macro_ovr"] = float(
                roc_auc_score(y_bin[:, present], proba[:, present],
                              average="macro", multi_class="ovr")
            )
        except ValueError:
            metrics["roc_auc_macro_ovr"] = float("nan")
    return metrics


def plot_confusion_matrix(y_true, y_pred, path: str, normalize: bool = True) -> str:
    cm = confusion_matrix(y_true, y_pred, labels=list(range(N_CLASSES))).astype(float)
    if normalize:
        cm = cm / (cm.sum(axis=1, keepdims=True) + 1e-12)
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm, cmap="Blues", vmin=0, vmax=1 if normalize else None)
    ax.set_xticks(range(N_CLASSES)); ax.set_yticks(range(N_CLASSES))
    ax.set_xticklabels(CLASSES, rotation=45, ha="right"); ax.set_yticklabels(CLASSES)
    ax.set_xlabel("Predicted"); ax.set_ylabel("True")
    ax.set_title("Confusion Matrix" + (" (normalised)" if normalize else ""))
    for i in range(N_CLASSES):
        for j in range(N_CLASSES):
            ax.text(j, i, f"{cm[i, j]:.2f}", ha="center", va="center",
                    color="white" if cm[i, j] > 0.5 else "black", fontsize=8)
    fig.colorbar(im, fraction=0.046, pad=0.04)
    fig.tight_layout(); fig.savefig(path, bbox_inches="tight"); plt.close(fig)
    return path


def plot_roc(y_true, proba, path: str) -> str:
    y_bin = label_binarize(y_true, classes=list(range(N_CLASSES)))
    fig, ax = plt.subplots(figsize=(6, 5))
    for i in range(N_CLASSES):
        if y_bin[:, i].sum() == 0:
            continue
        fpr, tpr, _ = roc_curve(y_bin[:, i], proba[:, i])
        ax.plot(fpr, tpr, lw=1.5, label=f"{CLASSES[i]} (AUC={auc(fpr, tpr):.3f})")
    ax.plot([0, 1], [0, 1], "k--", lw=0.8)
    ax.set_xlabel("False Positive Rate"); ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC (one-vs-rest)"); ax.legend(fontsize=7, loc="lower right")
    fig.tight_layout(); fig.savefig(path, bbox_inches="tight"); plt.close(fig)
    return path


def plot_pr(y_true, proba, path: str) -> str:
    y_bin = label_binarize(y_true, classes=list(range(N_CLASSES)))
    fig, ax = plt.subplots(figsize=(6, 5))
    for i in range(N_CLASSES):
        if y_bin[:, i].sum() == 0:
            continue
        prec, rec, _ = precision_recall_curve(y_bin[:, i], proba[:, i])
        ax.plot(rec, prec, lw=1.5, label=f"{CLASSES[i]} (AP={auc(rec, prec):.3f})")
    ax.set_xlabel("Recall"); ax.set_ylabel("Precision")
    ax.set_title("Precision-Recall (one-vs-rest)"); ax.legend(fontsize=7, loc="lower left")
    fig.tight_layout(); fig.savefig(path, bbox_inches="tight"); plt.close(fig)
    return path


def plot_feature_importance(importances: Dict[str, float], path: str) -> str:
    items = sorted(importances.items(), key=lambda kv: kv[1])
    names = [k for k, _ in items]; vals = [v for _, v in items]
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.barh(names, vals, color="#3b75af")
    ax.set_xlabel("Permutation importance (macro-F1 drop)")
    ax.set_title("Feature Importance")
    fig.tight_layout(); fig.savefig(path, bbox_inches="tight"); plt.close(fig)
    return path
