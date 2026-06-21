"""Generate all publication-quality figures.

Outputs are written to BOTH ``figures/`` (repo browsing) and ``paper/figures/``
(LaTeX ``\\includegraphics``). Run after generating the dataset; sweep panels
are read from ``results/`` when present, else computed at a small size.
"""
from __future__ import annotations

import argparse
import csv
import os
import sys
from typing import Dict, List, Optional

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from anomaly_detector.evaluate import (  # noqa: E402
    plot_confusion_matrix, plot_feature_importance, plot_pr, plot_roc,
)
from anomaly_detector.explain import permutation_importance  # noqa: E402
from bifurcation import reliability  # noqa: E402
from experiments._common import build_detector, load_dataset, stratified_split  # noqa: E402
from neural_crypto.attacks import AttackConfig  # noqa: E402
from neural_crypto.sync import SyncConfig, run_session  # noqa: E402
from neural_crypto.tpm import TPMConfig  # noqa: E402

FIG_DIRS = [os.path.join(_ROOT, "figures"), os.path.join(_ROOT, "paper", "figures")]

plt.rcParams.update({"figure.dpi": 150, "savefig.dpi": 300, "font.size": 10,
                     "axes.grid": True, "grid.alpha": 0.3})


def _dirs() -> None:
    for d in FIG_DIRS:
        os.makedirs(d, exist_ok=True)


def _save_both(fig, name: str) -> None:
    for d in FIG_DIRS:
        fig.savefig(os.path.join(d, name), bbox_inches="tight")
    plt.close(fig)


def detector_figures(dataset: str, fnn_epochs: int, seed: int) -> None:
    X, y = load_dataset(dataset)
    Xtr, Xte, ytr, yte = stratified_split(X, y, 0.25, seed)
    det = build_detector("hybrid", fnn_epochs=fnn_epochs)
    det.fit(Xtr, ytr)
    proba = det.predict_proba(Xte)
    pred = proba.argmax(1)

    for d in FIG_DIRS:
        plot_confusion_matrix(yte, pred, os.path.join(d, "confusion_matrix.png"))
        plot_roc(yte, proba, os.path.join(d, "roc_curves.png"))
        plot_pr(yte, proba, os.path.join(d, "pr_curves.png"))

    imp = permutation_importance(det, Xte, yte, n_repeats=8, seed=seed)
    for d in FIG_DIRS:
        plot_feature_importance(imp, os.path.join(d, "feature_importance.png"))


def reliability_figure() -> None:
    curve = reliability.reliability_curve(n_pdus=8)
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(curve.loss_probs, curve.single, "-o", ms=3, label="Single carrier")
    ax.plot(curve.loss_probs, curve.bifurcation_22, "-s", ms=3, label="(2,2) split bearer")
    ax.plot(curve.loss_probs, curve.duplicated, "-^", ms=3, label="Duplicated (2 of 2)")
    ax.set_xlabel("Per-PDU loss probability"); ax.set_ylabel("Reconstruction availability")
    ax.set_title("Availability vs loss (8 PDUs/share)")
    ax.legend(fontsize=8)
    _save_both(fig, "reliability_curve.png")


def threshold_frontier_figure() -> None:
    """(T,N) confidentiality/availability frontier vs per-PDU loss, N=5."""
    import numpy as _np
    ps = _np.linspace(0.0, 0.2, 21)
    N = 5
    fig, ax = plt.subplots(figsize=(6, 4))
    for T in range(1, N + 1):
        av = [reliability.availability_t_of_n(p, 8, T, N) for p in ps]
        ax.plot(ps, av, "-o", ms=2.5,
                label=f"({T},{N}) conf-thr={T}")
    ax.set_xlabel("Per-PDU loss probability")
    ax.set_ylabel("Reconstruction availability")
    ax.set_title("Tunable (T,N) frontier: confidentiality vs availability (N=5)")
    ax.legend(fontsize=7, title="scheme (higher T = more confidential)")
    _save_both(fig, "threshold_frontier.png")


def sync_trace_figure(N: int, seed: int) -> None:
    cfg = SyncConfig(tpm=TPMConfig(K=3, N=N, L=4), policy="early_confidence",
                     seed_a=seed + 1, seed_b=seed + 2, seed_e=seed + 3, seed_input=seed + 100)
    r = run_session(cfg, AttackConfig(kind="geometric"), record_traces=True)
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(r.overlap_trace, label="Partner A-B overlap", lw=1.5)
    ax.plot(r.attacker_overlap_trace, label="Eavesdropper overlap", lw=1.2, alpha=0.85)
    if r.stop_step:
        ax.axvline(r.stop_step, color="k", ls="--", lw=0.8, label="Termination")
    ax.set_xlabel("Round"); ax.set_ylabel("Weight overlap")
    ax.set_title("Synchronisation: parties vs geometric eavesdropper")
    ax.legend(fontsize=8, loc="lower right")
    _save_both(fig, "sync_trace.png")


def _read_csv(path: str) -> List[Dict]:
    with open(path, "r", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def sweep_figures(dataset: str, fnn_epochs: int, seed: int) -> None:
    # alpha sweep
    apath = os.path.join(_ROOT, "results", "sweep_alpha.csv")
    if os.path.exists(apath):
        rows = _read_csv(apath)
        alphas = [float(r["alpha"]) for r in rows]
        f1 = [float(r["macro_f1"]) for r in rows]
        auc = [float(r["roc_auc"]) for r in rows]
    else:
        from experiments.sweep import alpha_sweep
        rows = alpha_sweep(dataset, fnn_epochs, seed)
        alphas = [r["alpha"] for r in rows]; f1 = [r["macro_f1"] for r in rows]
        auc = [r["roc_auc"] for r in rows]
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(alphas, f1, "-o", ms=4, label="Macro-F1")
    ax.plot(alphas, auc, "-s", ms=4, label="ROC-AUC (macro-OvR)")
    ax.set_xlabel(r"Fusion weight $\alpha$ (0=WKNN, 1=FNN)"); ax.set_ylabel("Score")
    ax.set_title("Hybrid fusion sweep"); ax.legend(fontsize=8)
    _save_both(fig, "sweep_alpha.png")

    # TPM N sweep
    npath = os.path.join(_ROOT, "results", "sweep_tpm_N.csv")
    if os.path.exists(npath):
        rows = _read_csv(npath)
        Ns = [int(r["N"]) for r in rows]
        att = [float(r["attacker_success_geom"]) for r in rows]
        rounds = [float(r["mean_rounds"]) for r in rows]
        fig, ax1 = plt.subplots(figsize=(6, 4))
        ax2 = ax1.twinx()
        ax1.plot(Ns, att, "-o", color="#c0392b", label="Geom. attacker success")
        ax2.plot(Ns, rounds, "-s", color="#2c3e50", label="Mean sync rounds")
        ax1.set_xlabel("TPM input size N"); ax1.set_ylabel("Attacker success", color="#c0392b")
        ax2.set_ylabel("Mean sync rounds", color="#2c3e50")
        ax1.set_title("Security/cost vs TPM size")
        _save_both(fig, "sweep_tpm_N.png")


def main(argv: Optional[List[str]] = None) -> None:
    p = argparse.ArgumentParser(description="Generate all figures.")
    p.add_argument("--dataset", type=str, default=os.path.join(_ROOT, "datasets", "telemetry.csv"))
    p.add_argument("--fnn-epochs", type=int, default=120)
    p.add_argument("--tpm-N", type=int, default=100)
    p.add_argument("--seed", type=int, default=0)
    a = p.parse_args(argv)

    _dirs()
    print("[figures] detector (confusion/ROC/PR/importance) ...")
    detector_figures(a.dataset, a.fnn_epochs, a.seed)
    print("[figures] reliability curve ...")
    reliability_figure()
    print("[figures] (T,N) threshold frontier ...")
    threshold_frontier_figure()
    print("[figures] synchronisation trace ...")
    sync_trace_figure(a.tpm_N, a.seed)
    print("[figures] sweep panels ...")
    sweep_figures(a.dataset, a.fnn_epochs, a.seed)
    names = sorted({f for f in os.listdir(FIG_DIRS[1]) if f.endswith(".png")})
    print(f"[figures] wrote {len(names)} figures to figures/ and paper/figures/: {names}")


if __name__ == "__main__":
    main()
