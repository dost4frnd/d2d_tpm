"""Hyperparameter sweeps, written to ``results/`` (one CSV per sweep).

  (A) Hybrid fusion weight alpha : detector macro-F1 / AUC vs alpha in [0,1]
  (B) WKNN neighbours k          : macro-F1 vs k
  (C) TPM input size N           : sync rounds & geometric-attacker success vs N
"""
from __future__ import annotations

import argparse
import csv
import os
import sys
from typing import Dict, List, Optional

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from anomaly_detector.evaluate import compute_metrics  # noqa: E402
from anomaly_detector.hybrid import HybridConfig, HybridWKNNFNN  # noqa: E402
from anomaly_detector.fnn import FNNConfig  # noqa: E402
from anomaly_detector.wknn import WKNNConfig, WeightedKNN  # noqa: E402
from common.utils import ensure_dir, save_json  # noqa: E402
from experiments._common import load_dataset, stratified_split  # noqa: E402
from neural_crypto.attacks import AttackConfig  # noqa: E402
from neural_crypto.metrics import evaluate_security  # noqa: E402
from neural_crypto.sync import SyncConfig  # noqa: E402
from neural_crypto.tpm import TPMConfig  # noqa: E402

RESULTS_DIR = os.path.join(_ROOT, "results")


def alpha_sweep(dataset: str, fnn_epochs: int, seed: int) -> List[Dict]:
    X, y = load_dataset(dataset)
    Xtr, Xte, ytr, yte = stratified_split(X, y, 0.25, seed)
    rows = []
    for alpha in [0.0, 0.2, 0.4, 0.5, 0.6, 0.8, 1.0]:
        det = HybridWKNNFNN(HybridConfig(alpha=alpha, fusion="fixed",
                                         fnn=FNNConfig(epochs=fnn_epochs)))
        det.fit(Xtr, ytr)
        proba = det.predict_proba(Xte)
        m = compute_metrics(yte, proba.argmax(1), proba)
        rows.append({"alpha": alpha, "macro_f1": round(m["macro_f1"], 4),
                     "roc_auc": round(m.get("roc_auc_macro_ovr", float("nan")), 4)})
    return rows


def k_sweep(dataset: str, seed: int) -> List[Dict]:
    X, y = load_dataset(dataset)
    Xtr, Xte, ytr, yte = stratified_split(X, y, 0.25, seed)
    rows = []
    for k in [3, 5, 9, 15, 25, 41]:
        det = WeightedKNN(WKNNConfig(k=k))
        det.fit(Xtr, ytr)
        proba = det.predict_proba(Xte)
        m = compute_metrics(yte, proba.argmax(1), proba)
        rows.append({"k": k, "macro_f1": round(m["macro_f1"], 4),
                     "roc_auc": round(m.get("roc_auc_macro_ovr", float("nan")), 4)})
    return rows


def tpm_N_sweep(n_sessions: int, seed: int, Ns: List[int]) -> List[Dict]:
    rows = []
    for N in Ns:
        cfg = SyncConfig(tpm=TPMConfig(K=3, N=N, L=4), policy="early_confidence")
        none = evaluate_security(cfg, AttackConfig(kind="none"),
                                 n_sessions=n_sessions, base_seed=seed)
        geo = evaluate_security(cfg, AttackConfig(kind="geometric"),
                                n_sessions=n_sessions, base_seed=seed)
        rows.append({"N": N, "mean_rounds": round(none.mean_rounds, 1),
                     "key_agreement_rate": none.key_agreement_rate,
                     "attacker_success_geom": geo.attacker_success_rate})
    return rows


def threshold_sweep(loss: float, n_pdus: int) -> List[Dict]:
    """(T,N) confidentiality/availability frontier at fixed per-PDU loss.

    For N=5 we sweep T=1..5: confidentiality threshold = T (carriers an adversary
    must capture), availability = binomial tail. Recovers single/(2,2)/dup.
    """
    from bifurcation import reliability
    rows = []
    N = 5
    for T in range(1, N + 1):
        rows.append({
            "T": T, "N": N,
            "confidentiality_threshold": T,          # adversary needs >=T carriers
            "availability_margin": N - T,            # can lose up to N-T shares
            "availability": round(reliability.availability_t_of_n(loss, n_pdus, T, N), 4),
        })
    # also tabulate the canonical points for reference
    return rows


def _write(rows: List[Dict], path: str) -> None:
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


def main(argv: Optional[List[str]] = None) -> None:
    p = argparse.ArgumentParser(description="Hyperparameter sweeps.")
    p.add_argument("--dataset", type=str, default=os.path.join(_ROOT, "datasets", "telemetry.csv"))
    p.add_argument("--fnn-epochs", type=int, default=120)
    p.add_argument("--n-sessions", type=int, default=25)
    p.add_argument("--N-values", type=int, nargs="+", default=[40, 60, 80, 100])
    p.add_argument("--seed", type=int, default=0)
    a = p.parse_args(argv)

    ensure_dir(RESULTS_DIR)
    print("[sweep] hybrid alpha ...")
    al = alpha_sweep(a.dataset, a.fnn_epochs, a.seed)
    print("[sweep] WKNN k ...")
    ks = k_sweep(a.dataset, a.seed)
    print("[sweep] TPM N ...")
    Ns = tpm_N_sweep(a.n_sessions, a.seed, a.N_values)
    print("[sweep] (T,N) threshold frontier ...")
    tn = threshold_sweep(loss=0.05, n_pdus=8)

    _write(al, os.path.join(RESULTS_DIR, "sweep_alpha.csv"))
    _write(ks, os.path.join(RESULTS_DIR, "sweep_k.csv"))
    _write(Ns, os.path.join(RESULTS_DIR, "sweep_tpm_N.csv"))
    _write(tn, os.path.join(RESULTS_DIR, "sweep_threshold.csv"))
    save_json({"alpha": al, "k": ks, "tpm_N": Ns, "threshold_TN": tn},
              os.path.join(RESULTS_DIR, "sweep.json"))

    print("\n[sweep] alpha (0=KNN .. 1=FNN):")
    for r in al:
        print(f"  alpha={r['alpha']:.1f} macroF1={r['macro_f1']} auc={r['roc_auc']}")
    print("[sweep] TPM N:")
    for r in Ns:
        print(f"  N={r['N']:>3} rounds={r['mean_rounds']:>7} "
              f"key_ok={r['key_agreement_rate']:.2f} att_geo={r['attacker_success_geom']:.2f}")
    print("[sweep] (T,N) frontier (N=5, p=0.05, 8 PDUs/share):")
    for r in tn:
        print(f"  (T={r['T']},N={r['N']}) conf_thr={r['confidentiality_threshold']} "
              f"avail_margin={r['availability_margin']} availability={r['availability']}")
    print("\n[sweep] wrote results/sweep_*.csv and results/sweep.json")


if __name__ == "__main__":
    main()
