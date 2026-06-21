"""Ablation studies, written to ``results/``.

  (A) Detector architecture : WKNN vs FNN vs Hybrid (macro-F1, AUC)
  (B) Feature group         : crypto-only vs transport-only vs all
  (C) Early-termination      : confidence threshold / streak vs cost & security
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
from anomaly_detector.features import FEATURES  # noqa: E402
from common.utils import ensure_dir, save_json  # noqa: E402
from experiments._common import build_detector, load_dataset, stratified_split  # noqa: E402
from neural_crypto.attacks import AttackConfig  # noqa: E402
from neural_crypto.metrics import evaluate_security  # noqa: E402
from neural_crypto.sync import SyncConfig  # noqa: E402
from neural_crypto.tpm import TPMConfig  # noqa: E402

RESULTS_DIR = os.path.join(_ROOT, "results")

# Feature index groups (see anomaly_detector/features.py for the ordering).
CRYPTO_IDX = [FEATURES.index(f) for f in
              ["tpm_convergence", "sync_rate", "match_probability", "round_count"]]
TRANSPORT_IDX = [FEATURES.index(f) for f in
                 ["carrier_latency", "packet_loss", "throughput", "jitter",
                  "share_reconstruction_failures"]]


def architecture_ablation(dataset: str, fnn_epochs: int, seed: int) -> List[Dict]:
    X, y = load_dataset(dataset)
    Xtr, Xte, ytr, yte = stratified_split(X, y, 0.25, seed)
    rows = []
    for name in ["wknn", "fnn", "hybrid"]:
        det = build_detector(name, fnn_epochs=fnn_epochs)
        det.fit(Xtr, ytr)
        proba = det.predict_proba(Xte)
        m = compute_metrics(yte, proba.argmax(1), proba)
        rows.append({"detector": name, "macro_f1": round(m["macro_f1"], 4),
                     "accuracy": round(m["accuracy"], 4),
                     "roc_auc": round(m.get("roc_auc_macro_ovr", float("nan")), 4)})
    return rows


def feature_group_ablation(dataset: str, fnn_epochs: int, seed: int) -> List[Dict]:
    X, y = load_dataset(dataset)
    Xtr, Xte, ytr, yte = stratified_split(X, y, 0.25, seed)
    groups = {"crypto_only": CRYPTO_IDX, "transport_only": TRANSPORT_IDX,
              "all_features": list(range(len(FEATURES)))}
    rows = []
    for gname, idx in groups.items():
        # Zero out non-selected columns so the input dimensionality is constant
        # (keeps the fixed-architecture FNN/Hybrid comparable across groups).
        mask = np.zeros(X.shape[1]); mask[idx] = 1.0
        det = build_detector("hybrid", fnn_epochs=fnn_epochs)
        det.fit(Xtr * mask, ytr)
        proba = det.predict_proba(Xte * mask)
        m = compute_metrics(yte, proba.argmax(1), proba)
        rows.append({"feature_group": gname, "n_features": len(idx),
                     "macro_f1": round(m["macro_f1"], 4),
                     "roc_auc": round(m.get("roc_auc_macro_ovr", float("nan")), 4)})
    return rows


def early_termination_ablation(n_sessions: int, N: int, seed: int) -> List[Dict]:
    tpm = TPMConfig(K=3, N=N, L=4)
    rows = []
    # Baseline: fixed-interval hash verification.
    base = evaluate_security(SyncConfig(tpm=tpm, policy="hash_only"),
                             AttackConfig(kind="geometric"),
                             n_sessions=n_sessions, base_seed=seed)
    rows.append({"policy": "hash_only", "conf_threshold": None, "min_streak": None,
                 "mean_rounds": round(base.mean_rounds, 1),
                 "mean_verifications": round(base.mean_verifications, 2),
                 "attacker_success_geom": base.attacker_success_rate})
    # Early-confidence at several operating points.
    for thr, streak in [(0.95, 15), (0.98, 25), (0.99, 35)]:
        cfg = SyncConfig(tpm=tpm, policy="early_confidence",
                         conf_threshold=thr, min_streak=streak)
        s = evaluate_security(cfg, AttackConfig(kind="geometric"),
                              n_sessions=n_sessions, base_seed=seed)
        rows.append({"policy": "early_confidence", "conf_threshold": thr,
                     "min_streak": streak, "mean_rounds": round(s.mean_rounds, 1),
                     "mean_verifications": round(s.mean_verifications, 2),
                     "attacker_success_geom": s.attacker_success_rate})
    return rows


def _write(rows: List[Dict], path: str) -> None:
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


def main(argv: Optional[List[str]] = None) -> None:
    p = argparse.ArgumentParser(description="Ablation studies.")
    p.add_argument("--dataset", type=str, default=os.path.join(_ROOT, "datasets", "telemetry.csv"))
    p.add_argument("--fnn-epochs", type=int, default=120)
    p.add_argument("--n-sessions", type=int, default=30)
    p.add_argument("--tpm-N", type=int, default=100)
    p.add_argument("--seed", type=int, default=0)
    a = p.parse_args(argv)

    ensure_dir(RESULTS_DIR)
    print("[ablation] architecture ...")
    arch = architecture_ablation(a.dataset, a.fnn_epochs, a.seed)
    print("[ablation] feature groups ...")
    feat = feature_group_ablation(a.dataset, a.fnn_epochs, a.seed)
    print("[ablation] early termination ...")
    early = early_termination_ablation(a.n_sessions, a.tpm_N, a.seed)

    _write(arch, os.path.join(RESULTS_DIR, "ablation_architecture.csv"))
    _write(feat, os.path.join(RESULTS_DIR, "ablation_feature_groups.csv"))
    _write(early, os.path.join(RESULTS_DIR, "ablation_early_termination.csv"))
    save_json({"architecture": arch, "feature_groups": feat,
               "early_termination": early},
              os.path.join(RESULTS_DIR, "ablation.json"))

    print("\n[ablation] architecture:")
    for r in arch:
        print(f"  {r['detector']:7s} macroF1={r['macro_f1']} auc={r['roc_auc']}")
    print("[ablation] feature groups:")
    for r in feat:
        print(f"  {r['feature_group']:15s} macroF1={r['macro_f1']} auc={r['roc_auc']}")
    print("[ablation] early termination (vs geometric attacker):")
    for r in early:
        print(f"  {r['policy']:16s} thr={r['conf_threshold']} streak={r['min_streak']} "
              f"rounds={r['mean_rounds']} verif={r['mean_verifications']} "
              f"att={r['attacker_success_geom']:.2f}")
    print("\n[ablation] wrote results/ablation_*.csv and results/ablation.json")


if __name__ == "__main__":
    main()
