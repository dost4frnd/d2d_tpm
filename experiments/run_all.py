"""Run the five-configuration comparison and write results to ``results/``.

Configurations (cumulative ablation of the framework):
  C1  TPM baseline            : fixed-interval hash verification only
  C2  + Early Termination     : privacy-preserving confidence-gated stop
  C3  + Bifurcation           : (2,2) split-bearer transport
  C4  + Anomaly Detection      : hybrid WKNN-FNN telemetry classifier
  C5  Complete Framework       : early termination + bifurcation + detection

Each configuration reports the metrics that are *meaningful* for the components
it enables (crypto cost & eavesdropper success; transport availability/latency;
detector macro-F1/AUC). Cells that do not apply are recorded as ``None``.
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
from bifurcation import reliability  # noqa: E402
from common.utils import ensure_dir, save_json  # noqa: E402
from experiments._common import build_detector, load_dataset, stratified_split  # noqa: E402
from neural_crypto.attacks import AttackConfig  # noqa: E402
from neural_crypto.metrics import evaluate_security  # noqa: E402
from neural_crypto.sync import SyncConfig  # noqa: E402
from neural_crypto.tpm import TPMConfig  # noqa: E402

RESULTS_DIR = os.path.join(_ROOT, "results")


def crypto_block(policy: str, n_sessions: int, N: int, base_seed: int) -> Dict[str, Dict]:
    """Security summaries under no/geometric/majority attack for one policy."""
    tpm = TPMConfig(K=3, N=N, L=4)
    cfg = SyncConfig(tpm=tpm, policy=policy)
    out: Dict[str, Dict] = {}
    for kind, kw in [("none", {}), ("geometric", {}), ("majority", {"n_nets": 100})]:
        summ = evaluate_security(cfg, AttackConfig(kind=kind, **kw),
                                 n_sessions=n_sessions, base_seed=base_seed)
        out[kind] = summ.as_dict()
    return out


def bifurcation_block(loss: float, n_pdus: int, mc_trials: int) -> Dict[str, float]:
    from bifurcation.carriers import CarrierModel

    av_single = reliability.availability_single(loss, n_pdus)
    av_22 = reliability.availability_22(loss, n_pdus)
    av_dup = reliability.availability_dup(loss, n_pdus)
    # Build two carriers at the requested per-PDU loss so the Monte-Carlo
    # estimate is consistent with the analytic curve.
    carriers = [
        CarrierModel("CC0", base_latency_ms=7.0, jitter_ms=1.5,
                     loss_prob=loss, throughput_mbps=150.0),
        CarrierModel("CC1", base_latency_ms=10.0, jitter_ms=3.0,
                     loss_prob=loss, throughput_mbps=220.0),
    ]
    mc = reliability.monte_carlo_availability(payload_size=n_pdus * 1024,
                                              trials=mc_trials, carriers=carriers)
    return {
        "loss_prob": loss,
        "n_pdus": n_pdus,
        "availability_single": av_single,
        "availability_2of2": av_22,
        "availability_duplicated": av_dup,
        "mc_availability_2of2": mc["availability"],
        "mc_mean_latency_ms": mc["mean_latency_ms"],
        "mc_p95_latency_ms": mc["p95_latency_ms"],
    }


def detection_block(dataset: str, fnn_epochs: int, seed: int) -> Dict[str, float]:
    X, y = load_dataset(dataset)
    Xtr, Xte, ytr, yte = stratified_split(X, y, test_frac=0.25, seed=seed)
    det = build_detector("hybrid", fnn_epochs=fnn_epochs)
    det.fit(Xtr, ytr)
    proba = det.predict_proba(Xte)
    pred = proba.argmax(1)
    m = compute_metrics(yte, pred, proba)
    return {
        "macro_f1": m["macro_f1"],
        "accuracy": m["accuracy"],
        "roc_auc_macro_ovr": m.get("roc_auc_macro_ovr", float("nan")),
    }


def assemble_configs(baseline: Dict, early: Dict, bif: Dict,
                     det: Dict) -> List[Dict]:
    """Build the cumulative five-row comparison table."""
    def crypto_cells(block):
        return {
            "mean_rounds": round(block["none"]["mean_rounds"], 1),
            "mean_verifications": round(block["none"]["mean_verifications"], 2),
            "attacker_success_geometric": block["geometric"]["attacker_success_rate"],
            "attacker_success_majority": block["majority"]["attacker_success_rate"],
        }

    base_c = crypto_cells(baseline)
    early_c = crypto_cells(early)
    av = round(bif["availability_2of2"], 3)
    p95 = round(bif["mc_p95_latency_ms"], 2)
    f1 = round(det["macro_f1"], 3)
    auc = round(det["roc_auc_macro_ovr"], 3)

    rows = [
        {"config": "C1_TPM_baseline", "early_termination": False, "bifurcation": False,
         "anomaly_detection": False, **base_c,
         "availability_2of2": None, "p95_latency_ms": None,
         "detector_macro_f1": None, "detector_auc": None},
        {"config": "C2_plus_early_termination", "early_termination": True, "bifurcation": False,
         "anomaly_detection": False, **early_c,
         "availability_2of2": None, "p95_latency_ms": None,
         "detector_macro_f1": None, "detector_auc": None},
        {"config": "C3_plus_bifurcation", "early_termination": True, "bifurcation": True,
         "anomaly_detection": False, **early_c,
         "availability_2of2": av, "p95_latency_ms": p95,
         "detector_macro_f1": None, "detector_auc": None},
        {"config": "C4_plus_anomaly_detection", "early_termination": True, "bifurcation": False,
         "anomaly_detection": True, **early_c,
         "availability_2of2": None, "p95_latency_ms": None,
         "detector_macro_f1": f1, "detector_auc": auc},
        {"config": "C5_complete_framework", "early_termination": True, "bifurcation": True,
         "anomaly_detection": True, **early_c,
         "availability_2of2": av, "p95_latency_ms": p95,
         "detector_macro_f1": f1, "detector_auc": auc},
    ]
    return rows


def write_table(rows: List[Dict], path: str) -> None:
    fields = list(rows[0].keys())
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow(r)


def main(argv: Optional[List[str]] = None) -> None:
    p = argparse.ArgumentParser(description="Five-configuration comparison.")
    p.add_argument("--n-sessions", type=int, default=40)
    p.add_argument("--tpm-N", type=int, default=100)
    p.add_argument("--loss", type=float, default=0.05)
    p.add_argument("--n-pdus", type=int, default=8)
    p.add_argument("--mc-trials", type=int, default=3000)
    p.add_argument("--fnn-epochs", type=int, default=120)
    p.add_argument("--dataset", type=str, default=os.path.join(_ROOT, "datasets", "telemetry.csv"))
    p.add_argument("--seed", type=int, default=0)
    a = p.parse_args(argv)

    ensure_dir(RESULTS_DIR)
    print("[run_all] crypto baseline (hash_only) ...")
    baseline = crypto_block("hash_only", a.n_sessions, a.tpm_N, a.seed)
    print("[run_all] crypto early-termination (early_confidence) ...")
    early = crypto_block("early_confidence", a.n_sessions, a.tpm_N, a.seed)
    print("[run_all] bifurcation reliability ...")
    bif = bifurcation_block(a.loss, a.n_pdus, a.mc_trials)
    print("[run_all] anomaly detection ...")
    det = detection_block(a.dataset, a.fnn_epochs, a.seed)

    rows = assemble_configs(baseline, early, bif, det)
    write_table(rows, os.path.join(RESULTS_DIR, "configurations.csv"))
    save_json({
        "crypto_baseline_hash_only": baseline,
        "crypto_early_confidence": early,
        "bifurcation": bif,
        "detection_hybrid": det,
        "configurations": rows,
    }, os.path.join(RESULTS_DIR, "run_all.json"))

    print("\n[run_all] === Configuration comparison ===")
    for r in rows:
        print(f"  {r['config']:28s} rounds={r['mean_rounds']:>7} "
              f"verif={r['mean_verifications']:>5} "
              f"att_geo={r['attacker_success_geometric']:.2f} "
              f"att_maj={r['attacker_success_majority']:.2f} "
              f"avail={r['availability_2of2']} f1={r['detector_macro_f1']}")
    print(f"\n[run_all] wrote results/configurations.csv and results/run_all.json")


if __name__ == "__main__":
    main()
