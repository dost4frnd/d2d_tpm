"""Baseline comparison, repeated-run confidence intervals, and significance.

Compares the hybrid WKNN-FNN detector against standard baselines (random forest,
RBF-SVM, gradient boosting, logistic regression) and the two base learners, on
the same stratified splits. Reports mean +/- std macro-F1 / accuracy / ROC-AUC
over several seeds, and a McNemar test between the hybrid and the strongest
baseline to check whether the difference is statistically meaningful.

This is the ML-methodology backbone an ML-track reviewer expects. By default it
runs on the grounded synthetic telemetry; pass --dataset to point it at a
5G-NIDD-derived CSV (see data_generator/load_5g_nidd.py) with the same columns.
"""
from __future__ import annotations

import argparse
import csv
import os
import sys
from typing import Dict, List, Optional, Tuple

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.pipeline import make_pipeline  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402
from sklearn.svm import SVC  # noqa: E402

from anomaly_detector.evaluate import compute_metrics  # noqa: E402
from common.utils import save_json  # noqa: E402
from experiments._common import build_detector, load_dataset, stratified_split  # noqa: E402

RESULTS_DIR = os.path.join(_ROOT, "results")


def _sklearn_factory(name: str):
    if name == "random_forest":
        return RandomForestClassifier(n_estimators=300, random_state=0, n_jobs=-1)
    if name == "svm_rbf":
        return make_pipeline(StandardScaler(),
                             SVC(kernel="rbf", probability=True, random_state=0))
    if name == "grad_boosting":
        return GradientBoostingClassifier(random_state=0)
    if name == "logistic":
        return make_pipeline(StandardScaler(),
                             LogisticRegression(max_iter=2000))
    raise ValueError(name)


SKLEARN_MODELS = ["random_forest", "svm_rbf", "grad_boosting", "logistic"]
OUR_MODELS = ["wknn", "fnn", "hybrid"]


def _fit_predict(name: str, Xtr, ytr, Xte, fnn_epochs: int) -> np.ndarray:
    if name in OUR_MODELS:
        det = build_detector(name, fnn_epochs=fnn_epochs)
        det.fit(Xtr, ytr)
        return det.predict_proba(Xte)
    model = _sklearn_factory(name)
    model.fit(Xtr, ytr)
    return model.predict_proba(Xte)


def repeated_eval(dataset: str, seeds: List[int], fnn_epochs: int) -> Dict[str, Dict]:
    X, y = load_dataset(dataset)
    agg: Dict[str, Dict[str, List[float]]] = {
        m: {"macro_f1": [], "accuracy": [], "roc_auc": []}
        for m in SKLEARN_MODELS + OUR_MODELS}
    for s in seeds:
        Xtr, Xte, ytr, yte = stratified_split(X, y, 0.25, s)
        for m in SKLEARN_MODELS + OUR_MODELS:
            proba = _fit_predict(m, Xtr, ytr, Xte, fnn_epochs)
            met = compute_metrics(yte, proba.argmax(1), proba)
            agg[m]["macro_f1"].append(met["macro_f1"])
            agg[m]["accuracy"].append(met["accuracy"])
            agg[m]["roc_auc"].append(met.get("roc_auc_macro_ovr", float("nan")))
    out: Dict[str, Dict] = {}
    for m, d in agg.items():
        out[m] = {k: {"mean": round(float(np.mean(v)), 4),
                      "std": round(float(np.std(v)), 4)} for k, v in d.items()}
    return out


def mcnemar_hybrid_vs_best(dataset: str, seed: int, fnn_epochs: int
                           ) -> Dict:
    """McNemar test: hybrid vs the strongest sklearn baseline on one split."""
    X, y = load_dataset(dataset)
    Xtr, Xte, ytr, yte = stratified_split(X, y, 0.25, seed)
    # pick best baseline by macro-F1 on this split
    best, best_f1, best_pred = None, -1.0, None
    for m in SKLEARN_MODELS:
        pred = _fit_predict(m, Xtr, ytr, Xte, fnn_epochs).argmax(1)
        f1 = compute_metrics(yte, pred)["macro_f1"]
        if f1 > best_f1:
            best, best_f1, best_pred = m, f1, pred
    hyb_pred = _fit_predict("hybrid", Xtr, ytr, Xte, fnn_epochs).argmax(1)
    hyb_correct = (hyb_pred == yte)
    base_correct = (best_pred == yte)
    # discordant pairs
    b = int(np.sum(hyb_correct & ~base_correct))   # hybrid right, baseline wrong
    c = int(np.sum(~hyb_correct & base_correct))   # baseline right, hybrid wrong
    # McNemar statistic with continuity correction
    stat = ((abs(b - c) - 1) ** 2) / (b + c) if (b + c) > 0 else 0.0
    from math import erfc, sqrt
    p_value = erfc(sqrt(stat / 2.0)) if stat > 0 else 1.0  # chi2_1 survival
    return {"best_baseline": best, "best_baseline_macro_f1": round(best_f1, 4),
            "hybrid_macro_f1": round(compute_metrics(yte, hyb_pred)["macro_f1"], 4),
            "b_hybrid_only_correct": b, "c_baseline_only_correct": c,
            "mcnemar_stat": round(stat, 4), "p_value": round(p_value, 4)}


def _write(rows: List[Dict], path: str) -> None:
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


def main(argv: Optional[List[str]] = None) -> None:
    p = argparse.ArgumentParser(description="Baselines + significance.")
    p.add_argument("--dataset", type=str, default=os.path.join(_ROOT, "datasets", "telemetry.csv"))
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    p.add_argument("--fnn-epochs", type=int, default=120)
    a = p.parse_args(argv)

    print(f"[baselines] repeated evaluation over {len(a.seeds)} seeds ...")
    rep = repeated_eval(a.dataset, a.seeds, a.fnn_epochs)
    print("[baselines] McNemar hybrid vs best baseline ...")
    mc = mcnemar_hybrid_vs_best(a.dataset, a.seeds[0], a.fnn_epochs)

    # flat CSV for the paper
    rows = []
    pretty = {"random_forest": "Random Forest", "svm_rbf": "SVM (RBF)",
              "grad_boosting": "Grad. Boosting", "logistic": "Logistic Reg.",
              "wknn": "WKNN", "fnn": "FNN", "hybrid": "Hybrid (ours)"}
    for m in SKLEARN_MODELS + OUR_MODELS:
        rows.append({"model": pretty[m],
                     "macro_f1_mean": rep[m]["macro_f1"]["mean"],
                     "macro_f1_std": rep[m]["macro_f1"]["std"],
                     "accuracy_mean": rep[m]["accuracy"]["mean"],
                     "roc_auc_mean": rep[m]["roc_auc"]["mean"]})
    _write(rows, os.path.join(RESULTS_DIR, "baselines.csv"))
    save_json({"repeated": rep, "mcnemar": mc, "seeds": a.seeds},
              os.path.join(RESULTS_DIR, "baselines.json"))

    print("\n[baselines] === Macro-F1 (mean +/- std over seeds) ===")
    for r in rows:
        print(f"  {r['model']:16s} F1={r['macro_f1_mean']:.3f}+/-{r['macro_f1_std']:.3f} "
              f"AUC={r['roc_auc_mean']:.3f}")
    print(f"\n[baselines] McNemar: hybrid vs {mc['best_baseline']} -> "
          f"stat={mc['mcnemar_stat']} p={mc['p_value']} "
          f"(hybrid-only-correct={mc['b_hybrid_only_correct']}, "
          f"baseline-only-correct={mc['c_baseline_only_correct']})")
    print("[baselines] wrote results/baselines.csv and results/baselines.json")


if __name__ == "__main__":
    main()
