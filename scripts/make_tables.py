"""Emit LaTeX tables (paper/tables/*.tex) from the JSON results.

Keeps the paper numerically consistent with results/: re-run the experiments,
then re-run this to refresh the tables that main.tex \\input{}s.
"""
from __future__ import annotations

import json
import os

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
RES = os.path.join(_ROOT, "results")
TAB = os.path.join(_ROOT, "paper", "tables")
os.makedirs(TAB, exist_ok=True)


def _load(name):
    with open(os.path.join(RES, name), "r", encoding="utf-8") as fh:
        return json.load(fh)


def _fmt(v):
    if v is None:
        return "--"
    if isinstance(v, bool):
        return "yes" if v else "no"
    if isinstance(v, float):
        return f"{v:.3f}" if abs(v) < 100 else f"{v:.1f}"
    return str(v)


def configurations_table():
    d = _load("run_all.json")
    rows = d["configurations"]
    lines = [
        r"\begin{table}[t]", r"\centering",
        r"\caption{Cumulative ablation of the framework components. Crypto "
        r"columns use $K{=}3,N{=}100$; eavesdropper success is the operational "
        r"(majority-vote) key-recovery rate; availability is the (2,2) "
        r"reconstruction probability at per-PDU loss $p{=}0.05$.}",
        r"\label{tab:configs}",
        r"\resizebox{\columnwidth}{!}{%",
        r"\begin{tabular}{lccccc}", r"\toprule",
        r"Config & Verif. & Att.$_{geo}$ & Att.$_{maj}$ & Avail. & F1 \\",
        r"\midrule",
    ]
    label = {
        "C1_TPM_baseline": "TPM baseline",
        "C2_plus_early_termination": "+ Early term.",
        "C3_plus_bifurcation": "+ Bifurcation",
        "C4_plus_anomaly_detection": "+ Anomaly det.",
        "C5_complete_framework": "Complete",
    }
    for r in rows:
        lines.append(
            f"{label.get(r['config'], r['config'])} & "
            f"{_fmt(r['mean_verifications'])} & "
            f"{_fmt(r['attacker_success_geometric'])} & "
            f"{_fmt(r['attacker_success_majority'])} & "
            f"{_fmt(r['availability_2of2'])} & "
            f"{_fmt(r['detector_macro_f1'])} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}}", r"\end{table}"]
    return "\n".join(lines) + "\n"


def detector_table():
    d = _load("ablation.json")
    arch = d["architecture"]
    feat = d["feature_groups"]
    lines = [
        r"\begin{table}[t]", r"\centering",
        r"\caption{Detector ablations on the held-out test split. Left: "
        r"architecture. Right: feature-group restriction (non-selected features "
        r"zeroed). Neither cryptographic nor transport telemetry alone suffices; "
        r"their fusion drives performance.}",
        r"\label{tab:detector}",
        r"\begin{tabular}{lcc@{\hskip 1.5em}lcc}", r"\toprule",
        r"Model & F1 & AUC & Features & F1 & AUC \\", r"\midrule",
    ]
    name = {"wknn": "WKNN", "fnn": "FNN", "hybrid": "Hybrid"}
    fname = {"crypto_only": "Crypto", "transport_only": "Transport",
             "all_features": "All"}
    for i in range(max(len(arch), len(feat))):
        a = arch[i] if i < len(arch) else None
        f = feat[i] if i < len(feat) else None
        la = (f"{name.get(a['detector'], a['detector'])} & {_fmt(a['macro_f1'])} "
              f"& {_fmt(a['roc_auc'])}") if a else " & & "
        lf = (f"{fname.get(f['feature_group'], f['feature_group'])} & "
              f"{_fmt(f['macro_f1'])} & {_fmt(f['roc_auc'])}") if f else " & & "
        lines.append(f"{la} & {lf} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    return "\n".join(lines) + "\n"


def earlyterm_table():
    d = _load("ablation.json")
    rows = d["early_termination"]
    lines = [
        r"\begin{table}[t]", r"\centering",
        r"\caption{Effect of privacy-preserving early termination versus "
        r"fixed-interval verification, against the geometric eavesdropper "
        r"($K{=}3,N{=}100$). Rate-limited verification bounds the message count.}",
        r"\label{tab:earlyterm}",
        r"\begin{tabular}{lccc}", r"\toprule",
        r"Policy & Rounds & Verif. & Att.$_{geo}$ \\", r"\midrule",
    ]
    for r in rows:
        pol = "hash-only" if r["policy"] == "hash_only" else \
            f"early ($\\tau{{=}}{r['conf_threshold']}$)"
        lines.append(f"{pol} & {_fmt(r['mean_rounds'])} & "
                     f"{_fmt(r['mean_verifications'])} & "
                     f"{_fmt(r['attacker_success_geom'])} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    return "\n".join(lines) + "\n"


def reliability_table():
    d = _load("run_all.json")["bifurcation"]
    lines = [
        r"\begin{table}[t]", r"\centering",
        r"\caption{Transport reliability at per-PDU loss $p{=}0.05$, 8 PDUs per "
        r"share. (2,2) bifurcation maximises confidentiality but lowers "
        r"availability; duplication recovers availability at the cost of "
        r"confidentiality.}",
        r"\label{tab:reliability}",
        r"\begin{tabular}{lcc}", r"\toprule",
        r"Scheme & Analytic avail. & MC avail. \\", r"\midrule",
        f"Single carrier & {_fmt(d['availability_single'])} & -- \\\\",
        f"(2,2) split bearer & {_fmt(d['availability_2of2'])} & "
        f"{_fmt(d['mc_availability_2of2'])} \\\\",
        f"Duplicated (2 of 2) & {_fmt(d['availability_duplicated'])} & -- \\\\",
        r"\bottomrule", r"\end{tabular}", r"\end{table}",
    ]
    return "\n".join(lines) + "\n"


def pqc_table():
    d = _load("pqc_benchmark.json")
    lines = [
        r"\begin{table}[t]", r"\centering",
        r"\caption{Key establishment: standardized ML-KEM vs the TPM neural "
        r"key-agreement candidate vs the hybrid. ML-KEM timings are a "
        r"\emph{pure-Python reference} (relative only); byte sizes are "
        r"standardized (FIPS~203). TPM's sole advantage is wire footprint; it "
        r"carries no security proof.}",
        r"\label{tab:pqc}",
        r"\begin{tabular}{lcccc}", r"\toprule",
        r"Scheme & Estab. (ms) & Wire (B) & Secret (B) & Proof \\", r"\midrule",
    ]
    for r in d["mlkem"]:
        lines.append(f"{r['scheme']} & {_fmt(r['establish_ms'])} & {r['wire_bytes']} "
                     f"& {r['shared_bytes']} & yes \\\\")
    t = d["tpm"]
    lines.append(f"{t['scheme']} & {_fmt(t['establish_ms'])} & {t['wire_bytes']} "
                 f"& {t['shared_bytes']} & no \\\\")
    h = d["hybrid"]
    lines.append(r"\midrule")
    lines.append(f"Hybrid (768+TPM) & {_fmt(h['establish_ms'])} & {h['wire_bytes']} "
                 f"& {h['shared_bytes']} & yes \\\\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    return "\n".join(lines) + "\n"


def baselines_table():
    d = _load("baselines.json")
    rep = d["repeated"]
    mc = d["mcnemar"]
    order = ["random_forest", "svm_rbf", "grad_boosting", "logistic", "wknn", "fnn", "hybrid"]
    pretty = {"random_forest": "Random Forest", "svm_rbf": "SVM (RBF)",
              "grad_boosting": "Grad. Boosting", "logistic": "Logistic Reg.",
              "wknn": "WKNN", "fnn": "FNN", "hybrid": "Hybrid (ours)"}
    lines = [
        r"\begin{table}[t]", r"\centering",
        r"\caption{Detector vs standard baselines on the grounded telemetry "
        r"(mean$\pm$std macro-F1 over 5 seeds, all using both feature modalities). "
        r"The hybrid is competitive with the best calibration (AUC); a McNemar "
        r"test finds no significant gap to the strongest baseline, so the "
        r"contribution is cross-modal fusion and integration, not classifier "
        r"superiority.}",
        r"\label{tab:baselines}",
        r"\begin{tabular}{lcc}", r"\toprule",
        r"Model & Macro-F1 & ROC-AUC \\", r"\midrule",
    ]
    for m in order:
        f1 = rep[m]["macro_f1"]; auc = rep[m]["roc_auc"]
        bold = m == "hybrid"
        name = (r"\textbf{" + pretty[m] + "}") if bold else pretty[m]
        f1s = f"{f1['mean']:.3f}$\\pm${f1['std']:.3f}"
        lines.append(f"{name} & {f1s} & {auc['mean']:.3f} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}",
              r"\par\smallskip",
              r"{\footnotesize McNemar (hybrid vs " +
              str(mc["best_baseline"]).replace("_", "\\_") +
              f"): $\\chi^2{{=}}{mc['mcnemar_stat']}$, $p{{=}}{mc['p_value']}$.}}",
              r"\end{table}"]
    return "\n".join(lines) + "\n"


def main():
    tables = {
        "configurations.tex": configurations_table(),
        "detector.tex": detector_table(),
        "earlyterm.tex": earlyterm_table(),
        "reliability.tex": reliability_table(),
        "pqc.tex": pqc_table(),
        "baselines.tex": baselines_table(),
    }
    for name, body in tables.items():
        with open(os.path.join(TAB, name), "w", encoding="utf-8") as fh:
            fh.write(body)
        print(f"[tables] wrote paper/tables/{name}")


if __name__ == "__main__":
    main()
