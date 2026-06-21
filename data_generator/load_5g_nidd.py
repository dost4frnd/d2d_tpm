"""Loader/adapter for the public 5G-NIDD intrusion-detection dataset.

5G-NIDD (Samarakoon et al., IEEE DataPort doi:10.21227/xtep-hv36, 2022) was
captured on the University of Oulu 5G Test Network and contains ~1.2M labeled
flows across eight attack types (ICMP/UDP/SYN/HTTP floods, slow-rate DoS, and
SYN/TCP-Connect/UDP scans) plus benign traffic.

HONEST SCOPE NOTE
-----------------
5G-NIDD captures *transport/network-flow* telemetry. It does NOT contain the
cryptographic-layer signals our detector fuses (TPM convergence, key match
probability, sync round count). So:
  * On 5G-NIDD alone, the detector operates in a transport-only regime on REAL
    attack traffic -- a strong external-validity check for the transport half.
  * To exercise the cross-modal *fusion* claim on real transport data, the
    cryptographic features must be augmented/injected (clearly a semi-synthetic
    setup). Use --augment-crypto for that, and state it as a limitation.

The dataset is NOT redistributed here; download it from IEEE DataPort and pass
the combined CSV path. Column handling is auto-detecting and may need minor
adjustment to your 5G-NIDD release.
"""
from __future__ import annotations

import argparse
import os
import sys
from typing import List, Optional, Tuple

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

_LABEL_CANDIDATES = ["Attack Type", "attack_type", "AttackType", "Label", "label",
                     "Attack", "attack", "class", "Class"]
_DROP_HINTS = ["ip", "addr", "port", "proto", "state", "time", "stime", "ltime",
               "flow", "id", "seq", "dir", "service", "sport", "dport", "saddr",
               "daddr", "timestamp", "date"]


def _is_droppable(col: str) -> bool:
    c = col.lower()
    return any(h in c for h in _DROP_HINTS)


def load_5g_nidd(path: str, max_rows: Optional[int] = None,
                 augment_crypto: bool = False, seed: int = 0
                 ) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """Load 5G-NIDD into (X, y, class_names).

    Numeric flow columns become features; the label column is mapped to integer
    classes. With augment_crypto, four synthetic cryptographic-telemetry columns
    are appended (documented as semi-synthetic) so the fusion pipeline can run.
    """
    try:
        import pandas as pd
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("pandas is required to load 5G-NIDD") from exc

    df = pd.read_csv(path, low_memory=False)
    if max_rows is not None and len(df) > max_rows:
        df = df.sample(n=max_rows, random_state=seed).reset_index(drop=True)

    label_col = next((c for c in _LABEL_CANDIDATES if c in df.columns), None)
    if label_col is None:
        raise ValueError(f"no label column found among {_LABEL_CANDIDATES}; "
                         f"columns present: {list(df.columns)[:20]} ...")

    y_raw = df[label_col].astype(str).fillna("Unknown").values
    class_names = sorted(set(y_raw))
    name_to_id = {n: i for i, n in enumerate(class_names)}
    y = np.array([name_to_id[v] for v in y_raw], dtype=int)

    feat_cols = [c for c in df.columns
                 if c != label_col and not _is_droppable(c)
                 and np.issubdtype(df[c].dtype, np.number)]
    X = df[feat_cols].to_numpy(dtype=float)
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

    if augment_crypto:
        rng = np.random.default_rng(seed)
        # Append 4 synthetic crypto-telemetry columns correlated with the label
        # so the fusion path is exercisable on real transport data. SEMI-SYNTHETIC.
        n = X.shape[0]
        shift = (y.astype(float) / max(1, len(class_names) - 1))
        crypto = np.stack([
            0.9 - 0.4 * shift + 0.05 * rng.standard_normal(n),   # convergence
            0.95 - 0.5 * shift + 0.05 * rng.standard_normal(n),  # match prob
            1.0 + 0.6 * shift + 0.1 * rng.standard_normal(n),    # round count (norm)
            0.5 + 0.4 * shift + 0.1 * rng.standard_normal(n),    # sync rate proxy
        ], axis=1)
        X = np.concatenate([X, crypto], axis=1)

    return X, y, class_names


def main(argv: Optional[List[str]] = None) -> None:
    p = argparse.ArgumentParser(description="Inspect/convert a 5G-NIDD CSV.")
    p.add_argument("path", help="path to the 5G-NIDD combined CSV")
    p.add_argument("--max-rows", type=int, default=None)
    p.add_argument("--augment-crypto", action="store_true")
    p.add_argument("--out", type=str, default=os.path.join(_ROOT, "datasets", "fivegnidd.csv"))
    a = p.parse_args(argv)

    X, y, names = load_5g_nidd(a.path, a.max_rows, a.augment_crypto)
    print(f"[5g-nidd] loaded X={X.shape} classes={len(names)}: {names}")
    counts = {names[i]: int((y == i).sum()) for i in range(len(names))}
    print(f"[5g-nidd] class counts: {counts}")
    # write a normalized CSV (feature_0..feature_{d-1}, label)
    import csv as _c
    with open(a.out, "w", newline="", encoding="utf-8") as fh:
        w = _c.writer(fh)
        w.writerow([f"feature_{i}" for i in range(X.shape[1])] + ["label"])
        for row, lab in zip(X, y):
            w.writerow(list(row) + [int(lab)])
    print(f"[5g-nidd] wrote normalized CSV -> {a.out}")


if __name__ == "__main__":
    main()
