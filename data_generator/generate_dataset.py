"""Synthetic-but-grounded telemetry generator for the edge anomaly detector.

Design philosophy
-----------------
The detector must learn to separate cryptographic attacks, transport faults and
combined (multi-stage) events. We therefore *ground* the data in the project's
own simulators rather than sampling from hand-written Gaussians:

1.  A pool of **real** Tree-Parity-Machine synchronisation sessions
    (``neural_crypto.sync.run_session``) supplies the joint distribution of the
    cryptographic features (convergence, learning/sync rate, match probability,
    round count).
2.  A pool of **real** (2,2) split-bearer transmissions
    (``bifurcation.carriers.PDCPSplitBearer``) supplies the transport features
    (latency, loss, throughput, jitter, reconstruction failures) under both
    nominal and disrupted carrier conditions.

Per-class *artifacts* are then applied as documented, reproducible shifts on top
of the real base distributions. This reflects the threat model: attacks are
modelled as **active** (probing / injecting), which perturbs observable signals
(e.g. an active key-recovery adversary raises round count and depresses match
probability; a multi-stage adversary couples that with carrier disruption).

NOTE (honesty / limitation): a purely *passive* eavesdropper produces no host-
side artifact and is therefore not represented as a detectable class. This is an
explicit limitation, stated in the README and the paper.

Every generative parameter is written to a metadata JSON so the dataset is fully
reproducible from a seed.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import asdict, dataclass, field
from typing import Dict, List, Tuple

import numpy as np

# --- make the project importable when run as a script ---------------------- #
_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from anomaly_detector.features import (  # noqa: E402
    CLASS_TO_IDX,
    CLASSES,
    FEATURES,
)
from bifurcation.carriers import CarrierModel, PDCPSplitBearer  # noqa: E402
from neural_crypto.attacks import AttackConfig  # noqa: E402
from neural_crypto.sync import SyncConfig, run_session  # noqa: E402
from neural_crypto.tpm import TPMConfig  # noqa: E402

# Round-count normalisation constant (keeps the feature O(1)).
ROUND_NORM = 1000.0

# Valid clipping ranges per feature (after all shifts/noise).
CLIP: Dict[str, Tuple[float, float]] = {
    "tpm_convergence": (0.0, 1.0),
    "sync_rate": (0.0, 1.0),
    "match_probability": (0.0, 1.0),
    "round_count": (0.0, 10.0),
    "carrier_latency": (0.1, 1e4),
    "packet_loss": (0.0, 1.0),
    "throughput": (0.0, 1e4),
    "jitter": (0.0, 1e3),
    "share_reconstruction_failures": (0.0, 64.0),
    "flow_stats": (0.0, 1e3),
}


# --------------------------------------------------------------------------- #
#  Per-class generative specification (documented & serialised)
# --------------------------------------------------------------------------- #
@dataclass
class ClassSpec:
    """Multiplicative (``*_mul``) and additive (``*_add``) shifts per class.

    Crypto shifts model active key-recovery interference; transport shifts and
    ``transport_pool`` select nominal vs disrupted carrier conditions; flow_mul
    scales the aggregate flow statistic.
    """
    name: str
    transport_pool: str = "nominal"          # "nominal" | "disrupted"
    match_probability_add: float = 0.0
    tpm_convergence_add: float = 0.0
    sync_rate_add: float = 0.0
    round_count_mul: float = 1.0
    latency_mul: float = 1.0
    loss_add: float = 0.0
    throughput_mul: float = 1.0
    jitter_mul: float = 1.0
    recon_fail_add: float = 0.0
    flow_mul: float = 1.0


def default_class_specs() -> Dict[str, ClassSpec]:
    """The calibrated per-class artifact specification."""
    return {
        "Normal": ClassSpec(name="Normal"),
        "GeometricAttack": ClassSpec(
            name="GeometricAttack",
            match_probability_add=-0.08,
            tpm_convergence_add=-0.03,
            sync_rate_add=-0.03,
            round_count_mul=1.15,
            flow_mul=1.4,
        ),
        "MajorityAttack": ClassSpec(
            name="MajorityAttack",
            match_probability_add=-0.15,
            tpm_convergence_add=-0.06,
            sync_rate_add=-0.06,
            round_count_mul=1.30,
            flow_mul=1.8,
        ),
        "TrafficAnomaly": ClassSpec(
            name="TrafficAnomaly",
            throughput_mul=0.7,
            jitter_mul=1.6,
            flow_mul=3.0,
        ),
        "CarrierDisruption": ClassSpec(
            name="CarrierDisruption",
            transport_pool="disrupted",
            latency_mul=1.2,
            recon_fail_add=1.5,
        ),
        "MultiStageAttack": ClassSpec(
            name="MultiStageAttack",
            transport_pool="disrupted",
            match_probability_add=-0.15,
            tpm_convergence_add=-0.06,
            sync_rate_add=-0.06,
            round_count_mul=1.30,
            latency_mul=1.2,
            recon_fail_add=1.0,
            flow_mul=2.5,
        ),
    }


# --------------------------------------------------------------------------- #
#  Real base pools
# --------------------------------------------------------------------------- #
def build_crypto_pool(n_sessions: int, tpm: TPMConfig, seed: int) -> np.ndarray:
    """Run ``n_sessions`` real *normal* TPM sessions.

    Returns an array of shape (n_sessions, 4) with columns
    [tpm_convergence, sync_rate, match_probability, round_count_raw].
    """
    master = np.random.default_rng(seed)
    rows: List[List[float]] = []
    for _ in range(n_sessions):
        s = master.integers(0, 2 ** 31 - 1, size=4)
        cfg = SyncConfig(
            tpm=tpm,
            policy="early_confidence",
            seed_a=int(s[0]),
            seed_b=int(s[1]),
            seed_e=int(s[2]),
            seed_input=int(s[3]),
        )
        r = run_session(cfg, AttackConfig(kind="none"), record_traces=True)
        match_prob = r.confidence_trace[-1] if r.confidence_trace else 0.0
        rows.append([
            float(r.final_overlap_ab),
            float(r.total_agreements / max(1, r.rounds)),
            float(match_prob),
            float(r.rounds),
        ])
    return np.asarray(rows, dtype=float)


def _disrupted_carriers() -> List[CarrierModel]:
    """Degraded component carriers (congestion / interference)."""
    return [
        CarrierModel("CC0_n78_deg", base_latency_ms=18.0, jitter_ms=8.0,
                     loss_prob=0.08, throughput_mbps=60.0),
        CarrierModel("CC1_n258_deg", base_latency_ms=25.0, jitter_ms=12.0,
                     loss_prob=0.12, throughput_mbps=90.0),
    ]


def _nominal_carriers() -> List[CarrierModel]:
    return [
        CarrierModel("CC0_n78", base_latency_ms=7.0, jitter_ms=1.5,
                     loss_prob=0.008, throughput_mbps=150.0),
        CarrierModel("CC1_n258", base_latency_ms=10.0, jitter_ms=3.0,
                     loss_prob=0.015, throughput_mbps=220.0),
    ]


def build_transport_pool(n_windows: int, window: int, carriers: List[CarrierModel],
                         payload_bytes: int, seed: int) -> np.ndarray:
    """Aggregate ``n_windows`` observation windows of ``window`` real transmits.

    Returns array (n_windows, 5):
    [carrier_latency, packet_loss, throughput, jitter, recon_failures].
    """
    gen = np.random.default_rng(seed)
    bearer = PDCPSplitBearer(carriers)
    payload = b"\x00" * payload_bytes
    rows: List[List[float]] = []
    for _ in range(n_windows):
        lat, loss, tput, jit, fails = [], [], [], [], 0
        for _ in range(window):
            t = bearer.transmit(payload, gen)
            lat.append(t.e2e_latency_ms)
            loss.append(float(np.mean(t.per_carrier_loss)))
            tput.append(float(np.mean(t.per_carrier_throughput_mbps)))
            jit.append(float(np.mean(t.per_carrier_jitter_ms)))
            fails += t.reconstruction_failures
        rows.append([
            float(np.mean(lat)),
            float(np.mean(loss)),
            float(np.mean(tput)),
            float(np.mean(jit)),
            float(fails),
        ])
    return np.asarray(rows, dtype=float)


# --------------------------------------------------------------------------- #
#  Sampling
# --------------------------------------------------------------------------- #
@dataclass
class NoiseSpec:
    """Std-devs of the additive Gaussian noise applied per (grouped) feature.

    Calibrated so the six classes are *separable but genuinely confusable*
    (target macro-F1 ~0.85), reflecting real telemetry rather than a toy task.
    """
    crypto_overlap: float = 0.035
    crypto_rate: float = 0.035
    crypto_match: float = 0.040
    crypto_round_frac: float = 0.12      # fractional noise on round count
    latency_frac: float = 0.12
    loss: float = 0.015
    throughput_frac: float = 0.12
    jitter_frac: float = 0.25
    recon_fail: float = 0.8
    flow: float = 0.22


def _clip(name: str, x: np.ndarray) -> np.ndarray:
    lo, hi = CLIP[name]
    return np.clip(x, lo, hi)


def sample_class(
    spec: ClassSpec,
    n: int,
    crypto_pool: np.ndarray,
    transport_nominal: np.ndarray,
    transport_disrupted: np.ndarray,
    flow_mean: float,
    noise: NoiseSpec,
    gen: np.random.Generator,
) -> np.ndarray:
    """Generate ``n`` samples for one class as an (n, N_FEATURES) array."""
    # --- crypto base (bootstrap rows from the real normal pool) ------------- #
    ci = gen.integers(0, crypto_pool.shape[0], size=n)
    conv = crypto_pool[ci, 0] + gen.normal(0, noise.crypto_overlap, n) + spec.tpm_convergence_add
    rate = crypto_pool[ci, 1] + gen.normal(0, noise.crypto_rate, n) + spec.sync_rate_add
    match = crypto_pool[ci, 2] + gen.normal(0, noise.crypto_match, n) + spec.match_probability_add
    rounds_raw = crypto_pool[ci, 3] * spec.round_count_mul
    rounds_raw = rounds_raw * (1.0 + gen.normal(0, noise.crypto_round_frac, n))
    rounds = rounds_raw / ROUND_NORM

    # --- transport base (nominal or disrupted pool) ------------------------- #
    pool = transport_disrupted if spec.transport_pool == "disrupted" else transport_nominal
    ti = gen.integers(0, pool.shape[0], size=n)
    latency = pool[ti, 0] * spec.latency_mul * (1.0 + gen.normal(0, noise.latency_frac, n))
    loss = pool[ti, 1] + spec.loss_add + gen.normal(0, noise.loss, n)
    throughput = pool[ti, 2] * spec.throughput_mul * (1.0 + gen.normal(0, noise.throughput_frac, n))
    jitter = pool[ti, 3] * spec.jitter_mul * (1.0 + gen.normal(0, noise.jitter_frac, n))
    recon = pool[ti, 4] + spec.recon_fail_add + gen.normal(0, noise.recon_fail, n)
    recon = np.maximum(0.0, np.round(recon))

    # --- flow statistic ----------------------------------------------------- #
    flow = flow_mean * spec.flow_mul * (1.0 + gen.normal(0, noise.flow, n))

    cols = {
        "tpm_convergence": _clip("tpm_convergence", conv),
        "sync_rate": _clip("sync_rate", rate),
        "match_probability": _clip("match_probability", match),
        "round_count": _clip("round_count", rounds),
        "carrier_latency": _clip("carrier_latency", latency),
        "packet_loss": _clip("packet_loss", loss),
        "throughput": _clip("throughput", throughput),
        "jitter": _clip("jitter", jitter),
        "share_reconstruction_failures": _clip("share_reconstruction_failures", recon),
        "flow_stats": _clip("flow_stats", flow),
    }
    return np.column_stack([cols[f] for f in FEATURES])


# --------------------------------------------------------------------------- #
#  Top-level dataset generation
# --------------------------------------------------------------------------- #
@dataclass
class GenConfig:
    n_per_class: int = 1500
    seed: int = 7
    tpm_K: int = 3
    tpm_N: int = 100
    tpm_L: int = 4
    crypto_sessions: int = 48
    transport_windows: int = 256
    window: int = 8
    payload_bytes: int = 2048
    flow_mean: float = 10.0
    label_noise: float = 0.03                 # fraction of annotation errors
    out: str = "datasets/telemetry.csv"
    meta_out: str = "datasets/telemetry_metadata.json"


def generate(cfg: GenConfig) -> Tuple[np.ndarray, np.ndarray, dict]:
    tpm = TPMConfig(K=cfg.tpm_K, N=cfg.tpm_N, L=cfg.tpm_L)
    master = np.random.default_rng(cfg.seed)

    crypto_pool = build_crypto_pool(cfg.crypto_sessions, tpm, int(master.integers(0, 2 ** 31)))
    transport_nominal = build_transport_pool(
        cfg.transport_windows, cfg.window, _nominal_carriers(),
        cfg.payload_bytes, int(master.integers(0, 2 ** 31)))
    transport_disrupted = build_transport_pool(
        cfg.transport_windows, cfg.window, _disrupted_carriers(),
        cfg.payload_bytes, int(master.integers(0, 2 ** 31)))

    specs = default_class_specs()
    noise = NoiseSpec()
    gen = np.random.default_rng(int(master.integers(0, 2 ** 31)))

    X_parts, y_parts = [], []
    for cls in CLASSES:
        Xc = sample_class(specs[cls], cfg.n_per_class, crypto_pool,
                          transport_nominal, transport_disrupted,
                          cfg.flow_mean, noise, gen)
        X_parts.append(Xc)
        y_parts.append(np.full(cfg.n_per_class, CLASS_TO_IDX[cls], dtype=int))

    X = np.vstack(X_parts)
    y = np.concatenate(y_parts)

    # shuffle
    perm = gen.permutation(X.shape[0])
    X, y = X[perm], y[perm]

    # optional annotation noise: a small fraction of samples get a wrong label,
    # modelling imperfect ground truth in real telemetry capture.
    n_flip = int(round(cfg.label_noise * y.shape[0]))
    flipped = 0
    if n_flip > 0:
        flip_idx = gen.choice(y.shape[0], size=n_flip, replace=False)
        for fi in flip_idx:
            choices = [c for c in range(len(CLASSES)) if c != y[fi]]
            y[fi] = gen.choice(choices)
        flipped = n_flip

    meta = {
        "config": asdict(cfg),
        "features": FEATURES,
        "classes": CLASSES,
        "round_norm": ROUND_NORM,
        "label_noise_fraction": cfg.label_noise,
        "label_noise_count": int(flipped),
        "clip_ranges": CLIP,
        "noise_spec": asdict(noise),
        "class_specs": {k: asdict(v) for k, v in specs.items()},
        "crypto_pool_stats": {
            "n_sessions": int(crypto_pool.shape[0]),
            "mean_convergence": float(crypto_pool[:, 0].mean()),
            "mean_sync_rate": float(crypto_pool[:, 1].mean()),
            "mean_match_probability": float(crypto_pool[:, 2].mean()),
            "mean_round_count": float(crypto_pool[:, 3].mean()),
        },
        "transport_nominal_stats": {
            "mean_latency_ms": float(transport_nominal[:, 0].mean()),
            "mean_loss": float(transport_nominal[:, 1].mean()),
            "mean_throughput_mbps": float(transport_nominal[:, 2].mean()),
        },
        "transport_disrupted_stats": {
            "mean_latency_ms": float(transport_disrupted[:, 0].mean()),
            "mean_loss": float(transport_disrupted[:, 1].mean()),
            "mean_throughput_mbps": float(transport_disrupted[:, 2].mean()),
        },
    }
    return X, y, meta


def save_dataset(X: np.ndarray, y: np.ndarray, meta: dict, cfg: GenConfig) -> None:
    out_path = cfg.out if os.path.isabs(cfg.out) else os.path.join(_ROOT, cfg.out)
    meta_path = cfg.meta_out if os.path.isabs(cfg.meta_out) else os.path.join(_ROOT, cfg.meta_out)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    header = ",".join(FEATURES + ["label", "label_name"])
    names = np.array(CLASSES)
    with open(out_path, "w") as fh:
        fh.write(header + "\n")
        for i in range(X.shape[0]):
            vals = ",".join(f"{v:.6f}" for v in X[i])
            fh.write(f"{vals},{int(y[i])},{names[y[i]]}\n")

    with open(meta_path, "w") as fh:
        json.dump(meta, fh, indent=2)


def parse_args(argv: List[str] | None = None) -> GenConfig:
    p = argparse.ArgumentParser(description="Generate grounded edge telemetry dataset.")
    p.add_argument("--n-per-class", type=int, default=1500)
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--crypto-sessions", type=int, default=48)
    p.add_argument("--transport-windows", type=int, default=256)
    p.add_argument("--window", type=int, default=8)
    p.add_argument("--label-noise", type=float, default=0.03)
    p.add_argument("--tpm-N", type=int, default=100)
    p.add_argument("--out", type=str, default="datasets/telemetry.csv")
    p.add_argument("--meta-out", type=str, default="datasets/telemetry_metadata.json")
    a = p.parse_args(argv)
    return GenConfig(
        n_per_class=a.n_per_class, seed=a.seed, crypto_sessions=a.crypto_sessions,
        transport_windows=a.transport_windows, window=a.window, tpm_N=a.tpm_N,
        label_noise=a.label_noise,
        out=a.out, meta_out=a.meta_out,
    )


def main(argv: List[str] | None = None) -> None:
    cfg = parse_args(argv)
    print(f"[generate] building real base pools (crypto={cfg.crypto_sessions} "
          f"sessions, transport={cfg.transport_windows} windows) ...")
    X, y, meta = generate(cfg)
    save_dataset(X, y, meta, cfg)
    out_path = cfg.out if os.path.isabs(cfg.out) else os.path.join(_ROOT, cfg.out)
    print(f"[generate] wrote {X.shape[0]} samples x {X.shape[1]} features -> {out_path}")
    print(f"[generate] class balance: {np.bincount(y).tolist()}")


if __name__ == "__main__":
    main()
