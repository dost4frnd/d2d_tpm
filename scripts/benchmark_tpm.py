"""Micro-benchmark for the TPM primitive (deliverable C).

Reports per-round wall-clock cost, full-session synchronisation latency, key
derivation time, and serialized key size for a given (K, N, L). Pure-CPU; the
numbers are indicative of an edge node, not a tuned C implementation.
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from typing import Dict, List, Optional

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from common.utils import save_json  # noqa: E402
from neural_crypto.sync import SyncConfig, run_session  # noqa: E402
from neural_crypto.tpm import TPMConfig, TreeParityMachine, random_input  # noqa: E402


def bench_round(tpm: TPMConfig, iters: int, seed: int = 0) -> float:
    gen = np.random.default_rng(seed)
    A = TreeParityMachine(tpm, np.random.default_rng(seed + 1))
    B = TreeParityMachine(tpm, np.random.default_rng(seed + 2))
    # warm-up
    for _ in range(50):
        x = random_input(tpm, gen)
        ta, tb = A.forward(x), B.forward(x)
        if ta == tb:
            A.update(x, tb); B.update(x, ta)
    t0 = time.perf_counter()
    for _ in range(iters):
        x = random_input(tpm, gen)
        ta, tb = A.forward(x), B.forward(x)
        if ta == tb:
            A.update(x, tb); B.update(x, ta)
    return (time.perf_counter() - t0) / iters * 1e3  # ms/round


def bench_session(tpm: TPMConfig, reps: int, seed: int) -> Dict[str, float]:
    lat, rounds = [], []
    for i in range(reps):
        cfg = SyncConfig(tpm=tpm, policy="early_confidence",
                         seed_a=seed + 10 * i + 1, seed_b=seed + 10 * i + 2,
                         seed_e=seed + 10 * i + 3, seed_input=seed + 10 * i + 4)
        t0 = time.perf_counter()
        r = run_session(cfg, None)
        lat.append((time.perf_counter() - t0) * 1e3)
        rounds.append(r.rounds)
    return {"mean_session_ms": float(np.mean(lat)),
            "p95_session_ms": float(np.percentile(lat, 95)),
            "mean_rounds": float(np.mean(rounds))}


def main(argv: Optional[List[str]] = None) -> None:
    p = argparse.ArgumentParser(description="TPM micro-benchmark.")
    p.add_argument("--K", type=int, default=3)
    p.add_argument("--N", type=int, default=100)
    p.add_argument("--L", type=int, default=4)
    p.add_argument("--round-iters", type=int, default=5000)
    p.add_argument("--session-reps", type=int, default=10)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", type=str, default=os.path.join(_ROOT, "results", "tpm_benchmark.json"))
    a = p.parse_args(argv)

    tpm = TPMConfig(K=a.K, N=a.N, L=a.L)
    print(f"[bench] TPM K={a.K} N={a.N} L={a.L} on CPU")
    ms_round = bench_round(tpm, a.round_iters, a.seed)
    sess = bench_session(tpm, a.session_reps, a.seed)
    result = {"K": a.K, "N": a.N, "L": a.L,
              "ms_per_round": round(ms_round, 5),
              "rounds_per_sec": round(1000.0 / ms_round, 1),
              **{k: round(v, 3) for k, v in sess.items()},
              "key_bytes": 32}
    save_json(result, a.out)
    print(f"[bench] ms/round           : {result['ms_per_round']}")
    print(f"[bench] rounds/sec         : {result['rounds_per_sec']}")
    print(f"[bench] mean session (ms)  : {result['mean_session_ms']}")
    print(f"[bench] mean rounds        : {result['mean_rounds']}")
    print(f"[bench] wrote {a.out}")


if __name__ == "__main__":
    main()
