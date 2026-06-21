"""Head-to-head benchmark: standardized ML-KEM vs the TPM neural key-agreement
candidate, plus the hybrid ML-KEM+TPM key-establishment cost.

Reports, for ML-KEM-512/768/1024: keygen / encaps / decaps latency, total
key-establishment time, and the standardized bytes on the wire. For TPM: mean
synchronization rounds and wall-clock to a 32-byte key with early termination.
The hybrid adds one ML-KEM round + one HKDF over the concatenated secrets.

IMPORTANT: ML-KEM here is a *pure-Python reference* implementation; absolute
timings are far slower than optimized assembly on real edge hardware. We report
them for relative comparison and rely on the (implementation-independent) byte
sizes and the cited pqm4 / wolfSSL device numbers for hardware claims.
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

from common.utils import hkdf_sha256, save_json  # noqa: E402
from pqc.hybrid_kdf import derive_hybrid_key  # noqa: E402
from pqc.mlkem import MLKEM, PARAM_SIZES, available  # noqa: E402


def _time(fn, iters: int) -> float:
    fn()  # warm-up
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    return (time.perf_counter() - t0) / iters * 1e3  # ms


def bench_mlkem(level: str, iters: int) -> Dict:
    kem = MLKEM(level)
    ek, dk = kem.keygen()
    shared, ct = kem.encaps(ek)
    kg = _time(lambda: kem.keygen(), iters)
    en = _time(lambda: kem.encaps(ek), iters)
    de = _time(lambda: kem.decaps(dk, ct), iters)
    sz = PARAM_SIZES[level]
    # Initiator does keygen+decaps; responder does encaps. Establishment latency
    # on the initiator's critical path is keygen + decaps (encaps is concurrent).
    return {
        "scheme": level, "nist_level": sz["nist_level"],
        "keygen_ms": round(kg, 4), "encaps_ms": round(en, 4), "decaps_ms": round(de, 4),
        "establish_ms": round(kg + en + de, 4),
        "ek_bytes": sz["ek"], "dk_bytes": sz["dk"], "ct_bytes": sz["ct"],
        "shared_bytes": sz["ss"],
        "wire_bytes": sz["ek"] + sz["ct"],  # public key out + ciphertext back
    }


def bench_tpm(N: int, reps: int, seed: int) -> Dict:
    from neural_crypto.sync import SyncConfig, run_session
    from neural_crypto.tpm import TPMConfig

    lat, rounds = [], []
    for i in range(reps):
        cfg = SyncConfig(tpm=TPMConfig(K=3, N=N, L=4), policy="early_confidence",
                         seed_a=seed + 7 * i + 1, seed_b=seed + 7 * i + 2,
                         seed_e=seed + 7 * i + 3, seed_input=seed + 7 * i + 4)
        t0 = time.perf_counter()
        r = run_session(cfg, None)
        lat.append((time.perf_counter() - t0) * 1e3)
        rounds.append(r.rounds)
    return {
        "scheme": f"TPM(K=3,N={N},L=4)", "nist_level": None,
        "establish_ms": round(float(np.mean(lat)), 4),
        "p95_establish_ms": round(float(np.percentile(lat, 95)), 4),
        "mean_rounds": round(float(np.mean(rounds)), 1),
        "shared_bytes": 32,
        # TPM exchanges one bit per round over the public channel, plus a few
        # short equality tags; the dominant on-wire cost is ~mean_rounds bits.
        "wire_bytes": int(round(np.mean(rounds) / 8.0)),
        "security_proof": False,
    }


def bench_hybrid(level: str, N: int, iters: int, seed: int) -> Dict:
    kem = MLKEM(level)
    tpm = bench_tpm(N, max(3, iters // 10), seed)
    ek, dk = kem.keygen()
    shared, ct = kem.encaps(ek)
    tpm_secret = hkdf_sha256(b"tpm-weight-bytes", length=32)
    kdf_ms = _time(lambda: derive_hybrid_key(shared, tpm_secret, length=32), iters)
    mlkem = bench_mlkem(level, iters)
    return {
        "scheme": f"Hybrid({level} + TPM)",
        "mlkem_establish_ms": mlkem["establish_ms"],
        "tpm_establish_ms": tpm["establish_ms"],
        "kdf_ms": round(kdf_ms, 5),
        "establish_ms": round(mlkem["establish_ms"] + tpm["establish_ms"] + kdf_ms, 4),
        "wire_bytes": mlkem["wire_bytes"] + tpm["wire_bytes"],
        "shared_bytes": 32,
        "security_floor": level,
        "note": "security inherited from ML-KEM; TPM is defence-in-depth only",
    }


def main(argv: Optional[List[str]] = None) -> None:
    p = argparse.ArgumentParser(description="ML-KEM vs TPM key-establishment benchmark.")
    p.add_argument("--iters", type=int, default=200)
    p.add_argument("--tpm-N", type=int, default=100)
    p.add_argument("--tpm-reps", type=int, default=10)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", type=str, default=os.path.join(_ROOT, "results", "pqc_benchmark.json"))
    a = p.parse_args(argv)

    if not available():
        print("[pqc] ML-KEM backend unavailable; install kyber-py. Aborting.")
        return

    print("[pqc] benchmarking ML-KEM parameter sets (pure-Python reference) ...")
    mlkem_rows = [bench_mlkem(lvl, a.iters) for lvl in
                  ["ML-KEM-512", "ML-KEM-768", "ML-KEM-1024"]]
    print("[pqc] benchmarking TPM neural key agreement ...")
    tpm_row = bench_tpm(a.tpm_N, a.tpm_reps, a.seed)
    print("[pqc] benchmarking hybrid ML-KEM-768 + TPM ...")
    hybrid_row = bench_hybrid("ML-KEM-768", a.tpm_N, a.iters, a.seed)

    result = {"mlkem": mlkem_rows, "tpm": tpm_row, "hybrid": hybrid_row}
    save_json(result, a.out)

    print("\n[pqc] === Key establishment: scheme comparison ===")
    print(f"  {'scheme':24s} {'establish_ms':>12} {'wire_B':>8} {'ss_B':>5} {'proof':>6}")
    for r in mlkem_rows:
        print(f"  {r['scheme']:24s} {r['establish_ms']:>12.3f} {r['wire_bytes']:>8} "
              f"{r['shared_bytes']:>5} {'yes':>6}")
    print(f"  {tpm_row['scheme']:24s} {tpm_row['establish_ms']:>12.3f} "
          f"{tpm_row['wire_bytes']:>8} {tpm_row['shared_bytes']:>5} {'no':>6}")
    print(f"  {hybrid_row['scheme']:24s} {hybrid_row['establish_ms']:>12.3f} "
          f"{hybrid_row['wire_bytes']:>8} {hybrid_row['shared_bytes']:>5} {'yes':>6}")
    print(f"\n[pqc] wrote {a.out}")


if __name__ == "__main__":
    main()
