"""Tunable (T, N) threshold secret sharing over GF(2^8) for multi-carrier
bifurcation -- the principled generalization of the XOR (2,2) scheme.

A payload is split into ``N`` shares such that any ``T`` of them reconstruct it
and any ``T-1`` reveal nothing (Shamir's scheme; information-theoretic). Mapping
each share to an independent 5G carrier/path gives a tunable knob:

  * confidentiality threshold = T   (adversary capturing <= T-1 carriers learns nothing)
  * availability margin       = N-T (reconstruction tolerates losing up to N-T shares)

Special cases recovered:
  * (1,1)  -> single carrier
  * (2,2)  -> the XOR scheme in ``secret_sharing`` (max confidentiality, min availability)
  * (1,N)  -> N-way duplication (max availability, min confidentiality)

GF(2^8) arithmetic uses the AES reduction polynomial (0x11B) with precomputed
log/antilog tables, vectorized over payload bytes with NumPy.
"""
from __future__ import annotations

from typing import List, Optional, Sequence, Tuple

import numpy as np

# ---- GF(2^8) tables (generator g=3 over the AES field) -------------------- #
_EXP = np.zeros(512, dtype=np.uint16)
_LOG = np.zeros(256, dtype=np.uint16)


def _build_tables() -> None:
    x = 1
    for i in range(255):
        _EXP[i] = x
        _LOG[x] = i
        # multiply by 3 (generator) in GF(2^8) with 0x11B reduction
        x ^= (x << 1)
        if x & 0x100:
            x ^= 0x11B
        x &= 0xFF
    for i in range(255, 512):
        _EXP[i] = _EXP[i - 255]


_build_tables()


def _gf_mul(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Vectorized GF(2^8) multiply of two uint8 arrays."""
    a = a.astype(np.uint16)
    b = b.astype(np.uint16)
    out = np.zeros(np.broadcast(a, b).shape, dtype=np.uint8)
    nz = (a != 0) & (b != 0)
    la = _LOG[a[nz]].astype(np.uint16)
    lb = _LOG[b[nz]].astype(np.uint16)
    out[nz] = _EXP[(la + lb) % 255].astype(np.uint8)
    return out


def _gf_mul_scalar(a: np.ndarray, s: int) -> np.ndarray:
    if s == 0:
        return np.zeros_like(a, dtype=np.uint8)
    la = _LOG[a.astype(np.uint16)]
    res = _EXP[(la + _LOG[s]) % 255].astype(np.uint8)
    res[a == 0] = 0
    return res


def split(payload: bytes, T: int, N: int,
          generator: Optional[np.random.Generator] = None
          ) -> List[Tuple[int, bytes]]:
    """Split ``payload`` into ``N`` shares; any ``T`` reconstruct it.

    Returns a list of (x, share_bytes) with distinct x in 1..N.
    """
    if not (1 <= T <= N <= 255):
        raise ValueError("require 1 <= T <= N <= 255")
    gen = generator if generator is not None else np.random.default_rng()
    p = np.frombuffer(payload, dtype=np.uint8)
    L = p.shape[0]
    # Random coefficients a_1..a_{T-1} per byte; a_0 = secret byte.
    coeffs = gen.integers(0, 256, size=(T - 1, L), dtype=np.uint8) if T > 1 \
        else np.zeros((0, L), dtype=np.uint8)
    shares: List[Tuple[int, bytes]] = []
    for x in range(1, N + 1):
        # Horner evaluation over GF(2^8) of p + a1*x + ... + a_{T-1}*x^{T-1}.
        # coeffs[k-1] = a_k for k=1..T-1; constant term a0 = p (the secret).
        y = np.zeros(L, dtype=np.uint8)
        for k in range(T - 1, 0, -1):
            y = np.bitwise_xor(_gf_mul_scalar(y, x), coeffs[k - 1])
        y = np.bitwise_xor(_gf_mul_scalar(y, x), p)
        shares.append((x, y.tobytes()))
    return shares


def reconstruct(shares: Sequence[Tuple[int, bytes]], T: int) -> bytes:
    """Reconstruct the payload from at least ``T`` shares via Lagrange at x=0."""
    if len(shares) < T:
        raise ValueError(f"need >= {T} shares, got {len(shares)}")
    use = list(shares)[:T]
    xs = [x for x, _ in use]
    ys = [np.frombuffer(s, dtype=np.uint8) for _, s in use]
    L = ys[0].shape[0]
    secret = np.zeros(L, dtype=np.uint8)
    for j in range(T):
        # Lagrange basis l_j(0) = prod_{m!=j} x_m / (x_m XOR x_j)
        num, den = 1, 1
        for m in range(T):
            if m == j:
                continue
            num = _single_mul(num, xs[m])
            den = _single_mul(den, xs[m] ^ xs[j])
        coeff = _single_mul(num, _single_inv(den))
        secret = np.bitwise_xor(secret, _gf_mul_scalar(ys[j], coeff))
    return secret.tobytes()


def _single_mul(a: int, b: int) -> int:
    if a == 0 or b == 0:
        return 0
    return int(_EXP[(int(_LOG[a]) + int(_LOG[b])) % 255])


def _single_inv(a: int) -> int:
    if a == 0:
        raise ZeroDivisionError("GF(256) inverse of 0")
    return int(_EXP[(255 - int(_LOG[a])) % 255])


def verify_roundtrip(payload: bytes, T: int, N: int,
                     generator: Optional[np.random.Generator] = None) -> bool:
    """Split, then reconstruct from an arbitrary T-subset, and check equality."""
    gen = generator if generator is not None else np.random.default_rng()
    shares = split(payload, T, N, gen)
    # pick a random T-subset of the N shares
    idx = gen.choice(N, size=T, replace=False)
    subset = [shares[i] for i in idx]
    return reconstruct(subset, T) == payload
