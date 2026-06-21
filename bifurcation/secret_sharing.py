"""XOR-based (2,2) secret sharing for 5G payload bifurcation.

A payload ``P`` is split into two shares ``S1 = R`` (uniform random) and
``S2 = P XOR R``.  Either share alone is information-theoretically independent of
``P``; both are required to reconstruct (``P = S1 XOR S2``).  This is the
two-party special case of Shamir's threshold scheme realised over GF(2).

Security/availability note
--------------------------
A (2,2) scheme maximises *confidentiality* of a single intercepted carrier but
*reduces availability*: losing either share prevents reconstruction.  See
``bifurcation.reliability`` for the quantified trade-off and the (2,3)/duplication
extensions discussed as future work.
"""
from __future__ import annotations

from typing import Optional, Tuple

import numpy as np


def split(payload: bytes, generator: Optional[np.random.Generator] = None
          ) -> Tuple[bytes, bytes]:
    """Split ``payload`` into two XOR shares of equal length."""
    gen = generator if generator is not None else np.random.default_rng()
    p = np.frombuffer(payload, dtype=np.uint8)
    r = gen.integers(0, 256, size=p.shape, dtype=np.uint8)
    s1 = r
    s2 = np.bitwise_xor(p, r)
    return s1.tobytes(), s2.tobytes()


def reconstruct(share1: bytes, share2: bytes) -> bytes:
    """Reconstruct the payload from two XOR shares."""
    a = np.frombuffer(share1, dtype=np.uint8)
    b = np.frombuffer(share2, dtype=np.uint8)
    if a.shape != b.shape:
        raise ValueError("share length mismatch")
    return np.bitwise_xor(a, b).tobytes()


def verify_roundtrip(payload: bytes, generator: Optional[np.random.Generator] = None) -> bool:
    s1, s2 = split(payload, generator)
    return reconstruct(s1, s2) == payload
