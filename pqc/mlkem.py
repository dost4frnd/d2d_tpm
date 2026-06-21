"""ML-KEM (FIPS 203) key encapsulation -- the assured, standardized security floor.

This wraps a *pure-Python* reference implementation of ML-KEM (``kyber-py``), so
it runs anywhere in CI without a C toolchain. Pure-Python keygen/encaps/decaps is
one to two orders of magnitude slower than the optimized assembly used on real
hardware (pqm4 on Cortex-M4, AVX2 on x86); we therefore use these timings only
for *relative* comparison and quote the standardized byte sizes (which are
implementation-independent) and cite pqm4 / wolfSSL for optimized device numbers.

Security rationale (FIPS 203): ML-KEM is an IND-CCA2 key-encapsulation mechanism
whose security reduces to the hardness of Module-Learning-With-Errors (Module-LWE).
Unlike the TPM neural key-agreement candidate, this is a standardized primitive
with a security argument and is the component the framework relies on for
confidentiality against a quantum adversary.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, Tuple

try:
    from kyber_py.ml_kem import ML_KEM_512, ML_KEM_768, ML_KEM_1024
    _AVAILABLE = True
except Exception:  # pragma: no cover
    _AVAILABLE = False
    ML_KEM_512 = ML_KEM_768 = ML_KEM_1024 = None  # type: ignore


# Standardized sizes in bytes (FIPS 203, Table 3). Independent of implementation.
PARAM_SIZES: Dict[str, Dict[str, int]] = {
    "ML-KEM-512": {"ek": 800, "dk": 1632, "ct": 768, "ss": 32, "nist_level": 1},
    "ML-KEM-768": {"ek": 1184, "dk": 2400, "ct": 1088, "ss": 32, "nist_level": 3},
    "ML-KEM-1024": {"ek": 1568, "dk": 3168, "ct": 1568, "ss": 32, "nist_level": 5},
}

_IMPLS: Dict[str, object] = {
    "ML-KEM-512": ML_KEM_512,
    "ML-KEM-768": ML_KEM_768,
    "ML-KEM-1024": ML_KEM_1024,
}


def available() -> bool:
    """True iff the ML-KEM backend imported successfully."""
    return _AVAILABLE


@dataclass
class KEMResult:
    name: str
    ek: bytes          # encapsulation (public) key
    dk: bytes          # decapsulation (secret) key
    ct: bytes          # ciphertext
    shared: bytes      # 32-byte shared secret
    agree: bool        # decaps(dk, ct) == shared


class MLKEM:
    """Thin object wrapper around one ML-KEM parameter set."""

    def __init__(self, level: str = "ML-KEM-768"):
        if level not in PARAM_SIZES:
            raise ValueError(f"unknown ML-KEM level {level!r}")
        if not _AVAILABLE:
            raise RuntimeError(
                "ML-KEM backend unavailable; install with `pip install kyber-py`.")
        self.level = level
        self._impl = _IMPLS[level]
        self.sizes = PARAM_SIZES[level]

    def keygen(self) -> Tuple[bytes, bytes]:
        ek, dk = self._impl.keygen()
        return ek, dk

    def encaps(self, ek: bytes) -> Tuple[bytes, bytes]:
        shared, ct = self._impl.encaps(ek)
        return shared, ct

    def decaps(self, dk: bytes, ct: bytes) -> bytes:
        return self._impl.decaps(dk, ct)

    def run_once(self) -> KEMResult:
        """Full keygen -> encaps -> decaps round, with an agreement check."""
        ek, dk = self.keygen()
        shared_e, ct = self.encaps(ek)
        shared_d = self.decaps(dk, ct)
        return KEMResult(self.level, ek, dk, ct, shared_e, shared_e == shared_d)
