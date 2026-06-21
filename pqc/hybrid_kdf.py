"""Hybrid key establishment: combine the ML-KEM shared secret with the TPM-derived
secret into a single session key.

Design (follows the concatenation pattern of NIST SP 800-56C Rev. 2 and the IETF
hybrid-KEM drafts, e.g. X25519MLKEM768): the two shared secrets are concatenated
in a fixed order and fed through HKDF-SHA256 together with a context/transcript
label. The resulting key is secure as long as *at least one* input secret is
secure and the KDF is a secure PRF -- so the hybrid is no weaker than its
strongest component.

Honest positioning:
  * ML-KEM provides the assured, standardized, post-quantum confidentiality floor
    (Module-LWE, IND-CCA2). It is the component we *rely on* against a quantum
    adversary.
  * The TPM secret is defence-in-depth / auxiliary entropy only. It has no
    security proof and is broken by known attacks; it must never be the sole
    basis of confidentiality.
  * For a FIPS-approved key-derivation chain the *first* shared secret must come
    from a FIPS-approved scheme (ML-KEM), which is why ML-KEM is placed first in
    the concatenation.
"""
from __future__ import annotations

import hashlib
import hmac
from dataclasses import dataclass
from typing import Optional


def _hkdf_extract(salt: bytes, ikm: bytes) -> bytes:
    if not salt:
        salt = b"\x00" * hashlib.sha256().digest_size
    return hmac.new(salt, ikm, hashlib.sha256).digest()


def _hkdf_expand(prk: bytes, info: bytes, length: int) -> bytes:
    out, t, counter = b"", b"", 1
    while len(out) < length:
        t = hmac.new(prk, t + info + bytes([counter]), hashlib.sha256).digest()
        out += t
        counter += 1
    return out[:length]


@dataclass
class HybridKey:
    key: bytes                 # derived session key
    length: int
    transcript: bytes          # context/label bound into the KDF
    mlkem_used: bool
    tpm_used: bool


def derive_hybrid_key(mlkem_secret: Optional[bytes],
                      tpm_secret: Optional[bytes],
                      length: int = 32,
                      salt: bytes = b"",
                      label: bytes = b"pqedge/hybrid-v1") -> HybridKey:
    """Derive a session key from the ML-KEM and TPM shared secrets.

    ML-KEM is concatenated first (FIPS-approved-first ordering). Either input may
    be ``None`` (e.g. ML-KEM-only or TPM-only ablations), but at least one must be
    present.
    """
    if not mlkem_secret and not tpm_secret:
        raise ValueError("at least one of mlkem_secret / tpm_secret is required")

    # Length-prefix each secret so the concatenation is unambiguous.
    def _lp(b: Optional[bytes]) -> bytes:
        b = b or b""
        return len(b).to_bytes(2, "big") + b

    ikm = _lp(mlkem_secret) + _lp(tpm_secret)
    # Bind a transcript label (and which components are present) into info.
    info = (label + b"|mlkem=" + (b"1" if mlkem_secret else b"0")
            + b"|tpm=" + (b"1" if tpm_secret else b"0"))
    prk = _hkdf_extract(salt, ikm)
    key = _hkdf_expand(prk, info, length)
    return HybridKey(key=key, length=length, transcript=info,
                     mlkem_used=bool(mlkem_secret), tpm_used=bool(tpm_secret))


def key_confirmation_tag(key: bytes, nonce: bytes,
                         label: bytes = b"pqedge/kc") -> bytes:
    """HMAC-SHA256 key-confirmation tag over a public nonce (leaks <= 1 bit)."""
    return hmac.new(key, label + nonce, hashlib.sha256).digest()
