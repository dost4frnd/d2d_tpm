"""Shared utilities: reproducible seeding, config I/O, hashing/KDF, project paths.

These helpers are intentionally dependency-light so every module can import them.
"""
from __future__ import annotations

import hashlib
import hmac
import json
import os
import random
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any, Dict

import numpy as np

try:  # torch is optional for some scripts; import lazily where possible
    import torch
    _HAS_TORCH = True
except Exception:  # pragma: no cover
    _HAS_TORCH = False


# --------------------------------------------------------------------------- #
# Paths
# --------------------------------------------------------------------------- #
def project_root() -> Path:
    """Return the repository root (parent of the ``common`` package)."""
    return Path(__file__).resolve().parents[1]


def ensure_dir(path: os.PathLike | str) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


# --------------------------------------------------------------------------- #
# Reproducibility
# --------------------------------------------------------------------------- #
def set_global_seed(seed: int) -> None:
    """Seed Python, NumPy and (if present) torch for reproducible runs."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    if _HAS_TORCH:
        torch.manual_seed(seed)
        if torch.cuda.is_available():  # pragma: no cover - hw dependent
            torch.cuda.manual_seed_all(seed)
        # Determinism (best effort; some CUDA ops remain nondeterministic).
        torch.use_deterministic_algorithms(True, warn_only=True)


def rng(seed: int) -> np.random.Generator:
    """Return a dedicated NumPy Generator (preferred over global state)."""
    return np.random.default_rng(seed)


# --------------------------------------------------------------------------- #
# Config I/O
# --------------------------------------------------------------------------- #
def load_config(path: os.PathLike | str) -> Dict[str, Any]:
    import yaml  # imported here to keep import cost local

    with open(path, "r", encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh)
    return cfg or {}


def save_json(obj: Any, path: os.PathLike | str) -> None:
    ensure_dir(Path(path).parent)

    def _default(o: Any):
        if is_dataclass(o):
            return asdict(o)
        if isinstance(o, (np.integer,)):
            return int(o)
        if isinstance(o, (np.floating,)):
            return float(o)
        if isinstance(o, np.ndarray):
            return o.tolist()
        return str(o)

    with open(path, "w", encoding="utf-8") as fh:
        json.dump(obj, fh, indent=2, default=_default)


# --------------------------------------------------------------------------- #
# Cryptographic helpers (key derivation + privacy-preserving equality test)
# --------------------------------------------------------------------------- #
def sha256_bytes(data: bytes) -> bytes:
    return hashlib.sha256(data).digest()


def hkdf_sha256(ikm: bytes, length: int = 32, salt: bytes = b"", info: bytes = b"pqedge") -> bytes:
    """Minimal RFC-5869 HKDF (extract+expand) over SHA-256.

    Used to turn synchronised TPM weights into a fixed-length symmetric key.
    """
    if not salt:
        salt = b"\x00" * hashlib.sha256().digest_size
    prk = hmac.new(salt, ikm, hashlib.sha256).digest()
    okm = b""
    t = b""
    counter = 1
    while len(okm) < length:
        t = hmac.new(prk, t + info + bytes([counter]), hashlib.sha256).digest()
        okm += t
        counter += 1
    return okm[:length]


def equality_tag(secret: bytes, nonce: bytes) -> bytes:
    """Privacy-preserving equality tag.

    Both parties compute ``HMAC(SHA256(secret), nonce)`` over a *public* nonce.
    Exchanging the tag reveals at most a single equality bit about ``secret``
    (the tag is a PRF output; it does not leak the secret itself).
    """
    key = sha256_bytes(secret)
    return hmac.new(key, nonce, hashlib.sha256).digest()
