"""Tree Parity Machine (TPM) for neural key exchange.

A TPM has ``K`` hidden units, each with ``N`` inputs and integer weights in
``[-L, L]``.  Given a common random input, both parties update their weights with
a learning rule whenever their public outputs agree; the weight vectors converge
("synchronise") and are then hashed into a shared symmetric key.

References
----------
I. Kanter, W. Kinzel, E. Kanter, "Secure exchange of information by
synchronization of neural networks," Europhys. Lett., 2002.
A. Ruttor, "Neural Synchronization and Cryptography," PhD thesis, 2006.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal, Optional, Tuple

import numpy as np

LearningRule = Literal["hebbian", "anti_hebbian", "random_walk"]


def _theta(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Heaviside-style indicator: 1 where a == b else 0 (for sign arrays)."""
    return (a == b).astype(np.int64)


@dataclass
class TPMConfig:
    K: int = 3            # hidden units
    N: int = 100          # inputs per hidden unit
    L: int = 4            # weight range [-L, L]
    rule: LearningRule = "hebbian"


class TreeParityMachine:
    """A single Tree Parity Machine.

    Weights are stored as an int matrix of shape ``(K, N)`` in ``[-L, L]``.
    """

    def __init__(self, cfg: TPMConfig, generator: Optional[np.random.Generator] = None):
        self.cfg = cfg
        self.gen = generator if generator is not None else np.random.default_rng()
        self.W = self.gen.integers(-cfg.L, cfg.L + 1, size=(cfg.K, cfg.N), dtype=np.int64)
        # Cache of the most recent forward pass (used by attackers).
        self.sigma: Optional[np.ndarray] = None   # (K,) in {-1,+1}
        self.h: Optional[np.ndarray] = None        # (K,) local fields
        self.tau: Optional[int] = None             # scalar in {-1,+1}

    # ------------------------------------------------------------------ #
    # Forward pass
    # ------------------------------------------------------------------ #
    def forward(self, x: np.ndarray) -> int:
        """Compute hidden outputs and parity output for input ``x`` (K, N)."""
        # Local field (scaled by sqrt(N) only matters for the geometric attack
        # ordering, which is scale-invariant, so we keep the raw sum here).
        self.h = np.einsum("ij,ij->i", self.W, x).astype(np.float64)
        sigma = np.where(self.h >= 0, 1, -1).astype(np.int64)  # sign(0) := +1
        self.sigma = sigma
        self.tau = int(np.prod(sigma))
        return self.tau

    # ------------------------------------------------------------------ #
    # Weight update
    # ------------------------------------------------------------------ #
    def update(self, x: np.ndarray, tau_other: int) -> bool:
        """Apply the learning rule given the partner's output ``tau_other``.

        Returns True if an update was applied (i.e. outputs agreed).
        """
        if self.tau is None or self.sigma is None:
            raise RuntimeError("forward() must be called before update()")
        if self.tau != tau_other:
            return False  # learning only on output agreement
        self._apply_rule(x, self.sigma, self.tau)
        return True

    def _apply_rule(self, x: np.ndarray, sigma: np.ndarray, tau: int) -> None:
        # Only hidden units whose output equals the network output learn.
        mask = _theta(sigma, np.full_like(sigma, tau))[:, None]  # (K,1)
        if self.cfg.rule == "hebbian":
            delta = x * sigma[:, None]
        elif self.cfg.rule == "anti_hebbian":
            delta = -x * sigma[:, None]
        elif self.cfg.rule == "random_walk":
            delta = x.copy()
        else:  # pragma: no cover
            raise ValueError(f"unknown rule {self.cfg.rule}")
        self.W = np.clip(self.W + mask * delta, -self.cfg.L, self.cfg.L)

    # ------------------------------------------------------------------ #
    # Serialisation / key material
    # ------------------------------------------------------------------ #
    def weight_bytes(self) -> bytes:
        """Deterministic byte serialisation of the weight matrix."""
        # Shift to non-negative range so a single unsigned byte suffices for L<128.
        shifted = (self.W + self.cfg.L).astype(np.uint8)
        return shifted.tobytes()

    def copy(self) -> "TreeParityMachine":
        clone = TreeParityMachine.__new__(TreeParityMachine)
        clone.cfg = self.cfg
        clone.gen = self.gen
        clone.W = self.W.copy()
        clone.sigma = None
        clone.h = None
        clone.tau = None
        return clone


# ---------------------------------------------------------------------------- #
# Overlap / distance helpers
# ---------------------------------------------------------------------------- #
def weight_overlap(a: TreeParityMachine, b: TreeParityMachine) -> float:
    """Mean per-hidden-unit cosine overlap rho in [-1, 1] (1.0 == identical)."""
    wa, wb = a.W.astype(np.float64), b.W.astype(np.float64)
    num = np.einsum("ij,ij->i", wa, wb)
    den = np.linalg.norm(wa, axis=1) * np.linalg.norm(wb, axis=1) + 1e-12
    return float(np.mean(num / den))


def hamming_fraction(a: TreeParityMachine, b: TreeParityMachine) -> float:
    """Fraction of weights that differ (0.0 == fully synchronised)."""
    return float(np.mean(a.W != b.W))


def random_input(cfg: TPMConfig, generator: np.random.Generator) -> np.ndarray:
    """Common public input x in {-1,+1}^{K x N}."""
    return generator.integers(0, 2, size=(cfg.K, cfg.N), dtype=np.int64) * 2 - 1
