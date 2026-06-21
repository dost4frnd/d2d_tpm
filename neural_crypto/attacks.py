"""Eavesdropper attacks against TPM key exchange.

Implemented attacks
-------------------
* ``GeometricAttacker`` - a single attacker network that, when its output
  disagrees with the public output, flips the hidden unit with the smallest
  absolute local field before learning (the classic geometric attack).
* ``MajorityAttacker`` - an ensemble of geometric attackers ("majority/ensemble"
  attack family).  The attacker's key guess is taken from the ensemble; this is
  stronger than a single geometric attacker and motivates early termination.

The attacker observes only the *public* channel (the random input ``x`` and the
agreed output on learning steps) - never the partners' weights.

References
----------
A. Klimov, A. Mityagin, A. Shamir, "Analysis of neural cryptography,"
ASIACRYPT 2002.
A. Ruttor, "Neural Synchronization and Cryptography," PhD thesis, 2006.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import numpy as np

from .tpm import TPMConfig, TreeParityMachine, weight_overlap


def _geometric_correct(net: TreeParityMachine, tau_public: int) -> None:
    """Force ``net`` to output ``tau_public`` by flipping its least-confident unit."""
    if net.tau == tau_public:
        return
    j = int(np.argmin(np.abs(net.h)))
    net.sigma[j] *= -1
    net.tau = int(np.prod(net.sigma))


class GeometricAttacker:
    def __init__(self, cfg: TPMConfig, generator: Optional[np.random.Generator] = None):
        self.cfg = cfg
        self.gen = generator if generator is not None else np.random.default_rng()
        self.E = TreeParityMachine(cfg, self.gen)

    def step(self, x: np.ndarray, tau_public: int) -> None:
        """Called only on partner learning steps (when tau_A == tau_B)."""
        self.E.forward(x)
        _geometric_correct(self.E, tau_public)
        self.E.update(x, tau_public)  # tau now matches -> rule is applied

    def overlap_with(self, target: TreeParityMachine) -> float:
        return weight_overlap(self.E, target)

    def best_net(self) -> TreeParityMachine:
        return self.E


class MajorityAttacker:
    def __init__(self, cfg: TPMConfig, n_nets: int = 100,
                 generator: Optional[np.random.Generator] = None):
        self.cfg = cfg
        self.gen = generator if generator is not None else np.random.default_rng()
        self.nets: List[TreeParityMachine] = [
            TreeParityMachine(cfg, self.gen) for _ in range(n_nets)
        ]

    def step(self, x: np.ndarray, tau_public: int) -> None:
        for net in self.nets:
            net.forward(x)
            _geometric_correct(net, tau_public)
            net.update(x, tau_public)

    def majority_weights(self) -> np.ndarray:
        """Per-weight majority estimate (sign of the summed ensemble weights)."""
        stack = np.stack([n.W for n in self.nets], axis=0).astype(np.float64)
        return np.sign(stack.sum(axis=0)).astype(np.int64)

    def overlap_with(self, target: TreeParityMachine) -> float:
        """Operational overlap: the *majority-vote* key guess the attacker would
        actually output (not an optimistic best-of-N). See ``best_overlap_with``
        for the optimistic upper bound."""
        return self.majority_overlap_with(target)

    def best_overlap_with(self, target: TreeParityMachine) -> float:
        """Optimistic upper bound: best single-net overlap (diagnostic only)."""
        return max(weight_overlap(n, target) for n in self.nets)

    def majority_overlap_with(self, target: TreeParityMachine) -> float:
        maj = self.majority_weights().astype(np.float64)
        wt = target.W.astype(np.float64)
        num = np.einsum("ij,ij->i", maj, wt)
        den = (np.linalg.norm(maj, axis=1) * np.linalg.norm(wt, axis=1)) + 1e-12
        return float(np.mean(num / den))

    def best_net(self) -> TreeParityMachine:
        # Return the net closest to its own majority estimate as a representative.
        maj = self.majority_weights()
        scores = [float(np.mean(n.W == maj)) for n in self.nets]
        return self.nets[int(np.argmax(scores))]


@dataclass
class AttackConfig:
    kind: str = "none"          # none | geometric | majority
    n_nets: int = 100           # only used by the majority attack


def build_attacker(attack_cfg: AttackConfig, tpm_cfg: TPMConfig,
                   generator: Optional[np.random.Generator] = None):
    if attack_cfg.kind == "geometric":
        return GeometricAttacker(tpm_cfg, generator)
    if attack_cfg.kind == "majority":
        return MajorityAttacker(tpm_cfg, attack_cfg.n_nets, generator)
    return None
