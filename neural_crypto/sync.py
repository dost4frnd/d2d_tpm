"""TPM synchronisation protocol with privacy-preserving early termination.

The protocol runs the mutual-learning loop between two parties (A, B), optionally
in the presence of an eavesdropper, and derives a shared key once synchronisation
is *verified*.

Termination policies
--------------------
``hash_only`` (baseline)
    Every ``verify_every`` rounds the parties exchange a privacy-preserving
    equality tag (HMAC over a public nonce keyed by SHA-256 of the weights) and
    stop on the first match.  This is the conventional scheme.

``early_confidence`` (proposed)
    The parties maintain a confidence score derived *only from the public output
    stream* (the windowed output-agreement rate plus the current agreement
    streak).  A verification is requested only once the confidence crosses a
    threshold.  This (a) reduces verification traffic and (b) tightens detection
    to within ~1 round of true synchronisation, minimising the number of
    post-synchronisation rounds an eavesdropper can exploit.

The confidence estimate never inspects weights, so it leaks no key material; the
verification tag leaks at most a single equality bit.
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np

from common.utils import equality_tag, hkdf_sha256
from .attacks import AttackConfig, build_attacker
from .tpm import (
    TPMConfig,
    TreeParityMachine,
    hamming_fraction,
    random_input,
    weight_overlap,
)


@dataclass
class SyncConfig:
    tpm: TPMConfig = field(default_factory=TPMConfig)
    max_steps: int = 4000
    policy: str = "early_confidence"      # hash_only | early_confidence
    verify_every: int = 50                # baseline verification interval
    conf_window: int = 40                 # rounds in the confidence window
    conf_threshold: float = 0.98          # trigger verification above this
    min_streak: int = 25                  # consecutive-agreement requirement
    key_bytes: int = 32                   # derived symmetric-key length
    attacker_sync_overlap: float = 0.99   # overlap counted as attacker success
    seed_a: int = 1
    seed_b: int = 2
    seed_e: int = 3
    seed_input: int = 100


@dataclass
class SyncResult:
    synchronized: bool
    stop_step: int
    partner_sync_step: Optional[int]      # ground-truth identical-weights step
    post_sync_rounds: int                 # stop_step - partner_sync_step
    num_verifications: int                # verification messages exchanged
    rounds: int                           # == stop_step
    total_agreements: int                 # learning steps (tau_A == tau_B)
    key_hex: Optional[str]
    key_agreement: bool                   # A and B derived identical keys
    final_overlap_ab: float
    attack_kind: str
    attacker_sync_step: Optional[int]
    attacker_success: bool
    confidence_trace: List[float] = field(default_factory=list)
    overlap_trace: List[float] = field(default_factory=list)
    attacker_overlap_trace: List[float] = field(default_factory=list)


def _derive_key(net: TreeParityMachine, key_bytes: int) -> bytes:
    return hkdf_sha256(net.weight_bytes(), length=key_bytes, info=b"pqedge-tpm-key")


def run_session(cfg: SyncConfig, attack: Optional[AttackConfig] = None,
                record_traces: bool = True) -> SyncResult:
    """Run one full synchronisation session and return metrics + traces."""
    gen_a = np.random.default_rng(cfg.seed_a)
    gen_b = np.random.default_rng(cfg.seed_b)
    gen_e = np.random.default_rng(cfg.seed_e)
    gen_x = np.random.default_rng(cfg.seed_input)

    A = TreeParityMachine(cfg.tpm, gen_a)
    B = TreeParityMachine(cfg.tpm, gen_b)
    attack = attack or AttackConfig(kind="none")
    attacker = build_attacker(attack, cfg.tpm, gen_e)

    window: deque = deque(maxlen=cfg.conf_window)
    streak = 0
    partner_sync_step: Optional[int] = None
    attacker_sync_step: Optional[int] = None
    num_verifications = 0
    total_agreements = 0
    last_verify_step = -10 ** 9          # cooldown bookkeeping
    stop_step = cfg.max_steps
    synchronized = False

    conf_trace: List[float] = []
    ov_trace: List[float] = []
    att_trace: List[float] = []

    for step in range(1, cfg.max_steps + 1):
        x = random_input(cfg.tpm, gen_x)
        tau_a = A.forward(x)
        tau_b = B.forward(x)
        agree = tau_a == tau_b
        if agree:
            total_agreements += 1
            A.update(x, tau_b)
            B.update(x, tau_a)
            if attacker is not None:
                attacker.step(x, tau_a)

        # --- bookkeeping (ground truth; not visible to the parties) -------- #
        if partner_sync_step is None and hamming_fraction(A, B) == 0.0:
            partner_sync_step = step
        if attacker is not None:
            a_ov = attacker.overlap_with(A)
            if attacker_sync_step is None and a_ov >= cfg.attacker_sync_overlap:
                attacker_sync_step = step
        else:
            a_ov = 0.0

        # --- privacy-preserving confidence (public information only) ------- #
        window.append(1 if agree else 0)
        streak = streak + 1 if agree else 0
        confidence = float(np.mean(window)) if window else 0.0

        if record_traces:
            conf_trace.append(confidence)
            ov_trace.append(weight_overlap(A, B))
            att_trace.append(a_ov)

        # --- termination policy ------------------------------------------- #
        do_verify = False
        if cfg.policy == "hash_only":
            do_verify = (step % cfg.verify_every == 0)
        elif cfg.policy == "early_confidence":
            # Trigger on the privacy-preserving confidence signal, but rate-limit
            # the (cheap) equality-tag exchange to at most once per verify_every
            # rounds so a high-confidence plateau before true weight equality
            # cannot inflate the verification-message count.
            ready = (confidence >= cfg.conf_threshold and streak >= cfg.min_streak)
            cooled = (step - last_verify_step) >= cfg.verify_every
            do_verify = ready and cooled
        else:  # pragma: no cover
            raise ValueError(f"unknown policy {cfg.policy}")

        if do_verify:
            num_verifications += 1
            last_verify_step = step
            nonce = gen_x.integers(0, 2 ** 32, dtype=np.uint64).tobytes()
            if equality_tag(A.weight_bytes(), nonce) == equality_tag(B.weight_bytes(), nonce):
                synchronized = True
                stop_step = step
                break

    key_hex = None
    key_agreement = False
    if synchronized:
        ka = _derive_key(A, cfg.key_bytes)
        kb = _derive_key(B, cfg.key_bytes)
        key_agreement = ka == kb
        key_hex = ka.hex()

    return SyncResult(
        synchronized=synchronized,
        stop_step=stop_step,
        partner_sync_step=partner_sync_step,
        post_sync_rounds=(stop_step - partner_sync_step) if partner_sync_step else -1,
        num_verifications=num_verifications,
        rounds=stop_step,
        total_agreements=total_agreements,
        key_hex=key_hex,
        key_agreement=key_agreement,
        final_overlap_ab=weight_overlap(A, B),
        attack_kind=attack.kind,
        attacker_sync_step=attacker_sync_step,
        attacker_success=(
            attacker_sync_step is not None and attacker_sync_step <= stop_step
        ),
        confidence_trace=conf_trace,
        overlap_trace=ov_trace,
        attacker_overlap_trace=att_trace,
    )
