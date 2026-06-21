"""Security metrics for TPM key exchange, aggregated over many sessions.

We run repeated sessions with distinct seeds and report distributions of the
synchronisation cost and the eavesdropper's success rate.  The latter is the key
honesty-preserving number: TPM key exchange is *not* unconditionally secure, and
this module quantifies exactly how often an attacker keeps up.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Dict, List, Optional

import numpy as np

from .attacks import AttackConfig
from .sync import SyncConfig, SyncResult, run_session


@dataclass
class SecuritySummary:
    n_sessions: int
    policy: str
    attack_kind: str
    sync_success_rate: float
    key_agreement_rate: float
    mean_rounds: float
    std_rounds: float
    median_rounds: float
    mean_post_sync_rounds: float
    mean_verifications: float
    attacker_success_rate: float
    mean_attacker_sync_step: Optional[float]

    def as_dict(self) -> Dict:
        return asdict(self)


def evaluate_security(base_cfg: SyncConfig, attack: AttackConfig,
                      n_sessions: int = 50, base_seed: int = 0,
                      record_traces: bool = False) -> SecuritySummary:
    rounds: List[int] = []
    post: List[int] = []
    verifs: List[int] = []
    sync_ok = 0
    key_ok = 0
    att_ok = 0
    att_steps: List[int] = []

    for i in range(n_sessions):
        cfg = SyncConfig(**{**base_cfg.__dict__})
        # decorrelate every session
        cfg.seed_a = base_seed + 1000 + i
        cfg.seed_b = base_seed + 2000 + i
        cfg.seed_e = base_seed + 3000 + i
        cfg.seed_input = base_seed + 4000 + i
        res: SyncResult = run_session(cfg, attack, record_traces=record_traces)
        if res.synchronized:
            sync_ok += 1
            rounds.append(res.rounds)
            if res.partner_sync_step:
                post.append(res.post_sync_rounds)
            verifs.append(res.num_verifications)
        if res.key_agreement:
            key_ok += 1
        if res.attacker_success:
            att_ok += 1
        if res.attacker_sync_step is not None:
            att_steps.append(res.attacker_sync_step)

    rounds_arr = np.array(rounds) if rounds else np.array([np.nan])
    return SecuritySummary(
        n_sessions=n_sessions,
        policy=base_cfg.policy,
        attack_kind=attack.kind,
        sync_success_rate=sync_ok / n_sessions,
        key_agreement_rate=key_ok / n_sessions,
        mean_rounds=float(np.nanmean(rounds_arr)),
        std_rounds=float(np.nanstd(rounds_arr)),
        median_rounds=float(np.nanmedian(rounds_arr)),
        mean_post_sync_rounds=float(np.mean(post)) if post else float("nan"),
        mean_verifications=float(np.mean(verifs)) if verifs else float("nan"),
        attacker_success_rate=att_ok / n_sessions,
        mean_attacker_sync_step=float(np.mean(att_steps)) if att_steps else None,
    )
