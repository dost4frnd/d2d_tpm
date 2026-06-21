"""Reliability and latency analysis for multi-carrier bifurcation.

We compare the availability of three transport strategies under independent
per-PDU loss ``p`` and a share of ``n`` PDUs:

* single carrier          : A = (1-p)^n
* (2,2) bifurcation        : A = ((1-p)^n)^2          (both shares required)
* (2,3) / duplicated share : A = 1 - (1 - (1-p)^n)^2  (any 2 of 2 legs, modelled
                              here as a duplicated leg for availability)

The (2,2) scheme trades availability for confidentiality; this module quantifies
that trade-off both analytically and by Monte Carlo, and is used to justify the
limitations discussion in the paper.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import numpy as np

from .carriers import CarrierModel, PDCPSplitBearer, default_carriers


def share_delivery_prob(p: float, n_pdus: int) -> float:
    return (1.0 - p) ** n_pdus


def availability_single(p: float, n_pdus: int) -> float:
    return share_delivery_prob(p, n_pdus)


def availability_22(p: float, n_pdus: int) -> float:
    return share_delivery_prob(p, n_pdus) ** 2


def availability_dup(p: float, n_pdus: int) -> float:
    a = share_delivery_prob(p, n_pdus)
    return 1.0 - (1.0 - a) ** 2


def _binom(n: int, k: int) -> float:
    from math import comb
    return float(comb(n, k))


def availability_t_of_n(p: float, n_pdus: int, T: int, N: int) -> float:
    """Availability of a (T,N) threshold scheme: each of N shares rides an
    independent leg delivered with prob q=(1-p)^n_pdus; reconstruction needs >=T.

    A = sum_{k=T}^{N} C(N,k) q^k (1-q)^(N-k).  Recovers single (1,1), (2,2)=q^2,
    and N-way duplication (1,N).
    """
    if not (1 <= T <= N):
        raise ValueError("require 1 <= T <= N")
    q = share_delivery_prob(p, n_pdus)
    return float(sum(_binom(N, k) * (q ** k) * ((1.0 - q) ** (N - k))
                     for k in range(T, N + 1)))


@dataclass
class ReliabilityCurve:
    loss_probs: List[float]
    single: List[float]
    bifurcation_22: List[float]
    duplicated: List[float]


def reliability_curve(n_pdus: int = 8,
                      loss_probs: Optional[List[float]] = None) -> ReliabilityCurve:
    if loss_probs is None:
        loss_probs = list(np.linspace(0.0, 0.2, 21))
    return ReliabilityCurve(
        loss_probs=[float(p) for p in loss_probs],
        single=[availability_single(p, n_pdus) for p in loss_probs],
        bifurcation_22=[availability_22(p, n_pdus) for p in loss_probs],
        duplicated=[availability_dup(p, n_pdus) for p in loss_probs],
    )


def monte_carlo_availability(payload_size: int = 8192, trials: int = 2000,
                             carriers: Optional[List[CarrierModel]] = None,
                             mtu: int = 1024, seed: int = 0) -> dict:
    """Empirically estimate (2,2) availability and latency for a configuration."""
    gen = np.random.default_rng(seed)
    bearer = PDCPSplitBearer(carriers or default_carriers(), mtu=mtu)
    payload = gen.integers(0, 256, size=payload_size, dtype=np.uint8).tobytes()
    successes, latencies = 0, []
    for _ in range(trials):
        tm = bearer.transmit(payload, gen)
        successes += int(tm.reconstruction_success)
        latencies.append(tm.e2e_latency_ms)
    lat = np.asarray(latencies)
    return {
        "availability": successes / trials,
        "mean_latency_ms": float(lat.mean()),
        "p95_latency_ms": float(np.percentile(lat, 95)),
        "p99_latency_ms": float(np.percentile(lat, 99)),
        "trials": trials,
    }
