"""Carrier channel models and a simulated PDCP split-bearer.

We model a 5G *split bearer* (3GPP TS 38.323 PDCP) at a behavioural level: a
payload is bifurcated into two XOR shares, each share is segmented into PDCP PDUs,
and each share is mapped to a distinct component carrier / RLC leg.  Per-PDU
latency, jitter and independent loss are sampled from per-carrier models; the
payload reconstructs only if *both* shares are delivered (a property of the (2,2)
scheme).

This is a *simulation* of the transport behaviour, not a bit-accurate 5G stack.
Quantities (latency, loss, jitter, throughput, reconstruction failures) are
exported as telemetry features for the anomaly detector.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np

from . import secret_sharing


@dataclass
class CarrierModel:
    name: str
    base_latency_ms: float = 8.0      # one-way propagation + processing
    jitter_ms: float = 2.0           # std of per-PDU latency
    loss_prob: float = 0.01          # independent per-PDU loss probability
    throughput_mbps: float = 100.0   # effective goodput


@dataclass
class TransmitTelemetry:
    reconstruction_success: bool
    reconstruction_failures: int
    e2e_latency_ms: float
    per_carrier_latency_ms: List[float]
    per_carrier_jitter_ms: List[float]
    per_carrier_loss: List[float]
    per_carrier_throughput_mbps: List[float]
    n_pdus: List[int]


def _send_share(share: bytes, carrier: CarrierModel, mtu: int,
                gen: np.random.Generator) -> tuple[bool, float, float, float]:
    """Transmit one share over a carrier. Returns
    (delivered, mean_latency_ms, jitter_ms, observed_loss)."""
    n_pdus = max(1, math.ceil(len(share) / mtu))
    pdu_bits = mtu * 8
    tx_ms = (pdu_bits / (carrier.throughput_mbps * 1e6)) * 1e3  # per-PDU serialisation
    latencies = []
    delivered_pdus = 0
    for _ in range(n_pdus):
        lat = carrier.base_latency_ms + tx_ms + gen.normal(0.0, carrier.jitter_ms)
        latencies.append(max(0.1, lat))
        if gen.random() > carrier.loss_prob:
            delivered_pdus += 1
    latencies = np.asarray(latencies)
    observed_loss = 1.0 - delivered_pdus / n_pdus
    delivered = delivered_pdus == n_pdus  # no FEC: all PDUs needed
    # share latency dominated by the slowest PDU plus reassembly within the leg
    return delivered, float(latencies.mean()), float(latencies.std()), observed_loss


class PDCPSplitBearer:
    """Behavioural split-bearer over two component carriers."""

    def __init__(self, carriers: List[CarrierModel], mtu: int = 1024,
                 reassembly_overhead_ms: float = 1.0):
        if len(carriers) != 2:
            raise ValueError("PDCP (2,2) split bearer requires exactly two carriers")
        self.carriers = carriers
        self.mtu = mtu
        self.reassembly_overhead_ms = reassembly_overhead_ms

    def transmit(self, payload: bytes, generator: Optional[np.random.Generator] = None
                 ) -> TransmitTelemetry:
        gen = generator if generator is not None else np.random.default_rng()
        s1, s2 = secret_sharing.split(payload, gen)
        shares = [s1, s2]

        delivered_flags, lat_means, jitters, losses, n_pdus_list, tputs = (
            [], [], [], [], [], []
        )
        for share, carrier in zip(shares, self.carriers):
            delivered, lat_mean, jitter, loss = _send_share(share, carrier, self.mtu, gen)
            delivered_flags.append(delivered)
            lat_means.append(lat_mean)
            jitters.append(jitter)
            losses.append(loss)
            n_pdus_list.append(max(1, math.ceil(len(share) / self.mtu)))
            # effective throughput degraded by observed loss (retx-free model)
            tputs.append(carrier.throughput_mbps * (1.0 - loss))

        success = all(delivered_flags)
        if success:
            recon = secret_sharing.reconstruct(s1, s2)
            success = recon == payload
        # end-to-end latency: parallel legs -> max, plus reassembly
        e2e = max(lat_means) + self.reassembly_overhead_ms
        return TransmitTelemetry(
            reconstruction_success=success,
            reconstruction_failures=0 if success else 1,
            e2e_latency_ms=e2e,
            per_carrier_latency_ms=lat_means,
            per_carrier_jitter_ms=jitters,
            per_carrier_loss=losses,
            per_carrier_throughput_mbps=tputs,
            n_pdus=n_pdus_list,
        )


def default_carriers() -> List[CarrierModel]:
    return [
        CarrierModel("CC0_n78", base_latency_ms=7.0, jitter_ms=1.5,
                     loss_prob=0.008, throughput_mbps=150.0),
        CarrierModel("CC1_n258", base_latency_ms=10.0, jitter_ms=3.0,
                     loss_prob=0.015, throughput_mbps=220.0),
    ]
