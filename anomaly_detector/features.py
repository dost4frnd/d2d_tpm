"""Feature and class schema for edge intrusion / anomaly detection.

Telemetry fuses cryptographic-synchronisation signals with transport signals so a
single detector can separate cryptographic attacks from network faults and from
combined (multi-stage) events.
"""
from __future__ import annotations

from typing import List

# Ordered feature vector (do not reorder: model artefacts depend on this order).
FEATURES: List[str] = [
    "tpm_convergence",              # final A-B overlap / convergence quality
    "sync_rate",                    # weight-update (learning) rate over the run
    "match_probability",           # P(tau_A == tau_B) observed
    "round_count",                 # rounds to terminate (normalised)
    "carrier_latency",             # mean end-to-end latency (ms)
    "packet_loss",                 # mean per-carrier loss
    "throughput",                  # mean effective throughput (Mbps)
    "jitter",                      # mean per-carrier jitter (ms)
    "share_reconstruction_failures",  # failed (2,2) reconstructions in window
    "flow_stats",                  # aggregate flow statistic (bytes/s, scaled)
]

CLASSES: List[str] = [
    "Normal",
    "GeometricAttack",
    "MajorityAttack",
    "TrafficAnomaly",
    "CarrierDisruption",
    "MultiStageAttack",
]

N_FEATURES = len(FEATURES)
N_CLASSES = len(CLASSES)
CLASS_TO_IDX = {c: i for i, c in enumerate(CLASSES)}
IDX_TO_CLASS = {i: c for i, c in enumerate(CLASSES)}
