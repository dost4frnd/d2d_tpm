"""Fast smoke tests covering the core invariants of each subsystem.

Run with:  pytest -q     (or: python -m pytest tests/ -q)
These are intentionally quick (small sizes) so they can gate CI.
"""
from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from bifurcation import reliability, secret_sharing  # noqa: E402
from bifurcation.carriers import PDCPSplitBearer, default_carriers  # noqa: E402
from common.utils import equality_tag, hkdf_sha256  # noqa: E402
from neural_crypto.attacks import AttackConfig  # noqa: E402
from neural_crypto.sync import SyncConfig, run_session  # noqa: E402
from neural_crypto.tpm import TPMConfig  # noqa: E402


def test_xor_secret_sharing_roundtrip():
    rng = np.random.default_rng(0)
    payload = b"post-quantum-edge-payload-1234567890"
    s1, s2 = secret_sharing.split(payload, rng)
    assert s1 != payload and s2 != payload          # each share hides the payload
    assert secret_sharing.reconstruct(s1, s2) == payload


def test_hkdf_and_equality_tag():
    key = hkdf_sha256(b"weights-bytes", length=32)
    assert len(key) == 32
    nonce = b"\x01" * 16
    assert equality_tag(b"abc", nonce) == equality_tag(b"abc", nonce)
    assert equality_tag(b"abc", nonce) != equality_tag(b"abd", nonce)


def test_tpm_parties_synchronize_and_agree_key():
    cfg = SyncConfig(tpm=TPMConfig(K=3, N=40, L=4), policy="early_confidence")
    r = run_session(cfg, AttackConfig(kind="none"))
    assert r.synchronized
    assert r.key_agreement
    assert 0 < r.total_agreements <= r.rounds


def test_early_termination_bounds_verifications():
    tpm = TPMConfig(K=3, N=40, L=4)
    base = run_session(SyncConfig(tpm=tpm, policy="hash_only"), AttackConfig(kind="none"))
    early = run_session(SyncConfig(tpm=tpm, policy="early_confidence"), AttackConfig(kind="none"))
    # rate-limited early termination must not explode verification count
    assert early.num_verifications <= base.num_verifications + 2


def test_single_geometric_attacker_does_not_trivially_win_every_time():
    # Over a few seeds, the single geometric attacker should NOT succeed always
    # at K=3 (it is a real but probabilistic threat).
    wins = 0
    for s in range(5):
        cfg = SyncConfig(tpm=TPMConfig(K=3, N=40, L=4), policy="early_confidence",
                         seed_a=s + 1, seed_b=s + 2, seed_e=s + 3, seed_input=s + 100)
        wins += int(run_session(cfg, AttackConfig(kind="geometric")).attacker_success)
    assert wins < 5


def test_bifurcation_availability_ordering():
    p, n = 0.05, 8
    a_single = reliability.availability_single(p, n)
    a_22 = reliability.availability_22(p, n)
    a_dup = reliability.availability_dup(p, n)
    # (2,2) is strictly less available than a single carrier; duplication is more.
    assert a_22 < a_single < a_dup


def test_split_bearer_transmit_roundtrip():
    bearer = PDCPSplitBearer(default_carriers())
    tm = bearer.transmit(b"\x00" * 2048, np.random.default_rng(0))
    assert len(tm.per_carrier_latency_ms) == 2
    assert tm.e2e_latency_ms > 0


def test_dataset_generator_shapes_and_balance():
    from anomaly_detector.features import N_CLASSES, N_FEATURES
    from data_generator.generate_dataset import GenConfig, generate

    cfg = GenConfig(n_per_class=60, crypto_sessions=6, transport_windows=24, tpm_N=40)
    X, y, meta = generate(cfg)
    assert X.shape == (60 * N_CLASSES, N_FEATURES)
    assert set(np.unique(y)) == set(range(N_CLASSES))
    assert np.isfinite(X).all()


def test_detector_beats_chance():
    from anomaly_detector.evaluate import compute_metrics
    from data_generator.generate_dataset import GenConfig, generate
    from experiments._common import build_detector, stratified_split

    cfg = GenConfig(n_per_class=200, crypto_sessions=8, transport_windows=32, tpm_N=40)
    X, y, _ = generate(cfg)
    Xtr, Xte, ytr, yte = stratified_split(X, y, 0.25, 0)
    det = build_detector("hybrid", fnn_epochs=30)
    det.fit(Xtr, ytr)
    m = compute_metrics(yte, det.predict_proba(Xte).argmax(1))
    assert m["macro_f1"] > 0.6          # well above 1/6 chance, below suspicious 1.0
    assert m["macro_f1"] < 0.999


# ---------------------- post-quantum (ML-KEM) + hybrid KDF ------------------
import pytest  # noqa: E402

from pqc.hybrid_kdf import derive_hybrid_key  # noqa: E402
from pqc.mlkem import MLKEM, PARAM_SIZES, available  # noqa: E402


@pytest.mark.skipif(not available(), reason="kyber-py not installed")
def test_mlkem_roundtrip_and_sizes():
    kem = MLKEM("ML-KEM-768")
    r = kem.run_once()
    assert r.agree                                   # decaps reproduces the secret
    assert len(r.shared) == 32
    assert len(r.ek) == PARAM_SIZES["ML-KEM-768"]["ek"]   # standardized size
    assert len(r.ct) == PARAM_SIZES["ML-KEM-768"]["ct"]


@pytest.mark.skipif(not available(), reason="kyber-py not installed")
def test_hybrid_kdf_orders_and_separates():
    kem = MLKEM("ML-KEM-512")
    r = kem.run_once()
    tpm_secret = hkdf_sha256(b"weights", length=32)
    both = derive_hybrid_key(r.shared, tpm_secret, length=32)
    mlkem_only = derive_hybrid_key(r.shared, None, length=32)
    tpm_only = derive_hybrid_key(None, tpm_secret, length=32)
    assert len(both.key) == len(mlkem_only.key) == len(tpm_only.key) == 32
    # different component sets must yield different keys (domain separation)
    assert len({both.key, mlkem_only.key, tpm_only.key}) == 3
    assert both.mlkem_used and both.tpm_used


def test_hybrid_kdf_requires_at_least_one_secret():
    with pytest.raises(ValueError):
        derive_hybrid_key(None, None)


# ---------------------- tunable (T,N) threshold sharing ---------------------
from bifurcation import threshold as ts  # noqa: E402


def test_threshold_roundtrip_various_TN():
    payload = bytes(range(256)) * 2
    for T, N in [(2, 2), (2, 3), (3, 5), (4, 7), (1, 4)]:
        assert ts.verify_roundtrip(payload, T, N, np.random.default_rng(T * 10 + N))


def test_threshold_confidentiality_rejects_insufficient_shares():
    payload = b"a-secret-payload-of-some-length-..."
    shares = ts.split(payload, T=3, N=5, generator=np.random.default_rng(0))
    with pytest.raises(ValueError):
        ts.reconstruct(shares[:2], T=3)              # fewer than T shares
    # forcing interpolation with the wrong degree must not recover the payload
    assert ts.reconstruct(shares[:2], T=2) != payload


def test_threshold_availability_recovers_closed_forms():
    p, n = 0.05, 8
    assert round(reliability.availability_t_of_n(p, n, 1, 1), 6) == \
        round(reliability.availability_single(p, n), 6)
    assert round(reliability.availability_t_of_n(p, n, 2, 2), 6) == \
        round(reliability.availability_22(p, n), 6)
    assert round(reliability.availability_t_of_n(p, n, 1, 2), 6) == \
        round(reliability.availability_dup(p, n), 6)
    # monotonicity: higher threshold (more confidential) => lower availability
    avs = [reliability.availability_t_of_n(p, n, T, 5) for T in range(1, 6)]
    assert all(avs[i] >= avs[i + 1] for i in range(len(avs) - 1))
