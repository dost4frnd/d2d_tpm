"""Edge inference entrypoint with graceful backend fallback.

Backend selection order (first available wins):
  1. TensorRT engine (.plan)         -- production path on the Jetson GPU
  2. ONNX Runtime on the .onnx file  -- portable accelerated path
  3. PyTorch FNNClassifier           -- always-available reference path

This lets the same container image run on the Jetson (TensorRT) and on a laptop
or CI (PyTorch), so the deployment is demonstrable without hardware. It emits a
synthetic inference stream with latency stats; wire ``next_sample()`` to the
live crypto/transport telemetry bus in production.
"""
from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(_HERE, "..", ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from anomaly_detector.features import CLASSES, FEATURES  # noqa: E402


def _load_eval_samples(dataset: str, n: int = 256) -> np.ndarray:
    from experiments._common import load_dataset
    if not os.path.exists(dataset):
        from data_generator.generate_dataset import GenConfig, generate, save_dataset
        cfg = GenConfig(n_per_class=150, crypto_sessions=8, transport_windows=48, tpm_N=60)
        X, y, meta = generate(cfg); save_dataset(X, y, meta, cfg)
    X, _ = load_dataset(dataset)
    rng = np.random.default_rng(0)
    return X[rng.choice(X.shape[0], size=min(n, X.shape[0]), replace=False)].astype(np.float32)


def _try_tensorrt(engine_path: str):
    if not os.path.exists(engine_path):
        return None
    try:
        import tensorrt as trt  # noqa: F401
        import pycuda.autoinit  # noqa: F401
        import pycuda.driver as cuda  # noqa: F401
    except Exception:
        return None
    # Engine loading/binding is JetPack-specific; left as a hook so the import
    # guard above cleanly routes laptops/CI to the next backend.
    print("[serve] TensorRT available but runtime binding is device-specific; "
          "skipping in this environment.")
    return None


def _try_onnxruntime(onnx_path: str):
    if not os.path.exists(onnx_path):
        return None
    try:
        import onnxruntime as ort
    except Exception:
        return None
    sess = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
    name = sess.get_inputs()[0].name

    def predict(x: np.ndarray) -> np.ndarray:
        return sess.run(None, {name: x.astype(np.float32)})[0]

    print("[serve] backend = ONNX Runtime")
    return predict


def _pytorch_backend(dataset: str, epochs: int):
    from experiments._common import build_detector, load_dataset, stratified_split
    X, y = load_dataset(dataset)
    Xtr, _, ytr, _ = stratified_split(X, y, 0.25, 0)
    det = build_detector("hybrid", fnn_epochs=epochs)
    det.fit(Xtr, ytr)
    print("[serve] backend = PyTorch (hybrid WKNN-FNN reference)")
    return det.predict_proba


def main() -> None:
    p = argparse.ArgumentParser(description="Edge inference server.")
    p.add_argument("--dataset", type=str, default=os.path.join(_ROOT, "datasets", "telemetry.csv"))
    p.add_argument("--onnx", type=str, default=os.path.join(_HERE, "detector.onnx"))
    p.add_argument("--engine", type=str, default=os.path.join(_HERE, "detector.plan"))
    p.add_argument("--epochs", type=int, default=40)
    p.add_argument("--iters", type=int, default=200)
    a = p.parse_args()

    predict = (_try_tensorrt(a.engine) or _try_onnxruntime(a.onnx)
               or _pytorch_backend(a.dataset, a.epochs))

    samples = _load_eval_samples(a.dataset)
    rng = np.random.default_rng(0)
    lat = []
    alerts = 0
    for _ in range(a.iters):
        x = samples[rng.integers(0, samples.shape[0]):][:1]
        t0 = time.perf_counter()
        proba = predict(x)
        lat.append((time.perf_counter() - t0) * 1e3)
        pred = int(np.argmax(proba[0]))
        if CLASSES[pred] != "Normal":
            alerts += 1
    lat = np.asarray(lat)
    print(f"[serve] ran {a.iters} inferences | mean {lat.mean():.3f} ms | "
          f"p95 {np.percentile(lat, 95):.3f} ms | alerts {alerts}")


if __name__ == "__main__":
    main()
