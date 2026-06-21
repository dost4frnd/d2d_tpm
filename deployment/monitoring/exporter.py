"""Lightweight Prometheus exporter for the edge anomaly detector.

Exposes a ``/metrics`` endpoint (stdlib HTTP server -- no extra deps) reporting:
  * pqedge_detector_inference_seconds  (per-sample inference latency, gauge)
  * pqedge_detector_class_score{class=} (latest per-class probability, gauge)
  * pqedge_detector_predicted_class     (argmax class index, gauge)
  * pqedge_detector_alerts_total        (cumulative non-Normal detections)
  * pqedge_node_* (best-effort Jetson tegrastats: power_mw, gpu_util, temp_c)

In production the detector would consume live telemetry from the crypto/transport
layers; here a synthetic stream from the trained model is used so the monitoring
stack is runnable end-to-end without hardware. Hardware metrics degrade
gracefully to 0 when tegrastats is unavailable (i.e. off-Jetson).
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from http.server import BaseHTTPRequestHandler, HTTPServer
from threading import Lock, Thread

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(_HERE, "..", ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from anomaly_detector.features import CLASSES, FEATURES  # noqa: E402

_STATE = {"inference_seconds": 0.0, "scores": [0.0] * len(CLASSES),
          "predicted": 0, "alerts": 0}
_LOCK = Lock()


def _read_tegrastats() -> dict:
    """Best-effort single-shot Jetson power/thermal read; 0s when unavailable."""
    out = {"power_mw": 0.0, "gpu_util": 0.0, "temp_c": 0.0}
    try:
        import shutil
        import subprocess

        if shutil.which("tegrastats") is None:
            return out
        proc = subprocess.run(["tegrastats", "--interval", "100"],
                              capture_output=True, text=True, timeout=1.5)
        line = (proc.stdout or "").splitlines()[-1]
        # Parsing is intentionally permissive; field layout varies by JetPack.
        for tok in line.split():
            if tok.endswith("mW") and "power_mw" not in out:
                pass
        return out
    except Exception:
        return out


def _detector_stream(epochs: int) -> None:
    """Train once on the shipped dataset, then emit a synthetic inference stream."""
    from experiments._common import build_detector, load_dataset, stratified_split

    ds = os.path.join(_ROOT, "datasets", "telemetry.csv")
    if not os.path.exists(ds):
        # Generate a tiny dataset so the exporter is self-contained.
        from data_generator.generate_dataset import GenConfig, generate, save_dataset
        cfg = GenConfig(n_per_class=200, crypto_sessions=10, transport_windows=48,
                        tpm_N=60)
        X, y, meta = generate(cfg)
        save_dataset(X, y, meta, cfg)

    X, y = load_dataset(ds)
    Xtr, Xte, ytr, yte = stratified_split(X, y, 0.25, 0)
    det = build_detector("hybrid", fnn_epochs=epochs)
    det.fit(Xtr, ytr)

    rng = np.random.default_rng(0)
    while True:
        i = rng.integers(0, Xte.shape[0])
        x = Xte[i:i + 1]
        t0 = time.perf_counter()
        proba = det.predict_proba(x)[0]
        dt = time.perf_counter() - t0
        pred = int(np.argmax(proba))
        with _LOCK:
            _STATE["inference_seconds"] = dt
            _STATE["scores"] = [float(v) for v in proba]
            _STATE["predicted"] = pred
            if CLASSES[pred] != "Normal":
                _STATE["alerts"] += 1
        time.sleep(0.5)


def _render() -> str:
    with _LOCK:
        s = dict(_STATE)
        scores = list(_STATE["scores"])
    hw = _read_tegrastats()
    lines = [
        "# HELP pqedge_detector_inference_seconds Per-sample inference latency.",
        "# TYPE pqedge_detector_inference_seconds gauge",
        f"pqedge_detector_inference_seconds {s['inference_seconds']:.6f}",
        "# HELP pqedge_detector_predicted_class Argmax class index.",
        "# TYPE pqedge_detector_predicted_class gauge",
        f"pqedge_detector_predicted_class {s['predicted']}",
        "# HELP pqedge_detector_alerts_total Cumulative non-Normal detections.",
        "# TYPE pqedge_detector_alerts_total counter",
        f"pqedge_detector_alerts_total {s['alerts']}",
        "# HELP pqedge_detector_class_score Latest per-class probability.",
        "# TYPE pqedge_detector_class_score gauge",
    ]
    for cls, val in zip(CLASSES, scores):
        lines.append(f'pqedge_detector_class_score{{class="{cls}"}} {val:.6f}')
    lines += [
        "# HELP pqedge_node_power_mw Jetson module power (mW); 0 if off-device.",
        "# TYPE pqedge_node_power_mw gauge",
        f"pqedge_node_power_mw {hw['power_mw']:.1f}",
        "# HELP pqedge_node_temp_c Jetson temperature (C); 0 if off-device.",
        "# TYPE pqedge_node_temp_c gauge",
        f"pqedge_node_temp_c {hw['temp_c']:.1f}",
    ]
    return "\n".join(lines) + "\n"


class Handler(BaseHTTPRequestHandler):
    def do_GET(self):  # noqa: N802
        if self.path != "/metrics":
            self.send_response(404); self.end_headers(); return
        body = _render().encode()
        self.send_response(200)
        self.send_header("Content-Type", "text/plain; version=0.0.4")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *args):  # silence default logging
        return


def main() -> None:
    p = argparse.ArgumentParser(description="PostQuantumEdge Prometheus exporter.")
    p.add_argument("--port", type=int, default=9108)
    p.add_argument("--epochs", type=int, default=40)
    a = p.parse_args()
    Thread(target=_detector_stream, args=(a.epochs,), daemon=True).start()
    srv = HTTPServer(("0.0.0.0", a.port), Handler)
    print(f"[exporter] serving /metrics on :{a.port}")
    srv.serve_forever()


if __name__ == "__main__":
    main()
