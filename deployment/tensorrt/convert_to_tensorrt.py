"""Export the trained FNN detector to ONNX and build a TensorRT engine.

PIPELINE:  PyTorch (FNNClassifier) --> ONNX --> TensorRT (.plan / .engine)

The ONNX export step runs anywhere (CPU). The TensorRT build step requires the
``tensorrt`` Python package and a CUDA GPU, so it is guarded: on a machine
without TensorRT this script still produces the ONNX file and prints the exact
command to finish the build on the Jetson. This is a *prototype* deployment
path; it has not been validated on hardware within this repository.

Usage:
  python deployment/tensorrt/convert_to_tensorrt.py \
      --dataset datasets/telemetry.csv --onnx detector.onnx --engine detector.plan
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(_HERE, "..", ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from anomaly_detector.features import N_FEATURES  # noqa: E402
from anomaly_detector.fnn import FNNClassifier, FNNConfig  # noqa: E402
from experiments._common import load_dataset, stratified_split  # noqa: E402


def export_onnx(dataset: str, onnx_path: str, epochs: int, opset: int = 17) -> str:
    import torch

    X, y = load_dataset(dataset)
    Xtr, Xte, ytr, yte = stratified_split(X, y, 0.25, 0)
    clf = FNNClassifier(FNNConfig(epochs=epochs))
    clf.fit(Xtr, ytr)

    # Wrap so the ONNX graph includes input standardisation (matches inference).
    mean = torch.tensor(clf.scaler.mean_, dtype=torch.float32)
    scale = torch.tensor(clf.scaler.scale_, dtype=torch.float32)
    net = clf.model.eval()

    class Wrapped(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.register_buffer("mean", mean)
            self.register_buffer("scale", scale)
            self.net = net

        def forward(self, x):
            x = (x - self.mean) / self.scale
            return torch.softmax(self.net(x), dim=-1)

    wrapped = Wrapped().eval()
    dummy = torch.randn(1, N_FEATURES, dtype=torch.float32)
    export_kwargs = dict(
        input_names=["telemetry"], output_names=["class_proba"],
        dynamic_axes={"telemetry": {0: "batch"}, "class_proba": {0: "batch"}},
        opset_version=opset,
    )
    try:
        # Prefer the stable TorchScript exporter (no onnxscript dependency).
        torch.onnx.export(wrapped, dummy, onnx_path, dynamo=False, **export_kwargs)
    except TypeError:
        # Older torch without the `dynamo` kwarg.
        torch.onnx.export(wrapped, dummy, onnx_path, **export_kwargs)
    print(f"[onnx] exported -> {onnx_path}")
    return onnx_path


def build_engine(onnx_path: str, engine_path: str, fp16: bool = True) -> bool:
    try:
        import tensorrt as trt
    except Exception:
        print("[trt] TensorRT not available on this machine. ONNX is ready.")
        print("[trt] Finish on the Jetson with trtexec, e.g.:")
        flag = "--fp16 " if fp16 else ""
        print(f"      trtexec --onnx={onnx_path} {flag}--saveEngine={engine_path}")
        return False

    logger = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(logger)
    network = builder.create_network(
        1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    parser = trt.OnnxParser(network, logger)
    with open(onnx_path, "rb") as f:
        if not parser.parse(f.read()):
            for i in range(parser.num_errors):
                print("[trt] parse error:", parser.get_error(i))
            return False
    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 1 << 28)  # 256 MB
    if fp16 and builder.platform_has_fast_fp16:
        config.set_flag(trt.BuilderFlag.FP16)
    serialized = builder.build_serialized_network(network, config)
    if serialized is None:
        print("[trt] engine build failed")
        return False
    with open(engine_path, "wb") as f:
        f.write(serialized)
    print(f"[trt] engine written -> {engine_path}")
    return True


def main() -> None:
    p = argparse.ArgumentParser(description="Export detector to ONNX + TensorRT.")
    p.add_argument("--dataset", type=str, default=os.path.join(_ROOT, "datasets", "telemetry.csv"))
    p.add_argument("--onnx", type=str, default=os.path.join(_HERE, "detector.onnx"))
    p.add_argument("--engine", type=str, default=os.path.join(_HERE, "detector.plan"))
    p.add_argument("--epochs", type=int, default=120)
    p.add_argument("--no-fp16", action="store_true")
    a = p.parse_args()
    export_onnx(a.dataset, a.onnx, a.epochs)
    build_engine(a.onnx, a.engine, fp16=not a.no_fp16)


if __name__ == "__main__":
    main()
