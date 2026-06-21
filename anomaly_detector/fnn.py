"""Feedforward neural network (FNN) classifier in PyTorch.

A compact MLP suitable for edge inference (Jetson Orin Nano).  Features are
standardised; training uses Adam with early stopping on a validation split.  The
model is intentionally small so it can be exported to TorchScript/TensorRT (see
``deployment/tensorrt``).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np

import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler

from .features import N_CLASSES, N_FEATURES


@dataclass
class FNNConfig:
    hidden: List[int] = field(default_factory=lambda: [64, 32])
    dropout: float = 0.1
    lr: float = 1e-3
    weight_decay: float = 1e-4
    epochs: int = 120
    batch_size: int = 128
    patience: int = 15
    device: str = "cpu"          # "cuda" if available; orchestrator may override


class MLP(nn.Module):
    def __init__(self, in_dim: int, hidden: List[int], n_classes: int, dropout: float):
        super().__init__()
        layers: List[nn.Module] = []
        prev = in_dim
        for h in hidden:
            layers += [nn.Linear(prev, h), nn.ReLU(), nn.Dropout(dropout)]
            prev = h
        layers.append(nn.Linear(prev, n_classes))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class FNNClassifier:
    def __init__(self, cfg: Optional[FNNConfig] = None,
                 in_dim: int = N_FEATURES, n_classes: int = N_CLASSES):
        self.cfg = cfg or FNNConfig()
        self.in_dim = in_dim
        self.n_classes = n_classes
        self.scaler = StandardScaler()
        self.device = torch.device(
            self.cfg.device if (self.cfg.device != "cuda" or torch.cuda.is_available())
            else "cpu"
        )
        self.model = MLP(in_dim, self.cfg.hidden, n_classes, self.cfg.dropout).to(self.device)

    # ------------------------------------------------------------------ #
    def fit(self, X: np.ndarray, y: np.ndarray,
            X_val: Optional[np.ndarray] = None, y_val: Optional[np.ndarray] = None
            ) -> "FNNClassifier":
        Xs = self.scaler.fit_transform(X).astype(np.float32)
        yv = np.asarray(y).astype(np.int64)
        Xt = torch.from_numpy(Xs).to(self.device)
        yt = torch.from_numpy(yv).to(self.device)

        if X_val is not None:
            Xvs = self.scaler.transform(X_val).astype(np.float32)
            Xv = torch.from_numpy(Xvs).to(self.device)
            yvt = torch.from_numpy(np.asarray(y_val).astype(np.int64)).to(self.device)
        else:
            Xv = yvt = None

        opt = torch.optim.Adam(self.model.parameters(), lr=self.cfg.lr,
                               weight_decay=self.cfg.weight_decay)
        loss_fn = nn.CrossEntropyLoss()
        n = Xt.shape[0]
        best_state, best_val, wait = None, float("inf"), 0

        for _ in range(self.cfg.epochs):
            self.model.train()
            perm = torch.randperm(n, device=self.device)
            for s in range(0, n, self.cfg.batch_size):
                bi = perm[s:s + self.cfg.batch_size]
                opt.zero_grad()
                out = self.model(Xt[bi])
                loss = loss_fn(out, yt[bi])
                loss.backward()
                opt.step()

            # early stopping
            if Xv is not None:
                self.model.eval()
                with torch.no_grad():
                    vloss = loss_fn(self.model(Xv), yvt).item()
                if vloss < best_val - 1e-4:
                    best_val, wait = vloss, 0
                    best_state = {k: v.detach().clone() for k, v in self.model.state_dict().items()}
                else:
                    wait += 1
                    if wait >= self.cfg.patience:
                        break
        if best_state is not None:
            self.model.load_state_dict(best_state)
        return self

    # ------------------------------------------------------------------ #
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        Xs = self.scaler.transform(X).astype(np.float32)
        self.model.eval()
        with torch.no_grad():
            logits = self.model(torch.from_numpy(Xs).to(self.device))
            proba = torch.softmax(logits, dim=1).cpu().numpy()
        return proba

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.predict_proba(X).argmax(axis=1)

    # ------------------------------------------------------------------ #
    def save(self, path: str) -> None:
        torch.save({"state_dict": self.model.state_dict(),
                    "scaler_mean": self.scaler.mean_,
                    "scaler_scale": self.scaler.scale_,
                    "cfg": self.cfg.__dict__,
                    "in_dim": self.in_dim, "n_classes": self.n_classes}, path)

    def export_torchscript(self, path: str) -> None:
        """Export a TorchScript module for the TensorRT/edge path."""
        self.model.eval()
        example = torch.randn(1, self.in_dim, device=self.device)
        ts = torch.jit.trace(self.model, example)
        ts.save(path)
