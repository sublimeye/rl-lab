"""The pilot's network as plain numpy: inputs scaled, ReLU layers, one score per move."""

from __future__ import annotations

from pathlib import Path

import numpy as np


class SoloNet:
    def __init__(self, layers, mean, std):
        self.layers = [(np.asarray(w, np.float32), np.asarray(b, np.float32)) for w, b in layers]
        self.mean, self.std = np.asarray(mean, np.float32), np.asarray(std, np.float32)

    def __call__(self, x: np.ndarray) -> np.ndarray:
        h = ((x - self.mean) / self.std).astype(np.float32)
        for w, b in self.layers[:-1]:
            h = np.maximum(h @ w + b, 0.0)
        w, b = self.layers[-1]
        return h @ w + b

    def act(self, x: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """The top move among those on offer."""
        lg = self(x)
        return np.where(mask, lg[:, : mask.shape[1]], -np.inf).argmax(1)

    def sample(self, x: np.ndarray, mask: np.ndarray, rng, temp: float = 1.0) -> np.ndarray:
        """A move drawn from the network's distribution over those on offer."""
        lg = np.where(mask, self(x)[:, : mask.shape[1]], -np.inf) / temp
        u = rng.random(lg.shape).clip(1e-12, 1 - 1e-12)
        return (lg - np.log(-np.log(u))).argmax(1)


def save(net: SoloNet, path: Path) -> None:
    arrs = {"mean": net.mean, "std": net.std}
    for i, (w, b) in enumerate(net.layers):
        arrs[f"w{i}"], arrs[f"b{i}"] = w, b
    np.savez(path, **arrs)


def load(path: Path) -> SoloNet:
    z = np.load(path)
    n = (len(z.files) - 2) // 2
    return SoloNet([(z[f"w{i}"], z[f"b{i}"]) for i in range(n)], z["mean"], z["std"])
