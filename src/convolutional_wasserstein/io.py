"""Image loaders that yield normalized probability distributions."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image


def load_binary_image(path: str | Path) -> np.ndarray:
    """Load a bilevel image; foreground (black) carries unit mass."""
    img = Image.open(path).convert("1")
    mat = 1.0 - np.asarray(img, dtype=float)
    return mat / mat.sum()


def load_grayscale_image(path: str | Path, invert: bool = True) -> np.ndarray:
    """Load a grayscale image as a distribution summing to 1."""
    img = Image.open(path).convert("L")
    mat = np.asarray(img, dtype=float)
    if invert:
        mat = mat.max() - mat
    total = mat.sum()
    if total == 0:
        raise ValueError(f"Image at {path} is empty after preprocessing.")
    return mat / total


def load_color_image(path: str | Path) -> np.ndarray:
    """Load an RGB image as a float array with shape ``(H, W, 3)`` in ``[0, 255]``."""
    return np.asarray(Image.open(path).convert("RGB"), dtype=float)


def color_channel_distributions(rgb: np.ndarray, invert: bool = True) -> list[np.ndarray]:
    """Flattened per-channel distributions (each sums to 1) for grid barycenters."""
    channels: list[np.ndarray] = []
    for channel in range(3):
        mat = rgb[..., channel].copy()
        if invert:
            mat = 255.0 - mat
        total = mat.sum()
        if total == 0:
            raise ValueError("Channel is empty after preprocessing.")
        channels.append((mat / total).ravel())
    return channels
