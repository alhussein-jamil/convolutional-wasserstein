"""Image loaders that produce normalized probability distributions."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image


def load_binary_image(path: str | Path) -> np.ndarray:
    """Load an image as a binary distribution (foreground = mass)."""
    img = Image.open(path).convert("1")
    mat = 1.0 - np.asarray(img, dtype=float)  # invert: black = 1
    return mat / mat.sum()


def load_grayscale_image(path: str | Path, invert: bool = True) -> np.ndarray:
    """Load an image as a grayscale distribution (normalized to sum 1).

    With ``invert=True`` (default) dark pixels are treated as mass — matching
    the convention used by the paper's portrait demo.
    """
    img = Image.open(path).convert("L")
    mat = np.asarray(img, dtype=float)
    if invert:
        mat = mat.max() - mat
    total = mat.sum()
    if total == 0:
        raise ValueError(f"Image at {path} is empty after preprocessing.")
    return mat / total
