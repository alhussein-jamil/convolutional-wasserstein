"""Separable Gaussian heat convolution on uniform grids."""

from __future__ import annotations

from functools import lru_cache

import numpy as np
from scipy.ndimage import convolve1d


@lru_cache(maxsize=64)
def gaussian_kernel(n: int, gamma: float, truncate: float = 6.0) -> np.ndarray:
    """1-D heat kernel for an ``n``-cell grid on ``[0, 1]``."""
    sigma = n * np.sqrt(gamma) / 2.0
    full = truncate * sigma
    radius = n - 1 if not np.isfinite(full) else min(n - 1, int(np.ceil(full)))
    idx = np.arange(radius + 1, dtype=np.float64)
    half = np.exp(-((idx / n) ** 2) / (gamma / 2.0))
    return np.concatenate([half[:0:-1], half])


def heat_convolve(u: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    """Apply separable 1-D ``kernel`` along every axis of ``u``."""
    out = u
    for axis in range(out.ndim):
        out = convolve1d(out, kernel, axis=axis, mode="constant", cval=0.0)
    return out


def make_heat_operator(shape: tuple[int, ...], gamma: float):
    """Return ``K(v)`` — flat heat-kernel convolution on a grid of ``shape``."""
    kernel = gaussian_kernel(shape[0], gamma)
    size = int(np.prod(shape))

    def apply(v: np.ndarray) -> np.ndarray:
        return heat_convolve(v.reshape(shape), kernel).reshape(size)

    return apply


def naive_gaussian_convolution(u: np.ndarray, gamma: float) -> np.ndarray:
    """Dense ``O(n^{2d})`` reference implementation for tests."""
    n = u.shape[0]
    grid = np.arange(n, dtype=np.float64)
    diff = (grid[:, None] - grid[None, :]) / n
    kernel = np.exp(-(diff**2) / (gamma / 2.0))
    out = u
    for axis in range(u.ndim):
        out = np.tensordot(kernel, out, axes=([1], [axis]))
        out = np.moveaxis(out, 0, axis)
    return out
