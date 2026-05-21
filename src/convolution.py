"""Separable Gaussian (heat) convolution on a uniform grid.

The Sinkhorn iterations in Solomon et al. 2015 only need the operator
``K v`` where ``K_ij = exp(-d(x_i, x_j)^2 / gamma)``. On a regular grid this
kernel is separable, so applying it reduces to a 1-D Gaussian convolution
along each axis — overall ``O(n^d)`` instead of ``O(n^{2d})``.
"""

from __future__ import annotations

import numpy as np
from scipy.ndimage import convolve1d


def gaussian_kernel(n: int, gamma: float, truncate: float = 6.0) -> np.ndarray:
    """Symmetric 1-D heat kernel for an ``n``-cell grid covering ``[0, 1]``.

    Matches the paper's convention ``K(i, j) = exp(-(|i-j|/n)^2 / (gamma/2))``
    so that ``gamma`` is interpreted in normalized [0, 1]^d coordinates.

    The Gaussian has standard deviation ``sigma = n * sqrt(gamma) / 2`` cells.
    The returned kernel is truncated to ``radius = ceil(truncate * sigma)``
    cells on each side (capped at ``n - 1``). At ``truncate = 6.0`` the
    truncated tail is ``< 1e-15``, i.e. below float64 round-off, so the result
    is bitwise indistinguishable from the full ``2n-1`` kernel.
    """
    sigma = n * np.sqrt(gamma) / 2.0
    full = truncate * sigma
    radius = n - 1 if not np.isfinite(full) else min(n - 1, int(np.ceil(full)))
    idx = np.arange(radius + 1)
    half = np.exp(-((idx / n) ** 2) / (gamma / 2.0))
    return np.concatenate([half[:0:-1], half])


def heat_convolve(u: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    """Apply a separable 1-D ``kernel`` along every axis of ``u``."""
    out = u
    for axis in range(out.ndim):
        out = convolve1d(out, kernel, axis=axis, mode="constant", cval=0.0)
    return out


def make_heat_operator(shape: tuple[int, ...], gamma: float):
    """Return a callable ``K(v)`` that convolves a flat vector with the heat kernel."""
    kernel = gaussian_kernel(shape[0], gamma)

    def apply(v: np.ndarray) -> np.ndarray:
        return heat_convolve(v.reshape(shape), kernel).reshape(-1)

    return apply


def naive_gaussian_convolution(u: np.ndarray, gamma: float) -> np.ndarray:
    """Reference O(n^{2d}) implementation, used only as a sanity check in tests."""
    n = u.shape[0]
    grid = np.arange(n)
    diff = (grid[:, None] - grid[None, :]) / n
    K = np.exp(-(diff**2) / (gamma / 2.0))
    out = u
    for axis in range(u.ndim):
        out = np.tensordot(K, out, axes=([1], [axis]))
        out = np.moveaxis(out, 0, axis)
    return out
