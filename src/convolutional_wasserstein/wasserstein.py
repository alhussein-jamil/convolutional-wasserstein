"""Convolutional Sinkhorn barycenter (Algorithm 2, Solomon et al. 2015)."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass

import numpy as np
import scipy.linalg as slin

from convolutional_wasserstein.convolution import make_heat_operator
from convolutional_wasserstein.post_processing import entropic_sharpening, entropy

Operator = Callable[[np.ndarray], np.ndarray]
_EPS = 1e-20


def wasserstein_barycenter(
    mus: Sequence[np.ndarray],
    weights: Sequence[float],
    area: np.ndarray,
    apply_kernel: Operator,
    iterations: int = 100,
    sharpen: bool = True,
) -> np.ndarray:
    """Wasserstein barycenter via convolutional Sinkhorn iterations."""
    mus_arr = [np.asarray(mu, dtype=np.float64).reshape(-1) + _EPS for mu in mus]
    lengths = {mu.size for mu in mus_arr}
    if len(lengths) != 1:
        raise ValueError(f"all distributions must share the same length, got {sorted(lengths)}")
    k = len(mus_arr)
    n = mus_arr[0].size
    weights = np.asarray(weights, dtype=np.float64)
    area = np.asarray(area, dtype=np.float64)

    v = np.ones((k, n), dtype=np.float64)
    w = np.ones((k, n), dtype=np.float64)
    d = np.empty((k, n), dtype=np.float64)
    scratch = np.empty(n, dtype=np.float64)
    powered = np.empty(n, dtype=np.float64)
    sharpen_bound = max(entropy(mus_arr[i], area) for i in range(k)) if sharpen else None
    bary = np.ones(n, dtype=np.float64)

    for _ in range(iterations):
        bary.fill(1.0)
        for i in range(k):
            np.multiply(area, v[i], out=scratch)
            kv = apply_kernel(scratch)
            np.maximum(kv, _EPS, out=kv)
            np.divide(mus_arr[i], kv, out=w[i])

            np.multiply(area, w[i], out=scratch)
            kw = apply_kernel(scratch)
            np.maximum(kw, _EPS, out=kw)
            np.multiply(v[i], kw, out=d[i])
            np.maximum(d[i], _EPS, out=d[i])

            if weights[i] == 1.0:
                bary *= d[i]
            else:
                np.power(d[i], weights[i], out=powered)
                bary *= powered

        if sharpen:
            bary = entropic_sharpening(bary, sharpen_bound, area)
        v *= bary / d

    return bary


@dataclass(frozen=True, slots=True)
class GridSolver:
    """Reusable grid barycenter solver with a cached heat operator."""

    area: np.ndarray
    apply_kernel: Operator

    def __call__(
        self,
        mus: Sequence[np.ndarray],
        weights: Sequence[float],
        iterations: int = 100,
        sharpen: bool = True,
    ) -> np.ndarray:
        return wasserstein_barycenter(
            mus, weights, self.area, self.apply_kernel, iterations, sharpen
        )


def make_grid_solver(shape: tuple[int, ...], gamma: float) -> GridSolver:
    """Build a solver that reuses one heat operator for many barycenter calls."""
    n_total = int(np.prod(shape))
    area = np.full(n_total, 1.0 / n_total, dtype=np.float64)
    return GridSolver(area=area, apply_kernel=make_heat_operator(shape, gamma))


def grid_barycenter(
    mus: Sequence[np.ndarray],
    weights: Sequence[float],
    n: int,
    gamma: float = 0.01,
    iterations: int = 100,
    sharpen: bool = True,
    solver: GridSolver | None = None,
) -> np.ndarray:
    """Barycenter on an ``n^d`` uniform grid (``d`` inferred from ``mus[0].size``)."""
    if solver is None:
        n_total = mus[0].size
        d = int(round(np.log(n_total) / np.log(n)))
        solver = make_grid_solver((n,) * d, gamma)
    return solver(mus, weights, iterations=iterations, sharpen=sharpen)


def mesh_heat_operator(L_cholesky: np.ndarray) -> Operator:
    """Heat step ``(D_a + tL)^{-1}`` from Cholesky factor of ``D_a + (gamma/2) L``."""
    lower = L_cholesky
    upper = L_cholesky.T

    def apply(v: np.ndarray) -> np.ndarray:
        y = slin.solve_triangular(lower, v, lower=True, overwrite_b=True)
        return slin.solve_triangular(upper, y, lower=False, overwrite_b=True)

    return apply
