"""Wasserstein barycenter via the convolutional Sinkhorn iterations of
Solomon et al. 2015 (Algorithm 2).

The iteration only ever uses the kernel as a linear operator ``K v``. So a
single :func:`wasserstein_barycenter` works on any domain — uniform 2-D/3-D
grids or triangle meshes — by accepting the operator as a callable.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence

import numpy as np
import scipy.linalg as slin

from .convolution import make_heat_operator
from .post_processing import entropic_sharpening, entropy

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
    """Compute a Wasserstein barycenter on an arbitrary geometric domain.

    Parameters
    ----------
    mus
        Input distributions, each a flat ``(N,)`` non-negative vector summing
        to 1 (or that will be normalized implicitly by the Sinkhorn loop).
    weights
        Mixing coefficients (``alpha_i`` in the paper) summing to 1.
    area
        Per-cell area / mass-lumped vertex area, shape ``(N,)``.
    apply_kernel
        Callable mapping a flat ``(N,)`` vector to ``K v`` where ``K`` is the
        heat kernel of the domain. For grids use
        :func:`src.convolution.make_heat_operator`; for meshes use
        :func:`mesh_heat_operator` below.
    iterations
        Number of Sinkhorn iterations.
    sharpen
        Whether to apply entropic sharpening after each iteration.
    """
    mus = [np.asarray(mu, dtype=float) + _EPS for mu in mus]
    weights = np.asarray(weights, dtype=float)
    area = np.asarray(area, dtype=float)
    k = len(mus)
    n = mus[0].shape[0]

    v = np.ones((k, n))
    w = np.ones((k, n))
    d = np.zeros((k, n))

    sharpen_bound = max(entropy(mu, area) for mu in mus) if sharpen else None
    bary = np.ones(n)

    for _ in range(iterations):
        bary = np.ones(n)
        for i in range(k):
            kv = np.maximum(apply_kernel(area * v[i]), _EPS)
            w[i] = mus[i] / kv
            kw = np.maximum(apply_kernel(area * w[i]), _EPS)
            d[i] = np.maximum(v[i] * kw, _EPS)
            bary *= d[i] ** weights[i]
        if sharpen:
            bary = entropic_sharpening(bary, sharpen_bound, area)
        v *= bary / d

    return bary


def grid_barycenter(
    mus: Sequence[np.ndarray],
    weights: Sequence[float],
    n: int,
    gamma: float = 0.01,
    iterations: int = 100,
    sharpen: bool = True,
) -> np.ndarray:
    """Convenience wrapper: barycenter of distributions on an ``n^d`` grid.

    Each ``mu`` must be a flat array of length ``n^d`` (``d`` inferred).
    Returns the barycenter as a flat array of the same length.
    """
    n_total = mus[0].size
    d = int(round(np.log(n_total) / np.log(n)))
    shape = (n,) * d
    area = np.full(n_total, 1.0 / n_total)
    return wasserstein_barycenter(
        mus,
        weights,
        area,
        apply_kernel=make_heat_operator(shape, gamma),
        iterations=iterations,
        sharpen=sharpen,
    )


def mesh_heat_operator(L_cholesky: np.ndarray) -> Operator:
    """Heat-diffusion operator on a mesh from a Cholesky factor of ``D_a + (gamma/2) L``.

    Applies one backward Euler step of the heat equation, i.e. ``H_t = (D_a + tL)^{-1}``
    times the right-hand side.
    """

    def apply(v: np.ndarray) -> np.ndarray:
        y = slin.solve_triangular(L_cholesky, v, lower=True)
        return slin.solve_triangular(L_cholesky.T, y, lower=False)

    return apply
