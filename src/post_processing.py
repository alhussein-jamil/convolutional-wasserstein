"""Entropic sharpening (Algorithm 3 of Solomon et al. 2015).

After Sinkhorn iterations the barycenter is blurred by the entropic
regularizer. Sharpening rescales it by an exponent ``beta >= 1`` chosen so the
entropy lies at the maximum entropy of the input distributions.
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import brentq


def entropy(mu: np.ndarray, a: np.ndarray) -> float:
    """Weighted entropy ``-sum a_i mu_i log mu_i`` (``0 log 0 := 0``)."""
    mask = mu > 0
    return float(-np.sum(a[mask] * mu[mask] * np.log(mu[mask])))


def entropic_sharpening(mu: np.ndarray, entropy_bound: float, a: np.ndarray) -> np.ndarray:
    """Sharpen ``mu`` by raising it to a power ``beta >= 1``.

    Returns ``mu`` unchanged if its entropy is already below the bound; else
    finds the unique ``beta >= 1`` such that ``H(mu^beta) + <a, mu^beta> = 1 + H``.
    """
    if entropy(mu, a) + float(a @ mu) <= entropy_bound + 1.0:
        return mu

    def gap(beta: float) -> float:
        powered = mu**beta
        return float(a @ powered) + entropy(powered, a) - (1.0 + entropy_bound)

    # gap(1) > 0 by construction above; gap grows ↘ as beta ↑.
    try:
        beta = brentq(gap, 1.0, 50.0)
    except ValueError:
        # Either the bracket fails or the function is flat — fall back to no-op.
        return mu
    return mu**beta
