"""Entropic sharpening (Algorithm 3, Solomon et al. 2015)."""

from __future__ import annotations

import numpy as np
from scipy.optimize import brentq


def entropy(mu: np.ndarray, area: np.ndarray) -> float:
    """Weighted entropy ``-sum a_i mu_i log mu_i``."""
    mask = mu > 0
    return float(-np.sum(area[mask] * mu[mask] * np.log(mu[mask])))


def entropic_sharpening(mu: np.ndarray, entropy_bound: float, area: np.ndarray) -> np.ndarray:
    """Raise ``mu`` to power ``beta >= 1`` so entropy stays below ``entropy_bound``."""
    if entropy(mu, area) + float(area @ mu) <= entropy_bound + 1.0:
        return mu

    def gap(beta: float) -> float:
        powered = mu**beta
        return float(area @ powered) + entropy(powered, area) - (1.0 + entropy_bound)

    try:
        beta = brentq(gap, 1.0, 50.0)
    except ValueError:
        return mu
    return mu**beta
