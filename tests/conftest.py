import numpy as np
import pytest


@pytest.fixture
def uniform_grid():
    n = 16
    rng = np.random.default_rng(0)
    mu = rng.random(n * n)
    mu /= mu.sum()
    return mu, n
