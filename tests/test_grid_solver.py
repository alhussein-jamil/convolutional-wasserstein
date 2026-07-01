import numpy as np

from convolutional_wasserstein.wasserstein import grid_barycenter, make_grid_solver


def test_grid_solver_matches_grid_barycenter():
    n = 10
    rng = np.random.default_rng(3)
    mu1 = rng.random(n * n)
    mu2 = rng.random(n * n)
    mu1 /= mu1.sum()
    mu2 /= mu2.sum()
    weights = [0.4, 0.6]
    gamma = 0.01

    solver = make_grid_solver((n, n), gamma)
    a = grid_barycenter([mu1, mu2], weights, n, gamma=gamma, solver=solver)
    b = solver([mu1, mu2], weights, iterations=100)
    assert np.allclose(a, b)
