import numpy as np

from convolutional_wasserstein.wasserstein import grid_barycenter


def test_two_point_barycenter_on_line():
    n = 32
    left = np.zeros(n)
    left[0] = 1.0
    right = np.zeros(n)
    right[-1] = 1.0
    mid = grid_barycenter([left, right], [0.5, 0.5], n, gamma=0.05, iterations=50, sharpen=False)
    assert mid.argmax() in range(n // 4, 3 * n // 4)


def test_barycenter_stays_normalized():
    n = 10
    rng = np.random.default_rng(2)
    mu1 = rng.random(n * n)
    mu2 = rng.random(n * n)
    mu1 /= mu1.sum()
    mu2 /= mu2.sum()
    bary = grid_barycenter([mu1, mu2], [0.3, 0.7], n, gamma=0.01, iterations=20)
    assert np.isfinite(bary).all()
    assert bary.min() >= 0
