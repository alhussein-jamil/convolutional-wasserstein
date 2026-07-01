import numpy as np

from convolutional_wasserstein.post_processing import entropic_sharpening, entropy


def test_entropy_nonnegative():
    mu = np.array([0.25, 0.25, 0.25, 0.25])
    area = np.ones(4) / 4
    assert entropy(mu, area) >= 0


def test_sharpening_respects_bound():
    area = np.ones(8) / 8
    mu = np.full(8, 1 / 8)
    bound = entropy(mu, area) * 0.5
    sharp = entropic_sharpening(mu, bound, area)
    assert entropy(sharp, area) <= bound + 1e-6 or np.allclose(sharp, mu)
