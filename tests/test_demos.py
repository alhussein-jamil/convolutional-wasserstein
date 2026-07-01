import numpy as np

from scripts.assets import bilinear_coefs, synthetic_circle


def test_bilinear_coefs_sum_to_one():
    for coef in bilinear_coefs(4):
        assert np.isclose(coef.sum(), 1.0)


def test_synthetic_circle_normalized():
    dist = synthetic_circle(64)
    assert np.isclose(dist.sum(), 1.0)
    assert dist.shape == (64, 64)
