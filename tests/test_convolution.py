import numpy as np

from convolutional_wasserstein.convolution import (
    gaussian_kernel,
    heat_convolve,
    make_heat_operator,
    naive_gaussian_convolution,
)


def test_gaussian_kernel_symmetric():
    kernel = gaussian_kernel(32, gamma=0.01)
    assert np.allclose(kernel, kernel[::-1])
    assert kernel.argmax() == len(kernel) // 2


def test_separable_matches_naive():
    n, gamma = 12, 0.02
    rng = np.random.default_rng(1)
    u = rng.random((n, n))
    kernel = gaussian_kernel(n, gamma)
    fast = heat_convolve(u, kernel)
    ref = naive_gaussian_convolution(u, gamma)
    assert np.allclose(fast, ref, rtol=1e-10, atol=1e-12)


def test_heat_operator_shape_and_nonnegativity():
    n, gamma = 8, 0.05
    op = make_heat_operator((n, n), gamma)
    mu = np.ones(n * n) / (n * n)
    out = op(mu)
    assert out.shape == mu.shape
    assert (out >= 0).all()
    assert np.isfinite(out).all()
