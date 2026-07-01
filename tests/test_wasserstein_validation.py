import numpy as np
import pytest

from convolutional_wasserstein.wasserstein import wasserstein_barycenter


def test_mismatched_distribution_lengths():
    def op(v):
        return v

    area = np.array([0.5, 0.5])
    with pytest.raises(ValueError, match="same length"):
        wasserstein_barycenter(
            [np.array([1.0, 0.0]), np.array([1.0, 0.0, 0.0])],
            [0.5, 0.5],
            area,
            op,
            iterations=1,
            sharpen=False,
        )
