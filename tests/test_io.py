import numpy as np

from convolutional_wasserstein.io import (
    bary_channels_to_rgb,
    color_channel_distributions,
    load_color_image,
)
from convolutional_wasserstein.paths import portrait_color_path


def test_bary_channels_to_rgb_recovers_portrait_colors():
    rgb = load_color_image(portrait_color_path("monge"))
    n = rgb.shape[0]
    channels = color_channel_distributions(rgb)
    recovered = bary_channels_to_rgb(channels, n)
    expected = rgb / 255.0
    for channel in range(3):
        corr = np.corrcoef(expected[..., channel].ravel(), recovered[..., channel].ravel())[0, 1]
        assert corr > 0.95
