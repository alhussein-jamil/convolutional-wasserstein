"""Convolutional Wasserstein distances (Solomon et al. 2015)."""

from convolutional_wasserstein.convolution import (
    gaussian_kernel,
    heat_convolve,
    make_heat_operator,
    naive_gaussian_convolution,
)
from convolutional_wasserstein.io import (
    bary_channels_to_rgb,
    color_channel_distributions,
    load_binary_image,
    load_color_image,
    load_grayscale_image,
)
from convolutional_wasserstein.mesh import (
    VoxelMesh,
    cotangent_laplacian,
    gaussian_on_mesh,
    geodesic_distances,
    normalize_mesh,
    opposite_vertices,
)
from convolutional_wasserstein.post_processing import entropic_sharpening, entropy
from convolutional_wasserstein.wasserstein import (
    GridSolver,
    grid_barycenter,
    make_grid_solver,
    mesh_heat_operator,
    wasserstein_barycenter,
)

__version__ = "0.2.0"

__all__ = [
    "GridSolver",
    "VoxelMesh",
    "cotangent_laplacian",
    "entropic_sharpening",
    "entropy",
    "gaussian_kernel",
    "gaussian_on_mesh",
    "geodesic_distances",
    "grid_barycenter",
    "heat_convolve",
    "bary_channels_to_rgb",
    "color_channel_distributions",
    "load_binary_image",
    "load_color_image",
    "load_grayscale_image",
    "make_grid_solver",
    "make_heat_operator",
    "mesh_heat_operator",
    "naive_gaussian_convolution",
    "normalize_mesh",
    "opposite_vertices",
    "wasserstein_barycenter",
]
