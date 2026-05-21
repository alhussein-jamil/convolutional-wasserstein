"""Reimplementation of Solomon et al. 2015,
'Convolutional Wasserstein Distances: Efficient Optimal Transportation on Geometric Domains'.
"""

from .convolution import gaussian_kernel, heat_convolve, make_heat_operator
from .io import load_binary_image, load_grayscale_image
from .mesh import (
    Mesh,
    cotangent_laplacian,
    gaussian_on_mesh,
    geodesic_distances,
    index_to_xyz,
    normalize,
    write_off,
    xyz_to_index,
)
from .post_processing import entropic_sharpening, entropy
from .wasserstein import (
    grid_barycenter,
    mesh_heat_operator,
    wasserstein_barycenter,
)

__all__ = [
    "Mesh",
    "cotangent_laplacian",
    "entropic_sharpening",
    "entropy",
    "gaussian_kernel",
    "gaussian_on_mesh",
    "geodesic_distances",
    "grid_barycenter",
    "heat_convolve",
    "index_to_xyz",
    "load_binary_image",
    "load_grayscale_image",
    "make_heat_operator",
    "mesh_heat_operator",
    "normalize",
    "wasserstein_barycenter",
    "write_off",
    "xyz_to_index",
]
