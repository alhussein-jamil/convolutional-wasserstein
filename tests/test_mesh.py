import numpy as np
import trimesh

from convolutional_wasserstein.mesh import (
    cotangent_laplacian,
    gaussian_on_mesh,
    geodesic_distances,
    opposite_vertices,
)


def test_cotangent_laplacian_diagonal():
    mesh = trimesh.creation.box()
    laplacian, areas = cotangent_laplacian(mesh)
    row_sums = np.asarray(laplacian.sum(axis=1)).ravel()
    assert np.allclose(row_sums, 0, atol=1e-10)
    assert (areas > 0).all()


def test_opposite_vertices_are_far_apart():
    mesh = trimesh.creation.icosphere(subdivisions=3)
    i, j = opposite_vertices(mesh)
    assert i != j
    dist = geodesic_distances(mesh, i)[j]
    assert dist > 0.8 * geodesic_distances(mesh, i).max()


def test_mesh_gaussian_sums_to_one():
    mesh = trimesh.creation.icosphere(subdivisions=2)
    g = gaussian_on_mesh(mesh, source=0, sigma=0.15)
    assert np.isclose(g.sum(), 1.0)
    assert (g >= 0).all()
