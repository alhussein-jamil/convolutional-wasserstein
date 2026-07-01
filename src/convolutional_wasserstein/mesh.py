"""Mesh geometry, geodesics, and voxel distributions."""

from __future__ import annotations

from dataclasses import dataclass
from heapq import heappop, heappush
from pathlib import Path

import numpy as np
import scipy.sparse as sp
import trimesh


def normalize_mesh(mesh: trimesh.Trimesh, padding: float = 0.05) -> None:
    """In-place: fit mesh into ``[padding, 1 - padding]^3``."""
    bounds = mesh.bounds
    extent = bounds[1] - bounds[0]
    scale = (1.0 - 2.0 * padding) / max(extent.max(), 1e-12)
    mesh.apply_scale(scale)
    center = (mesh.bounds[0] + mesh.bounds[1]) / 2.0
    mesh.apply_translation(0.5 - center)


def cotangent_laplacian(mesh: trimesh.Trimesh) -> tuple[sp.csr_matrix, np.ndarray]:
    """Cotangent Laplacian and lumped vertex areas."""
    vertices = np.asarray(mesh.vertices, dtype=float)
    faces = np.asarray(mesh.faces, dtype=int)
    n = len(vertices)

    v0, v1, v2 = vertices[faces[:, 0]], vertices[faces[:, 1]], vertices[faces[:, 2]]
    e01, e12, e20 = v1 - v0, v2 - v1, v0 - v2
    twice_area = np.linalg.norm(np.cross(e01, -e20), axis=1)
    safe = np.maximum(twice_area, 1e-12)

    cot0 = -np.einsum("ij,ij->i", e01, e20) / safe
    cot1 = -np.einsum("ij,ij->i", e12, e01) / safe
    cot2 = -np.einsum("ij,ij->i", e20, e12) / safe

    rows = np.concatenate(
        [faces[:, 1], faces[:, 2], faces[:, 2], faces[:, 0], faces[:, 0], faces[:, 1]]
    )
    cols = np.concatenate(
        [faces[:, 2], faces[:, 1], faces[:, 0], faces[:, 2], faces[:, 1], faces[:, 0]]
    )
    weights = 0.5 * np.concatenate([cot0, cot0, cot1, cot1, cot2, cot2])

    off_diag = sp.csr_matrix((-weights, (rows, cols)), shape=(n, n))
    diag = -np.asarray(off_diag.sum(axis=1)).ravel()
    laplacian = off_diag + sp.diags(diag)

    areas = np.zeros(n)
    contrib = twice_area / 6.0
    np.add.at(areas, faces[:, 0], contrib)
    np.add.at(areas, faces[:, 1], contrib)
    np.add.at(areas, faces[:, 2], contrib)
    return laplacian, areas


def geodesic_distances(mesh: trimesh.Trimesh, source: int) -> np.ndarray:
    """Dijkstra geodesic distances from vertex ``source``."""
    vertices = np.asarray(mesh.vertices, dtype=np.float64)
    adjacency = mesh.vertex_neighbors
    n = len(vertices)
    dist = np.full(n, np.inf, dtype=np.float64)
    dist[source] = 0.0
    heap: list[tuple[float, int]] = [(0.0, source)]
    while heap:
        d, u = heappop(heap)
        if d > dist[u]:
            continue
        vu = vertices[u]
        for v in adjacency[u]:
            new = d + float(np.linalg.norm(vertices[v] - vu))
            if new < dist[v]:
                dist[v] = new
                heappush(heap, (new, v))
    return dist


def opposite_vertices(mesh: trimesh.Trimesh) -> tuple[int, int]:
    """Return two nearly antipodal vertices — good endpoints for transport demos."""
    vertices = np.asarray(mesh.vertices, dtype=float)
    centered = vertices - vertices.mean(axis=0)
    unit = centered / np.maximum(np.linalg.norm(centered, axis=1, keepdims=True), 1e-12)
    i = int(np.argmax(unit[:, 0]))
    j = int(np.argmin(unit @ unit[i]))
    return i, j


def gaussian_on_mesh(mesh: trimesh.Trimesh, source: int, sigma: float = 0.1) -> np.ndarray:
    """Normalized geodesic Gaussian centered at ``source``."""
    distances = geodesic_distances(mesh, source)
    bump = np.exp(-(distances**2) / (2.0 * sigma**2))
    return bump / bump.sum()


@dataclass
class VoxelMesh:
    """Solid mesh voxelized onto an ``n^3`` grid in ``[0, 1]^3``."""

    n: int = 40
    mesh: trimesh.Trimesh | None = None
    distribution: np.ndarray | None = None
    binary: np.ndarray | None = None
    binary_distribution: np.ndarray | None = None

    @classmethod
    def from_file(cls, path: str | Path, n: int = 40) -> VoxelMesh:
        obj = cls(n=n)
        obj.mesh = trimesh.load(path, process=False)
        normalize_mesh(obj.mesh)
        return obj

    def voxelize(self) -> None:
        """Rasterize the normalized mesh onto an ``n^3`` occupancy grid."""
        if self.mesh is None:
            raise RuntimeError("No mesh loaded.")
        pitch = 1.0 / self.n
        vox = self.mesh.voxelized(pitch=pitch).fill()
        try:
            self.mesh._cache.clear()
        except AttributeError:
            pass

        matrix = np.ascontiguousarray(vox.matrix)
        start = np.round(np.asarray(vox.translation) / pitch).astype(int)
        end = start + np.asarray(matrix.shape)

        full = np.zeros((self.n, self.n, self.n), dtype=bool)
        dst_lo = np.maximum(start, 0)
        dst_hi = np.minimum(end, self.n)
        if np.any(dst_hi <= dst_lo):
            raise RuntimeError("Voxelization fell outside the unit cube.")
        src_lo = dst_lo - start
        src_hi = src_lo + (dst_hi - dst_lo)
        full[dst_lo[0] : dst_hi[0], dst_lo[1] : dst_hi[1], dst_lo[2] : dst_hi[2]] = matrix[
            src_lo[0] : src_hi[0], src_lo[1] : src_hi[1], src_lo[2] : src_hi[2]
        ]

        self.binary = full.astype(float).ravel()
        total = self.binary.sum()
        if total == 0:
            raise RuntimeError("Voxelization produced no occupied cells.")
        self.distribution = self.binary / total
        self.binary_distribution = self.binary / total
