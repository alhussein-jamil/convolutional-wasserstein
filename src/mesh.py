"""Mesh utilities: voxelization, cotangent Laplacian, geodesic helpers."""

from __future__ import annotations

from dataclasses import dataclass
from heapq import heappop, heappush
from pathlib import Path

import numpy as np
import scipy.sparse as sp
import trimesh
from scipy.ndimage import binary_dilation

# --------------------------------------------------------------------------- #
# Grid index helpers
# --------------------------------------------------------------------------- #


def index_to_xyz(index: int, n: int) -> tuple[int, int, int]:
    """Inverse of :func:`xyz_to_index`."""
    z, rem = divmod(index, n * n)
    y, x = divmod(rem, n)
    return x, y, z


def xyz_to_index(x: int, y: int, z: int, n: int) -> int:
    """Lexicographic flatten of a 3-D grid index."""
    return n * n * x + n * y + z


def write_off(path: str | Path, vertices: np.ndarray, faces: np.ndarray) -> None:
    """Write a triangular mesh to an ``.off`` file."""
    with open(path, "w") as f:
        f.write("OFF\n")
        f.write(f"{len(vertices)} {len(faces)} 0\n")
        np.savetxt(f, vertices, fmt="%g")
        np.savetxt(f, np.column_stack([np.full(len(faces), 3), faces]), fmt="%d")


# --------------------------------------------------------------------------- #
# Geometry
# --------------------------------------------------------------------------- #


def normalize(mesh: trimesh.Trimesh, padding: float = 0.05) -> None:
    """In-place: rescale and center ``mesh`` so it fits in ``[padding, 1-padding]^3``."""
    bounds = mesh.bounds
    extent = bounds[1] - bounds[0]
    scale = (1.0 - 2.0 * padding) / max(extent.max(), 1e-12)
    mesh.apply_scale(scale)
    bbox_center = (mesh.bounds[0] + mesh.bounds[1]) / 2.0
    mesh.apply_translation(0.5 - bbox_center)


def cotangent_laplacian(mesh: trimesh.Trimesh) -> tuple[sp.csr_matrix, np.ndarray]:
    """Discrete (positive-semidefinite) cotangent Laplacian and lumped vertex areas.

    Returns ``(L, a)`` where ``L`` is sparse ``(n, n)`` and ``a`` is the
    Voronoi-area (mass-lumped) vector of length ``n``.
    """
    V = np.asarray(mesh.vertices, dtype=float)
    F = np.asarray(mesh.faces, dtype=int)
    n = len(V)

    v0, v1, v2 = V[F[:, 0]], V[F[:, 1]], V[F[:, 2]]
    e01 = v1 - v0
    e12 = v2 - v1
    e20 = v0 - v2

    twice_area = np.linalg.norm(np.cross(e01, -e20), axis=1)
    safe = np.maximum(twice_area, 1e-12)

    cot0 = -np.einsum("ij,ij->i", e01, e20) / safe
    cot1 = -np.einsum("ij,ij->i", e12, e01) / safe
    cot2 = -np.einsum("ij,ij->i", e20, e12) / safe

    # Edge (j, k) opposite vertex 0 gets weight cot0/2, etc.
    rows = np.concatenate([F[:, 1], F[:, 2], F[:, 2], F[:, 0], F[:, 0], F[:, 1]])
    cols = np.concatenate([F[:, 2], F[:, 1], F[:, 0], F[:, 2], F[:, 1], F[:, 0]])
    weights = 0.5 * np.concatenate([cot0, cot0, cot1, cot1, cot2, cot2])

    L_off = sp.csr_matrix((-weights, (rows, cols)), shape=(n, n))
    diag = -np.asarray(L_off.sum(axis=1)).ravel()
    L = L_off + sp.diags(diag)

    areas = np.zeros(n)
    contrib = twice_area / 6.0  # face area / 3 = (2A)/6
    np.add.at(areas, F[:, 0], contrib)
    np.add.at(areas, F[:, 1], contrib)
    np.add.at(areas, F[:, 2], contrib)
    return L, areas


def geodesic_distances(mesh: trimesh.Trimesh, source: int) -> np.ndarray:
    """Dijkstra from a vertex index ``source``; returns one distance per vertex."""
    adjacency = mesh.vertex_neighbors
    V = mesh.vertices
    dist = np.full(len(V), np.inf)
    dist[source] = 0.0
    heap: list[tuple[float, int]] = [(0.0, source)]
    while heap:
        d, u = heappop(heap)
        if d > dist[u]:
            continue
        for v in adjacency[u]:
            new = d + float(np.linalg.norm(V[v] - V[u]))
            if new < dist[v]:
                dist[v] = new
                heappush(heap, (new, v))
    return dist


def gaussian_on_mesh(mesh: trimesh.Trimesh, source: int, sigma: float = 0.1) -> np.ndarray:
    """Geodesic Gaussian centered at vertex ``source``, normalized to sum 1."""
    d = geodesic_distances(mesh, source)
    bump = np.exp(-(d**2) / (2.0 * sigma**2))
    return bump / bump.sum()


# --------------------------------------------------------------------------- #
# Mesh ↔ voxel distribution
# --------------------------------------------------------------------------- #


@dataclass
class Mesh:
    """Solid mesh with a Monte-Carlo voxel distribution on ``[0, 1]^3``.

    The mesh is normalized to fit in the unit cube, then voxelized at
    resolution ``n``. ``distribution`` is the per-voxel probability (length
    ``n^3``), ``binary`` is the 0/1 occupancy of voxels.
    """

    n: int = 40
    mesh: trimesh.Trimesh | None = None
    count: np.ndarray | None = None
    distribution: np.ndarray | None = None
    binary: np.ndarray | None = None
    binary_distribution: np.ndarray | None = None

    @classmethod
    def from_file(cls, path: str | Path, n: int = 40) -> Mesh:
        m = cls(n=n)
        m.mesh = trimesh.load(path, process=False)
        normalize(m.mesh)
        return m

    # -- voxelization ------------------------------------------------------- #

    def voxelize(self) -> None:
        """Voxelize the (normalized) solid mesh deterministically onto an ``n^3`` grid.

        Uses ``trimesh.Trimesh.voxelized(pitch).fill()`` — a sweep-based
        voxelizer that runs in seconds and bounded memory for any mesh size,
        without the rtree caches that ``mesh.contains`` builds. Memory and
        runtime are both ``O(n^3)``.
        """
        n = self.n
        pitch = 1.0 / n
        vox = self.mesh.voxelized(pitch=pitch).fill()
        try:  # trimesh caches the ray index and matrix queries — drop them.
            self.mesh._cache.clear()
        except AttributeError:
            pass

        matrix = np.ascontiguousarray(vox.matrix)
        start = np.round(np.asarray(vox.translation) / pitch).astype(int)
        end = start + np.asarray(matrix.shape)

        full = np.zeros((n, n, n), dtype=bool)
        dst_lo = np.maximum(start, 0)
        dst_hi = np.minimum(end, n)
        if np.any(dst_hi <= dst_lo):
            raise RuntimeError("Voxelization fell outside the unit cube — check normalize().")
        src_lo = dst_lo - start
        src_hi = src_lo + (dst_hi - dst_lo)
        full[dst_lo[0] : dst_hi[0], dst_lo[1] : dst_hi[1], dst_lo[2] : dst_hi[2]] = matrix[
            src_lo[0] : src_hi[0], src_lo[1] : src_hi[1], src_lo[2] : src_hi[2]
        ]

        self.binary = full.astype(float).ravel()
        self.count = self.binary.copy()
        total = self.binary.sum()
        if total == 0:
            raise RuntimeError("Voxelization produced no occupied cells.")
        self.distribution = self.count / total
        self.binary_distribution = self.binary / total

    def fill_holes(self, iterations: int = 3, connectivity: int = 26) -> None:
        """Close small holes in the voxel occupancy via binary dilation.

        ``connectivity`` is 6 (face neighbors only) or 26 (full 3x3x3 block).
        The original implementation used 26-neighborhood; that is the default.
        """
        if self.binary is None:
            raise RuntimeError("Call voxelize() before fill_holes().")
        if connectivity == 26:
            structure = np.ones((3, 3, 3), dtype=bool)
        elif connectivity == 6:
            structure = None  # scipy default
        else:
            raise ValueError("connectivity must be 6 or 26")
        vol = self.binary.reshape(self.n, self.n, self.n).astype(bool)
        vol = binary_dilation(vol, structure=structure, iterations=iterations)
        self.binary = vol.astype(float).ravel()
        self.binary_distribution = self.binary / self.binary.sum()
