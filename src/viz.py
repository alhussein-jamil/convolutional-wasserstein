"""Plotting helpers for 2-D distributions, voxel grids and triangle meshes."""

from __future__ import annotations

from pathlib import Path

import imageio.v2 as imageio
import matplotlib.pyplot as plt
import numpy as np
import plotly.graph_objects as go
import trimesh
from skimage.measure import marching_cubes

from .mesh import normalize

PLOT_RANGE = [-0.05, 1.05]

# Phong-ish lighting shared by every 3-D plot in this module.
_LIGHTING = dict(ambient=0.45, diffuse=0.75, specular=0.35, fresnel=0.15, roughness=0.5)
_LIGHT_POS = dict(x=1.5, y=1.5, z=2.5)
_CAMERA = dict(eye=dict(x=1.6, y=1.6, z=1.1))


def _style_3d(fig: go.Figure, *, show_axes: bool = False, auto_range: bool = False) -> go.Figure:
    """Apply consistent 3-D styling (axes off, white background, fixed camera).

    With ``auto_range=True`` the scene fits to the data instead of clamping
    to the unit cube — use for meshes whose vertex coords aren't already
    normalized to ``[0, 1]^3``.
    """
    axis_kw: dict = {} if auto_range else dict(range=PLOT_RANGE)
    if not show_axes:
        axis_kw.update(showbackground=False, showgrid=False, zeroline=False, visible=False)
    fig.update_layout(
        scene=dict(
            xaxis=axis_kw,
            yaxis=axis_kw,
            zaxis=axis_kw,
            bgcolor="white",
            camera=_CAMERA,
            aspectmode="data" if auto_range else "cube",
        ),
        paper_bgcolor="white",
        margin=dict(l=0, r=0, b=0, t=0),
    )
    return fig


# --------------------------------------------------------------------------- #
# 2-D
# --------------------------------------------------------------------------- #


def save_image_sequence(
    images: list[np.ndarray], gif_path: str | Path, fps: float = 10.0
) -> None:
    """Encode a list of frames as an infinitely-looping gif at ``fps`` frames/second.

    ``duration`` is passed to imageio in **milliseconds** (integer) and
    ``loop=0`` is set explicitly — imageio v2.37 silently drops sub-second
    floats and writes a gif with 0 ms / frame, which viewers freeze on frame 0.
    """
    imageio.mimsave(str(gif_path), images, duration=int(1000 / fps), loop=0)


def render_2d(
    bary: np.ndarray, threshold: float | None = 1e-6, ax: plt.Axes | None = None
) -> plt.Axes:
    """Show a square distribution as a binary heatmap."""
    n = int(round(bary.size**0.5))
    img = bary.reshape(n, n)
    if threshold is not None:
        img = (img > threshold).astype(float)
    ax = ax or plt.gca()
    ax.imshow(img, cmap="binary")
    ax.set_axis_off()
    return ax


# --------------------------------------------------------------------------- #
# 3-D
# --------------------------------------------------------------------------- #


def distribution_to_binary(distribution: np.ndarray, divisor: float = 8.0) -> np.ndarray:
    """Threshold a voxel distribution by ``max(distribution)/divisor``."""
    return (distribution > distribution.max() / divisor).astype(float)


def distribution_to_point_cloud(distribution: np.ndarray, scale: int = 1) -> np.ndarray:
    """Sample one point per voxel proportional to its mass.

    Returns coordinates in ``[0, 1]^3`` (normalized by the grid resolution).
    """
    if distribution.ndim == 1:
        n = int(round(distribution.size ** (1 / 3)))
        distribution = distribution.reshape(n, n, n)
    n = distribution.shape[0]
    n_pts = scale * int(1.0 / max(distribution.max(), 1e-12) * (1.28747 + 468.153 / n**3))
    counts = np.minimum(1, (distribution * n_pts).astype(int))
    idx = np.argwhere(counts > 0)
    jitter = np.random.rand(*idx.shape)
    return (idx + jitter) / n


def point_cloud(points: np.ndarray, color: str = "#1f77b4", size: float = 2.5) -> go.Figure:
    """3-D scatter plot of a point cloud in ``[0, 1]^3``."""
    if points.size == 0:
        return go.Figure()
    fig = go.Figure(
        go.Scatter3d(
            x=points[:, 0],
            y=points[:, 1],
            z=points[:, 2],
            mode="markers",
            marker=dict(size=size, color=color, opacity=0.7),
        )
    )
    return _style_3d(fig)


def voxel_isosurface(
    binary: np.ndarray,
    smooth: int = 30,
    color: str = "#7CB9E8",
) -> tuple[go.Figure, trimesh.Trimesh]:
    """Marching-cubes isosurface (Laplacian smoothed) of a binary voxel grid.

    Returns the plotly figure and the smoothed trimesh object.
    """
    if binary.ndim == 1:
        n = int(round(binary.size ** (1 / 3)))
        binary = binary.reshape(n, n, n)
    n = binary.shape[0]
    padded = np.zeros((n + 2, n + 2, n + 2))
    padded[1:-1, 1:-1, 1:-1] = binary

    verts, faces, _, _ = marching_cubes(padded, level=0.5, spacing=(1.0, 1.0, 1.0))
    mesh = trimesh.Trimesh(vertices=verts, faces=faces)
    normalize(mesh)
    if smooth:
        mesh = trimesh.smoothing.filter_laplacian(mesh, iterations=smooth)

    fig = go.Figure(
        go.Mesh3d(
            x=mesh.vertices[:, 0],
            y=mesh.vertices[:, 1],
            z=mesh.vertices[:, 2],
            i=mesh.faces[:, 0],
            j=mesh.faces[:, 1],
            k=mesh.faces[:, 2],
            color=color,
            flatshading=False,
            lighting=_LIGHTING,
            lightposition=_LIGHT_POS,
        )
    )
    return _style_3d(fig), mesh


def voxel_cubes(binary: np.ndarray, surface_only: bool = True) -> go.Figure:
    """Minecraft-style: render every occupied voxel as a cube.

    With ``surface_only=True`` (default), interior voxels — those whose six
    face neighbors are all occupied — are dropped: they're invisible from any
    camera angle, and removing them keeps the plotly JSON manageable at
    ``n >= 50``.
    """
    if binary.ndim == 1:
        n = int(round(binary.size ** (1 / 3)))
        vol = binary.reshape(n, n, n) > 0
    else:
        n = binary.shape[0]
        vol = binary > 0

    if surface_only:
        from scipy.ndimage import binary_erosion

        interior = binary_erosion(vol, iterations=1, border_value=0)
        vol = vol & ~interior

    xs, ys, zs = np.nonzero(vol)
    if xs.size == 0:
        return go.Figure()

    unit_v = np.array(
        [[0, 0, 0], [0, 1, 0], [1, 1, 0], [1, 0, 0], [0, 0, 1], [0, 1, 1], [1, 1, 1], [1, 0, 1]],
        dtype=float,
    )
    unit_faces_i = np.array([7, 0, 0, 0, 4, 4, 6, 6, 4, 0, 3, 2])
    unit_faces_j = np.array([3, 4, 1, 2, 5, 6, 5, 2, 0, 1, 6, 3])
    unit_faces_k = np.array([0, 7, 2, 3, 6, 7, 1, 1, 5, 5, 7, 6])

    origins = np.stack([xs, ys, zs], axis=1).astype(float)
    vertices = (origins[:, None, :] + unit_v[None, :, :]).reshape(-1, 3) / n
    offsets = 8 * np.arange(len(origins))[:, None]
    ii = (offsets + unit_faces_i).ravel()
    jj = (offsets + unit_faces_j).ravel()
    kk = (offsets + unit_faces_k).ravel()

    fig = go.Figure(
        go.Mesh3d(
            x=vertices[:, 0],
            y=vertices[:, 1],
            z=vertices[:, 2],
            i=ii,
            j=jj,
            k=kk,
            color="#7CB9E8",
            flatshading=True,
            lighting=_LIGHTING,
            lightposition=_LIGHT_POS,
        )
    )
    return _style_3d(fig)


def mesh_scalar_field(
    mesh: trimesh.Trimesh, values: np.ndarray, colorscale: str = "Viridis"
) -> go.Figure:
    """Plotly mesh3d colored by a per-vertex scalar field."""
    fig = go.Figure(
        go.Mesh3d(
            x=mesh.vertices[:, 0],
            y=mesh.vertices[:, 1],
            z=mesh.vertices[:, 2],
            i=mesh.faces[:, 0],
            j=mesh.faces[:, 1],
            k=mesh.faces[:, 2],
            intensity=values,
            colorscale=colorscale,
            showscale=False,
            flatshading=False,
            lighting=_LIGHTING,
            lightposition=_LIGHT_POS,
        )
    )
    return _style_3d(fig, auto_range=True)
