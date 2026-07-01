"""Plotting helpers used by demo scripts."""

from __future__ import annotations

from pathlib import Path

import imageio.v2 as imageio
import matplotlib.pyplot as plt
import numpy as np
import plotly.graph_objects as go
import trimesh
from skimage.measure import marching_cubes

from convolutional_wasserstein.mesh import normalize_mesh

PLOT_RANGE = [-0.05, 1.05]
_LIGHTING = dict(ambient=0.45, diffuse=0.75, specular=0.35, fresnel=0.15, roughness=0.5)
_LIGHT_POS = dict(x=1.5, y=1.5, z=2.5)
_CAMERA = dict(eye=dict(x=1.6, y=1.6, z=1.1))


def _style_3d(fig: go.Figure, *, auto_range: bool = False) -> go.Figure:
    axis_kw: dict = {} if auto_range else dict(range=PLOT_RANGE)
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


def save_gif(frames: list[np.ndarray], path: str | Path, fps: float = 10.0) -> None:
    imageio.mimsave(str(path), frames, duration=int(1000 / fps), loop=0)


def render_distribution(
    distribution: np.ndarray,
    *,
    threshold: float | None = 1e-6,
    ax: plt.Axes | None = None,
    cmap: str = "binary",
) -> plt.Axes:
    side = int(round(distribution.size**0.5))
    img = distribution.reshape(side, side)
    if threshold is not None:
        img = (img > threshold).astype(float)
    ax = ax or plt.gca()
    ax.imshow(img, cmap=cmap)
    ax.set_axis_off()
    return ax


def distribution_to_binary(distribution: np.ndarray, divisor: float = 8.0) -> np.ndarray:
    peak = distribution.max()
    return (distribution > peak / divisor).astype(float)


def distribution_to_point_cloud(distribution: np.ndarray, scale: int = 1) -> np.ndarray:
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
    if binary.ndim == 1:
        n = int(round(binary.size ** (1 / 3)))
        binary = binary.reshape(n, n, n)
    n = binary.shape[0]
    padded = np.zeros((n + 2, n + 2, n + 2))
    padded[1:-1, 1:-1, 1:-1] = binary

    verts, faces, _, _ = marching_cubes(padded, level=0.5, spacing=(1.0, 1.0, 1.0))
    mesh = trimesh.Trimesh(vertices=verts, faces=faces)
    normalize_mesh(mesh)
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
    if binary.ndim == 1:
        n = int(round(binary.size ** (1 / 3)))
        vol = binary.reshape(n, n, n) > 0
    else:
        n = binary.shape[0]
        vol = binary > 0

    if surface_only:
        from scipy.ndimage import binary_erosion

        vol &= ~binary_erosion(vol, iterations=1, border_value=0)

    xs, ys, zs = np.nonzero(vol)
    if xs.size == 0:
        return go.Figure()

    unit_v = np.array(
        [[0, 0, 0], [0, 1, 0], [1, 1, 0], [1, 0, 0], [0, 0, 1], [0, 1, 1], [1, 1, 1], [1, 0, 1]],
        dtype=float,
    )
    fi = np.array([7, 0, 0, 0, 4, 4, 6, 6, 4, 0, 3, 2])
    fj = np.array([3, 4, 1, 2, 5, 6, 5, 2, 0, 1, 6, 3])
    fk = np.array([0, 7, 2, 3, 6, 7, 1, 1, 5, 5, 7, 6])

    origins = np.stack([xs, ys, zs], axis=1).astype(float)
    vertices = (origins[:, None, :] + unit_v[None, :, :]).reshape(-1, 3) / n
    offsets = 8 * np.arange(len(origins))[:, None]

    fig = go.Figure(
        go.Mesh3d(
            x=vertices[:, 0],
            y=vertices[:, 1],
            z=vertices[:, 2],
            i=(offsets + fi).ravel(),
            j=(offsets + fj).ravel(),
            k=(offsets + fk).ravel(),
            color="#7CB9E8",
            flatshading=True,
            lighting=_LIGHTING,
            lightposition=_LIGHT_POS,
        )
    )
    return _style_3d(fig)


_MESH_COLORSCALE = [
    [0.0, "#e8e8e8"],
    [0.06, "#3b6ea8"],
    [0.35, "#f58518"],
    [1.0, "#d62728"],
]


def mesh_scalar_field(
    mesh: trimesh.Trimesh,
    values: np.ndarray,
    *,
    colorscale: str | list | None = None,
    normalize: bool = True,
    contrast: float = 0.4,
    showscale: bool = False,
    cmin: float | None = 0.0,
    cmax: float | None = 1.0,
) -> go.Figure:
    display = np.asarray(values, dtype=float)
    if normalize:
        peak = display.max()
        if peak > 0:
            display = display / peak
    if contrast != 1.0:
        display = display**contrast

    fig = go.Figure(
        go.Mesh3d(
            x=mesh.vertices[:, 0],
            y=mesh.vertices[:, 1],
            z=mesh.vertices[:, 2],
            i=mesh.faces[:, 0],
            j=mesh.faces[:, 1],
            k=mesh.faces[:, 2],
            intensity=display,
            colorscale=colorscale or _MESH_COLORSCALE,
            cmin=cmin,
            cmax=cmax,
            showscale=showscale,
            flatshading=False,
            lighting=_LIGHTING,
            lightposition=_LIGHT_POS,
        )
    )
    return _style_3d(fig, auto_range=True)


def mesh_scalar_field_grid(
    mesh: trimesh.Trimesh,
    fields: list[np.ndarray],
    *,
    rows: int,
    cols: int,
    subtitles: list[str] | None = None,
    normalize: bool = True,
    contrast: float = 0.4,
    height: int = 900,
    width: int = 1600,
    title: str | None = None,
) -> go.Figure:
    """Plotly grid of scalar fields on the same mesh (e.g. transport frames)."""
    import plotly.subplots as sp

    fig = sp.make_subplots(
        rows=rows,
        cols=cols,
        specs=[[{"type": "scene"}] * cols] * rows,
        horizontal_spacing=0.01,
        vertical_spacing=0.03,
        subplot_titles=subtitles,
    )
    scene_layout = None
    for idx, values in enumerate(fields):
        panel = mesh_scalar_field(
            mesh, values, normalize=normalize, contrast=contrast, showscale=idx == 0
        )
        if scene_layout is None:
            scene_layout = panel.layout.scene.to_plotly_json()
        fig.add_trace(panel.data[0], row=idx // cols + 1, col=idx % cols + 1)

    if scene_layout is not None:
        fig.update_scenes(scene_layout)
    fig.update_layout(
        height=height,
        width=width,
        paper_bgcolor="white",
        margin=dict(l=0, r=0, t=50 if title else 30, b=0),
        title=dict(text=title, x=0.5, xanchor="center") if title else None,
    )
    return fig
