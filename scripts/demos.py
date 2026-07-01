"""Demo implementations for convolutional Wasserstein barycenters."""

from __future__ import annotations

import logging
import pickle
import tempfile
from pathlib import Path

import imageio.v2 as imageio
import matplotlib.pyplot as plt
import numpy as np
import plotly.subplots as sp
import scipy.linalg as slin
import trimesh

from convolutional_wasserstein import (
    VoxelMesh,
    cotangent_laplacian,
    gaussian_on_mesh,
    load_binary_image,
    load_grayscale_image,
    mesh_heat_operator,
    normalize_mesh,
    opposite_vertices,
    wasserstein_barycenter,
)
from convolutional_wasserstein.paths import DEFAULT_OUTPUT, IMAGES_DIR, MESHES_DIR, portrait_path
from scripts.assets import bilinear_coefs, ensure_demo_images
from scripts.parallel import parallel_barycenters
from scripts.viz import (
    distribution_to_binary,
    distribution_to_point_cloud,
    mesh_scalar_field_grid,
    point_cloud,
    render_distribution,
    save_gif,
    voxel_cubes,
    voxel_isosurface,
)

log = logging.getLogger("convolutional_wasserstein.demos")


def demo_shapes(
    grid_size: int = 5,
    gamma: float = 0.0015,
    iterations: int = 3,
    workers: int | None = None,
    output: Path = DEFAULT_OUTPUT,
) -> None:
    ensure_demo_images()
    shape_dir = IMAGES_DIR / "shapes"
    shapes = {
        name: load_binary_image(shape_dir / f"shape{i}filled.png").ravel()
        for name, i in [("circle", 1), ("dots", 2), ("star", 3), ("fivestar", 4)]
    }
    mus = [shapes["circle"], shapes["star"], shapes["fivestar"], shapes["dots"]]
    n = int(round(mus[0].size ** 0.5))
    barys = parallel_barycenters(
        mus, bilinear_coefs(grid_size), n, gamma, iterations, True, workers
    )

    _, axes = plt.subplots(grid_size, grid_size, figsize=(11, 11))
    for k, bary in enumerate(barys):
        render_distribution(bary, ax=axes[k // grid_size, k % grid_size])
    plt.subplots_adjust(left=0, bottom=0, right=1, top=1, wspace=0, hspace=0)
    grid_path = output / "shapes_grid.png"
    plt.savefig(grid_path, bbox_inches="tight")
    plt.close()
    log.info("wrote %s", grid_path)

    interp_coefs = [np.array([t, 1 - t]) for t in np.linspace(0, 1, 20)]
    interp_barys = parallel_barycenters(
        [shapes["dots"], shapes["star"]], interp_coefs, n, gamma, iterations, False, workers
    )
    frames: list[np.ndarray] = []
    with tempfile.TemporaryDirectory() as tmp:
        for i, bary in enumerate(interp_barys):
            render_distribution(bary, threshold=5e-6)
            frame = Path(tmp) / f"f{i:02d}.png"
            plt.savefig(frame, bbox_inches="tight")
            plt.close()
            frames.append(imageio.imread(frame))
    gif_path = output / "dots_to_star.gif"
    save_gif(frames, gif_path)
    log.info("wrote %s", gif_path)


def demo_portrait(
    gamma: float = 0.0002,
    iterations: int = 100,
    n_frames: int = 20,
    workers: int | None = None,
    output: Path = DEFAULT_OUTPUT,
) -> None:
    ensure_demo_images()
    monge = load_grayscale_image(portrait_path("monge")).ravel()
    kant = load_grayscale_image(portrait_path("kantorovich")).ravel()
    n = int(round(monge.size**0.5))

    coefs = [np.array([t, 1 - t]) for t in np.linspace(0, 1, n_frames + 1)]
    barys = parallel_barycenters([monge, kant], coefs, n, gamma, iterations, False, workers)

    frames: list[np.ndarray] = []
    with tempfile.TemporaryDirectory() as tmp:
        for i, bary in enumerate(barys):
            render_distribution(bary, threshold=None)
            frame = Path(tmp) / f"f{i:02d}.png"
            plt.savefig(frame, bbox_inches="tight")
            plt.close()
            frames.append(imageio.imread(frame))
    gif_path = output / "portrait.gif"
    save_gif(frames, gif_path)
    log.info("wrote %s", gif_path)


def _load_voxel_meshes(n: int, names: list[str], cache_dir: Path) -> list[VoxelMesh]:
    cache_dir.mkdir(parents=True, exist_ok=True)
    meshes: list[VoxelMesh] = []
    for name in names:
        cache = cache_dir / f"{name}_{n}.pkl"
        if cache.is_file():
            with open(cache, "rb") as f:
                meshes.append(pickle.load(f))
            log.info("loaded %s from cache", name)
            continue
        vm = VoxelMesh.from_file(MESHES_DIR / f"{name}.off", n=n)
        log.info("voxelizing %s", name)
        vm.voxelize()
        vm.mesh = None
        with open(cache, "wb") as f:
            pickle.dump(vm, f)
        meshes.append(vm)
    return meshes


def demo_meshes(
    names: tuple[str, ...] = ("duck", "torus", "mushroom", "sphere_102"),
    n: int = 50,
    gamma: float = 0.0015,
    iterations: int = 4,
    grid_size: int = 4,
    workers: int | None = None,
    output: Path = DEFAULT_OUTPUT,
) -> None:
    cache_dir = output / "meshes_cache"
    meshes = _load_voxel_meshes(n, list(names), cache_dir)

    point_cloud(distribution_to_point_cloud(meshes[0].distribution, scale=3)).write_html(
        str(output / f"{names[0]}_points.html")
    )
    voxel_cubes(meshes[0].binary).write_html(str(output / f"{names[0]}_cubes.html"))

    out_dir = output / "generated" / "_".join(names)
    out_dir.mkdir(parents=True, exist_ok=True)

    coefs = bilinear_coefs(grid_size)
    barys = parallel_barycenters(
        [m.binary_distribution for m in meshes], coefs, n, gamma, iterations, False, workers
    )

    corner_rgb = np.array(
        [[0.93, 0.27, 0.27], [0.30, 0.69, 0.31], [0.13, 0.45, 0.86], [0.98, 0.75, 0.18]]
    )
    fig = sp.make_subplots(
        rows=grid_size,
        cols=grid_size,
        specs=[[{"type": "scene"}] * grid_size] * grid_size,
        horizontal_spacing=0.005,
        vertical_spacing=0.005,
    )
    subplot = None
    for idx, (coef, bary) in enumerate(zip(coefs, barys, strict=True)):
        i, j = divmod(idx, grid_size)
        rgb = np.clip(coef @ corner_rgb, 0, 1)
        color = f"rgb({int(rgb[0] * 255)},{int(rgb[1] * 255)},{int(rgb[2] * 255)})"
        subplot, mesh = voxel_isosurface(distribution_to_binary(bary), smooth=30, color=color)
        tag = "_".join(f"{w:.2f}{nm}" for w, nm in zip(coef, names, strict=True))
        trimesh.exchange.export.export_mesh(mesh, out_dir / f"{tag}.off", file_type="off")
        fig.add_trace(subplot.data[0], row=i + 1, col=j + 1)

    fig.update_scenes(subplot.layout.scene.to_plotly_json())
    fig.update_layout(
        height=1800,
        width=1800,
        paper_bgcolor="white",
        margin=dict(l=0, r=0, t=40, b=0),
        title=dict(text=" × ".join(names), x=0.5, xanchor="center"),
    )
    out_path = output / f"{'_'.join(names)}_grid.html"
    fig.write_html(str(out_path))
    log.info("wrote %s", out_path)


def demo_gaussian(
    mesh_name: str = "sphere_1300",
    gamma: float = 0.001,
    iterations: int = 30,
    sigma: float = 0.12,
    output: Path = DEFAULT_OUTPUT,
) -> None:
    mesh: trimesh.Trimesh = trimesh.load(MESHES_DIR / f"{mesh_name}.off", process=False)
    normalize_mesh(mesh)
    laplacian, areas = cotangent_laplacian(mesh)
    cholesky = slin.cholesky(np.diag(areas) + (gamma / 2.0) * laplacian.toarray(), lower=True)
    apply_kernel = mesh_heat_operator(cholesky)

    src_a, src_b = opposite_vertices(mesh)
    g1 = gaussian_on_mesh(mesh, source=src_a, sigma=sigma)
    g2 = gaussian_on_mesh(mesh, source=src_b, sigma=sigma)
    mesh_scalar_field_grid(
        mesh,
        [g1, g2],
        rows=1,
        cols=2,
        subtitles=["source A", "source B"],
        title=f"{mesh_name}: geodesic sources",
        height=420,
        width=900,
    ).write_html(str(output / f"{mesh_name}_gaussians.html"))

    frame_ts = [0.0, 0.15, 0.3, 0.45, 0.55, 0.7, 0.85, 1.0]
    barys = [
        wasserstein_barycenter(
            [g1, g2],
            [1 - t, t],
            area=areas,
            apply_kernel=apply_kernel,
            iterations=iterations,
            sharpen=False,
        )
        for t in frame_ts
    ]
    fig_grid = mesh_scalar_field_grid(
        mesh,
        barys,
        rows=2,
        cols=4,
        subtitles=[f"t={t:g}" for t in frame_ts],
        title=f"{mesh_name}: heat-kernel barycenter transport",
        height=900,
        width=1600,
    )
    out_path = output / f"{mesh_name}_barycenter.html"
    fig_grid.write_html(str(out_path))
    log.info("wrote %s", out_path)
