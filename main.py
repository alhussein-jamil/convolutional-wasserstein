"""Demos for the Convolutional Wasserstein Distances reimplementation.

Usage
-----
    python -m main shapes     # 5x5 grid of 2-D shape barycenters + dots-to-star gif
    python -m main portrait   # 1-D portrait morph (Monge <-> Kantorovich)
    python -m main meshes     # 3-D mesh barycenters via grid convolution
    python -m main gaussian   # Heat-kernel Gaussians + interpolation on a mesh
"""

from __future__ import annotations

import argparse
import os
import pickle
import tempfile
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import imageio.v2 as imageio
import matplotlib.pyplot as plt
import numpy as np
import plotly.subplots as sp
import scipy.linalg as slin
import trimesh

from src import (
    Mesh,
    cotangent_laplacian,
    gaussian_on_mesh,
    grid_barycenter,
    load_binary_image,
    load_grayscale_image,
    mesh_heat_operator,
    wasserstein_barycenter,
)
from src.viz import (
    distribution_to_binary,
    distribution_to_point_cloud,
    mesh_scalar_field,
    point_cloud,
    save_image_sequence,
    voxel_cubes,
    voxel_isosurface,
)

REPO = Path(__file__).resolve().parent
OUTPUT = REPO / "output"


# --------------------------------------------------------------------------- #
# Parallel barycenter map
# --------------------------------------------------------------------------- #

# Workers receive ``mus`` once via ``initializer``; only the per-task coefficient
# is shipped through the queue, keeping IPC cost ~O(grid_size^2) integers.

_CTX: dict = {}


def _init_worker(mus: list, n: int, gamma: float, iterations: int, sharpen: bool) -> None:
    _CTX.update(mus=mus, n=n, gamma=gamma, iterations=iterations, sharpen=sharpen)


def _bary_worker(coef: np.ndarray) -> np.ndarray:
    return grid_barycenter(
        _CTX["mus"],
        coef,
        _CTX["n"],
        gamma=_CTX["gamma"],
        iterations=_CTX["iterations"],
        sharpen=_CTX["sharpen"],
    )


def _bary_map(
    mus: list,
    coefs: list,
    n: int,
    gamma: float,
    iterations: int,
    sharpen: bool,
    workers: int | None,
) -> list[np.ndarray]:
    """Compute ``grid_barycenter`` for many ``coefs`` in parallel (or serial if workers<=1)."""
    workers = workers or os.cpu_count() or 1
    workers = min(workers, len(coefs))
    if workers <= 1:
        return [
            grid_barycenter(mus, c, n, gamma=gamma, iterations=iterations, sharpen=sharpen)
            for c in coefs
        ]
    with ProcessPoolExecutor(
        max_workers=workers,
        initializer=_init_worker,
        initargs=(mus, n, gamma, iterations, sharpen),
    ) as ex:
        return list(ex.map(_bary_worker, coefs))


# --------------------------------------------------------------------------- #
# 2-D shapes
# --------------------------------------------------------------------------- #


def demo_shapes(
    grid_size: int = 5,
    gamma: float = 0.0015,
    iterations: int = 3,
    workers: int | None = None,
) -> None:
    shape_dir = REPO / "data" / "images" / "shapes"
    shapes = {
        name: load_binary_image(shape_dir / f"shape{i}filled.png").ravel()
        for name, i in [("circle", 1), ("dots", 2), ("star", 3), ("fivestar", 4)]
    }
    mus = [shapes["circle"], shapes["star"], shapes["fivestar"], shapes["dots"]]
    n = int(round(mus[0].size ** 0.5))

    s = grid_size - 1
    coefs = [
        np.array([(s - j) * (s - i), j * (s - i), j * i, (s - j) * i]) / s**2
        for i in range(grid_size)
        for j in range(grid_size)
    ]
    barys = _bary_map(mus, coefs, n, gamma, iterations, True, workers)

    _, axes = plt.subplots(grid_size, grid_size, figsize=(11, 11))
    for k, bary in enumerate(barys):
        ax = axes[k // grid_size, k % grid_size]
        ax.imshow((bary.reshape(n, n) > 1e-6), cmap="binary")
        ax.set_axis_off()
    plt.subplots_adjust(left=0, bottom=0, right=1, top=1, wspace=0, hspace=0)
    plt.savefig(OUTPUT / "shapes_grid.png", bbox_inches="tight")
    plt.close()
    print(f"wrote {OUTPUT / 'shapes_grid.png'}")

    n_frames = 20
    ts = np.linspace(0, 1, n_frames)
    interp_coefs = [np.array([t, 1 - t]) for t in ts]
    interp_barys = _bary_map(
        [shapes["dots"], shapes["star"]],
        interp_coefs,
        n,
        gamma,
        iterations,
        sharpen=False,
        workers=workers,
    )

    frames: list[np.ndarray] = []
    with tempfile.TemporaryDirectory() as tmp:
        for i, bary in enumerate(interp_barys):
            plt.imshow((bary.reshape(n, n) > 5e-6).astype(float), cmap="binary")
            plt.axis("off")
            frame = Path(tmp) / f"f{i:02d}.png"
            plt.savefig(frame, bbox_inches="tight")
            plt.close()
            frames.append(imageio.imread(frame))
    save_image_sequence(frames, OUTPUT / "dots_to_star.gif", duration=0.1)
    print(f"wrote {OUTPUT / 'dots_to_star.gif'}")


# --------------------------------------------------------------------------- #
# Portrait morph
# --------------------------------------------------------------------------- #


def demo_portrait(
    gamma: float = 0.0002,
    iterations: int = 100,
    n_frames: int = 20,
    workers: int | None = None,
) -> None:
    monge = load_grayscale_image(REPO / "data" / "monge.png").ravel()
    kant = load_grayscale_image(REPO / "data" / "kantorowich.png").ravel()
    n = int(round(monge.size**0.5))

    coefs = [np.array([t, 1 - t]) for t in np.linspace(0, 1, n_frames + 1)]
    barys = _bary_map([monge, kant], coefs, n, gamma, iterations, False, workers)

    frames: list[np.ndarray] = []
    with tempfile.TemporaryDirectory() as tmp:
        for i, bary in enumerate(barys):
            plt.imshow(bary.reshape(n, n), cmap="binary")
            plt.axis("off")
            frame = Path(tmp) / f"f{i:02d}.png"
            plt.savefig(frame, bbox_inches="tight")
            plt.close()
            frames.append(imageio.imread(frame))
    save_image_sequence(frames, OUTPUT / "portrait.gif", duration=0.1)
    print(f"wrote {OUTPUT / 'portrait.gif'}")


# --------------------------------------------------------------------------- #
# 3-D meshes via voxel grid + convolutional heat kernel
# --------------------------------------------------------------------------- #


def _load_or_build_meshes(n: int, names: list[str]) -> list[Mesh]:
    cache_dir = OUTPUT / "meshes_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    meshes_dir = REPO / "data" / "meshes"
    out = []
    for name in names:
        cache = cache_dir / f"{name}_{n}.pkl"
        if cache.is_file():
            with open(cache, "rb") as f:
                out.append(pickle.load(f))
            print(f"loaded {name} from cache")
            continue
        m = Mesh.from_file(meshes_dir / f"{name}.off", n=n)
        print(f"voxelizing {name} ...")
        m.voxelize()
        m.mesh = None  # drop the trimesh + its caches before pickling
        with open(cache, "wb") as f:
            pickle.dump(m, f)
        out.append(m)
    return out


def demo_meshes(
    names: tuple[str, ...] = ("duck", "torus", "mushroom", "sphere_102"),
    n: int = 50,
    gamma: float = 0.0015,
    iterations: int = 4,
    grid_size: int = 4,
    workers: int | None = None,
) -> None:
    meshes = _load_or_build_meshes(n, list(names))

    fig_pts = point_cloud(distribution_to_point_cloud(meshes[0].distribution, scale=3))
    fig_pts.write_html(str(OUTPUT / f"{names[0]}_points.html"))

    fig_cubes = voxel_cubes(meshes[0].binary)
    fig_cubes.write_html(str(OUTPUT / f"{names[0]}_cubes.html"))

    out_dir = OUTPUT / "generated" / "_".join(names)
    out_dir.mkdir(parents=True, exist_ok=True)

    s = grid_size - 1
    coefs = [
        np.array([(s - j) * (s - i), j * (s - i), j * i, (s - j) * i]) / s**2
        for i in range(grid_size)
        for j in range(grid_size)
    ]
    mus = [m.binary_distribution for m in meshes]
    barys = _bary_map(mus, coefs, n, gamma, iterations, False, workers)

    # Per-corner pure colors (one per input mesh); each grid cell is the
    # barycentric blend, so the gradient across the grid is visible at a glance.
    corner_rgb = np.array(
        [
            [0.93, 0.27, 0.27],  # red
            [0.30, 0.69, 0.31],  # green
            [0.13, 0.45, 0.86],  # blue
            [0.98, 0.75, 0.18],  # yellow
        ]
    )

    fig = sp.make_subplots(
        rows=grid_size,
        cols=grid_size,
        specs=[[{"type": "scene"}] * grid_size] * grid_size,
        horizontal_spacing=0.005,
        vertical_spacing=0.005,
    )
    for idx, (coef, bary) in enumerate(zip(coefs, barys, strict=False)):
        i, j = divmod(idx, grid_size)
        rgb = np.clip(coef @ corner_rgb, 0, 1)
        color = f"rgb({int(rgb[0] * 255)},{int(rgb[1] * 255)},{int(rgb[2] * 255)})"
        subplot, mesh = voxel_isosurface(distribution_to_binary(bary), smooth=30, color=color)
        tag = "_".join(f"{w:.2f}{nm}" for w, nm in zip(coef, names, strict=False))
        trimesh.exchange.export.export_mesh(mesh, out_dir / f"{tag}.off", file_type="off")
        fig.add_trace(subplot.data[0], row=i + 1, col=j + 1)

    scene_kw = subplot.layout.scene.to_plotly_json()
    fig.update_scenes(scene_kw)
    fig.update_layout(
        height=1800,
        width=1800,
        paper_bgcolor="white",
        margin=dict(l=0, r=0, t=40, b=0),
        title=dict(text=" × ".join(names), x=0.5, xanchor="center"),
    )
    out_path = OUTPUT / f"{'_'.join(names)}_grid.html"
    fig.write_html(str(out_path))
    print(f"wrote {out_path}")


# --------------------------------------------------------------------------- #
# Mesh-intrinsic heat-kernel demo
# --------------------------------------------------------------------------- #


def demo_gaussian(
    mesh_name: str = "sphere_1300", gamma: float = 0.001, iterations: int = 5
) -> None:
    mesh: trimesh.Trimesh = trimesh.load(
        REPO / "data" / "meshes" / f"{mesh_name}.off", process=False
    )
    L, areas = cotangent_laplacian(mesh)
    T = np.diag(areas) + (gamma / 2.0) * L.toarray()
    Lcho = slin.cholesky(T, lower=True)

    g1 = gaussian_on_mesh(mesh, source=0)
    g2 = gaussian_on_mesh(mesh, source=len(mesh.vertices) // 2)

    fig = mesh_scalar_field(mesh, g1 + g2)
    fig.write_html(str(OUTPUT / f"{mesh_name}_gaussians.html"))

    apply_kernel = mesh_heat_operator(Lcho)
    n_frames = 10
    rows, cols = 2, 5
    fig_grid = sp.make_subplots(
        rows=rows,
        cols=cols,
        specs=[[{"type": "scene"}] * cols] * rows,
        horizontal_spacing=0.005,
        vertical_spacing=0.01,
    )
    subplot = None
    for k in range(n_frames):
        t = k / (n_frames - 1)
        bary = wasserstein_barycenter(
            [g1, g2],
            [1 - t, t],
            area=areas,
            apply_kernel=apply_kernel,
            iterations=iterations,
            sharpen=False,
        )
        subplot = mesh_scalar_field(mesh, bary)
        fig_grid.add_trace(subplot.data[0], row=k // cols + 1, col=k % cols + 1)

    fig_grid.update_scenes(subplot.layout.scene.to_plotly_json())
    fig_grid.update_layout(
        height=600,
        width=1500,
        paper_bgcolor="white",
        margin=dict(l=0, r=0, t=40, b=0),
        title=dict(text=f"{mesh_name}: heat-kernel barycenter", x=0.5, xanchor="center"),
    )
    out_path = OUTPUT / f"{mesh_name}_barycenter.html"
    fig_grid.write_html(str(out_path))
    print(f"wrote {out_path}")


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #

DEMOS = {
    "shapes": demo_shapes,
    "portrait": demo_portrait,
    "meshes": demo_meshes,
    "gaussian": demo_gaussian,
}


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("demo", choices=list(DEMOS), help="which demo to run")
    parser.add_argument(
        "--workers",
        type=int,
        default=None,
        help="parallel workers for the demo grid (default: all CPUs; pass 1 for serial)",
    )
    args = parser.parse_args(argv)
    OUTPUT.mkdir(parents=True, exist_ok=True)
    fn = DEMOS[args.demo]
    if "workers" in fn.__code__.co_varnames:
        fn(workers=args.workers)
    else:
        fn()


if __name__ == "__main__":
    main()
