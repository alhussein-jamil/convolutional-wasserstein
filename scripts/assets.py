"""Synthetic demo assets and barycentric coefficient grids."""

from __future__ import annotations

import numpy as np
from PIL import Image, ImageDraw

from convolutional_wasserstein.paths import IMAGES_DIR, portrait_color_path, portrait_path


def _grid(n: int) -> tuple[np.ndarray, np.ndarray]:
    y, x = np.mgrid[0:n, 0:n]
    center = (n - 1) / 2.0
    return (x - center) / n, (y - center) / n


def synthetic_circle(n: int, radius: float = 0.28) -> np.ndarray:
    x, y = _grid(n)
    dist = (x**2 + y**2 <= radius**2).astype(np.float64)
    return dist / dist.sum()


def synthetic_dots(n: int) -> np.ndarray:
    x, y = _grid(n)
    dist = np.zeros((n, n), dtype=np.float64)
    for ox, oy in ((-0.2, -0.2), (0.2, -0.2), (0.0, 0.2)):
        dist += np.exp(-((x - ox) ** 2 + (y - oy) ** 2) / 0.002)
    return dist / dist.sum()


def synthetic_star(n: int, points: int = 5) -> np.ndarray:
    angles = np.linspace(0, 2 * np.pi, 2 * points, endpoint=False)
    radii = np.tile([0.32, 0.12], points)
    xs = 0.5 + radii * np.cos(angles)
    ys = 0.5 + radii * np.sin(angles)
    img = Image.new("L", (n, n), 255)
    ImageDraw.Draw(img).polygon(list(zip(xs * n, ys * n, strict=False)), fill=0)
    mat = 1.0 - np.asarray(img, dtype=np.float64) / 255.0
    return mat / mat.sum()


def bilinear_coefs(grid_size: int) -> list[np.ndarray]:
    """Corner weights for a ``grid_size x grid_size`` barycentric lattice."""
    s = float(grid_size - 1)
    ij = np.arange(grid_size, dtype=np.float64)
    i, j = np.meshgrid(ij, ij, indexing="ij")
    stacked = np.stack(
        [(s - j) * (s - i), j * (s - i), j * i, (s - j) * i],
        axis=-1,
    ).reshape(-1, 4)
    return [row / (s * s) for row in stacked]


def ensure_demo_images(portrait_size: int = 202) -> None:
    """Write demo PNG assets under ``data/images`` if they are missing."""
    from scripts.prepare_portraits import PORTRAIT_SIZE, write_portraits

    shapes_dir = IMAGES_DIR / "shapes"
    shapes_dir.mkdir(parents=True, exist_ok=True)
    n = portrait_size
    for name, dist in {
        "shape1filled.png": synthetic_circle(n),
        "shape2filled.png": synthetic_dots(n),
        "shape3filled.png": synthetic_star(n, points=5),
        "shape4filled.png": synthetic_star(n, points=8),
    }.items():
        path = shapes_dir / name
        if path.is_file():
            continue
        Image.fromarray((255 * (1 - dist / dist.max())).astype(np.uint8)).save(path)

    need_portraits = False
    for name in ("monge", "kantorovich"):
        for path in (portrait_path(name), portrait_color_path(name)):
            if not path.is_file():
                need_portraits = True
                break
            with Image.open(path) as img:
                if img.size != (n, n):
                    need_portraits = True
                    break
        if need_portraits:
            break
    if need_portraits:
        write_portraits(size=portrait_size or PORTRAIT_SIZE)
