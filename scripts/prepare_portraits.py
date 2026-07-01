"""Prepare portrait demo assets from ``data/images/portraits/raw/``."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image, ImageOps
from scipy.ndimage import binary_closing, label

from convolutional_wasserstein.paths import (
    PORTRAIT_COLOR_DIR,
    PORTRAIT_RAW_DIR,
    PORTRAITS_DIR,
    portrait_color_path,
    portrait_path,
)

PORTRAIT_SIZE = 202
NAMES = ("monge", "kantorovich")


def _foreground_mask(rgb: np.ndarray, thresh: int = 32) -> np.ndarray:
    """Foreground = not reachable from the border through near-black pixels."""
    dark = np.max(rgb, axis=2) <= thresh
    h, w = dark.shape
    bg = np.zeros((h, w), dtype=bool)
    stack: list[tuple[int, int]] = []
    for x in range(w):
        if dark[0, x]:
            stack.append((0, x))
        if dark[h - 1, x]:
            stack.append((h - 1, x))
    for y in range(h):
        if dark[y, 0]:
            stack.append((y, 0))
        if dark[y, w - 1]:
            stack.append((y, w - 1))
    while stack:
        y, x = stack.pop()
        if bg[y, x] or not dark[y, x]:
            continue
        bg[y, x] = True
        if y > 0:
            stack.append((y - 1, x))
        if y + 1 < h:
            stack.append((y + 1, x))
        if x > 0:
            stack.append((y, x - 1))
        if x + 1 < w:
            stack.append((y, x + 1))
    mask = ~bg
    mask = binary_closing(mask, iterations=1)
    labels, n = label(mask)
    if n > 1:
        sizes = np.bincount(labels.ravel())
        sizes[0] = 0
        mask = labels == sizes.argmax()
    return mask


def _crop_to_mask(img: Image.Image, mask: np.ndarray, pad: float = 0.04) -> Image.Image:
    ys, xs = np.where(mask)
    if ys.size == 0:
        return img
    y0, y1 = ys.min(), ys.max()
    x0, x1 = xs.min(), xs.max()
    h, w = y1 - y0, x1 - x0
    py, px = int(h * pad), int(w * pad)
    return img.crop(
        (
            max(0, x0 - px),
            max(0, y0 - py),
            min(img.width, x1 + px),
            min(img.height, y1 + py),
        )
    )


def _prepare_portrait_base(path: Path) -> tuple[Image.Image, np.ndarray]:
    img = Image.open(path).convert("RGB")
    rgb = np.asarray(img, dtype=np.float64)
    mask = _foreground_mask(rgb)
    img = _crop_to_mask(img, mask)
    mask = _foreground_mask(np.asarray(img.convert("RGB"), dtype=np.float64))
    return img, mask


def prepare_portrait(path: Path, size: int = PORTRAIT_SIZE) -> Image.Image:
    """Square grayscale portrait on white (dark pixels carry transport mass)."""
    img, mask = _prepare_portrait_base(path)
    alpha = mask.astype(np.float64)
    gray = np.asarray(img.convert("L"), dtype=np.float64)
    composed = 255.0 * (1.0 - alpha) + gray * alpha
    out = Image.fromarray(composed.astype(np.uint8), mode="L")
    return ImageOps.fit(out, (size, size), method=Image.Resampling.LANCZOS, centering=(0.5, 0.45))


def prepare_color_portrait(path: Path, size: int = PORTRAIT_SIZE) -> Image.Image:
    """Square RGB portrait on white (same crop as the grayscale asset)."""
    img, mask = _prepare_portrait_base(path)
    rgb = np.asarray(img, dtype=np.float64)
    alpha = mask[..., np.newaxis]
    composed = 255.0 * (1.0 - alpha) + rgb * alpha
    out = Image.fromarray(composed.astype(np.uint8), mode="RGB")
    return ImageOps.fit(out, (size, size), method=Image.Resampling.LANCZOS, centering=(0.5, 0.45))


def write_portraits(size: int = PORTRAIT_SIZE) -> None:
    PORTRAITS_DIR.mkdir(parents=True, exist_ok=True)
    PORTRAIT_COLOR_DIR.mkdir(parents=True, exist_ok=True)
    for name in NAMES:
        src = PORTRAIT_RAW_DIR / f"{name}.png"
        if not src.is_file():
            raise FileNotFoundError(f"missing portrait source: {src}")
        for prepare, dest in (
            (prepare_portrait, portrait_path(name)),
            (prepare_color_portrait, portrait_color_path(name)),
        ):
            out = prepare(src, size=size)
            out.save(dest)
            print(f"wrote {dest}")


if __name__ == "__main__":
    write_portraits()
