"""Filesystem paths for package data and demo output."""

from __future__ import annotations

from pathlib import Path

PACKAGE_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = PACKAGE_ROOT.parent.parent
_CANDIDATES = (PROJECT_ROOT / "data", PACKAGE_ROOT / "data")
DATA_DIR = next((p for p in _CANDIDATES if (p / "meshes").is_dir()), _CANDIDATES[0])
MESHES_DIR = DATA_DIR / "meshes"
IMAGES_DIR = DATA_DIR / "images"
SHAPES_DIR = IMAGES_DIR / "shapes"
PORTRAITS_DIR = IMAGES_DIR / "portraits"
PORTRAIT_RAW_DIR = PORTRAITS_DIR / "raw"
PORTRAIT_COLOR_DIR = PORTRAITS_DIR / "color"
DEFAULT_OUTPUT = PROJECT_ROOT / "output"


def portrait_path(name: str) -> Path:
    """Processed grayscale portrait PNG used by CLI demos."""
    return PORTRAITS_DIR / f"{name}.png"


def portrait_color_path(name: str) -> Path:
    """Processed RGB portrait PNG (white background, square crop)."""
    return PORTRAIT_COLOR_DIR / f"{name}.png"
