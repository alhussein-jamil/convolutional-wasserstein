"""CLI entry point for convolutional Wasserstein demos."""

from __future__ import annotations

import argparse
import logging
from collections.abc import Callable
from pathlib import Path

from convolutional_wasserstein.logging import setup_logging
from convolutional_wasserstein.paths import DEFAULT_OUTPUT
from scripts.demos import demo_gaussian, demo_meshes, demo_portrait, demo_shapes

log = logging.getLogger("convolutional_wasserstein.demos")

DEMOS: dict[str, Callable[..., None]] = {
    "shapes": demo_shapes,
    "portrait": demo_portrait,
    "meshes": demo_meshes,
    "gaussian": demo_gaussian,
}


def main(argv: list[str] | None = None) -> None:
    setup_logging()
    parser = argparse.ArgumentParser(description="Convolutional Wasserstein demos")
    parser.add_argument("demo", choices=sorted(DEMOS))
    parser.add_argument(
        "--workers", type=int, default=None, help="parallel workers (default: all CPUs)"
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args(argv)
    if args.verbose:
        log.setLevel(logging.DEBUG)
    args.output.mkdir(parents=True, exist_ok=True)
    fn = DEMOS[args.demo]
    if "workers" in fn.__code__.co_varnames:
        fn(workers=args.workers, output=args.output)
    else:
        fn(output=args.output)


if __name__ == "__main__":
    main()
