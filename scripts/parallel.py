"""Parallel grid barycenter evaluation for demo coefficient lattices."""

from __future__ import annotations

import os
from concurrent.futures import ProcessPoolExecutor

import numpy as np

from convolutional_wasserstein.wasserstein import grid_barycenter, make_grid_solver

_CTX: dict = {}


def _grid_shape(n: int, n_total: int) -> tuple[int, ...]:
    d = int(round(np.log(n_total) / np.log(n)))
    return (n,) * d


def _init_worker(
    mus: list[np.ndarray],
    shape: tuple[int, ...],
    gamma: float,
    iterations: int,
    sharpen: bool,
) -> None:
    _CTX.update(
        mus=mus,
        solver=make_grid_solver(shape, gamma),
        iterations=iterations,
        sharpen=sharpen,
    )


def _bary_worker(coef: np.ndarray) -> np.ndarray:
    return _CTX["solver"](
        _CTX["mus"],
        coef,
        iterations=_CTX["iterations"],
        sharpen=_CTX["sharpen"],
    )


def parallel_barycenters(
    mus: list[np.ndarray],
    coefs: list[np.ndarray],
    n: int,
    gamma: float,
    iterations: int,
    sharpen: bool,
    workers: int | None,
) -> list[np.ndarray]:
    shape = _grid_shape(n, mus[0].size)
    solver = make_grid_solver(shape, gamma)
    workers = min(workers or os.cpu_count() or 1, len(coefs))

    if workers <= 1:
        return [
            grid_barycenter(
                mus, coef, n, gamma=gamma, iterations=iterations, sharpen=sharpen, solver=solver
            )
            for coef in coefs
        ]

    with ProcessPoolExecutor(
        max_workers=workers,
        initializer=_init_worker,
        initargs=(mus, shape, gamma, iterations, sharpen),
    ) as pool:
        return list(pool.map(_bary_worker, coefs))
