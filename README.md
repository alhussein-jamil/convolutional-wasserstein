# Convolutional Wasserstein Distances

A clean reimplementation of

> Solomon, de Goes, Peyré, Cuturi, Butscher, Nguyen, Du, Guibas (2015).
> **Convolutional Wasserstein Distances: Efficient Optimal Transportation on Geometric Domains.**
> *ACM Transactions on Graphics (SIGGRAPH).*
> [Paper](https://people.csail.mit.edu/jsolomon/assets/convolutional_w2.compressed.pdf)

The paper's key idea: in entropy-regularized optimal transport, every step
of the Sinkhorn iteration only ever needs the kernel `K` as a linear
operator `K v`. On a Euclidean grid that operator is a Gaussian
convolution — separable in `O(n^d)` per step — and on a triangle mesh it
is one backward-Euler step of the heat equation `(D_a + tL)^{-1}`. The
same Sinkhorn loop therefore solves Wasserstein barycenters on images,
voxel grids, and curved surfaces by swapping a single callable.

## Repository layout

```
src/
  convolution.py      separable Gaussian convolution (O(n^d))
  wasserstein.py      unified Sinkhorn barycenter (Algorithm 2)
  post_processing.py  entropic sharpening (Algorithm 3)
  mesh.py             cotangent Laplacian, voxelization, geodesics
  io.py               image loaders
  viz.py              plotly / matplotlib helpers
data/                 meshes (.off) and shape / portrait images
main.py               CLI driving the four demos below
EA Convolution.ipynb  the same demos, interactive
build_notebook.py     regenerates the notebook from its cell sources
```

## Install

```sh
make install           # creates ./venv and installs the package in editable mode
# or
python -m venv venv && ./venv/bin/pip install -e .
```

Add `[notebook]` / `[dev]` extras (or `make dev`) for Jupyter and ruff.

## Run the demos

```sh
make demo-2d           # 5x5 grid of bilinear shape barycenters + dots-to-star gif
make demo-portrait     # grayscale portrait morph (writes output/portrait.gif)
make demo-3d           # voxelized solid-mesh barycenter grid
make demo-gaussian     # heat-kernel barycenter between two Gaussians on a sphere
make notebook          # interactive notebook of the same demos
```

Each demo accepts `--workers N` (`python -m main shapes --workers 8`) to
parallelize the coefficient grid across processes. Default is "all CPUs".
Outputs land in `output/`.

## Performance notes

* The Gaussian kernel is truncated at `6σ` (where `σ = n·√γ / 2` cells).
  At that cutoff the dropped tail is below float64 round-off, so the result
  is numerically identical to using the full `2n−1` kernel — but ~10–20×
  faster per convolution for typical `γ`.
* Barycenter grids run in parallel via `ProcessPoolExecutor` with `mus`
  shared once through a worker initializer; serial and parallel runs are
  bitwise identical.
* Combined, the 5×5 shape demo at n=404 goes from ~200 s serial-naive to
  ~3 s on a 20-core machine.

## API in one minute

```python
from src import grid_barycenter, wasserstein_barycenter, mesh_heat_operator

# 2-D image barycenter
bary = grid_barycenter([mu1, mu2], weights=[0.5, 0.5], n=128, gamma=1e-3)

# Mesh barycenter (factor T = D_a + (gamma/2) L once with Cholesky, reuse it)
from src import cotangent_laplacian
import numpy as np, scipy.linalg as slin
L, areas = cotangent_laplacian(mesh)
Lcho = slin.cholesky(np.diag(areas) + 0.5 * gamma * L.toarray(), lower=True)
bary = wasserstein_barycenter(
    [mu1, mu2], weights=[0.5, 0.5], area=areas,
    apply_kernel=mesh_heat_operator(Lcho),
)
```

## Notes

* Originally built as a course project for *MAP588 — Emerging Topics in Machine
  Learning* (École Polytechnique).
* The `convolutional_w2.compressed.tex` in the repository root is the
  authoritative paper, included for reference.
