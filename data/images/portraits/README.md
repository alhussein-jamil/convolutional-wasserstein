# Portrait assets

| Path | Role |
| --- | --- |
| `raw/monge.png` | Source illustration (Gaspard Monge) |
| `raw/kantorovich.png` | Source illustration (Leonid Kantorovich) |
| `monge.png` | Processed 202×202 grayscale for CLI demo |
| `kantorovich.png` | Processed 202×202 grayscale for CLI demo |
| `color/monge.png` | Processed 202×202 RGB for the notebook morph |
| `color/kantorovich.png` | Processed 202×202 RGB for the notebook morph |

Processed files are built from `raw/` with background removal:

```sh
make portraits
```

Demos load grayscale PNGs via `portrait_path(...)`. The notebook uses
`portrait_color_path(...)` for the color morph.
