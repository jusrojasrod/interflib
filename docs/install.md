# Installation

interflib needs Python 3.10 or later. Its runtime dependencies are numpy, scipy,
matplotlib and opencv-python.

```bash
git clone https://github.com/jusrojasrod/interflib.git
cd interflib
python -m venv .venv
.venv/Scripts/activate          # Windows; use source .venv/bin/activate on POSIX
pip install -e ".[dev]"         # editable install with pytest, ruff and nbstripout
pytest
```

## Optional extras

| Extra | Adds | Needed for |
|---|---|---|
| `dev` | pytest, ruff, nbstripout | running the tests and the lint, committing notebooks |
| `tools` | pyyaml | `tools/run_calibration.py --series` (reads the `.meta.yaml` sidecar) |
| `notebooks` | scikit-image | one of the legacy exploration notebooks |
| `docs` | Sphinx, pydata-sphinx-theme, myst-nb, sphinx-argparse, sphinx-copybutton, ipykernel | building this documentation |

Several extras can be combined: `pip install -e ".[dev,tools,docs]"`.

## Layout

The project uses a `src` layout. Only `src/interflib/` is installed; the rest of the
repository supports development.

```text
interflib/
├── src/interflib/
│   ├── constants.py        physical constants, each with its source
│   ├── simulation/         layered synthetic interferograms with ground truth
│   ├── fringe/             fringe image -> phase -> fraction
│   ├── dimensional/        air index and exact fractions -> length
│   ├── io/                 image loading (sidecar validation: placeholder)
│   ├── preprocessing/      placeholder
│   ├── uncertainty/        placeholder (GUM and Monte Carlo propagation)
│   └── vibration/          placeholder (heterodyne, fringe counting)
├── tests/                  pytest suite, synthetic signals and published reference values
├── tools/                  run_calibration.py, select_roi.py, make_readme_figures.py
├── notebooks/              worked studies and exploration
├── docs/                   this documentation (Sphinx)
└── simulations/            interferograms of the earlier prototype
```
