# interflib

[![Documentation](https://github.com/jusrojasrod/interflib/actions/workflows/docs.yml/badge.svg)](https://jusrojasrod.github.io/interflib/)
[![License: BSD-3-Clause](https://img.shields.io/badge/license-BSD--3--Clause-blue.svg)](LICENSE)

Analysis library for the interferometry project of the Instituto Nacional de Metrología
(INM, Colombia): absolute length of gauge blocks from Twyman-Green interferograms.

**Documentation: <https://jusrojasrod.github.io/interflib/>**

The library covers the whole chain:

1. **Read the fringe fraction from the image.** Several readers are available: a
   sinusoidal fit, single-shot Fourier demodulation, or phase stepping. Each compares
   the block and platen phases at the edge column, where every phase term common to
   both surfaces cancels.
2. **Correct the wavelength** for the refractive index of air.
3. **Resolve the integer fringe order** by the method of exact fractions, with several
   wavelengths or with one wavelength and a mechanical prior.

A layered interferogram simulator supplies images with a known truth for every step, and
`tools/run_calibration.py` runs the whole chain on a simulated block or a real series.

![Synthetic interferogram](docs/images/interferogram.png)

*A synthetic frame from the simulator: a 9 mm wide gauge block wrung to a platen, about
30 fringes, CS505MU camera noise. The fraction is read from column bands on each side of
the edge (boxes).*

## Install

```bash
git clone https://github.com/jusrojasrod/interflib.git
cd interflib
python -m venv .venv
.venv/Scripts/activate          # Windows; use source .venv/bin/activate on POSIX
pip install -e ".[dev]"         # editable install with pytest, ruff and nbstripout
pytest
```

The library needs Python 3.10 or later. Its runtime dependencies are numpy, scipy,
matplotlib and opencv-python. The extras `tools` (pyyaml, for real series) and `docs`
(Sphinx, to build the documentation) are optional.

## A first measurement

A simulated 2 mm block, 51 nm long, read at 543.5 nm: the fraction from one frame, then the
length from the fraction and a mechanical prior.

```python
from interflib.dimensional.mefm import MonochromaticEFMSolver
from interflib.fringe.edge_comparison import detect_edge_column, edge_fraction
from interflib.simulation import PhaseModel, SceneGeometry, SensorModel, synthesize_interferogram

geometry = SceneGeometry.cs505mu(scale=0.5, fov_mm=25.0)      # 1224 x 1024 px, block centred
model = PhaseModel(wavelength_nm=543.5, central_length_nm=2_000_051.0, fringes_y=30.0, fringes_x=1.0)
img = synthesize_interferogram(geometry, model, sensor=SensorModel.cs505mu(), seed=0).image.astype(float)

height, width = img.shape
edge_x = detect_edge_column(img, search=(0, width // 2)) - 0.5   # boundary between columns
eval_row = height // 2
res = edge_fraction(img, edge_x, (eval_row - 75, eval_row + 75), eval_row, period_guess=height / 30)
print(f"F = {res.fraction:.4f}")                                 # -> F = 0.8934

block = MonochromaticEFMSolver(wavelength=543.5).solve(
    nominal_length=2e6, prior_length_dev=55.0, prior_U=30.0,     # nm: L', mechanical deviation, its U
    fringe_fraction=res.fraction, cte=11.5e-6, temp_c=20.0)
print(f"{block.length_dev:.1f} nm")                              # -> 51.0 nm
```

The [user guide](https://jusrojasrod.github.io/interflib/guide/index.html) covers every
reader and solver. The [command-line tools](https://jusrojasrod.github.io/interflib/tools/index.html)
page covers `run_calibration.py`, and the [API reference](https://jusrojasrod.github.io/interflib/api/index.html)
covers every module.

## Development

```bash
pytest                                  # full suite, including the examples of the docs
ruff check src tests tools              # lint rules in pyproject.toml
python tools/make_readme_figures.py     # regenerate docs/images/
sphinx-build -W -b html docs docs/_build/html   # the documentation (pip install -e ".[docs]")
```

Notebooks are committed without outputs (nbstripout). Once a method works, it is promoted
into the package with its tests. See [Development](https://jusrojasrod.github.io/interflib/development.html).

## License and citation

BSD 3-Clause License. Copyright (c) 2026, Instituto Nacional de Metrología de Colombia
(INM); see [LICENSE](LICENSE). The authors are listed in [AUTHORS.md](AUTHORS.md). To cite
the library, use [CITATION.cff](CITATION.cff) (GitHub shows it as *Cite this repository*).
