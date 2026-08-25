# interflib

Analysis library for the interferometry project of the Instituto Nacional de
Metrología (INM, Colombia): absolute length determination of gauge blocks from
Twyman-Green interferograms, and the supporting layered interferogram simulator used
as ground truth.

It turns a set of measured fringe fractions (one per laser wavelength) into an
absolute length traceable to the metre, resolving the integer fringe order by the
method of exact fractions and correcting the vacuum wavelength for the refractive
index of air.

## Layout

The project follows a `src` layout: the importable, shippable package is
`src/interflib/`; everything else (tests, notebooks, tools, simulation outputs)
supports development but does **not** travel in a `pip install`.

```
interflib/
├── pyproject.toml
├── src/interflib/          importable package
│   ├── constants.py        physical constants, each citing its source
│   ├── io/                 interferogram loading + .meta.yaml sidecar validation
│   ├── preprocessing/      ROI, illumination flattening, denoising pipeline
│   ├── fringe/             L2: fringe -> phase/fraction (sinusoidal fit; phase-shifting, Fourier later)
│   ├── dimensional/        L1: refractive index of air, excess fraction (single- and multi-wavelength)
│   ├── simulation/         layered synthetic interferograms (optical phase -> image -> sensor)
│   ├── vibration/          L3: heterodyne, fringe counting (later)
│   └── uncertainty/        L4: GUM propagation, Monte Carlo, budget (later)
├── tests/                  synthetic-signal tests; analytic cases checked to sub-nm
├── notebooks/              exploration only
├── tools/                  run_calibration.py, select_roi.py
└── simulations/            generated interferograms
```

## Status

Version 0.1.0, under active development. The pipeline that recovers an absolute length
from fringe fractions is implemented and tested end to end; the preprocessing,
vibration and uncertainty back-ends are still stubs. Each submodule states its own
status in its docstring.

| Submodule | State | What is there |
|---|---|---|
| `dimensional.refraction` | working | Modified Edlén (Birch & Downs), wavelength in air and refractive index from T, p, and humidity/dew/frost point, with input-range guards |
| `dimensional.multiwavelength` | working | `MultiwavelengthEFSolver`: standard multi-wavelength method of exact fractions; resolves the integer order from two or more wavelengths and reports the residual and the margin to the runner-up |
| `dimensional.excess_fraction` | working | `GaugeBlockLength`: single-wavelength excess fraction; air index delegated to `refraction` via `from_environment` |
| `dimensional.mefm` | working | `MonochromaticEFMSolver`: single-wavelength order resolution from a prior calibration bound |
| `fringe.profile_fit` | working | Fringe fraction by sinusoidal fit of block-vs-platen intensity profiles |
| `fringe.models` | working | Interference model and curve-fit quality metrics |
| `simulation` | working | `synthesize_interferogram`: three physical layers (phase, image, sensor) with ground truth; CS505MU camera and geometry presets |
| `io.images` | working | Unicode-path-safe grayscale interferogram loading |
| `io.meta` | stub | `.meta.yaml` sidecar validator; written when the first real data file arrives |
| `preprocessing` | stub | shared ROI / flattening / denoising pipeline |
| `uncertainty` | stub | GUM first-order and Monte Carlo propagation |
| `vibration` | stub | heterodyne detection, fringe counting |

Test suite: 42 passing (`pytest`).

## Install and development

```bash
cd code/interflib
python -m venv .venv
.venv/Scripts/activate          # Windows; use source .venv/bin/activate on POSIX
pip install -e ".[dev]"         # editable install with pytest, nbstripout
pytest                          # src is on the path via pyproject
```

Requires Python ≥ 3.10. Runtime dependencies: numpy, scipy, matplotlib,
opencv-python, pandas, pyyaml.

## Quick start

Resolve the absolute length of a gauge block from fringe fractions measured at several
wavelengths. This is the worked four-wavelength example of Liu et al. (2022) (Aoki
1400 µm block); no single wavelength pair resolves the integer order, but the joint
solve over all four pins one length.

```python
from interflib.dimensional.multiwavelength import MultiwavelengthEFSolver

# Measurement wavelengths in air (nm); the first entry is the anchor.
wavelengths = [667.8184, 587.5649, 501.5702, 447.1501]
fractions   = [0.28, 0.85, 0.90, 0.23]   # measured fringe fractions in [0, 1)

solver = MultiwavelengthEFSolver(wavelengths)
solution = solver.solve(
    nominal_length=1_400_000.0,   # prior length L' (nm), e.g. from a mechanical comparison
    nominal_tolerance=1_000.0,    # +/- search window (nm)
    fractions=fractions,
)

print(solution.length / 1000.0)   # -> 1399.8409 (um)
print(solution.integer_orders)    # resolved integer order per wavelength
print(solution.margin)            # residual gap to the runner-up; small => ambiguous
```

Correct a vacuum wavelength for the refractive index of air (modified Edlén):

```python
from interflib.dimensional.refraction import EdlenRefractiveIndex

air = EdlenRefractiveIndex(temperature=20.0, pressure=101_325.0, relative_humidity=45.0)
lambda_air = air.wavelength_in_air(632.9905846)   # HeNe vacuum wavelength (nm)
n = air.index_refraction(632.9905846)
```

Generate a synthetic interferogram with ground truth for testing the analysis:

```python
from interflib.simulation import synthesize_interferogram, SceneGeometry, SensorModel

result = synthesize_interferogram(
    geometry=SceneGeometry.cs505mu(),   # real 2448x2048 sensor geometry
    sensor=SensorModel.cs505mu(),       # 12-bit, full well >=10 ke-, read noise <2.5 e-
)
image = result.image                    # simulated frame
truth = result.fraction_truth           # exact fringe fraction that produced it
```

## Conventions

- English identifiers, NumPy-style docstrings.
- Physical constants live in `src/interflib/constants.py`, each citing its source in a
  comment; no magic numbers in the algorithms.
- Every implemented method has a test; where an analytic solution exists, the test
  checks against it (recovering simulated block lengths to sub-nanometre).
- Notebooks are for exploration only. Nothing that other code imports lives solely in a
  notebook: once a method works, it is promoted into the package.
- The repository is self-contained. Where a step, algorithm or value relies on a
  reference, the docstring cites that source directly (paper, standard or datasheet):
  Birch & Downs (modified Edlén) for the air index, Liu et al. 2022 and the
  Michelson-Benoit / Decker formulation for the multi-wavelength method, and the
  Thorlabs CS505MU datasheet for the sensor preset.

## License

Proprietary. Developed for the interferometry project of the Instituto Nacional de
Metrología (INM). Not for redistribution.
