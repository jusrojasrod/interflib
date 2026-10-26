# interflib

Analysis library for the interferometry project of the Instituto Nacional de Metrología
(INM, Colombia): absolute length of gauge blocks from Twyman-Green interferograms.

The library covers the whole chain:

1. **Read the fringe fraction from the image.** Several readers are available: a
   sinusoidal fit, single-shot Fourier demodulation, or phase stepping. Each compares
   the block and platen phases at the edge column, where every phase term common to
   both surfaces cancels.
2. **Correct the wavelength** for the refractive index of air.
3. **Resolve the integer fringe order** by the method of exact fractions, with several
   wavelengths or with one wavelength and a mechanical prior.

A layered interferogram simulator supplies images with a known truth for every step.

![Synthetic interferogram](docs/images/interferogram.png)

*A synthetic frame from the simulator: a 9 mm wide gauge block wrung to a platen, about
30 fringes, CS505MU camera noise. The fraction is read from column bands on each side of
the edge (boxes). Every figure in this README is made with `tools/make_readme_figures.py`.*

## Contents

- [Install](#install)
- [From an interferogram to the fringe fraction](#from-an-interferogram-to-the-fringe-fraction)
  - [Synthetic interferograms](#synthetic-interferograms)
  - [Single shot: comparison at the edge column](#single-shot-comparison-at-the-edge-column)
  - [Single shot: Fourier demodulation](#single-shot-fourier-demodulation)
  - [Phase stepping](#phase-stepping)
  - [Sign of the fraction](#sign-of-the-fraction)
- [From the fraction to the length](#from-the-fraction-to-the-length)
- [Conventions](#conventions)
- [Package layout and status](#package-layout-and-status)
- [Development](#development)
- [References](#references)

## Install

```bash
cd code/interflib
python -m venv .venv
.venv/Scripts/activate          # Windows; use source .venv/bin/activate on POSIX
pip install -e ".[dev]"         # editable install with pytest, ruff and nbstripout
pytest
```

The library needs Python 3.10 or later. Its runtime dependencies are numpy, scipy,
matplotlib and opencv-python. The `notebooks` extra adds scikit-image, which one of the
exploration notebooks uses.

## From an interferogram to the fringe fraction

The block surface stands `L` above the platen. In double pass, the fringes on the block
are shifted against those on the platen by the **fringe fraction**
`F = frac(2L/λ)`, the part of the interference order that an image can show.

Every reader below returns `F` at a chosen **edge column** and **evaluation row**. The
reason is that `F` is only free of the instrument when both phases refer to the same
pixel: there, any term common to block and platen cancels, such as the reference tilt or
an air or vibration piston. Comparing two strips at different columns adds `g_x·Δx` as
soon as the fringes are not exactly along the rows.

### Synthetic interferograms

`interflib.simulation` builds a frame in three separable layers:

1. **Optical path.** Reference tilt, block step, flatness of block and platen, wringing
   wedge and low-order aberrations.
2. **Image formation.** Gaussian beam with vignetting, fringe visibility and blur.
3. **Sensor.** Shot and read noise, gain, quantisation and saturation.

It returns the image together with the truth. The examples in this section share the
scene below: a 2 mm block that is 51 nm long, read at 543.5 nm.

```python
from dataclasses import replace

import numpy as np

from interflib.simulation import PhaseModel, SceneGeometry, SensorModel, synthesize_interferogram

geometry = SceneGeometry.cs505mu(scale=0.5, fov_mm=25.0)      # 1224 x 1024 px, block centred
model = PhaseModel(wavelength_nm=543.5,                        # wavelength in the medium
                   central_length_nm=2_000_051.0,              # L = 2 mm + 51 nm
                   fringes_y=30.0, fringes_x=1.0)              # reference tilt, slightly rotated
sim = synthesize_interferogram(geometry, model, sensor=SensorModel.cs505mu(), seed=0)

img = sim.image.astype(float)        # 12-bit counts
# the truth: fraction at the block centre
print(round(sim.fraction_truth, 4))  # -> 0.8933
```

### Single shot: comparison at the edge column

`edge_fraction` fits a sinusoid along the rows to narrow column bands, from 30 to 180 px on
each side of the edge. It then fits a straight line to the band phases of each side and
takes the difference of the two lines at the edge column. The platen must lie to the left
of the edge; mirror the image for the opposite layout.

```python
from interflib.fringe.edge_comparison import detect_edge_column, edge_fraction

height, width = img.shape
edge_x = detect_edge_column(img, search=(0, width // 2)) - 0.5   # boundary between columns
eval_row = height // 2
rows = (eval_row - 75, eval_row + 75)                            # fit window along y

res = edge_fraction(img, edge_x, rows, eval_row, period_guess=height / 30)
print(f"F = {res.fraction:.4f} ± {res.fraction_error:.4f}")      # -> F = 0.8934 ± 0.0001
```

`fraction_error` is the standard uncertainty of `F` that comes from the extrapolation to
the edge. It does not cover systematic effects. On real images, refine the edge with
`edge_line_centre`, which finds the centre of the dark line that the chamfer draws.

![Edge comparison](docs/images/edge_comparison.png)

### Single shot: Fourier demodulation

With a spatial carrier, the Fourier transform of each column separates the fringe term
from the background. A band-pass filter around the carrier gives the phase at every
pixel. `fourier_edge_fraction` does this column by column. `fourier_edge_fraction_2d`
demodulates one region on each surface in two dimensions. Both use the same comparison at
the edge.

```python
from interflib.fringe.fourier import fourier_edge_fraction, fourier_edge_fraction_2d

fft_rows = (100, height - 100)
by_column = fourier_edge_fraction(img, edge_x, rows, eval_row, fft_rows=fft_rows)
two_d = fourier_edge_fraction_2d(img, edge_x, rows, eval_row, fft_rows=fft_rows)
print(f"{by_column.fraction:.4f} {two_d.fraction:.4f}")   # -> 0.8934 0.8934
```

The same demodulation gives a **continuous map** of a surface once the carrier is removed
and the phase is unwrapped. The map is defined up to a plane.

```python
from interflib.fringe.fourier import demodulate_2d, unwrap_phase_map

curved = synthesize_interferogram(geometry, replace(model, block_flatness_pv_nm=600.0),
                                  sensor=SensorModel.cs505mu(), seed=0).image.astype(float)
x0 = int(edge_x) + 20
face = demodulate_2d(curved, fft_rows, (x0, x0 + 400))           # a region on the block face
relief = unwrap_phase_map(face).object_phase / (2 * np.pi)       # fringes
```

![Fourier demodulation](docs/images/fourier.png)

`interflib.fringe.diagnostics` adds tools to inspect a real frame:

- `field_overview`: maps of phase, modulation and background over the whole field;
- `edge_from_modulation`: the block edge, taken from the dip in fringe modulation;
- `surface_phase`: the phase of one surface, split into a plane (the wedge) and a residual (the curvature);
- `bandwidth_sensitivity`: how much `F` moves with the filter width.

### Phase stepping

With a stack of frames taken at known phase steps, the phase comes from each pixel alone,
with no carrier needed. `hariharan_five_step` is insensitive to a linear miscalibration of
the step. `edge_fraction_from_phase` reads the fraction from the wrapped map with the
same comparison at the edge.

```python
from interflib.fringe.phase_shifting import edge_fraction_from_phase, hariharan_five_step

frames = [synthesize_interferogram(geometry, replace(model, phase_shift_rad=k * np.pi / 2),
                                   sensor=SensorModel.cs505mu(), seed=k).image.astype(float)
          for k in range(5)]
psi = hariharan_five_step(frames)            # phase, background, modulation, visibility
stepped = edge_fraction_from_phase(psi.phase, edge_x, rows, eval_row, modulation=psi.modulation)
print(f"{stepped.fraction:.4f}")             # -> 0.8934
```

`least_squares_psa` accepts any set of known steps. `interflib.fringe.unwrap` provides
three unwrappers for full maps: Itoh, quality-guided and Goldstein branch cuts.

![Phase stepping](docs/images/phase_stepping.png)

### Sign of the fraction

One intensity image cannot tell which way the interference order increases, because the
cosine is even. Read the wrong way, the fraction comes out as `1 − F`. The direction is a
property of the set-up's alignment and has to be determined on the bench. One way is a
known displacement; another is a block of certified length. Declare it to the readers
with `order_sign`:

- `+1`, the default: the order increases towards `+y`, down the image.
- `−1`: it decreases towards `+y`.

```python
reversed_tilt = synthesize_interferogram(geometry, replace(model, fringes_y=-30.0),
                                         sensor=SensorModel.cs505mu(), seed=0).image.astype(float)
wrong = edge_fraction(reversed_tilt, edge_x, rows, eval_row, height / 30)
right = edge_fraction(reversed_tilt, edge_x, rows, eval_row, height / 30, order_sign=-1)
print(f"{wrong.fraction:.4f} {right.fraction:.4f}")   # -> 0.1067 0.8933
```

## From the fraction to the length

**Refractive index of air.** This uses the modified Edlén equation of Birch & Downs, in
the form and with the input limits of the NIST Engineering Metrology Toolbox. It
reproduces that document's reference table to within 4·10⁻¹⁰.

```python
from interflib.dimensional.refraction import EdlenRefractiveIndex

air = EdlenRefractiveIndex(temperature=20.0, pressure=75_300.0, relative_humidity=50.0)
lambda_air = air.wavelength_in_air(543.5163)     # vacuum wavelength in, nm
print(f"{air.index_refraction(543.5163) - 1:.4e} {lambda_air:.4f}")   # -> 2.0260e-04 543.4062
```

**Several wavelengths.** Each wavelength allows a comb of lengths spaced `λ/2` apart;
only one length is consistent with all of them. `MultiwavelengthEFSolver` anchors the
sub-fringe position on the first wavelength and picks the integer orders that minimise
the residual. It also reports the `margin` to the runner-up, which warns of an ambiguous
choice. `CoincidenceEFSolver` scans the combs directly and serves as an independent
cross-check. The example below is the four-wavelength measurement of a 1400 µm block
published by Liu et al. (2022).

```python
from interflib.dimensional.coincidence import CoincidenceEFSolver
from interflib.dimensional.multiwavelength import MultiwavelengthEFSolver

wavelengths = [667.8184, 587.5649, 501.5702, 447.1501]   # in air, nm; the first is the anchor
fractions = [0.28, 0.85, 0.90, 0.23]

multi = MultiwavelengthEFSolver(wavelengths).solve(
    nominal_length=1_400_000.0, nominal_tolerance=1_000.0, fractions=fractions)
print(f"{multi.length / 1000:.6f} {multi.integer_orders}")   # -> 1399.840861 [4192, 4764, 5581, 6261]

comb = CoincidenceEFSolver(wavelengths).solve(1_400_000.0, 1_000.0, fractions)
print(f"{comb.length / 1000:.4f} {comb.is_unique}")          # -> 1399.8481 True
```

The two solvers agree on the integer orders. Their lengths differ by 7 nm because each
places the length within the fringe differently:

- `CoincidenceEFSolver` averages the nearest comb line of every wavelength, weighted by
  `(2/λ)²`.
- `MultiwavelengthEFSolver` takes the position within the fringe from the first wavelength.

The spread reflects how consistent the published fractions are, which are given to two
decimals only.

![Exact fractions](docs/images/exact_fractions.png)

**One wavelength and a mechanical prior.** `MonochromaticEFMSolver` (MEFM) picks the only
candidate that falls within `δ ± U` of the mechanically measured deviation. For the
candidate to be unique, `U` must stay below `λ/4`. Here it recovers the simulated block
from the fraction measured above.

```python
from interflib.dimensional.mefm import MonochromaticEFMSolver

mefm = MonochromaticEFMSolver(wavelength=543.5)   # the wavelength of the simulated scene
block = mefm.evaluate_block(nominal_length=2e6, delta_nominal_length=55.0, uncertainty=30.0,
                            measured_fraction=res.fraction)
# deviation from nominal, nm
print(f"{block['c_difference_nm']:.1f}")           # -> 51.0
```

`GaugeBlockLength` performs the classical single-wavelength computation. It takes the
nominal length as prior and reduces the result to 20 °C. Its `from_environment` method
applies the air index itself.

## Conventions

- **Units.** Wavelengths and lengths are in nanometres throughout the public API. The
  exception is `GaugeBlockLength`, which uses millimetres. Images are rows × columns:
  `y` is the row index, increasing downwards, and `x` is the column index.
- **Phase.** The column phases are those of `A·sin(w·y + φ) + C` with `w > 0`, so the
  phase increases towards `+y`. The fraction is
  `F = frac(order_sign·(φ_block − φ_platen)/2π)`.
- **Constants.** Physical constants live in `src/interflib/constants.py`, each with its
  source in a comment. The algorithms contain no magic numbers.
- **Tests.** Every method has tests. Where an analytic solution or a published reference
  value exists, the test checks against it. `tests/test_readme.py` runs the code of this
  README.
- **Sources.** The repository is self-contained. Where a step, algorithm or value rests on
  a reference, the docstring cites that source directly.

## Package layout and status

The project uses a `src` layout. Only `src/interflib/` is installed; everything else
supports development.

```
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
├── notebooks/              exploration only
├── docs/images/            figures of this README
└── simulations/            interferograms of the earlier prototype
```

| Module | What it provides |
|---|---|
| `simulation` | `synthesize_interferogram` with `SceneGeometry`, `PhaseModel`, `Illumination`, `SensorModel`; presets for the Thorlabs CS505MU camera |
| `fringe.edge_comparison` | `edge_fraction`, `fraction_at_edge`, `column_phases`, `detect_edge_column`, `edge_line_centre` |
| `fringe.fourier` | column and 2-D demodulation, `fourier_edge_fraction(_2d)`, carrier-free phase, `unwrap_phase_map` |
| `fringe.phase_shifting` | `least_squares_psa`, 3/4/5-step algorithms, `edge_fraction_from_phase`, `fraction_from_phase` (strips) |
| `fringe.unwrap` | Itoh, quality-guided and Goldstein branch-cut unwrapping, residues |
| `fringe.diagnostics` | whole-field maps, edge from the modulation, surface phase, filter-width sensitivity |
| `fringe.wedge` | block-platen wedge along `x` and `y`, in fringes/px and µrad |
| `fringe.profile_fit` | two-strip sinusoidal fit (`FringeFractionMeasurement`), kept for strip analyses |
| `fringe.models` | sine model, seed and branch, fit quality, `signed_fraction` |
| `dimensional.refraction` | `EdlenRefractiveIndex`: index of air and wavelength in air |
| `dimensional.multiwavelength` | `MultiwavelengthEFSolver` |
| `dimensional.coincidence` | `CoincidenceEFSolver` |
| `dimensional.mefm` | `MonochromaticEFMSolver` |
| `dimensional.excess_fraction` | `GaugeBlockLength` |
| `io.images` | Unicode-path-safe image loading |

The modules `io.meta`, `preprocessing`, `uncertainty` and `vibration` are still
placeholders.

## Development

```bash
pytest                                  # full suite
ruff check src tests tools              # lint rules in pyproject.toml
python tools/make_readme_figures.py     # regenerate docs/images/
```

Notebooks are for exploration only. Once a method works, it is promoted into the package
with its tests.

## References

- K. P. Birch, M. J. Downs, "An updated Edlén equation for the refractive index of air",
  *Metrologia* 30, 155–162 (1993); correction, *Metrologia* 31, 315–316 (1994).
- J. A. Stone, J. H. Zimmerman, "Index of Refraction of Air", NIST Engineering Metrology
  Toolbox, https://emtoolbox.nist.gov/Wavelength/Documentation.asp.
- T. Liu, J. Wu, R. Sato, H. Matsukuma, W. Gao, "Theoretical analysis on gauge block
  length measurement by Excess Fraction method", Nanoman-AETS, IEEE (2022).
- J. E. Decker et al., "Increasing the range of unambiguity in step-height measurement
  with multiple-wavelength interferometry", *Appl. Opt.* 42(28), 5670–5678 (2003).
- J. E. Decker, R. Schödel, G. Bönsch, *Metrologia* 41, L11–L17 (2004),
  doi:10.1088/0026-1394/41/3/L03.
- T. Doiron, J. Beers, *The Gage Block Handbook*, NIST Monograph 180 (1995).
- J. S. Rojas-Rodríguez, J. L. Galvis Arroyave, H. Vinck-Posada, "Absolute Gauge Block
  Calibration: A Single-Wavelength Excess Fraction Method Assisted by High-Precision
  Mechanical Pre-measurement", preprint, INM (2026).
- M. Takeda, H. Ina, S. Kobayashi, *J. Opt. Soc. Am.* 72, 156–160 (1982).
- M. Servín, J. A. Quiroga, J. M. Padilla, *Fringe Pattern Analysis for Optical
  Metrology*, Wiley-VCH (2014).
- P. Hariharan, B. F. Oreb, T. Eiju (1987), five-step phase-shifting algorithm.
- R. M. Goldstein, H. A. Zebker, C. L. Werner, *Radio Sci.* 23, 713 (1988).
- Thorlabs, CS505MU camera datasheet (ITN002518-D02).

## License

Proprietary. Developed for the interferometry project of the Instituto Nacional de
Metrología (INM). Not for redistribution.
