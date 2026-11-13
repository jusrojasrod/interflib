# interflib

Analysis library for the interferometry project of the Instituto Nacional de Metrología
de Colombia (INM): **absolute length of gauge blocks from Twyman-Green interferograms**.

```{image} images/interferogram.png
:alt: Synthetic interferogram of a gauge block wrung to a platen
:width: 620px
```

*A synthetic frame from the simulator: a 9 mm wide gauge block wrung to a platen, about
30 fringes, CS505MU camera noise. The fraction is read from column bands on each side of
the edge (boxes).*

The library covers the whole measurement chain:

1. **Read the fringe fraction `F` from the image.** Three readers are available: a
   sinusoidal fit, single-shot Fourier demodulation and phase stepping. Each one compares
   the block and platen phases at a single column, where every phase term common to both
   surfaces cancels.
2. **Correct the wavelength** for the refractive index of air.
3. **Resolve the integer fringe order** by the method of exact fractions, with several
   wavelengths or with one wavelength and a mechanical prior.
4. **Reduce the length to 20 °C**, and check the parity of the reading (`F` against `1 − F`).

A layered interferogram simulator supplies images with a known truth for every step, and
`tools/run_calibration.py` runs the whole chain on a simulated block or a real series.

| Section | What you find there |
|---|---|
| [Installation](install.md) | Requirements, editable install, optional extras |
| [User guide](guide/index.md) | The methods, one page each, with examples that run |
| [Command-line tools](tools/index.md) | `run_calibration.py` and the helper scripts |
| [Notebooks](notebooks/index.md) | Worked studies, executed when the site is built (in Spanish) |
| [API reference](api/index.rst) | Every module, class and function, from the docstrings |
| [Development](development.md) | Tests, lint, figures, notebooks and this documentation |
| [References](references.md) | The sources the methods rest on |

interflib is free software under the BSD 3-Clause License. Copyright © 2026, Instituto
Nacional de Metrología de Colombia (INM). To cite it, use the repository's
[`CITATION.cff`](https://github.com/jusrojasrod/interflib/blob/main/CITATION.cff).

```{toctree}
:hidden:
:maxdepth: 2

install
guide/index
tools/index
notebooks/index
api/index
development
references
```
