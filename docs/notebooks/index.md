# Notebooks

Worked studies of the methods. Each one imports and runs the real functions of the
library, step by step and with its figures, and explains how to read every figure. They
are **in Spanish**. The notebooks are executed every time this site is built, so what they
show is what the current code does.

Some sections read frames of the INM measurement campaign of October 2026, which are not
part of the repository. On this site those sections report that the data are not found and
are skipped; everything else runs on simulated scenes with a known truth.

| Notebook | What it covers |
|---|---|
| [Simulator of the interferogram](_generated/20260821-desarrollo-simulador-interferograma.ipynb) | the three layers of `interflib.simulation` (optical path, image formation, sensor), the coordinate systems of non-square frames and the CS505MU camera presets |
| [Phase stepping and unwrapping](_generated/20260903-phase-stepping-y-desenrollado-de-fase.ipynb) | from a stack of phase-stepped frames to the wrapped phase, and from the wrapped phase to a continuous map with the Itoh, quality-guided and Goldstein unwrappers |
| [The Fourier method](_generated/20261022-metodo-de-fourier.ipynb) | single-shot demodulation: the carrier, the filter, the phase of each band, the fraction at the edge, and the two-dimensional variant |
| [Automatic detection of the analysis areas](_generated/20261027-deteccion-automatica-de-areas.ipynb) | how the library finds the fringe zone, the block edge, the outline and the centre of the face, without coordinates given by hand |
| [Parity of the reading](_generated/20261027-paridad-de-la-lectura.ipynb) | `F` or `1 − F`: the order direction and the platen side, what a wrong declaration costs and how a prior tests it |

The notebooks numbered `00` to `08` in `notebooks/` come from the first prototype of the
project. They are kept in the repository for reference but are not part of this site.

```{toctree}
:hidden:
:glob:

_generated/*
```
