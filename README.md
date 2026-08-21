# interflib

Analysis library for the INM interferometry project: absolute length determination
of gauge blocks from Twyman-Green interferograms.

This directory is the **library project** (isolated from the report/wiki tooling that
lives under `code/apoyo/`). It follows a `src` layout: the importable, shippable
package is `src/interflib/`; everything else here (tests, notebooks, tools,
simulations) supports development but does **not** travel in a `pip install`.

## Layout

```
interflib/
├── pyproject.toml
├── src/interflib/        importable package (fringe analysis only)
│   ├── io/               interferogram loading + .meta.yaml sidecar validation
│   ├── preprocessing/    ROI, background flattening, denoising pipeline
│   ├── fringe/           fringe -> phase/fraction: sinusoidal fit, phase-shifting, Fourier
│   ├── dimensional/      refractive index (Birch & Downs), excess fraction, MEFM
│   ├── vibration/        heterodyne, fringe counting (later)
│   └── uncertainty/      GUM propagation, Monte Carlo, budget
├── tests/                test code only (synthetic signals generated in-test)
├── notebooks/            exploration
├── tools/                library tools (build_budget, select_roi, ...)
└── simulations/          synthetic interferogram generator and outputs
```

## Status

Under construction. The package is being ported from the `Exccess_fractions`
prototype one method at a time; each module documents its own status in its
docstring. Architecture decision: `wiki/decisiones/ADR-004-arquitectura-interflib.md`.

## Development

```bash
cd code/interflib
pip install -e ".[dev]"   # editable install with pytest, nbstripout
pytest                    # runs tests/ (src is on pythonpath via pyproject)
```

Conventions (see `CLAUDE.md`): English identifiers; every non-trivial function opens
its docstring with a `Wiki:` anchor to the page it implements; physical constants live
in `src/interflib/constants.py`, each with a source citation; no magic numbers.
