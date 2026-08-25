# interflib

Analysis library for the INM interferometry project: absolute length determination
of gauge blocks from Twyman-Green interferograms.

This directory is the **library project**. It follows a `src` layout: the importable,
shippable package is `src/interflib/`; everything else here (tests, notebooks, tools,
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
docstring.

## Development

```bash
cd code/interflib
pip install -e ".[dev]"   # editable install with pytest, nbstripout
pytest                    # runs tests/ (src is on pythonpath via pyproject)
```

Conventions: English identifiers; physical constants live in
`src/interflib/constants.py`, each citing its source; no magic numbers. Where a step,
algorithm or value relies on a reference, the docstring cites that source directly
(paper, standard or datasheet) — this repository is self-contained and does not point
to any external knowledge base.
