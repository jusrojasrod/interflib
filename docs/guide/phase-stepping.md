# Phase stepping

With a stack of frames taken at known phase steps, the phase comes from each pixel alone,
with no carrier needed. `hariharan_five_step` (Hariharan, Oreb & Eiju 1987) is insensitive
to a linear miscalibration of the step. `edge_fraction_from_phase` reads the fraction from
the wrapped map with the same comparison at the edge as the single-shot readers.

```python
from dataclasses import replace

import numpy as np

from interflib.fringe.edge_comparison import detect_edge_column
from interflib.simulation import PhaseModel, SceneGeometry, SensorModel, synthesize_interferogram

geometry = SceneGeometry.cs505mu(scale=0.5, fov_mm=25.0)
model = PhaseModel(wavelength_nm=543.5, central_length_nm=2_000_051.0, fringes_y=30.0, fringes_x=1.0)
img = synthesize_interferogram(geometry, model, sensor=SensorModel.cs505mu(), seed=0).image.astype(float)

height, width = img.shape
edge_x = detect_edge_column(img, search=(0, width // 2)) - 0.5
eval_row = height // 2
rows = (eval_row - 75, eval_row + 75)
```

```python
from interflib.fringe.phase_shifting import edge_fraction_from_phase, hariharan_five_step

frames = [synthesize_interferogram(geometry, replace(model, phase_shift_rad=k * np.pi / 2),
                                   sensor=SensorModel.cs505mu(), seed=k).image.astype(float)
          for k in range(5)]
psi = hariharan_five_step(frames)            # phase, background, modulation, visibility
stepped = edge_fraction_from_phase(psi.phase, edge_x, rows, eval_row, modulation=psi.modulation)
print(f"{stepped.fraction:.4f}")             # -> 0.8934
```

```{image} ../images/phase_stepping.png
:alt: Wrapped phase from five phase-stepped frames
```

## Other algorithms

- `least_squares_psa` accepts any set of known steps; `three_step` and `four_step` are that
  fit with three or four steps of π/2.
- `fraction_from_phase` compares two strips of the phase map instead of the edge column.
- `interflib.fringe.unwrap` provides three unwrappers for full maps: Itoh along one axis,
  quality-guided flood fill and the Goldstein branch-cut method (Goldstein, Zebker & Werner
  1988), with the residue detection it rests on.

The [phase-stepping notebook](../notebooks/_generated/20260903-phase-stepping-y-desenrollado-de-fase.ipynb)
compares the algorithms and the unwrappers on simulated stacks.
