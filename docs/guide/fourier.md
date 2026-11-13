# Fourier demodulation

With a spatial carrier, the Fourier transform of each column separates the fringe term
from the background, and a band-pass filter around the carrier gives the phase at every
pixel (Takeda, Ina & Kobayashi 1982). `fourier_edge_fraction` does this column by column;
`fourier_edge_fraction_2d` demodulates one region on each surface in two dimensions. Both
read `F` with the same comparison at the edge as the [sine fit](edge-comparison.md).

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

## The fraction at the edge

```python
from interflib.fringe.fourier import fourier_edge_fraction, fourier_edge_fraction_2d

fft_rows = (100, height - 100)
by_column = fourier_edge_fraction(img, edge_x, rows, eval_row, fft_rows=fft_rows)
two_d = fourier_edge_fraction_2d(img, edge_x, rows, eval_row, fft_rows=fft_rows)
print(f"{by_column.fraction:.4f} {two_d.fraction:.4f}")   # -> 0.8934 0.8934
```

## A map of the surface

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

```{image} ../images/fourier.png
:alt: Wrapped phase of the block face and the unwrapped relief
```

## Diagnostics

`interflib.fringe.diagnostics` adds tools to inspect a real frame:

- `field_overview`: maps of phase, modulation and background over the whole field;
- `edge_from_modulation`: the block edge, taken from the dip in fringe modulation;
- `surface_phase`: the phase of one surface, split into a plane (the wedge) and a residual
  (the curvature);
- `bandwidth_sensitivity`: how much `F` moves with the filter width.

The [Fourier notebook](../notebooks/_generated/20261022-metodo-de-fourier.ipynb) develops
the method step by step, including the choice of the filter.
