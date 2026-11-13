# Edge comparison (sine fit)

`interflib.fringe.edge_comparison` reads the fraction from a single frame. It fits a
sinusoid along the rows to narrow column bands, from 30 to 180 px on each side of the
edge, fits a straight line to the band phases of each side, and takes the difference of
the two lines at the edge column. There both phases refer to one pixel, so every term
common to block and platen cancels. The platen must lie to the left of the edge; mirror
the image for the opposite layout.

The scene is the one of [Synthetic interferograms](simulation.md):

```python
import numpy as np

from interflib.simulation import PhaseModel, SceneGeometry, SensorModel, synthesize_interferogram

geometry = SceneGeometry.cs505mu(scale=0.5, fov_mm=25.0)
model = PhaseModel(wavelength_nm=543.5, central_length_nm=2_000_051.0, fringes_y=30.0, fringes_x=1.0)
img = synthesize_interferogram(geometry, model, sensor=SensorModel.cs505mu(), seed=0).image.astype(float)
```

## The fraction at the edge

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

## How much the bands move it

The lines cross the 30 px next to the edge where no band is read, so `F` assumes that the
side phases stay straight up to the edge. `span_sensitivity` repeats the comparison with
the bands from 45 or 60 px, or up to 120 px, and gives the largest change of `F`, which is
also its standard uncertainty (`model_error`). With the image blurred coherently across
the edge, simulations gave biases of up to 1.2 times that change. On the 2026-10 bench
images `F` moved by 1 to 15 nm this way, well above `fraction_error`.

```python
from interflib.fringe.edge_comparison import span_sensitivity

sens = span_sensitivity(res.platen, res.block, res.edge_x, res.eval_row)
print(f"{sens.deviation:.4f} {sens.model_error:.4f}")          # -> 0.0001 0.0001
```

```{image} ../images/edge_comparison.png
:alt: Band phases on each side of the edge and the two lines extended to it
```

## At the centre of the face

The fraction that enters the central length belongs to the centre of the measuring face
(ISO 3650:1998, clause 3), away from the edge. There the block face is in the image but
the platen is not. With `at_x`, the readers measure the block phase **at that column**
(bands `at_x ± centre_half`, the line interpolated at `at_x`) and extend only the platen
line to it (`fraction_at_point`). The comparison is still made at one pixel, so the
common terms still cancel; what remains is the assumption that the platen is straight
over the extension. `interflib.fringe.geometry.locate_face_centre` places the centre; see
[`run_calibration.py`](../tools/run-calibration.md) for how the chain uses it.

## Related tools

- `interflib.fringe.geometry.locate_block` finds the block without assuming where it is or
  how it is oriented, and gives the rotation and mirroring that bring the edge to a
  vertical line with the platen on the left.
- `interflib.fringe.wedge.measure_wedge` measures the block-platen wedge along `x` and `y`
  from the same comparison, in fringes/px and, with the pixel scale, in microradians.
- `interflib.fringe.profile_fit.FringeFractionMeasurement` is the earlier two-strip fit,
  kept for strip analyses. It is exact only for fringes exactly along the rows.

The sinusoidal model and its linear least-squares seed follow Servín, Quiroga & Padilla
(2014), ch. 2. Extending the platen phase to the point of the block where the fraction is
read is the treatment of Decker, Schödel & Bönsch (2004). See [References](../references.md).
