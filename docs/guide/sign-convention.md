# Sign of the fraction

One intensity image cannot tell which way the interference order increases, because the
cosine is even. Read the wrong way, the fraction comes out as `1 − F`. The direction is a
property of the set-up's alignment and has to be determined on the bench, for instance
with a known displacement or with a block of certified length. Declare it to the readers
with `order_sign`:

- `+1`, the default: the order increases towards `+y`, down the image.
- `−1`: it decreases towards `+y`.

Every band phase is taken on the branch whose phase increases towards `+y`, and the
fraction is

$$F = \operatorname{frac}\!\left(\text{order\_sign}\cdot\frac{\varphi_\text{block} - \varphi_\text{platen}}{2\pi}\right)$$

(`interflib.fringe.models.signed_fraction`). The example reverses the reference tilt of
the scene, so the order now decreases down the image:

```python
from dataclasses import replace

from interflib.fringe.edge_comparison import detect_edge_column, edge_fraction
from interflib.simulation import PhaseModel, SceneGeometry, SensorModel, synthesize_interferogram

geometry = SceneGeometry.cs505mu(scale=0.5, fov_mm=25.0)
model = PhaseModel(wavelength_nm=543.5, central_length_nm=2_000_051.0, fringes_y=30.0, fringes_x=1.0)
img = synthesize_interferogram(geometry, model, sensor=SensorModel.cs505mu(), seed=0).image.astype(float)
height, width = img.shape
edge_x = detect_edge_column(img, search=(0, width // 2)) - 0.5
eval_row = height // 2
rows = (eval_row - 75, eval_row + 75)

reversed_tilt = synthesize_interferogram(geometry, replace(model, fringes_y=-30.0),
                                         sensor=SensorModel.cs505mu(), seed=0).image.astype(float)
wrong = edge_fraction(reversed_tilt, edge_x, rows, eval_row, height / 30)
right = edge_fraction(reversed_tilt, edge_x, rows, eval_row, height / 30, order_sign=-1)
print(f"{wrong.fraction:.4f} {right.fraction:.4f}")   # -> 0.1067 0.8933
```

The side of the platen has the same effect: a platen declared on the wrong side also reads
`1 − F`, and two wrong declarations cancel. Neither shows in the image.
[Parity of the reading](parity.md) gives what a wrong declaration would cost and tests it
against a prior.
