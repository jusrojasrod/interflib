# make_readme_figures.py

Regenerates the figures in `docs/images/`, which the README and this documentation show:

| File | Shows |
|---|---|
| `interferogram.png` | a synthetic frame with the column bands on each side of the edge |
| `edge_comparison.png` | the band phases on each side and the two lines extended to the edge |
| `fourier.png` | a column spectrum with the carrier lobe and the filter, the carrier-free wrapped phase, and the unwrapped relief of a curved face |
| `phase_stepping.png` | five frames stepped by π/2 and the wrapped phase they give, with the step `2πF` at the edge |
| `exact_fractions.png` | the combs of candidate lengths of several wavelengths and their coincidence |
| `pipeline.png` | the chain from the bands to `F` to the length, for the home page and the guide |
| `simulation_layers.png` | the three layers of a synthetic frame: optical path, irradiance, sensor counts |
| `sign_convention.png` | the same block with the reference tilt reversed, read with both `order_sign` |
| `air_index.png` | `n − 1` against pressure, temperature and humidity |
| `mefm.png` | the candidate lengths of one wavelength and the nominal and mechanical priors |
| `parity.png` | what a wrong parity costs, and the test of both readings against a prior |
| `field_overview.png` | the whole-field carrier-free phase and the fringe modulation |
| `run_calibration_areas.png` | the `--plot` figure of `run_calibration.py` on a simulated block |

Every figure comes from the layered simulator or from published reference values, so it can be
rebuilt anywhere and shows a known truth. The script takes no arguments:

```bash
python tools/make_readme_figures.py
```

Run it after a change that alters what a figure shows, and commit the new images with the
change.
