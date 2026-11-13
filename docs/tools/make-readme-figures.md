# make_readme_figures.py

Regenerates the figures in `docs/images/`, which the README and the user guide show:

| File | Shows |
|---|---|
| `interferogram.png` | a synthetic frame with the column bands on each side of the edge |
| `edge_comparison.png` | the band phases on each side and the two lines extended to the edge |
| `fourier.png` | a column spectrum with the carrier lobe and the filter, the carrier-free wrapped phase, and the unwrapped relief of a curved face |
| `phase_stepping.png` | five frames stepped by π/2 and the wrapped phase they give, with the step `2πF` at the edge |
| `exact_fractions.png` | the combs of candidate lengths of several wavelengths and their coincidence |

Every figure comes from the layered simulator, so it can be rebuilt anywhere and shows a
known truth. The script takes no arguments:

```bash
python tools/make_readme_figures.py
```

Run it after a change that alters what a figure shows, and commit the new images with the
change.
