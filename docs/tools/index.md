# Command-line tools

The scripts in `tools/` are not part of the installed package: they use the library and
report, which the library itself never does. Run them from the repository root.

| Script | What it does |
|---|---|
| [`run_calibration.py`](run-calibration.md) | the whole measurement chain on a simulated block or a real series, with a summary report |
| [`make_readme_figures.py`](make-readme-figures.md) | regenerates the figures of the README and of this guide |
| [`select_roi.py`](select-roi.md) | draws a region of interest on a prototype interferogram by hand (legacy) |

```{toctree}
:hidden:

run-calibration
make-readme-figures
select-roi
```
