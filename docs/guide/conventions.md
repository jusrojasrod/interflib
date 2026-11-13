# Conventions

## Units and axes

Wavelengths and lengths are in **nanometres** throughout the public API. Images are
rows × columns: `y` is the row index, increasing downwards, and `x` is the column index.
Fringes are expected to run roughly along the rows (spatial carrier along `y`), with the
platen left of the block edge; `interflib.fringe.geometry.locate_block` gives the rotation
and mirroring that bring any frame to that layout.

## Names

Arguments and results use the same names across the package:

| Name | Symbol | Meaning |
|---|---|---|
| `wavelength` | `λ` | wavelength in air, nm |
| `wavelength_vac` | `λ_vac` | vacuum wavelength, nm |
| `n_air` | `n` | refractive index of air |
| `nominal_length` | `L'` | nominal length, nm |
| `length` | `L` | length reduced to 20 °C, nm |
| `length_dev` | `L − L'` | deviation from nominal, nm |
| `fringe_fraction` | `F` | fringe fraction, in `[0, 1)` |
| `integer_orders` | `κ` | integer interference orders |
| `cte` | `α` | thermal expansion coefficient, 1/K |
| `temp_c` | `t` | block temperature, °C |

## Deprecated names

The former names still work during a transition and raise a `DeprecationWarning`:

- `GaugeBlockLength(lambda_air, Lp [mm], C, t_block, obs_fraction)` and `block_length()`
  (mm);
- `MonochromaticEFMSolver.evaluate_block()`, which returns a dict;
- `fractions=` in the multi-wavelength solvers, and `.correction` in their results;
- `pixels_between_fringes=` in `FringeFractionMeasurement`.

## Phase and sign

The column phases are those of $A\sin(w y + \varphi) + C$ with $w > 0$, so the phase
increases towards `+y`. The fraction is
$F = \operatorname{frac}\left(\text{order\_sign}\cdot(\varphi_\text{block} - \varphi_\text{platen})/2\pi\right)$;
see [Sign of the fraction](sign-convention.md).

## Constants, tests and sources

- Physical constants live in `interflib.constants`, each with its source in a comment. The
  algorithms contain no magic numbers.
- Every method has tests. Where an analytic solution or a published reference value
  exists, the test checks against it. `tests/test_docs_examples.py` runs the examples of
  the README and of this guide.
- The repository is self-contained. Where a step, algorithm or value rests on a reference,
  the docstring cites that source directly; the list is in [References](../references.md).
- The library has no I/O side effects and does not print: it takes and returns arrays and
  dataclasses. Reading files is the job of `interflib.io`, and reporting that of the tools.
