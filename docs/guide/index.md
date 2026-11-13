# User guide

The block surface stands `L` above the platen. In double pass, the fringes on the block
are shifted against those on the platen by the **fringe fraction**
$F = \operatorname{frac}(2L/\lambda)$, the part of the interference order that an image
can show.

```{figure} ../images/pipeline.png
:alt: From the interferogram to the fraction to the length

The chain on the simulated 2 mm block, 51 nm long: the column bands on each side of the edge (1), their phases extrapolated to the edge column, whose difference is the fraction `F` (2), and the candidate lengths spaced `λ/2` apart, of which the mechanical prior (shaded) keeps one (3).
```

The guide follows the chain from the image to the length:

| Step | Page | Module |
|---|---|---|
| Images with a known truth | [Synthetic interferograms](simulation.md) | `interflib.simulation` |
| `F` from one frame, sine fit | [Edge comparison](edge-comparison.md) | `interflib.fringe.edge_comparison` |
| `F` from one frame, Fourier | [Fourier demodulation](fourier.md) | `interflib.fringe.fourier`, `interflib.fringe.diagnostics` |
| `F` from a stack of frames | [Phase stepping](phase-stepping.md) | `interflib.fringe.phase_shifting`, `interflib.fringe.unwrap` |
| Which way the order runs | [Sign of the fraction](sign-convention.md) | `interflib.fringe.models` |
| The wavelength in air | [Refractive index of air](air-index.md) | `interflib.dimensional.refraction` |
| The integer order and `L` | [Exact fractions](exact-fractions.md) | `interflib.dimensional` |
| `F` or `1 − F` | [Parity of the reading](parity.md) | `interflib.dimensional.parity` |
| Units, names and signs | [Conventions](conventions.md) | — |

Every reader returns `F` at a chosen **edge column** and **evaluation row**. The reason is
that `F` is only free of the instrument when both phases refer to the same pixel: there,
any term common to block and platen cancels, such as the reference tilt or an air or
vibration piston. Comparing two strips at different columns adds $g_x\,\Delta x$ as soon as
the fringes are not exactly along the rows.

The examples of every page run as written, and print what their comments say: the test
suite executes them (`tests/test_docs_examples.py`). Each page starts from its own scene,
so the pages can be read in any order.

```{toctree}
:maxdepth: 1

simulation
edge-comparison
fourier
phase-stepping
sign-convention
air-index
exact-fractions
parity
conventions
```
