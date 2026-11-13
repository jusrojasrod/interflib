# run_calibration.py

`tools/run_calibration.py` runs the whole chain on one gauge block, from the frames to the
length at 20 °C, and prints a summary. Without arguments it measures a simulated 2 mm block
and also reports the error against the truth. With `--series` it reads a real burst of
frames and its `.meta.yaml` sidecar; this needs the `tools` extra (pyyaml).

```bash
python tools/run_calibration.py                                   # simulated block
python tools/run_calibration.py --block-size-mm 9 14              # simulated block, whole face in the field
python tools/run_calibration.py --series DATA/20261005_L1_tg-inm_2mm_repet-2_serie-010 \
    --pressure 75300 --face-mm 9 30 --json summary.json --plot areas.png
```

## What it does

1. **Finds the block.** The block is found without assuming where it is or how it is
   oriented (`interflib.fringe.geometry.locate_block`, see below).
2. **Reads the fringe fraction** with the three single-shot readers: `sine`
   ([edge comparison](../guide/edge-comparison.md)), `fourier` and `fourier2d`
   ([Fourier demodulation](../guide/fourier.md)). `--method` picks the primary one (Fourier
   by default) and the others are a cross-check. `F` is read at the centre of the measuring
   face when it can be placed, otherwise at the edge column, with a warning.
3. **Corrects the wavelength** for the air (modified Edlén; see
   [Refractive index of air](../guide/air-index.md)).
4. **Resolves the integer order** from the nominal length and, given a prior deviation with
   its expanded uncertainty, also by MEFM ([Exact fractions](../guide/exact-fractions.md)).
5. **Reduces the length to 20 °C.**
6. **Checks the parity** of the reading ([Parity of the reading](../guide/parity.md)).

## The summary

- the inputs, each with its origin (sidecar, command line, assumption);
- the air correction;
- the image: edge, face centre, period, visibility, saturation and wedge;
- the fraction from the three readers, with their spread, the repeatability over the
  frames, how far the choice of bands moves it (`span_sensitivity`, which enters `u(F)`),
  and the value at the face centre with the uncertainty of its position;
- the integer order, from the nominal and, given a prior, by MEFM;
- the length at 20 °C;
- the parity of the reading: the length with `1 − F`, and the test of both readings against
  the prior (`--prior-dev-nm`, `--prior-U-nm` with `k = 2`, and `--u-other-nm` for the rest of
  the optical uncertainty); without a prior the parity is reported as not verified;
- the warnings that apply.

```text
Length at 20 C : 2.0000510 mm   (d = +51.0 nm from nominal; F contribution u = 0.02 nm)
...
Fringe fraction (order_sign +1; primary: fourier)
    sine       F = 0.1637   u_e(F) = 0.0000
  * fourier    F = 0.1636   u_e(F) = 0.0000
    fourier2d  F = 0.1636   u_e(F) = nan
  spread between methods 0.0001 fringe (0.0 nm)
  repeatability (5 frames) s = 0.0001, s/sqrt(n) = 0.0000 fringe
  ...
  band choice   gap-span px: 30-180 0.1636, 45-180 0.1636, 60-180 0.1636, 30-120 0.1635; u_m(F) = 0.0000 (0.0 nm)
...
Truth          F = 0.1636, L = 2.0000510 mm;   error F -0.0000, L -0.00 nm
```

`--json PATH` writes the same summary to a file, with the keys `inputs`, `air`, `image`,
`fraction`, `order`, `parity`, `length`, `warnings` and, for a simulated scene, `truth`.
`--plot PATH` saves a figure of the areas used, so that each series can be checked: the
frame, a zoom around the edge with the platen and block bands and the Fourier regions, and
the band phases with the two lines extended to the edge.

```{warning}
The contribution `u(F)·λ/2` covers only the reading of the fraction. It is **not** the
calibration uncertainty, which also includes the air, the temperatures, the wavelength, the
phase change on reflection and the wringing film.
```

## Real series and the sidecar

`--series FOLDER` reads every `.tif`, `.tiff` or `.png` frame of one burst in `FOLDER`, and
the sidecar `FOLDER.meta.yaml` next to it. A series without a sidecar is not processed. The
command line wins over the sidecar. A value that is in neither, such as the pressure of the
2026-10 campaign, has to be given on the command line: the script stops rather than assume
it.

| Sidecar key | Option | Meaning |
|---|---|---|
| `patron` | `--nominal-mm` | the block; the nominal length is read from its `NN mm` |
| `longitud_onda_nm` | `--wavelength-vac` | vacuum wavelength, nm |
| `condiciones.temperatura_C` | `--temperature` | air temperature, °C |
| `condiciones.presion_Pa` | `--pressure` | air pressure, Pa |
| `condiciones.humedad_relativa_pct` | `--humidity` | relative humidity, % |
| `lado_platina` | `--platen-side` | side of the platen: `izquierda`/`left` or `derecha`/`right` |
| `sentido_orden` | `--order-sign` | direction of the interference order, `1` or `-1` |
| `cara_mm` | `--face-mm` | measured width × length of the measuring face, mm |
| `u_cara_mm` | `--u-face-mm` | their standard uncertainties, mm |

```yaml
# the keys the script reads; the values are illustrative
fecha: 2026-10-05
linea: L1
instrumento: tg-inm
patron: BP-2mm            # the nominal length is read from "2mm"
longitud_onda_nm: 543.5163
condiciones:
  temperatura_C: 20.1
  humedad_relativa_pct: 45.0
lado_platina: izquierda
cara_mm: [9.0, 30.0]      # optional: width x length of the measured face
u_cara_mm: [0.02, 0.05]   # optional: their standard uncertainties
```

The block temperature defaults to the air temperature and the expansion coefficient to that
of gauge-block steel, 11.5·10⁻⁶ /K (the manufacturer's value); both are reported as
assumptions. When the side of the platen or the order direction is not declared, the
script takes the left side and `+1` and warns: the phase cannot tell them, and a wrong one
reads `1 − F`.

## Finding the block and the face centre

The fringe zone comes from the Fourier modulation, and the edge from a line of phase steps
inside it that the dark chamfer line confirms. With wide fringes the demodulation also
draws stray lines of phase steps, with as much support as the edge, and these have no dark
line along them. When no phase-step line has the dark line, the edge is the dark line
nearest the platen side. When there is no dark line at all, the edge rests on the phase
step alone and the summary warns that it is unconfirmed (`edge_source` in the result). The
frames are then rotated (if the edge is more than 1° from vertical) and mirrored (if the
platen is on the right) so that the readers see a vertical edge with the platen on the
left. With too few fringes for the demodulation the script falls back to the bench
heuristic.

The central length is defined at the centre of the free measuring face (ISO 3650:1998,
clause 3), so `F` is read there when the centre can be placed
(`interflib.fringe.geometry.locate_face_centre`). The modes are tried in this order:

1. **outline**: both lateral edges and both ends of the face are found, and the centre is
   their midpoint. No input is needed. This needs the whole block in the field with platen
   around it.
2. **outline and dimensions**: `--face-mm WIDTH LENGTH` (with `--u-face-mm`), or `cara_mm`
   and `u_cara_mm` in the sidecar, fills in what the field does not show. The scale comes
   from the pair of lateral edges, or else the pair of ends, or else `--pixel-mm`. An axis
   with one feature only takes that feature plus half the dimension. The dimensions belong
   to the block, so they are given once and hold for every series and position.
3. **manual**: `--face-centre Y X` in pixels of the input frames, with
   `--u-face-centre-px`.

The far edge is taken with the same order of preference as the edge. When none is found,
the end of the lit field is taken, which may be the border of the beam. A far edge without
the dark line is checked against the declared pixel size (`--pixel-mm`) when the dimensions
are given: the end of the lit field is used only if the width it gives agrees within 5 %,
and an unconfirmed phase-step line is dropped if it disagrees. In the 2026-10 position
series the beam clipped the block, and without this check the centre would have moved by up
to 325 px. An end of the face is a line of phase steps across the face that the platen does
not share at the same rows (the border of the beam crosses both), with fringes on both
sides.

The evaluation row moves to the centre, and the uncertainty of the position enters `u(F)`
through the wedge, $u_\text{pos}(F)^2 = (\Delta g_x\,u(x_c))^2 + (\Delta g_y\,u(y_c))^2$.

At the centre the block phase is measured there: the readers take `at_x`, read the block
bands at $x_c \pm 75$ px and interpolate their line at $x_c$. Only the platen line is
extended, about 280 px, because the block hides the platen. Both phases still refer to one
pixel, so the reference tilt still cancels, but the platen is assumed straight over the
extension. `u(F)` therefore adds a type B bound, $u_\text{ext}(F)$
(`--u-platen-extension`, 0.065 fringe by default). It comes from the one check available on
the 2026-10 bench: there the straight platen line missed the platen visible above the 2 mm
block by that much. With platen on both sides of the block it should be replaced by a check
in every image. `--face-centre-x` keeps its earlier meaning: `F` is extended to that column
at the evaluation row.

## Options

```{eval-rst}
.. argparse::
   :module: run_calibration
   :func: build_parser
   :prog: python tools/run_calibration.py
```

## As a module

The script can also be loaded from Python, as the notebooks do. `run(parse_args([...]))`
returns the summary as a dictionary without printing it, and `format_summary` turns it into
the text above.

```python
import importlib.util
from pathlib import Path

spec = importlib.util.spec_from_file_location("run_calibration", Path("tools/run_calibration.py"))
run_calibration = importlib.util.module_from_spec(spec)
spec.loader.exec_module(run_calibration)

summary = run_calibration.run(run_calibration.parse_args(["--frames", "2"]))
print(round(summary["length"]["deviation_from_nominal_nm"]))   # -> 51
```
