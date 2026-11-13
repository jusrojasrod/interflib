# Synthetic interferograms

`interflib.simulation` builds a Twyman-Green frame of a gauge block wrung to a platen in
three separable layers:

1. **Optical path.** Reference tilt, block step, flatness of block and platen, wringing
   wedge and low-order aberrations.
2. **Image formation.** Gaussian beam with vignetting, fringe visibility and blur.
3. **Sensor.** Shot and read noise, gain, quantisation and saturation.

It returns the image together with the truth: the optical path difference, the phase, the
block mask and the fraction at the block centre. The guide pages use the scene below, a
2 mm block that is 51 nm long, read at 543.5 nm.

```python
import numpy as np

from interflib.simulation import PhaseModel, SceneGeometry, SensorModel, synthesize_interferogram

geometry = SceneGeometry.cs505mu(scale=0.5, fov_mm=25.0)      # 1224 x 1024 px, block centred
model = PhaseModel(wavelength_nm=543.5,                        # wavelength in the medium
                   central_length_nm=2_000_051.0,              # L = 2 mm + 51 nm
                   fringes_y=30.0, fringes_x=1.0)              # reference tilt, slightly rotated
sim = synthesize_interferogram(geometry, model, sensor=SensorModel.cs505mu(), seed=0)

img = sim.image.astype(float)        # 12-bit counts
print(img.shape)                     # -> (1024, 1224)
# the truth: fraction at the block centre
print(round(sim.fraction_truth, 4))  # -> 0.8933
```

```{figure} ../images/simulation_layers.png
:alt: The three layers of a synthetic interferogram

The three layers of the scene above. (a) The optical path difference: the reference tilt gives the fringes, the block adds a step of `2L` at its edges. (b) Its wrapped phase. (c) The irradiance after image formation: Gaussian beam, visibility and blur. (d) What the sensor records, in 12-bit counts, over a small patch: shot and read noise on top of the irradiance.
```

## The configuration objects

| Object | Layer | Holds |
|---|---|---|
| `SceneGeometry` | — | frame size, field of view, block size, position and rotation |
| `PhaseModel` | optical path | wavelength, central length, reference tilt, block-platen tilt mismatch (wedge), flatness of block and platen, defocus, astigmatism and coma, phase change on reflection, phase step |
| `Illumination` | image formation | background, beam waist, vignetting, visibility and its fall-off, blur (PSF) |
| `SensorModel` | sensor | bit depth, full well, read noise, dark offset, shot noise, saturation, hot pixels, dust |

The result, `SyntheticInterferogram`, carries the `image` and the truth behind it:
`irradiance`, `phase`, `opd_nm`, `block_mask` and `fraction_truth`.

`SceneGeometry.cs505mu()` and `SensorModel.cs505mu()` are presets for the Thorlabs CS505MU
camera of the bench (2448 × 2048 px, 12 bits). The geometry handles non-square frames with
square pixels, so circular features stay circular.

`PhaseModel.phase_shift_rad` adds a global phase step to the frame. A phase-stepped stack
is a list of frames with different steps; see [Phase stepping](phase-stepping.md).
`fringe_fraction_of_length` gives the fraction that a length shows at a wavelength.

The [simulator notebook](../notebooks/_generated/20260821-desarrollo-simulador-interferograma.ipynb)
walks through each layer with its figures.
