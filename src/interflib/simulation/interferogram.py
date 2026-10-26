"""Physically layered synthesis of a Twyman-Green gauge-block interferogram.

The interferogram is built in three separable layers so that each physical effect
lives in exactly one place:

1. **Optical phase** ``phi(x, y)`` from an optical-path-difference (OPD) map, in
   length units. Contributions: shared reference tilt (the fringe carrier), the block
   step (central length reduced modulo ``lambda/2``, plus an optional phase-change
   difference on reflection), flatness departures of block and platen, a block-vs-platen
   tilt mismatch (imperfect wringing) and a low-order wavefront aberration of the optics.
2. **Image formation** ``I = I0(x, y) * (1 + V(x, y) * cos(phi))``: a non-uniform
   Gaussian illumination ``I0`` with optional vignetting, a fringe visibility ``V`` that
   may fall below one and off-axis, and a point-spread-function blur (finite optics MTF).
3. **Sensor**: photon shot (Poisson) noise, Gaussian read noise, gain, bias, optional
   hot pixels and dust, saturation and quantization to a finite bit depth.

Sign and unit conventions
--------------------------
Round trip: a surface height ``h`` produces ``OPD = 2 h`` and a phase
``phi = (2 pi / lambda) * OPD``. Tilt is parametrised directly in **number of fringes
across the field** (this is the physically correct count; the earlier prototype was a
factor of two off because it dropped the double pass). Wavelengths and OPD are in
nanometres; the field of view and block footprint are in millimetres.

The block step is injected as its **observable reduced value** ``frac(2 L / lambda)``
rather than the full (huge) central length, which is both numerically clean and exactly
what a fringe-fraction reader recovers. The recovered ground truth is returned in
:attr:`SyntheticInterferogram.fraction_truth`; feeding a synthesized image back through
``interflib.fringe.profile_fit`` returns the same fraction up to noise.

The phase-change difference defaults to zero, matching the current excess-fraction
pipeline (which does not yet apply the correction). Provide a non-zero value to test the
correction once it is quantified for the INM materials.

This module has no I/O side effects: it returns arrays. Saving to disk belongs to a tool
or notebook, not the library.
"""
from dataclasses import dataclass, field

import numpy as np
from scipy.ndimage import gaussian_filter


@dataclass
class SceneGeometry:
    """Field of view and gauge-block footprint, in millimetres unless noted.

    ``resolution`` is either an ``int`` (square, ``n x n``) or a ``(height_px, width_px)``
    tuple for a non-square sensor. ``fov_mm`` is the field of view across the **width**; the
    height field of view follows from the pixel aspect ratio, so pixels stay square.
    """

    resolution: "int | tuple[int, int]" = 512  # square side, or (height_px, width_px)
    fov_mm: float = 25.0                # field of view across the width, mm
    block_width_mm: float = 9.0        # block footprint along x (ISO 3650 face ~9 mm)
    block_height_mm: float = 35.0      # block footprint along y (~35 mm)
    block_center_mm: tuple = (0.0, 0.0)  # (x, y) offset of the block from the frame centre
    block_rotation_deg: float = 0.0    # in-plane rotation of the block footprint

    @classmethod
    def cs505mu(cls, *, fov_mm: float = 25.0, scale: float = 1.0,
                **block) -> "SceneGeometry":
        """Full-frame CS505MU geometry (2448 x 2048), optionally downscaled by ``scale``.

        The sensor is 2448 (W) x 2048 (H); ``scale`` < 1 keeps that aspect ratio with fewer
        pixels for lighter simulations. ``fov_mm`` and any ``block_*`` keyword pass straight
        through. The object-plane pixel pitch follows from ``fov_mm`` and the width (i.e.
        from the imaging magnification), not from the 3.45 um sensor pitch directly.
        """
        return cls(resolution=(round(2048 * scale), round(2448 * scale)),
                   fov_mm=fov_mm, **block)


@dataclass
class PhaseModel:
    """Optical-path model. Wavelength and all height/OPD terms in nanometres."""

    wavelength_nm: float = 632.8
    central_length_nm: float = 0.0       # L above the platen at the block centre (the truth)
    fringes_x: float = 0.0               # reference tilt: true fringe count across x
    fringes_y: float = 8.0               # reference tilt: true fringe count across y
    block_tilt_mismatch_x: float = 0.0   # extra fringes across the block only (bad wringing)
    block_tilt_mismatch_y: float = 0.0
    block_flatness_pv_nm: float = 0.0    # peak-to-valley flatness departure of the block face
    platen_flatness_pv_nm: float = 0.0   # peak-to-valley flatness departure of the platen
    defocus_nm: float = 0.0              # low-order optics wavefront aberration (Z4-like)
    astigmatism_nm: float = 0.0          # (Z5-like, 0 deg)
    coma_nm: float = 0.0                 # (Z7-like, along y)
    phase_change_diff_rad: float = 0.0   # 2*(phi_block - phi_platen); 0 matches the pipeline
    phase_shift_rad: float = 0.0         # global phase offset (hook for phase-shifting stacks)


@dataclass
class Illumination:
    """Illumination profile, fringe visibility and imaging blur."""

    background: float = 0.5          # mean I0 (normalised irradiance scale)
    beam_waist_frac: float = 1.4     # Gaussian beam waist as a fraction of the half-field
    vignette: float = 0.15           # additional radial darkening at the edges, 0..1
    visibility: float = 0.85         # base fringe visibility V0, in [0, 1]
    visibility_falloff: float = 0.1  # radial reduction of V towards the edges, 0..1
    psf_sigma_px: float = 0.8        # imaging point-spread-function sigma, pixels


@dataclass
class SensorModel:
    """Digital-sensor model: noise, gain, quantization and defects."""

    bit_depth: int = 8
    full_well_e: float = 12000.0     # electrons mapped to the saturation irradiance (=1.0)
    read_noise_e: float = 5.0        # Gaussian read noise, electrons RMS
    dark_offset_adu: float = 0.0     # additive bias, in ADU
    shot_noise: bool = True          # photon (Poisson) noise
    saturate: bool = True            # clip at the full-scale ADU
    hot_pixel_fraction: float = 0.0  # fraction of pixels forced to saturation
    dust_specks: int = 0             # number of dark circular occlusions
    dust_radius_px: float = 6.0

    @classmethod
    def cs505mu(cls) -> "SensorModel":
        """Preset for the Thorlabs Kiralux CS505MU acquisition camera.

        Values from the datasheet (Thorlabs ITN002518-D02, ch. 11, CS505MU/MU1
        specifications): 12-bit ADC, full well >= 10000 e-, read noise < 2.5 e- RMS
        (dynamic range up to 71 dB), global shutter.

        Only the noise chain is set here. The sensor geometry (3.45 um pixels,
        2448 x 2048) and the quantum efficiency (72 % peak, 525-580 nm) are not part
        of the noise model: geometry belongs to ``SceneGeometry`` and QE would only
        matter for an absolute photon budget, not the relative fringe contrast.
        """
        return cls(bit_depth=12, full_well_e=10000.0, read_noise_e=2.5,
                   shot_noise=True, saturate=True)


@dataclass
class SyntheticInterferogram:
    """Result of one synthesis, with the ground truth attached."""

    image: np.ndarray            # digital counts, shape (height, width), integer dtype
    irradiance: np.ndarray       # continuous irradiance before the sensor, float
    phase: np.ndarray            # wrapped optical phase, radians in [-pi, pi]
    opd_nm: np.ndarray           # optical-path-difference map, nanometres
    block_mask: np.ndarray       # boolean footprint of the gauge block
    fraction_truth: float        # injected fringe fraction at the block centre, [0, 1)
    wavelength_nm: float
    central_length_nm: float


def fringe_fraction_of_length(central_length_nm: float, wavelength_nm: float) -> float:
    """Observable fringe fraction ``frac(2 L / lambda)`` of a central length."""
    return float((2.0 * central_length_nm / wavelength_nm) % 1.0)


def _pixel_dims(geometry: SceneGeometry):
    """Return ``(height_px, width_px)`` from ``geometry.resolution`` (int means square)."""
    r = geometry.resolution
    if isinstance(r, (tuple, list)):
        return int(r[0]), int(r[1])
    return int(r), int(r)


def _coordinate_grids(geometry: SceneGeometry):
    """Return ``(Xn, Yn, Xr, Yr)`` coordinate meshgrids of shape ``(height, width)``.

    ``Xn, Yn`` are normalised to each axis half-extent (``[-1, 1]`` across columns and rows).
    They drive the *per-axis* quantities: the tilt fringe count and the block mask.

    ``Xr, Yr`` are *isotropic*: they share one physical scale so that circular features
    (beam, vignetting, wavefront aberration, flatness) stay circular on a non-square frame.
    ``Yr = Yn * (height - 1) / (width - 1)``, which makes the pixel pitch identical on both
    axes (square pixels); for a square frame ``Xr, Yr == Xn, Yn``.
    """
    height, width = _pixel_dims(geometry)
    xn = np.linspace(-1.0, 1.0, width)
    yn = np.linspace(-1.0, 1.0, height)
    Xn, Yn = np.meshgrid(xn, yn)
    return Xn, Yn, Xn, Yn * _pixel_aspect(height, width)


def _pixel_aspect(height: int, width: int) -> float:
    """Height/width aspect that keeps pixels square under the ``[-1, 1]`` endpoint grid."""
    return (height - 1) / (width - 1) if width > 1 else 1.0


def _block_mask(geom: SceneGeometry, Xn, Yn):
    """Boolean mask of the (optionally rotated, off-centre) rectangular block face."""
    height, width = _pixel_dims(geom)
    half_x = geom.fov_mm / 2.0
    half_y = geom.fov_mm * _pixel_aspect(height, width) / 2.0
    x_mm = Xn * half_x - geom.block_center_mm[0]
    y_mm = Yn * half_y - geom.block_center_mm[1]

    theta = np.deg2rad(geom.block_rotation_deg)
    cos_t, sin_t = np.cos(theta), np.sin(theta)
    x_rot = x_mm * cos_t + y_mm * sin_t
    y_rot = -x_mm * sin_t + y_mm * cos_t

    return (np.abs(x_rot) <= geom.block_width_mm / 2.0) & \
           (np.abs(y_rot) <= geom.block_height_mm / 2.0)


def _block_centre_normalised(geom: SceneGeometry):
    """Block centre ``(x, y)`` in the per-axis normalised coordinates ``Xn, Yn``."""
    height, width = _pixel_dims(geom)
    half_x = geom.fov_mm / 2.0
    half_y = geom.fov_mm * _pixel_aspect(height, width) / 2.0
    return geom.block_center_mm[0] / half_x, geom.block_center_mm[1] / half_y


def _tilt_height_nm(X, Y, fringes_x, fringes_y, wavelength_nm):
    """Surface height (nm) of a tilt giving ``fringes_x``/``fringes_y`` across the field.

    ``OPD = 2 h`` and one fringe per ``lambda`` of OPD, so a full-field span of ``N``
    fringes needs a height span of ``N * lambda / 2``. With ``X, Y`` in ``[-1, 1]`` the
    half-field factor ``/ 2`` yields exactly ``N`` fringes edge to edge.
    """
    return (fringes_x * X + fringes_y * Y) * (wavelength_nm / 2.0) / 2.0


def _low_order_flatness_nm(X, Y, pv_nm):
    """A smooth bowl of the requested peak-to-valley amplitude (nm), zero-mean-ish."""
    if pv_nm == 0.0:
        return np.zeros_like(X)
    r2 = X ** 2 + Y ** 2
    shape = 2.0 * r2 - 1.0            # spans [-1, 1] over the inscribed disk
    return 0.5 * pv_nm * shape


def _aberration_opd_nm(X, Y, phase: PhaseModel):
    """Low-order wavefront aberration as a single-pass OPD map (nm)."""
    if phase.defocus_nm == phase.astigmatism_nm == phase.coma_nm == 0.0:
        return np.zeros_like(X)
    rho2 = X ** 2 + Y ** 2
    defocus = phase.defocus_nm * (2.0 * rho2 - 1.0)
    astigmatism = phase.astigmatism_nm * (X ** 2 - Y ** 2)
    coma = phase.coma_nm * (3.0 * rho2 - 2.0) * Y
    return defocus + astigmatism + coma


def _illumination(X, Y, illum: Illumination):
    """Non-uniform background ``I0(x, y)`` with a Gaussian beam and vignetting."""
    r2 = X ** 2 + Y ** 2
    beam = np.exp(-r2 / (2.0 * illum.beam_waist_frac ** 2))
    vignette = 1.0 - illum.vignette * r2 / 2.0
    return illum.background * beam * np.clip(vignette, 0.0, 1.0)


def _visibility(X, Y, illum: Illumination):
    """Fringe visibility map ``V(x, y)`` with an optional radial falloff."""
    r2 = X ** 2 + Y ** 2
    return illum.visibility * (1.0 - illum.visibility_falloff * r2 / 2.0)


def _apply_sensor(irradiance, sensor: SensorModel, rng):
    """Convert continuous irradiance to digital counts through the sensor model."""
    max_adu = 2 ** sensor.bit_depth - 1

    electrons = np.clip(irradiance, 0.0, None) * sensor.full_well_e
    if sensor.shot_noise:
        electrons = rng.poisson(electrons).astype(np.float64)
    if sensor.read_noise_e > 0.0:
        electrons = electrons + rng.normal(0.0, sensor.read_noise_e, electrons.shape)

    if sensor.hot_pixel_fraction > 0.0:
        hot = rng.random(electrons.shape) < sensor.hot_pixel_fraction
        electrons[hot] = sensor.full_well_e

    gain = max_adu / sensor.full_well_e
    adu = electrons * gain + sensor.dark_offset_adu

    if sensor.dust_specks > 0:
        adu = _stamp_dust(adu, sensor, rng)

    adu = np.floor(adu)
    lo, hi = 0.0, (max_adu if sensor.saturate else np.inf)
    adu = np.clip(adu, lo, hi)

    dtype = np.uint8 if sensor.bit_depth <= 8 else np.uint16
    return adu.astype(dtype)


def _stamp_dust(adu, sensor: SensorModel, rng):
    """Darken a few small circular regions to mimic dust on the optics."""
    n_rows, n_cols = adu.shape
    yy, xx = np.mgrid[0:n_rows, 0:n_cols]
    for _ in range(sensor.dust_specks):
        cy = rng.integers(0, n_rows)
        cx = rng.integers(0, n_cols)
        speck = (yy - cy) ** 2 + (xx - cx) ** 2 <= sensor.dust_radius_px ** 2
        adu[speck] *= 0.2
    return adu


def synthesize_interferogram(
    geometry: SceneGeometry = None,
    phase: PhaseModel = None,
    illumination: Illumination = None,
    sensor: SensorModel = None,
    *,
    seed: int = None,
) -> SyntheticInterferogram:
    """Synthesize one Twyman-Green gauge-block interferogram.

    Parameters
    ----------
    geometry : SceneGeometry, optional
        Field of view and block footprint. Defaults to a centred 9x35 mm block.
    phase : PhaseModel, optional
        Optical-path model (tilt, block step, flatness, aberration). Defaults to a
        flat block on a 8-fringe vertical tilt with zero central length.
    illumination : Illumination, optional
        Beam, visibility and PSF. Defaults to a mildly non-uniform, high-contrast beam.
    sensor : SensorModel, optional
        Digital-sensor model. Defaults to an 8-bit camera with shot and read noise.
    seed : int, optional
        Seed for the noise generator; pass a value for reproducible images.

    Returns
    -------
    SyntheticInterferogram
        The digital image and the attached ground truth.
    """
    geometry = geometry or SceneGeometry()
    phase = phase or PhaseModel()
    illumination = illumination or Illumination()
    sensor = sensor or SensorModel()
    rng = np.random.default_rng(seed)

    # Per-axis grids (Xn, Yn) drive the tilt and the mask; isotropic grids (Xr, Yr) drive
    # the radial terms so circular features stay circular on a non-square frame.
    Xn, Yn, Xr, Yr = _coordinate_grids(geometry)
    mask = _block_mask(geometry, Xn, Yn)

    # --- Layer 1: optical phase --------------------------------------------------------
    lam = phase.wavelength_nm

    height_nm = _tilt_height_nm(Xn, Yn, phase.fringes_x, phase.fringes_y, lam)
    height_nm = height_nm + _low_order_flatness_nm(Xr, Yr, phase.platen_flatness_pv_nm)

    # The central length is defined at the block centre, so the block-only terms (wringing
    # tilt, face flatness) are centred on it and vanish there: the step at the block centre
    # is exactly frac(2 L / lambda), wherever the block sits in the frame.
    xc, yc = _block_centre_normalised(geometry)
    aspect = _pixel_aspect(*_pixel_dims(geometry))
    block_extra_nm = (
        _tilt_height_nm(Xn - xc, Yn - yc, phase.block_tilt_mismatch_x,
                        phase.block_tilt_mismatch_y, lam)
        + _low_order_flatness_nm(Xr - xc, Yr - yc * aspect, phase.block_flatness_pv_nm)
        + 0.5 * phase.block_flatness_pv_nm     # lift the bowl to zero at the centre
    )
    height_nm = np.where(mask, height_nm + block_extra_nm, height_nm)

    opd_nm = 2.0 * height_nm + _aberration_opd_nm(Xr, Yr, phase)

    fraction_truth = fringe_fraction_of_length(phase.central_length_nm, lam)
    step_nm = fraction_truth * lam + phase.phase_change_diff_rad * lam / (2.0 * np.pi)
    opd_nm = np.where(mask, opd_nm + step_nm, opd_nm)

    phi = (2.0 * np.pi / lam) * opd_nm + phase.phase_shift_rad

    # --- Layer 2: image formation ------------------------------------------------------
    i0 = _illumination(Xr, Yr, illumination)
    vis = _visibility(Xr, Yr, illumination)
    irradiance = i0 * (1.0 + vis * np.cos(phi))
    if illumination.psf_sigma_px > 0.0:
        irradiance = gaussian_filter(irradiance, illumination.psf_sigma_px)

    # --- Layer 3: sensor ---------------------------------------------------------------
    image = _apply_sensor(irradiance, sensor, rng)

    wrapped_phase = np.angle(np.exp(1j * phi))
    return SyntheticInterferogram(
        image=image,
        irradiance=irradiance,
        phase=wrapped_phase,
        opd_nm=opd_nm,
        block_mask=mask,
        fraction_truth=fraction_truth,
        wavelength_nm=lam,
        central_length_nm=phase.central_length_nm,
    )
