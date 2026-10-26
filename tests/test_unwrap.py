"""Tests for 2-D phase unwrapping (Itoh, quality-guided, Goldstein branch-cut)."""
import numpy as np

from interflib.fringe.unwrap import (
    find_residues,
    unwrap_goldstein,
    unwrap_itoh,
    unwrap_quality_guided,
    wrap_phase,
)


# --------------------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------------------
def _ramp(shape=(60, 80), slope_row=0.30, slope_col=0.22):
    """A smooth continuous phase ramp (per-pixel gradient below pi, so wrapping is exact)."""
    ny, nx = shape
    i = np.arange(ny)[:, None]
    j = np.arange(nx)[None, :]
    return slope_row * i + slope_col * j * np.ones((ny, 1))


def _demean(values):
    return values - np.mean(values)


def _vortex(n=40):
    """A single phase vortex: genuinely wrapped, one residue at the (off-grid) core."""
    x = np.linspace(-1.0, 1.0, n)
    X, Y = np.meshgrid(x, x)
    return np.arctan2(Y, X)


def _plane_residual_std(surface, mask):
    """Std of the residual after removing a best-fit plane over the masked pixels."""
    ys, xs = np.nonzero(mask)
    values = surface[ys, xs]
    design = np.column_stack([np.ones_like(xs, dtype=float), xs, ys])
    coeffs, *_ = np.linalg.lstsq(design, values, rcond=None)
    return float(np.std(values - design @ coeffs))


# --------------------------------------------------------------------------------------
# wrap / residues
# --------------------------------------------------------------------------------------
def test_wrap_phase_range():
    x = np.linspace(-10.0, 10.0, 1001)
    w = wrap_phase(x)
    assert np.all(w > -np.pi - 1e-12) and np.all(w <= np.pi + 1e-12)
    np.testing.assert_allclose(np.cos(w), np.cos(x), atol=1e-12)


def test_no_residues_on_clean_ramp():
    residues = find_residues(wrap_phase(_ramp()))
    assert np.count_nonzero(residues) == 0


def test_vortex_has_one_residue():
    residues = find_residues(_vortex())
    assert np.count_nonzero(residues) == 1
    assert abs(int(residues.sum())) == 1


# --------------------------------------------------------------------------------------
# Itoh baseline
# --------------------------------------------------------------------------------------
def test_itoh_recovers_clean_ramp():
    truth = _ramp()
    recovered = unwrap_itoh(wrap_phase(truth))
    np.testing.assert_allclose(_demean(recovered), _demean(truth), atol=1e-9)


# --------------------------------------------------------------------------------------
# Quality-guided flood fill
# --------------------------------------------------------------------------------------
def test_quality_guided_recovers_ramp_plus_bowl():
    ny, nx = 60, 80
    i = np.arange(ny)[:, None]
    j = np.arange(nx)[None, :]
    bowl = 0.001 * ((i - ny / 2) ** 2 + (j - nx / 2) ** 2)
    truth = 0.25 * i + 0.18 * j + bowl
    quality = np.ones((ny, nx))

    recovered = unwrap_quality_guided(wrap_phase(truth), quality=quality)
    np.testing.assert_allclose(_demean(recovered), _demean(truth), atol=1e-9)


def test_quality_guided_unwraps_disconnected_regions_independently():
    """Two masked regions separated by a real step must not bridge across the gap."""
    ny, nx = 50, 90
    truth = _ramp((ny, nx), 0.2, 0.15)
    truth[:, 45:] += 100.0                       # a large step, like a block above a platen
    wrapped = wrap_phase(truth)

    mask = np.zeros((ny, nx), dtype=bool)
    mask[:, 5:30] = True                          # platen region
    mask[:, 60:85] = True                         # block region (disconnected)

    recovered = unwrap_quality_guided(wrapped, mask=mask)

    left = mask.copy()
    left[:, 40:] = False
    right = mask.copy()
    right[:, :40] = False
    # Each region is recovered up to its own piston; the 100-rad step is not bridged.
    assert np.nanstd(_demean_masked(recovered - truth, left)) < 1e-9
    assert np.nanstd(_demean_masked(recovered - truth, right)) < 1e-9
    # Outside the mask stays nan.
    assert np.all(np.isnan(recovered[~mask]))


def _demean_masked(values, mask):
    v = values[mask]
    return v - np.mean(v)


# --------------------------------------------------------------------------------------
# Goldstein branch-cut
# --------------------------------------------------------------------------------------
def test_goldstein_matches_itoh_on_clean_data():
    truth = _ramp()
    wrapped = wrap_phase(truth)
    goldstein = unwrap_goldstein(wrapped)
    np.testing.assert_allclose(_demean(goldstein), _demean(truth), atol=1e-9)


def test_goldstein_isolates_noise_and_recovers_clean_region():
    """Local corruption creates residues; branch cuts confine it and the clean bulk is exact."""
    truth = _ramp((60, 80), 0.30, 0.22)
    wrapped = wrap_phase(truth)
    # Inject isolated corrupted pixels -> residue dipoles the branch cuts must fence off.
    wrapped[12, 15] = wrap_phase(wrapped[12, 15] + 2.4)
    wrapped[13, 40] = wrap_phase(wrapped[13, 40] - 2.7)

    recovered = unwrap_goldstein(wrapped)

    clean = np.zeros_like(truth, dtype=bool)
    clean[35:, 55:] = True                         # far from both injected defects
    diff = (recovered - truth)[clean]
    assert np.std(diff - np.mean(diff)) < 1e-9


def test_goldstein_cuts_a_single_vortex_to_the_border():
    _, cuts = unwrap_goldstein(_vortex(), return_cuts=True)
    assert cuts.sum() > 0
    touches_border = (cuts[0, :].any() or cuts[-1, :].any()
                      or cuts[:, 0].any() or cuts[:, -1].any())
    assert touches_border


def test_goldstein_result_has_no_residues_away_from_cuts():
    """A valid single-valued unwrap is residue-free except where a cut jumps across."""
    phase = _vortex()
    recovered, cuts = unwrap_goldstein(phase, return_cuts=True)

    residues = find_residues(recovered)
    # A plaquette is "at a cut" if any of its four corner pixels is a branch-cut pixel.
    at_cut = (cuts[:-1, :-1] | cuts[:-1, 1:] | cuts[1:, 1:] | cuts[1:, :-1])
    assert np.count_nonzero(residues[~at_cut]) == 0


# --------------------------------------------------------------------------------------
# Integration with phase stepping and the simulator
# --------------------------------------------------------------------------------------
def test_unwrap_of_phase_stepped_tilt_is_planar():
    """A phase-shifted tilt, demodulated and unwrapped, is a plane to machine precision."""
    from interflib.fringe.phase_shifting import four_step
    from interflib.simulation.interferogram import (
        Illumination,
        PhaseModel,
        SceneGeometry,
        SensorModel,
        synthesize_interferogram,
    )

    geometry = SceneGeometry(resolution=(120, 160), fov_mm=25.0,
                             block_width_mm=0.0, block_height_mm=0.0)  # platen only
    illumination = Illumination(psf_sigma_px=0.0, vignette=0.0, visibility=0.9)
    sensor = SensorModel(shot_noise=False, read_noise_e=0.0, bit_depth=16, saturate=False)

    frames = []
    for k in range(4):
        model = PhaseModel(wavelength_nm=632.8, fringes_x=1.5, fringes_y=5.0,
                           phase_shift_rad=k * np.pi / 2.0)
        frames.append(synthesize_interferogram(geometry, model, illumination, sensor,
                                                seed=k).irradiance)

    result = four_step(frames)
    unwrapped = unwrap_quality_guided(result.phase, quality=result.modulation)

    assert np.all(np.isfinite(unwrapped))
    assert _plane_residual_std(unwrapped, np.ones_like(unwrapped, dtype=bool)) < 1e-6
