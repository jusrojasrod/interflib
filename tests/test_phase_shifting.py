"""Unit and end-to-end tests for phase-stepping (PSI) phase recovery."""
import numpy as np
import pytest

from interflib.fringe.phase_shifting import (
    PhaseStepResult,
    four_step,
    fraction_from_phase,
    hariharan_five_step,
    least_squares_psa,
    three_step,
)


# --------------------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------------------
def _tilt_phase(shape=(48, 64), span_x=3.0, span_y=2.0):
    """A 2-D wrapped phase ramp that exercises a broad range of phase values."""
    ny, nx = shape
    y = np.linspace(-1.0, 1.0, ny)[:, None]
    x = np.linspace(-1.0, 1.0, nx)[None, :]
    return np.angle(np.exp(1j * (span_x * np.pi * x + span_y * np.pi * y)))


def _stack(phase, steps, background, modulation):
    """Synthesise ``I_k = a + b cos(phi + delta_k)`` for the given steps."""
    steps = np.asarray(steps, dtype=float)
    return np.stack([background + modulation * np.cos(phase + d) for d in steps])


def _wrap(values):
    return np.angle(np.exp(1j * np.asarray(values)))


def _assert_phase_close(recovered, truth, atol=1e-9):
    """Compare phase maps modulo 2*pi, so the +-pi wrap boundary is not a false failure."""
    assert float(np.max(np.abs(_wrap(recovered - truth)))) < atol


def _rms_phase_error(recovered, truth):
    """RMS of the wrapped phase error after removing the constant piston offset."""
    err = _wrap(recovered - truth)
    err = _wrap(err - _circular_mean(err))
    return float(np.sqrt(np.mean(err ** 2)))


def _circular_mean(values):
    return np.angle(np.mean(np.exp(1j * np.asarray(values))))


def _fraction_gap(a, b):
    """Distance between two fractions on the unit circle (handles the 0/1 wrap)."""
    d = abs(a - b) % 1.0
    return min(d, 1.0 - d)


# --------------------------------------------------------------------------------------
# General least-squares solver
# --------------------------------------------------------------------------------------
def test_least_squares_recovers_phase_uniform_steps():
    phase = _tilt_phase()
    steps = np.arange(4) * np.pi / 2.0
    frames = _stack(phase, steps, background=1.0, modulation=0.7)

    result = least_squares_psa(frames, steps)

    assert isinstance(result, PhaseStepResult)
    _assert_phase_close(result.phase, phase)


def test_least_squares_recovers_phase_nonuniform_steps():
    """Arbitrary (unevenly spaced) known steps must still demodulate exactly."""
    phase = _tilt_phase()
    steps = np.array([0.0, 0.6, 1.9, 3.1, 4.4])
    frames = _stack(phase, steps, background=1.0, modulation=0.5)

    result = least_squares_psa(frames, steps)

    _assert_phase_close(result.phase, phase)


def test_least_squares_recovers_background_modulation_visibility():
    """Background and modulation may vary per pixel; the per-pixel solve recovers both."""
    phase = _tilt_phase()
    ny, nx = phase.shape
    background = 0.8 + 0.2 * np.linspace(-1.0, 1.0, nx)[None, :] * np.ones((ny, 1))
    modulation = 0.4 * (1.0 - 0.3 * np.linspace(0.0, 1.0, ny)[:, None]) * np.ones((1, nx))
    steps = np.arange(5) * np.pi / 2.0
    frames = _stack(phase, steps, background, modulation)

    result = least_squares_psa(frames, steps)

    np.testing.assert_allclose(result.background, background, atol=1e-9)
    np.testing.assert_allclose(result.modulation, modulation, atol=1e-9)
    np.testing.assert_allclose(result.visibility, modulation / background, atol=1e-9)


def test_least_squares_rejects_too_few_frames():
    phase = _tilt_phase()
    frames = _stack(phase, [0.0, np.pi], 1.0, 0.5)
    with pytest.raises(ValueError, match="at least 3 frames"):
        least_squares_psa(frames, [0.0, np.pi])


def test_least_squares_rejects_mismatched_steps():
    phase = _tilt_phase()
    frames = _stack(phase, [0.0, np.pi / 2, np.pi], 1.0, 0.5)
    with pytest.raises(ValueError, match="one entry per frame"):
        least_squares_psa(frames, [0.0, np.pi / 2, np.pi, 3 * np.pi / 2])


def test_least_squares_rejects_degenerate_steps():
    """Steps spaced by pi cannot separate the sine term -> rank-deficient design."""
    phase = _tilt_phase()
    steps = [0.0, np.pi, 2.0 * np.pi]
    frames = _stack(phase, steps, 1.0, 0.5)
    with pytest.raises(ValueError, match="degenerate"):
        least_squares_psa(frames, steps)


# --------------------------------------------------------------------------------------
# Named fixed-step algorithms
# --------------------------------------------------------------------------------------
def test_three_and_four_step_recover_phase():
    phase = _tilt_phase()
    f3 = _stack(phase, [0.0, np.pi / 2, np.pi], 1.0, 0.6)
    f4 = _stack(phase, [0.0, np.pi / 2, np.pi, 3 * np.pi / 2], 1.0, 0.6)

    r3 = three_step(f3)
    r4 = four_step(f4)

    assert r3.algorithm == "three_step"
    assert r4.algorithm == "four_step"
    _assert_phase_close(r3.phase, phase)
    _assert_phase_close(r4.phase, phase)


def test_hariharan_recovers_phase_and_modulation():
    phase = _tilt_phase()
    steps = np.arange(5) * np.pi / 2.0
    frames = _stack(phase, steps, background=1.2, modulation=0.5)

    result = hariharan_five_step(frames)

    assert result.algorithm == "hariharan_5"
    _assert_phase_close(result.phase, phase)
    np.testing.assert_allclose(result.modulation, 0.5, atol=1e-9)
    np.testing.assert_allclose(result.background, 1.2, atol=1e-9)


def test_hariharan_rejects_wrong_frame_count():
    phase = _tilt_phase()
    frames = _stack(phase, np.arange(4) * np.pi / 2, 1.0, 0.5)
    with pytest.raises(ValueError, match="exactly 5 frames"):
        hariharan_five_step(frames)


def test_hariharan_is_robust_to_linear_detuning():
    """A miscalibrated step biases the four-step form (first order) but not Hariharan.

    Frames are generated with the *actual* (miscalibrated) steps and demodulated assuming
    the nominal pi/2. The Hariharan estimator's double FTF zero at the carrier cancels the
    linear term, so its residual ripple is far smaller and scales quadratically with the
    miscalibration, versus the four-step form's linear scaling.
    """
    phase = _tilt_phase(span_x=4.0, span_y=3.0)
    nominal4 = np.arange(4) * np.pi / 2.0
    nominal5 = np.arange(5) * np.pi / 2.0

    def errors(eps):
        f4 = _stack(phase, nominal4 * (1.0 + eps), 1.0, 0.7)
        f5 = _stack(phase, nominal5 * (1.0 + eps), 1.0, 0.7)
        e4 = _rms_phase_error(four_step(f4).phase, phase)
        e5 = _rms_phase_error(hariharan_five_step(f5).phase, phase)
        return e4, e5

    e4_eps, e5_eps = errors(0.05)
    e4_2eps, e5_2eps = errors(0.10)

    # Hariharan is markedly more accurate at a given miscalibration.
    assert e5_eps < 0.1 * e4_eps
    # Four-step error scales ~linearly (ratio ~2); Hariharan ~quadratically (ratio ~4).
    assert 1.6 < e4_2eps / e4_eps < 2.6
    assert 3.0 < e5_2eps / e5_eps < 5.5


# --------------------------------------------------------------------------------------
# Bridge to L1: fringe fraction from a wrapped-phase map
# --------------------------------------------------------------------------------------
def test_fraction_from_phase_recovers_injected_fraction():
    ny, nx = 120, 160
    fringes_y = 5.0
    fraction = 0.37

    y = np.linspace(-1.0, 1.0, ny)[:, None]
    tilt = 2.0 * np.pi * fringes_y * y * np.ones((1, nx))   # ramps along rows
    block_cols = slice(90, 130)
    unwrapped = tilt.copy()
    unwrapped[:, block_cols] += 2.0 * np.pi * fraction
    phase = np.angle(np.exp(1j * unwrapped))

    roi_platen = (0, ny, 10, 50)     # off the block, same rows
    roi_block = (0, ny, 95, 125)     # on the block, same rows

    recovered = fraction_from_phase(phase, roi_platen, roi_block)
    assert _fraction_gap(recovered, fraction) < 1e-6


def test_fraction_from_phase_requires_matching_rows():
    phase = np.zeros((40, 40))
    with pytest.raises(ValueError, match="same rows"):
        fraction_from_phase(phase, (0, 40, 0, 10), (0, 20, 20, 30))


# --------------------------------------------------------------------------------------
# End-to-end: synthesise a phase-shifted stack, recover phase, read the fraction
# --------------------------------------------------------------------------------------
def test_end_to_end_stack_recovers_ground_truth_fraction():
    """Feed a phase-shifted stack from the simulator back through PSI and the reader."""
    from interflib.simulation.interferogram import (
        Illumination,
        PhaseModel,
        SceneGeometry,
        SensorModel,
        synthesize_interferogram,
    )

    geometry = SceneGeometry(resolution=(160, 220), fov_mm=25.0,
                             block_width_mm=6.0, block_height_mm=10.0)
    illumination = Illumination(psf_sigma_px=0.0, vignette=0.0, visibility=0.9)
    sensor = SensorModel(shot_noise=False, read_noise_e=0.0, bit_depth=16, saturate=False)
    wavelength_nm = 632.8
    central_length_nm = 7777.0

    frames = []
    result_truth = None
    for k in range(5):
        phase_model = PhaseModel(
            wavelength_nm=wavelength_nm, central_length_nm=central_length_nm,
            fringes_x=0.0, fringes_y=6.0, phase_shift_rad=k * np.pi / 2.0,
        )
        synthetic = synthesize_interferogram(geometry, phase_model, illumination, sensor,
                                             seed=k)
        frames.append(synthetic.irradiance)
        result_truth = synthetic

    recovered = hariharan_five_step(frames)

    # Locate block-interior and off-block platen ROIs sharing the same rows.
    mask = result_truth.block_mask
    rows = np.where(mask.any(axis=1))[0]
    r1, r2 = rows[len(rows) // 4], rows[3 * len(rows) // 4]
    block_cols = np.where(mask[(r1 + r2) // 2])[0]
    cb1, cb2 = block_cols[len(block_cols) // 4], block_cols[3 * len(block_cols) // 4]
    roi_block = (r1, r2, cb1, cb2)
    roi_platen = (r1, r2, 5, block_cols[0] - 10)

    fraction = fraction_from_phase(recovered.phase, roi_platen, roi_block)
    assert _fraction_gap(fraction, result_truth.fraction_truth) < 0.01
