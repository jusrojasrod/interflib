"""Tests for the single-shot Fourier demodulation and its edge-column fringe fraction."""
import numpy as np
import pytest

from interflib.fringe.edge_comparison import column_phases
from interflib.fringe.fourier import (
    demodulate_columns,
    estimate_carrier,
    fourier_column_phases,
    fourier_edge_fraction,
)

# --------------------------------------------------------------------------------------
# Synthetic gauge-block scene (same geometry as test_edge_comparison)
# --------------------------------------------------------------------------------------
SHAPE = (600, 480)          # rows, columns
EDGE_X = 230                # platen left of it, block right of it
EVAL_ROW = 300
PERIOD = 30.0               # px per fringe along y
ROWS = (EVAL_ROW - 75, EVAL_ROW + 75)
FFT_ROWS = (100, 500)       # 13.3 fringes: a non-integer number, so the ends do leak


def _scene(fraction, tilt_deg=0.0, wedge_x=0.0, wedge_y=0.0, illumination=False, noise=0.0, seed=0,
           period=PERIOD):
    """Platen and block fringes with a known fraction at (EDGE_X, EVAL_ROW).

    ``illumination`` replaces the uniform background and modulation by a Gaussian beam
    off the centre of the scene with a modulation that also falls along ``y``.
    """
    ny, nx = SHAPE
    y = np.arange(ny, dtype=float)[:, None] - EVAL_ROW
    x = np.arange(nx, dtype=float)[None, :] - EDGE_X
    f_y = 1.0 / period
    g_x = f_y * np.tan(np.radians(tilt_deg))
    phase_platen = 2 * np.pi * (f_y * y + g_x * x) + 0.7
    phase_block = 2 * np.pi * ((f_y + wedge_y) * y + (g_x + wedge_x) * x + fraction) + 0.7
    phase = np.where(x < 0, phase_platen, phase_block)
    if illumination:
        a = 1000.0 * np.exp(-((y + 120) / 400) ** 2 - ((x + 60) / 500) ** 2)
        b = 0.8 * a * (1.0 - 0.3 * y / 300)
    else:
        a, b = 1000.0, 800.0
    img = (a + b * np.cos(phase)) * (1.0 - 0.8 * np.exp(-((x + 0.5) / 3.0) ** 2))
    if noise:
        img = img + np.random.default_rng(seed).normal(0.0, noise, img.shape)
    return img


def _circular_gap(a, b):
    return abs(((a - b) + 0.5) % 1.0 - 0.5)


# --------------------------------------------------------------------------------------
# Carrier and demodulation
# --------------------------------------------------------------------------------------
@pytest.mark.parametrize("period", [18.0, 30.0, 47.3])
def test_estimate_carrier(period):
    img = _scene(0.3, period=period)
    profiles = img[FFT_ROWS[0]:FFT_ROWS[1], 20:200:5]
    assert estimate_carrier(profiles) == pytest.approx(1.0 / period, rel=1e-3)


def test_too_few_fringes_raise():
    img = _scene(0.3, period=200.0)            # 2 fringes over the 400-row span
    with pytest.raises(ValueError):
        estimate_carrier(img[FFT_ROWS[0]:FFT_ROWS[1], 20:200:5])
    with pytest.raises(ValueError):
        demodulate_columns(img, [100], FFT_ROWS, carrier=1 / 200.0)


def test_bandwidth_must_exclude_the_background():
    with pytest.raises(ValueError):
        demodulate_columns(_scene(0.3), [100], FFT_ROWS, bandwidth=1.0)


def test_demodulation_recovers_background_and_modulation():
    d = demodulate_columns(_scene(0.3), [60, 120, 180], FFT_ROWS)
    inner = slice(100, 300)                    # away from the ends of the span
    assert np.allclose(d.modulation[inner], 800.0, rtol=5e-3)
    assert np.allclose(d.background[inner], 1000.0, rtol=5e-3)


def test_phase_convention_matches_the_sinusoidal_fit():
    """Both methods give the phase of A sin(w y + phi) + C at the evaluation row."""
    img = _scene(0.3, tilt_deg=2.0)
    xs = np.arange(40, 200, 10)
    fit = column_phases(img, xs, ROWS, EVAL_ROW, PERIOD)
    fou = fourier_column_phases(img, xs, ROWS, EVAL_ROW, fft_rows=FFT_ROWS)
    diff = np.angle(np.exp(1j * (fou.phase - fit.phase)))
    assert np.max(np.abs(diff)) < 1e-3
    assert np.allclose(fou.period, PERIOD, rtol=1e-4)
    assert np.min(fou.r2) > 0.999


def test_rows_must_lie_inside_the_transform_span():
    with pytest.raises(ValueError):
        fourier_column_phases(_scene(0.3), [100], (90, 300), EVAL_ROW, fft_rows=FFT_ROWS)


# --------------------------------------------------------------------------------------
# Fraction at the edge column
# --------------------------------------------------------------------------------------
@pytest.mark.parametrize("fraction", [0.05, 0.3, 0.62, 0.95])
def test_fourier_edge_fraction_recovers_the_true_fraction(fraction):
    res = fourier_edge_fraction(_scene(fraction), EDGE_X, ROWS, EVAL_ROW, fft_rows=FFT_ROWS)
    assert _circular_gap(res.fraction, fraction) < 1e-4


@pytest.mark.parametrize("tilt_deg", [-1.0, 0.0, 2.0, 5.0, 14.0])
def test_fourier_edge_fraction_is_invariant_to_a_common_fringe_tilt(tilt_deg):
    res = fourier_edge_fraction(_scene(0.3, tilt_deg=tilt_deg, wedge_x=1e-4, wedge_y=5e-4), EDGE_X,
                                ROWS, EVAL_ROW, fft_rows=FFT_ROWS)
    assert _circular_gap(res.fraction, 0.3) < 1e-4
    g_x = np.tan(np.radians(tilt_deg)) / PERIOD
    assert res.slope_platen == pytest.approx(g_x, abs=1e-6)


def test_fourier_edge_fraction_under_non_uniform_illumination():
    """A Gaussian beam and a modulation falling along y leave the fraction unbiased."""
    res = fourier_edge_fraction(_scene(0.3, tilt_deg=1.0, illumination=True), EDGE_X, ROWS, EVAL_ROW,
                                fft_rows=FFT_ROWS)
    assert _circular_gap(res.fraction, 0.3) < 1e-4


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_fourier_edge_fraction_with_noise(seed):
    res = fourier_edge_fraction(_scene(0.42, tilt_deg=1.0, noise=40.0, seed=seed), EDGE_X, ROWS,
                                EVAL_ROW, fft_rows=FFT_ROWS)
    assert _circular_gap(res.fraction, 0.42) < 3e-3
