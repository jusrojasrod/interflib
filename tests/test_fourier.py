"""Tests for the single-shot Fourier demodulation and its edge-column fringe fraction."""
import numpy as np
import pytest

from interflib.fringe.edge_comparison import column_phases
from interflib.fringe.fourier import (
    demodulate_2d,
    demodulate_columns,
    estimate_carrier,
    estimate_carrier_2d,
    fourier_column_phases,
    fourier_edge_fraction,
    fourier_edge_fraction_2d,
    map_column_phases,
    unwrap_phase_map,
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


# --------------------------------------------------------------------------------------
# Two-dimensional demodulation
# --------------------------------------------------------------------------------------
PLATEN_COLS = (EDGE_X - 182, EDGE_X - 27)      # the platen region used by the 2-D edge fraction


@pytest.mark.parametrize("tilt_deg", [0.0, 2.0, 14.0])
def test_estimate_carrier_2d(tilt_deg):
    img = _scene(0.3, tilt_deg=tilt_deg)
    g_x, g_y = estimate_carrier_2d(img[FFT_ROWS[0]:FFT_ROWS[1], PLATEN_COLS[0]:PLATEN_COLS[1]])
    assert g_y == pytest.approx(1.0 / PERIOD, rel=1e-3)
    assert g_x == pytest.approx(np.tan(np.radians(tilt_deg)) / PERIOD, abs=2e-5)


def test_2d_too_few_fringes_and_bad_inputs_raise():
    img = _scene(0.3, period=200.0)
    with pytest.raises(ValueError):
        estimate_carrier_2d(img[FFT_ROWS[0]:FFT_ROWS[1], PLATEN_COLS[0]:PLATEN_COLS[1]])
    with pytest.raises(ValueError):
        demodulate_2d(_scene(0.3), FFT_ROWS, PLATEN_COLS, bandwidth=1.0)
    with pytest.raises(ValueError):
        demodulate_2d(_scene(0.3), (100, 700), PLATEN_COLS)


def test_demodulate_2d_recovers_phase_background_and_modulation():
    """Inside the region, away from its ends, the map matches the true platen phase.

    Near the ends the background and the conjugate lobe leak into the pass band (about
    0.01 rad within 60 rows or 30 columns of an end in this scene), so the check stays
    100 rows and 50 columns inside.
    """
    tilt_deg = 3.0
    m = demodulate_2d(_scene(0.3, tilt_deg=tilt_deg), FFT_ROWS, PLATEN_COLS)
    y = m.y[:, None] - EVAL_ROW
    x = m.x[None, :] - EDGE_X
    true = 2 * np.pi * (y / PERIOD + x * np.tan(np.radians(tilt_deg)) / PERIOD) + 0.7 + np.pi / 2
    inner = (slice(100, -100), slice(50, -50))
    diff = np.angle(np.exp(1j * (m.phase - true)))[inner]
    assert np.max(np.abs(diff)) < 5e-3
    assert np.allclose(m.modulation[inner], 800.0, rtol=5e-3)
    assert np.allclose(m.background[inner], 1000.0, rtol=5e-3)


def test_map_column_phases_match_the_column_method():
    img = _scene(0.3, tilt_deg=2.0)
    xs = np.arange(PLATEN_COLS[0] + 2, PLATEN_COLS[1] - 2, 10)
    m = demodulate_2d(img, FFT_ROWS, PLATEN_COLS)
    two_d = map_column_phases(m, img, xs, ROWS, EVAL_ROW)
    cols = fourier_column_phases(img, xs, ROWS, EVAL_ROW, fft_rows=FFT_ROWS)
    assert np.max(np.abs(np.angle(np.exp(1j * (two_d.phase - cols.phase))))) < 1e-3
    with pytest.raises(ValueError):
        map_column_phases(m, img, [EDGE_X + 50], ROWS, EVAL_ROW)


@pytest.mark.parametrize("fraction", [0.05, 0.3, 0.62, 0.95])
def test_fourier_edge_fraction_2d_recovers_the_true_fraction(fraction):
    res = fourier_edge_fraction_2d(_scene(fraction), EDGE_X, ROWS, EVAL_ROW, fft_rows=FFT_ROWS)
    assert _circular_gap(res.fraction, fraction) < 1e-4


@pytest.mark.parametrize("tilt_deg", [-1.0, 0.0, 2.0, 5.0, 14.0])
def test_fourier_edge_fraction_2d_is_invariant_to_a_common_fringe_tilt(tilt_deg):
    res = fourier_edge_fraction_2d(_scene(0.3, tilt_deg=tilt_deg, wedge_x=1e-4, wedge_y=5e-4), EDGE_X,
                                   ROWS, EVAL_ROW, fft_rows=FFT_ROWS)
    assert _circular_gap(res.fraction, 0.3) < 1e-4
    assert res.slope_platen == pytest.approx(np.tan(np.radians(tilt_deg)) / PERIOD, abs=1e-6)


def test_fourier_edge_fraction_2d_under_non_uniform_illumination_with_wedge():
    res = fourier_edge_fraction_2d(_scene(0.3, tilt_deg=1.0, wedge_x=1e-4, wedge_y=5e-4, illumination=True),
                                   EDGE_X, ROWS, EVAL_ROW, fft_rows=FFT_ROWS)
    assert _circular_gap(res.fraction, 0.3) < 1e-4


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_fourier_edge_fraction_2d_with_noise(seed):
    res = fourier_edge_fraction_2d(_scene(0.42, tilt_deg=1.0, noise=40.0, seed=seed), EDGE_X, ROWS,
                                   EVAL_ROW, fft_rows=FFT_ROWS)
    assert _circular_gap(res.fraction, 0.42) < 3e-3


# --------------------------------------------------------------------------------------
# Carrier-free phase and unwrapping of a two-dimensional map
# --------------------------------------------------------------------------------------
def _relief(amplitude, ny=400, nx=300, period=25.0, tilt_deg=3.0, dark_band=False):
    """One surface with a Gaussian relief of ``amplitude`` fringes on a tilted carrier."""
    y = np.arange(ny, dtype=float)[:, None]
    x = np.arange(nx, dtype=float)[None, :]
    carrier = 2 * np.pi * (np.tan(np.radians(tilt_deg)) * x + y) / period
    relief = 2 * np.pi * amplitude * np.exp(-(((x - 150) / 150) ** 2 + ((y - 200) / 200) ** 2))
    phase = carrier + relief + 0.4
    b = np.full((ny, nx), 800.0)
    if dark_band:
        b[:, 200:] = 20.0                      # a 100 px strip with almost no fringes
    return 1000.0 + b * np.cos(phase), phase, relief


def test_object_phase_is_the_phase_without_the_carrier():
    img, phase, _ = _relief(0.0)
    m = demodulate_2d(img, (0, 400), (0, 300))
    g_x, g_y = m.carrier
    ramp = 2 * np.pi * (g_x * m.x[None, :] + g_y * m.y[:, None])
    assert np.allclose(np.angle(np.exp(1j * (m.phase - m.object_phase - ramp))), 0.0, atol=1e-9)
    # a flat, tilted surface has a constant carrier-free phase, up to the small plane left
    # by the error of the carrier estimate (indistinguishable from a tilt of the surface)
    inner = (slice(60, -60), slice(40, -40))
    spread = np.angle(np.exp(1j * (m.object_phase[inner] - m.object_phase[200, 150])))
    assert np.max(np.abs(spread)) < 5e-4


@pytest.mark.parametrize("method", ["quality", "goldstein", "itoh"])
def test_unwrapped_phase_of_a_flat_surface_is_exact(method):
    img, phase, _ = _relief(0.0)
    u = unwrap_phase_map(demodulate_2d(img, (0, 400), (0, 300)), method=method)
    inner = (slice(60, -60), slice(40, -40))
    err = (u.phase - phase)[inner]
    assert np.max(np.abs(err - np.median(err))) < 1e-4


@pytest.mark.parametrize("method", ["quality", "goldstein", "itoh"])
def test_unwrapping_a_curved_surface(method):
    """1.2 fringes of relief need unwrapping; the result is continuous and keeps the
    relief, with the edge error of the demodulation (see unwrap_phase_map)."""
    img, phase, relief = _relief(1.2)
    m = demodulate_2d(img, (0, 400), (0, 300))
    u = unwrap_phase_map(m, method=method)
    inner = (slice(100, -100), slice(80, -80))
    # continuous: no 2 pi jumps between neighbours, and it rewraps to the demodulated phase
    assert np.nanmax(np.abs(np.diff(u.object_phase, axis=0))) < np.pi / 2
    assert np.nanmax(np.abs(np.diff(u.object_phase, axis=1))) < np.pi / 2
    assert np.allclose(np.angle(np.exp(1j * (u.object_phase - m.object_phase))), 0.0, atol=1e-9)
    err = (u.phase - phase)[inner]
    assert np.max(np.abs(err - np.median(err))) < 0.06


def test_unwrap_mask_and_modulation_threshold():
    img, _, _ = _relief(0.0, dark_band=True)
    m = demodulate_2d(img, (0, 400), (0, 300))
    u = unwrap_phase_map(m, min_modulation=0.3)
    # the filter smooths the modulation over some 1 / (bandwidth * |g|) px across the edge
    # of the dark strip (and across the periodic ends of the region), so only the interiors
    # are checked
    assert not u.mask[100:300, 235:275].any() and u.mask[100:300, 40:160].all()
    assert np.isnan(u.object_phase[100:300, 235:275]).all()
    mask = np.ones((400, 300), dtype=bool)
    mask[:, :20] = False
    u2 = unwrap_phase_map(m, method="goldstein", mask=mask)
    assert np.isnan(u2.object_phase[:, :20]).all()
    with pytest.raises(ValueError):
        unwrap_phase_map(m, method="itoh", mask=mask)
    with pytest.raises(ValueError):
        unwrap_phase_map(m, method="lsq")
    with pytest.raises(ValueError):
        unwrap_phase_map(m, mask=np.ones((10, 10), dtype=bool))


@pytest.mark.parametrize("method", [fourier_edge_fraction, fourier_edge_fraction_2d])
@pytest.mark.parametrize("fraction", [0.05, 0.3, 0.62])
def test_reversed_order_reads_one_minus_f_unless_declared(method, fraction):
    """With the order decreasing towards +y (reference tilted the other way) the image is
    the same kind of fringe pattern; the default reads 1 - F and order_sign=-1 gives F."""
    img = _scene(fraction, period=-PERIOD)
    default = method(img, EDGE_X, ROWS, EVAL_ROW, fft_rows=FFT_ROWS)
    declared = method(img, EDGE_X, ROWS, EVAL_ROW, fft_rows=FFT_ROWS, order_sign=-1)
    assert _circular_gap(default.fraction, 1.0 - fraction) < 1e-4
    assert _circular_gap(declared.fraction, fraction) < 1e-4
    assert declared.order_sign == -1
