"""Tests for the edge-column fringe fraction and the block-platen wedge measurement."""
import numpy as np
import pytest

from interflib.fringe.edge_comparison import (
    ColumnPhases,
    detect_edge_column,
    edge_fraction,
    edge_line_centre,
    centre_band_centres,
    fraction_at_edge,
    fraction_at_point,
    side_band_centres,
    span_sensitivity,
)
from interflib.fringe.wedge import fringe_gradient_to_urad, measure_wedge

# --------------------------------------------------------------------------------------
# Synthetic gauge-block scene
# --------------------------------------------------------------------------------------
SHAPE = (600, 480)          # rows, columns
EDGE_X = 230                # platen left of it, block right of it
EVAL_ROW = 300
PERIOD = 30.0               # px per fringe along y (about 20 fringes in the window)
ROWS = (EVAL_ROW - 75, EVAL_ROW + 75)


def _scene(fraction, tilt_deg=0.0, wedge_x=0.0, wedge_y=0.0, dark_edge=True, noise=0.0, seed=0,
           period=PERIOD):
    """Platen and block fringes with a known fraction at (EDGE_X, EVAL_ROW).

    ``tilt_deg`` rotates the fringes of both surfaces alike (a reference-mirror tilt);
    ``wedge_x`` and ``wedge_y`` (fringes/px) are extra phase gradients of the block only.
    The phases are referred to the edge column and evaluation row, so the true fraction
    there is exactly ``fraction``.
    """
    ny, nx = SHAPE
    y = np.arange(ny, dtype=float)[:, None] - EVAL_ROW
    x = np.arange(nx, dtype=float)[None, :] - EDGE_X
    f_y = 1.0 / period
    g_x = f_y * np.tan(np.radians(tilt_deg))
    phase_platen = 2 * np.pi * (f_y * y + g_x * x) + 0.7
    phase_block = 2 * np.pi * ((f_y + wedge_y) * y + (g_x + wedge_x) * x + fraction) + 0.7
    phase = np.where(x < 0, phase_platen, phase_block)
    img = 1000.0 * (1.0 + 0.8 * np.cos(phase))
    if dark_edge:
        # the edge images as a dark valley some 10 px wide, centred at EDGE_X - 0.5
        img = img * (1.0 - 0.8 * np.exp(-((x + 0.5) / 3.0) ** 2))
    if noise:
        img = img + np.random.default_rng(seed).normal(0.0, noise, img.shape)
    return img


def _circular_gap(a, b):
    return abs(((a - b) + 0.5) % 1.0 - 0.5)


# --------------------------------------------------------------------------------------
# Edge location
# --------------------------------------------------------------------------------------
def test_edge_line_centre_finds_the_dark_line():
    img = _scene(0.3)
    approx = detect_edge_column(img, rows=(100, 500), search=(150, 330))
    assert abs(approx - EDGE_X) <= 6
    centre = edge_line_centre(img, approx, rows=(100, 500))
    assert abs(centre - (EDGE_X - 0.5)) < 0.3


# --------------------------------------------------------------------------------------
# Fraction at the edge column
# --------------------------------------------------------------------------------------
@pytest.mark.parametrize("fraction", [0.05, 0.3, 0.62, 0.95])
def test_edge_fraction_recovers_the_true_fraction(fraction):
    res = edge_fraction(_scene(fraction), EDGE_X, ROWS, EVAL_ROW, PERIOD)
    assert _circular_gap(res.fraction, fraction) < 1e-3


@pytest.mark.parametrize("tilt_deg", [-1.0, 0.0, 2.0, 5.0])
def test_edge_fraction_is_invariant_to_a_common_fringe_tilt(tilt_deg):
    """A reference tilt is common to both surfaces and must cancel at the edge."""
    res = edge_fraction(_scene(0.3, tilt_deg=tilt_deg, wedge_x=1e-4, wedge_y=5e-4), EDGE_X, ROWS,
                        EVAL_ROW, PERIOD)
    assert _circular_gap(res.fraction, 0.3) < 1e-3


def test_strip_comparison_error_matches_gx_times_separation():
    """Comparing at two separated columns picks up g_x * dx; the edge comparison does not."""
    tilt_deg = 2.0
    img = _scene(0.3, tilt_deg=tilt_deg)
    res = edge_fraction(img, EDGE_X, ROWS, EVAL_ROW, PERIOD)
    g_x = np.tan(np.radians(tilt_deg)) / PERIOD
    x_p, x_b = EDGE_X - 76, EDGE_X + 64
    i_p = int(np.argmin(np.abs(res.platen.x - x_p)))
    i_b = int(np.argmin(np.abs(res.block.x - x_b)))
    strip = ((res.block.phase[i_b] - res.platen.phase[i_p]) / (2 * np.pi)) % 1.0
    expected = (0.3 + g_x * (res.block.x[i_b] - res.platen.x[i_p])) % 1.0
    assert _circular_gap(strip, expected) < 2e-3
    assert _circular_gap(res.fraction, 0.3) < 1e-3


def test_edge_fraction_with_noise():
    res = edge_fraction(_scene(0.42, tilt_deg=1.0, noise=40.0, seed=3), EDGE_X, ROWS, EVAL_ROW, PERIOD)
    assert _circular_gap(res.fraction, 0.42) < 5e-3


def test_edge_fraction_reports_the_common_slope():
    tilt_deg = 3.0
    res = edge_fraction(_scene(0.3, tilt_deg=tilt_deg), EDGE_X, ROWS, EVAL_ROW, PERIOD)
    g_x = np.tan(np.radians(tilt_deg)) / PERIOD
    assert res.slope_platen == pytest.approx(g_x, abs=2e-5)
    assert res.slope_block == pytest.approx(g_x, abs=2e-5)


def test_bands_outside_the_image_raise():
    with pytest.raises(ValueError):
        edge_fraction(_scene(0.3), 20, ROWS, EVAL_ROW, PERIOD)


# --------------------------------------------------------------------------------------
# Wedge
# --------------------------------------------------------------------------------------
@pytest.mark.parametrize("wedge_x, wedge_y", [(0.0, 0.0), (2e-4, 0.0), (0.0, 8e-4), (-3e-4, 6e-4)])
def test_measure_wedge_recovers_the_injected_wedge(wedge_x, wedge_y):
    m = measure_wedge(_scene(0.3, tilt_deg=1.5, wedge_x=wedge_x, wedge_y=wedge_y), EDGE_X, ROWS,
                      EVAL_ROW, PERIOD)
    assert m.wedge_x == pytest.approx(wedge_x, abs=2e-5)
    assert m.wedge_y == pytest.approx(wedge_y, abs=2e-5)
    assert m.angle_urad is None and m.within_tolerance is None


def test_wedge_angle_conversion():
    # 1e-3 fringe/px with lambda/2 = 271.75 nm per fringe and 0.02 mm per px on the object
    assert fringe_gradient_to_urad(1e-3, 543.5, 0.02) == pytest.approx(13.5875)


def test_measure_wedge_angle_height_and_tolerance():
    m = measure_wedge(_scene(0.3, wedge_x=3e-4, wedge_y=4e-4), EDGE_X, ROWS, EVAL_ROW, PERIOD,
                      wavelength_nm=543.5, pixel_mm=0.02, max_angle_urad=5.0)
    expected = np.hypot(3e-4, 4e-4) * 271.75 / 0.02           # = 6.79 urad
    assert m.angle_urad == pytest.approx(expected, rel=0.03)
    assert m.within_tolerance is False
    assert m.height_change_nm(9.0) == pytest.approx(9.0 * m.angle_urad)


def test_measure_wedge_needs_both_scale_inputs():
    with pytest.raises(ValueError):
        measure_wedge(_scene(0.3), EDGE_X, ROWS, EVAL_ROW, PERIOD, wavelength_nm=543.5)


def test_height_change_needs_the_scale():
    m = measure_wedge(_scene(0.3), EDGE_X, ROWS, EVAL_ROW, PERIOD)
    with pytest.raises(ValueError):
        m.height_change_nm(9.0)


@pytest.mark.parametrize("fraction", [0.05, 0.3, 0.62])
def test_reversed_order_reads_one_minus_f_unless_declared(fraction):
    """Order decreasing towards +y: the default reads 1 - F, order_sign=-1 gives F."""
    img = _scene(fraction, period=-PERIOD)
    default = edge_fraction(img, EDGE_X, ROWS, EVAL_ROW, PERIOD)
    declared = edge_fraction(img, EDGE_X, ROWS, EVAL_ROW, PERIOD, order_sign=-1)
    assert _circular_gap(default.fraction, 1.0 - fraction) < 1e-3
    assert _circular_gap(declared.fraction, fraction) < 1e-3


def test_fraction_error_matches_the_scatter_over_noise():
    """The extrapolation uncertainty matches the scatter of F over noise realisations
    (within the sampling error of 40 runs) and vanishes on a clean scene."""
    assert edge_fraction(_scene(0.3), EDGE_X, ROWS, EVAL_ROW, PERIOD).fraction_error < 1e-9
    runs = [edge_fraction(_scene(0.3, tilt_deg=2.0, noise=200.0, seed=s), EDGE_X, ROWS, EVAL_ROW, PERIOD)
            for s in range(40)]
    scatter = np.std([((r.fraction - 0.3) + 0.5) % 1.0 - 0.5 for r in runs], ddof=1)
    reported = np.mean([r.fraction_error for r in runs])
    assert 0.6 < reported / scatter < 1.5


def test_edge_detection_defaults_to_the_whole_image():
    """Without rows and search the edge is searched over the middle half of the rows and
    the whole width."""
    img = _scene(0.3)
    approx = detect_edge_column(img)
    assert abs(approx - EDGE_X) <= 6
    assert abs(edge_line_centre(img, approx) - (EDGE_X - 0.5)) < 0.3


# --------------------------------------------------------------------------------------
# Sensitivity to the bands in the side fits
# --------------------------------------------------------------------------------------
def _side_phases(xs, phase):
    return ColumnPhases(x=xs, phase=phase % (2 * np.pi), phase_unwrapped=phase, period=np.full(xs.size, PERIOD),
                        r2=np.ones(xs.size), phase_error=np.zeros(xs.size))


def _straight_sides(fraction, curvature_block=0.0, g_p=2e-3, g_b=5e-3):
    """Band phases of both sides: straight lines that meet ``fraction`` at the edge, the
    block side optionally bent by ``curvature_block`` (fringes/px^2) about the edge."""
    xs_p, xs_b = side_band_centres(EDGE_X, 30, 180, 5)
    u_b = xs_b - EDGE_X
    platen = 2 * np.pi * (0.1 + g_p * (xs_p - EDGE_X))
    block = 2 * np.pi * (0.1 + fraction + g_b * u_b + curvature_block * u_b ** 2)
    return _side_phases(xs_p, platen), _side_phases(xs_b, block)


def test_span_sensitivity_vanishes_for_straight_sides():
    platen, block = _straight_sides(0.3)
    sens = span_sensitivity(platen, block, EDGE_X, EVAL_ROW)
    assert np.allclose(sens.fraction, 0.3, atol=1e-9)
    assert sens.deviation < 1e-9 and sens.model_error < 1e-9


def test_span_sensitivity_matches_the_line_error_of_a_curved_side():
    """A straight line fitted to c*u^2 over band distances u misses the edge by
    c*(var(u) - mean(u)^2): each variant shifts F by that amount."""
    c = 2e-6
    platen, block = _straight_sides(0.3, curvature_block=c)
    sens = span_sensitivity(platen, block, EDGE_X, EVAL_ROW)
    for (gap, span), f in zip(sens.variants, sens.fraction, strict=True):
        u = np.arange(gap, span + 1, 5, dtype=float)
        assert f - 0.3 == pytest.approx(c * (u.var() - u.mean() ** 2), rel=1e-6)
    shifts = sens.fraction - sens.fraction[0]
    assert sens.deviation == pytest.approx(np.abs(shifts).max())
    assert sens.model_error == pytest.approx(sens.deviation)


def test_span_sensitivity_at_another_column_extends_each_variant():
    """At the face centre each variant extends its own lines: exact for straight sides,
    and the reading of the first variant equals the extended edge reading."""
    platen, block = _straight_sides(0.3)
    at_x = EDGE_X + 300
    sens = span_sensitivity(platen, block, EDGE_X, EVAL_ROW, at_x=at_x)
    assert np.allclose(sens.fraction, (0.3 + (5e-3 - 2e-3) * 300) % 1.0, atol=1e-9)
    curved = span_sensitivity(*_straight_sides(0.3, curvature_block=2e-6), EDGE_X, EVAL_ROW, at_x=at_x)
    edge = fraction_at_edge(*_straight_sides(0.3, curvature_block=2e-6), EDGE_X, EVAL_ROW)
    assert curved.fraction[0] == pytest.approx((edge.fraction + (edge.slope_block - edge.slope_platen) * 300) % 1.0)
    assert curved.deviation > span_sensitivity(*_straight_sides(0.3, curvature_block=2e-6), EDGE_X, EVAL_ROW).deviation


# --------------------------------------------------------------------------------------
# Block read at the face centre: only the platen is extended
# --------------------------------------------------------------------------------------
AT_X = EDGE_X + 250                # the face centre, half a 500 px face away from the edge


def _sides_with_centre(fraction, curvature_block=0.0, g_p=2e-3, g_b=5e-3):
    """Platen bands left of the edge and block bands around AT_X. The block phase is
    ``fraction + g_b u`` above the platen line extended there, bent by ``curvature_block``
    about AT_X, so the true fraction at AT_X is ``fraction`` whatever the curvature."""
    xs_p = side_band_centres(EDGE_X, 30, 180, 5)[0]
    xs_b = centre_band_centres(AT_X)
    platen = 2 * np.pi * (0.1 + g_p * (xs_p - EDGE_X))
    platen_at = 0.1 + g_p * (AT_X - EDGE_X)
    v = xs_b - AT_X
    block = 2 * np.pi * (platen_at + fraction + g_b * v + curvature_block * v ** 2)
    return _side_phases(xs_p, platen), _side_phases(xs_b, block)


def test_fraction_at_point_is_exact_for_straight_sides():
    platen, block = _sides_with_centre(0.3)
    res = fraction_at_point(platen, block, EDGE_X, AT_X, EVAL_ROW)
    assert res.fraction == pytest.approx(0.3, abs=1e-9)
    assert res.at_x == AT_X and res.edge_x == EDGE_X
    assert res.slope_block == pytest.approx(5e-3) and res.slope_platen == pytest.approx(2e-3)


def test_a_curved_face_only_biases_the_point_by_the_curvature_inside_the_bands():
    """A line fitted to c*v^2 over bands symmetric about AT_X misses AT_X by c*var(v): the
    block is interpolated, not extended, so a curved face does not move F further."""
    c = 2e-6
    res = fraction_at_point(*_sides_with_centre(0.3, curvature_block=c), EDGE_X, AT_X, EVAL_ROW)
    v = centre_band_centres(AT_X) - AT_X
    assert res.fraction - 0.3 == pytest.approx(c * v.var(), rel=1e-6)
    # the same face read from the edge bands and extended to AT_X misses far more
    xs_b = side_band_centres(EDGE_X, 30, 180, 5)[1]
    w = xs_b - AT_X
    block_edge = _side_phases(xs_b, 2 * np.pi * (0.1 + 2e-3 * 250 + 0.3 + 5e-3 * w + c * w ** 2))
    platen = _sides_with_centre(0.3)[0]
    edge = fraction_at_edge(platen, block_edge, EDGE_X, EVAL_ROW)
    extended = (edge.fraction + (edge.slope_block - edge.slope_platen) * 250) % 1.0
    assert abs(extended - 0.3) > 5 * abs(res.fraction - 0.3)


def test_the_reader_reads_the_block_at_the_point():
    """On an image the block bands lie around at_x and the result refers to it."""
    img = _scene(0.3, wedge_x=1e-3)
    edge = edge_fraction(img, EDGE_X, ROWS, EVAL_ROW, PERIOD)
    at = edge_fraction(img, EDGE_X, ROWS, EVAL_ROW, PERIOD, at_x=EDGE_X + 150, centre_half=60)
    assert at.at_x == EDGE_X + 150 and edge.at_x == EDGE_X
    assert at.block.x.min() >= EDGE_X + 90 and at.block.x.max() <= EDGE_X + 210
    # straight phases: the point reading is the edge reading extended by the wedge
    assert (at.fraction - (edge.fraction + 1e-3 * 150) + 0.5) % 1.0 - 0.5 == pytest.approx(0.0, abs=2e-3)
    with pytest.raises(ValueError, match="within 30 px of the edge"):
        edge_fraction(img, EDGE_X, ROWS, EVAL_ROW, PERIOD, at_x=EDGE_X + 80)


def test_span_sensitivity_at_the_point_only_varies_the_platen():
    platen, block = _sides_with_centre(0.3)
    sens = span_sensitivity(platen, block, EDGE_X, EVAL_ROW, at_x=AT_X, centre_half=75)
    assert np.allclose(sens.fraction, 0.3, atol=1e-9)
