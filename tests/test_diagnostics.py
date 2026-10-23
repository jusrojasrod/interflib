"""Tests for the Fourier-based diagnostics: whole-field overview, modulation edge and mask,
and the sensitivity of the fraction to the filter width."""
import numpy as np
import pytest

from interflib.fringe.diagnostics import (bandwidth_sensitivity, edge_from_modulation, field_overview,
                                         surface_phase)
from interflib.fringe.fourier import modulation_mask

from .test_fourier import EDGE_X, EVAL_ROW, FFT_ROWS, ROWS, _scene

PLATEN_COLS = (EDGE_X - 182, EDGE_X - 27)


def test_edge_from_modulation_finds_the_dark_line():
    m = edge_from_modulation(_scene(0.3, tilt_deg=2.0), (150, 450), (150, 330), fft_rows=FFT_ROWS)
    assert m.edge_x == pytest.approx(EDGE_X - 0.5, abs=0.2)
    assert m.contrast > 0.5
    assert m.x.shape == m.profile.shape


def test_edge_from_modulation_without_a_dark_line():
    """A pure phase step (no dark line, as in the simulator) shows only through bands
    that straddle it; single columns see no dip. The 10 mm block gives F = 0.56 here; with
    F near 0 the straddling bands would not dip either."""
    from interflib.simulation import Illumination, PhaseModel, SceneGeometry, SensorModel, synthesize_interferogram

    geom = SceneGeometry(resolution=(480, 640), fov_mm=25.0, block_width_mm=9.0, block_height_mm=10.0)
    scene = synthesize_interferogram(geom, PhaseModel(central_length_nm=1.0e7, fringes_x=1.5, fringes_y=24.0),
                                     Illumination(psf_sigma_px=0.0),
                                     SensorModel(shot_noise=False, read_noise_e=0.0, bit_depth=16, saturate=False),
                                     seed=0)
    edge = np.where(scene.block_mask.any(axis=0))[0][0] - 0.5
    band = edge_from_modulation(scene.irradiance, (180, 300), (144, 264), fft_rows=(140, 340), half_width=2)
    assert band.edge_x == pytest.approx(edge, abs=0.2)
    assert band.contrast > 0.5
    single = edge_from_modulation(scene.irradiance, (180, 300), (144, 264), fft_rows=(140, 340), half_width=0)
    assert single.contrast < 0.2


def test_edge_from_modulation_reports_no_clear_dip_away_from_the_edge():
    m = edge_from_modulation(_scene(0.3, tilt_deg=2.0), (150, 450), (40, 180), fft_rows=FFT_ROWS)
    assert m.contrast < 0.1


def test_field_overview_shows_the_fraction_as_a_phase_step():
    o = field_overview(_scene(0.3, tilt_deg=2.0), FFT_ROWS, (20, 460), carrier_cols=PLATEN_COLS)
    platen = o.object_phase[EVAL_ROW - FFT_ROWS[0], 150 - 20]
    block = o.object_phase[EVAL_ROW - FFT_ROWS[0], 300 - 20]
    step = (np.angle(np.exp(1j * (block - platen))) / (2 * np.pi)) % 1.0
    assert step == pytest.approx(0.3, abs=0.01)
    assert o.carrier[1] == pytest.approx(1 / 30.0, rel=1e-3)


def test_modulation_mask():
    b = np.array([[800.0, 790.0, 100.0, np.nan], [810.0, 805.0, 300.0, 820.0]])
    mask = modulation_mask(b, 0.5)
    assert mask.tolist() == [[True, True, False, False], [True, True, False, True]]
    assert modulation_mask(b, 0.5, reference=100.0)[0, 2]


@pytest.mark.parametrize("method", ["2d", "columns"])
def test_bandwidth_sensitivity_is_small_on_a_clean_scene(method):
    s = bandwidth_sensitivity(_scene(0.3, tilt_deg=2.0), EDGE_X, ROWS, EVAL_ROW, method=method, fft_rows=FFT_ROWS)
    assert s.bandwidth.shape == s.fraction.shape == (6,)
    assert s.spread < 5e-4
    assert np.all(np.abs(((s.fraction - 0.3) + 0.5) % 1.0 - 0.5) < 5e-4)


def test_bandwidth_sensitivity_rejects_an_unknown_method():
    with pytest.raises(ValueError):
        bandwidth_sensitivity(_scene(0.3), EDGE_X, ROWS, EVAL_ROW, method="pll")


def test_surface_phase_separates_the_wedge_from_the_curvature():
    """Platen and block of one overview: the block plane minus the platen plane is the
    injected wedge, and with flat surfaces the residual is nil."""
    wedge_x, wedge_y = -1.5e-4, 6e-4
    img = _scene(0.25, tilt_deg=2.0, wedge_x=wedge_x, wedge_y=wedge_y)
    o = field_overview(img, FFT_ROWS, (20, 460), carrier_cols=PLATEN_COLS)
    yy, xx = np.meshgrid(o.y, o.x, indexing="ij")
    inner = (yy > 160) & (yy < 440)
    platen = surface_phase(o, inner & (xx > 60) & (xx < EDGE_X - 40))
    block = surface_phase(o, inner & (xx > EDGE_X + 40) & (xx < 420))
    assert block.plane[1] - platen.plane[1] == pytest.approx(wedge_x, abs=2e-5)
    assert block.plane[2] - platen.plane[2] == pytest.approx(wedge_y, abs=2e-5)
    assert np.nanmax(np.abs(platen.residual)) < 0.01 and np.nanmax(np.abs(block.residual)) < 0.01
    assert np.isnan(platen.phase[~platen.mask]).all()
    with pytest.raises(ValueError):
        surface_phase(o, np.zeros_like(inner))
