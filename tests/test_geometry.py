"""Block location and geometry normalisation on simulated interferograms."""
import numpy as np
import pytest

from interflib.fringe.fourier import fourier_edge_fraction
from interflib.fringe.geometry import locate_block
from interflib.simulation import PhaseModel, SceneGeometry, SensorModel, synthesize_interferogram


def _scene(rotation=0.0, centre=(0.0, 0.0), fringes_y=30.0, height_mm=14.0):
    geometry = SceneGeometry(resolution=(1024, 1224), fov_mm=25.0, block_rotation_deg=rotation,
                             block_center_mm=centre, block_height_mm=height_mm)
    model = PhaseModel(wavelength_nm=543.4, central_length_nm=2_000_051.0, fringes_y=fringes_y, fringes_x=1.0)
    return synthesize_interferogram(geometry, model, sensor=SensorModel.cs505mu(), seed=0)


def _mask_edge(geo, mask, row):
    cols = np.flatnonzero(geo.normalise(mask.astype(float))[row] > 0.5)
    return cols[0] - 0.5


def _fraction(geo, row):
    rows = (row - 75, row + 75)
    return fourier_edge_fraction(geo.image, geo.edge_x, rows, row, fft_rows=(row - 250, row + 250)).fraction


def _gap(a, b):
    return abs((a - b + 0.5) % 1.0 - 0.5)


@pytest.fixture(scope="module")
def upright():
    return _scene()


def _stepped_field(steps, dark_lines=(), bright_from=None, fringes_y=30.0, seed=0):
    # Horizontal fringes over a lit window, with phase steps of steps[x] fringes at columns
    # x, dark lines (depth 0.5, sigma 2.5 px) at columns dark_lines, and the illumination
    # 1.5 times brighter from column bright_from on.
    height, width = 1024, 1224
    y, x = np.mgrid[0:height, 0:width].astype(float)
    phase = 2 * np.pi * fringes_y * y / height
    for x_step, fraction in steps.items():
        phase = phase + 2 * np.pi * fraction * (x >= x_step)
    lit = ((x >= 100) & (x < 1124) & (y >= 60) & (y < 964)).astype(float) * 0.95 + 0.05
    if bright_from is not None:
        lit = lit * np.where(x >= bright_from, 1.5, 1.0)
    for x_dark in dark_lines:
        lit = lit * (1 - 0.5 * np.exp(-0.5 * ((x - x_dark) / 2.5) ** 2))
    rng = np.random.default_rng(seed)
    return 1000 * lit * (1 + 0.8 * np.cos(phase)) + rng.normal(0, 10, (height, width))


def test_a_stray_phase_step_nearer_the_platen_is_not_the_edge():
    # Two strong phase-step lines; only the one at 600 has the dark chamfer line.
    img = _stepped_field({350: 0.25, 600: 0.4}, dark_lines=(600,))
    geo = locate_block(img, platen_side="left", eval_row=512)
    assert geo.edge_source == "phase step and dark line" and geo.refined_by_dark_line
    assert abs(geo.edge_x - 600) < 1.0
    assert geo.dark_line_depth > 0.25


def test_the_dark_line_stands_in_when_no_phase_step_line_has_it():
    img = _stepped_field({350: 0.25}, dark_lines=(800,))
    geo = locate_block(img, platen_side="left", eval_row=512)
    assert geo.edge_source == "dark line"
    assert abs(geo.edge_x - 800) < 1.0
    assert np.isnan(geo.edge_x_phase) and np.isnan(geo.support)


def test_without_a_dark_line_the_edge_is_the_phase_step_and_says_so():
    # An illumination step is not a dark line: it is darker on one side only.
    img = _stepped_field({350: 0.25, 600: 0.4}, bright_from=900)
    geo = locate_block(img, platen_side="left", eval_row=512)
    assert geo.edge_source == "phase step" and not geo.refined_by_dark_line
    assert abs(geo.edge_x - 350) < 3.0


def test_vertical_edge_is_found_at_the_block_mask(upright):
    geo = locate_block(upright.image, platen_side="left", eval_row=512)
    assert abs(geo.edge_x - _mask_edge(geo, upright.block_mask, 512)) < 1.0
    assert geo.rotation_deg == 0.0 and not geo.mirrored
    assert geo.edge_source == "phase step"                  # the simulator draws no chamfer line
    assert geo.face_width_px * 25.0 / 1223 == pytest.approx(9.0, abs=0.3)
    assert _gap(_fraction(geo, 512), upright.fraction_truth) < 1e-3


def test_platen_on_the_right_is_mirrored(upright):
    geo = locate_block(upright.image[:, ::-1], platen_side="right", eval_row=512)
    assert geo.mirrored
    assert abs(geo.edge_x - _mask_edge(geo, upright.block_mask[:, ::-1], 512)) < 1.0
    assert _gap(_fraction(geo, 512), upright.fraction_truth) < 1e-3


def test_rotated_block_is_rotated_upright():
    sim = _scene(rotation=8.0, centre=(2.0, 0.0))
    geo = locate_block(sim.image, platen_side="left", eval_row=512)
    assert geo.angle_deg == pytest.approx(-8.0, abs=0.2)
    assert geo.rotation_deg == geo.angle_deg
    assert abs(geo.edge_x - _mask_edge(geo, sim.block_mask, 512)) < 1.5
    assert _gap(_fraction(geo, 512), sim.fraction_truth) < 1e-3
    # a point of the input maps onto the same block pixel of the normalised image
    ys, xs = np.nonzero(sim.block_mask)
    y_n, x_n = geo.to_normalised(ys.mean(), xs.mean())
    yn, xn = np.nonzero(geo.normalise(sim.block_mask.astype(float)) > 0.5)
    assert abs(y_n - yn.mean()) < 1.0 and abs(x_n - xn.mean()) < 1.0


def test_face_rows_stop_short_of_the_block_ends(upright):
    geo = locate_block(upright.image, platen_side="left", eval_row=512)
    rows = np.flatnonzero(upright.block_mask.any(axis=1))
    assert rows.min() <= geo.face_rows[0] < 512 < geo.face_rows[1] <= rows.max() + 1


def test_too_few_fringes_cannot_be_demodulated():
    with pytest.raises(ValueError):
        locate_block(_scene(fringes_y=2.0).image)


def test_platen_side_is_validated(upright):
    with pytest.raises(ValueError, match="platen_side"):
        locate_block(upright.image, platen_side="top")
