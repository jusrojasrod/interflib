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


def test_vertical_edge_is_found_at_the_block_mask(upright):
    geo = locate_block(upright.image, platen_side="left", eval_row=512)
    assert abs(geo.edge_x - _mask_edge(geo, upright.block_mask, 512)) < 1.0
    assert geo.rotation_deg == 0.0 and not geo.mirrored
    assert not geo.refined_by_dark_line                     # the simulator draws no chamfer line
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
