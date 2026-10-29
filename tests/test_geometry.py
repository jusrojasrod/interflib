"""Block location and geometry normalisation on simulated interferograms."""
from dataclasses import replace

import numpy as np
import pytest

from interflib.fringe.fourier import fourier_edge_fraction
from interflib.fringe.geometry import locate_block, locate_face_centre
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


PX_PER_MM = 1223 / 25.0          # the 25 mm field across the 1224 columns of _scene


def _true_centre(geo, centre=(0.0, 0.0)):
    # Geometric centre of the simulated face in the normalised image (the mask may be clipped).
    row = 511.5 + centre[1] * PX_PER_MM
    col = 611.5 + centre[0] * PX_PER_MM
    return geo.to_normalised(row, col)


@pytest.mark.parametrize("rotation, centre, side", [(0.0, (0.0, 0.0), "left"), (8.0, (2.0, 1.5), "left"),
                                                    (-5.0, (-1.0, -1.0), "right")])
def test_the_whole_outline_places_the_centre(rotation, centre, side):
    sim = _scene(rotation=rotation, centre=centre)
    geo = locate_block(sim.image, platen_side=side)
    assert geo.far_edge_source == "phase step"              # platen beyond it; no chamfer line drawn
    assert np.all(np.isfinite(geo.face_ends))
    face = locate_face_centre(geo)
    y, x = _true_centre(geo, centre)
    assert face.source == "outline"
    assert abs(face.x - x) < 1.0 and abs(face.y - y) < 3.0
    assert face.u_y >= face.u_x > 0


def test_the_dimensions_check_a_whole_outline(upright):
    face = locate_face_centre(locate_block(upright.image), face_mm=(9.0, 14.0))
    assert face.source == "outline" and face.scale_source == "face width"
    assert face.scale_px_per_mm == pytest.approx(PX_PER_MM, rel=0.01)
    assert face.aspect[0] == pytest.approx(face.aspect[1], rel=0.01)


def test_one_end_and_the_dimensions_place_the_centre():
    # A 30 mm face with only its top end in the field.
    sim = _scene(centre=(0.0, 9.0), height_mm=30.0)
    geo = locate_block(sim.image)
    assert np.isfinite(geo.face_ends[0]) and np.isnan(geo.face_ends[1])
    with pytest.raises(ValueError, match="incomplete"):
        locate_face_centre(geo)
    face = locate_face_centre(geo, face_mm=(9.0, 30.0), u_face_mm=(0.0, 0.1))
    y, x = _true_centre(geo, (0.0, 9.0))
    assert face.source == "outline and dimensions"
    assert abs(face.x - x) < 1.0 and abs(face.y - y) < 3.0
    # half the length times the scale carries the scale and length uncertainties
    assert face.u_y > 0.1 / 2 * face.scale_px_per_mm


def test_no_end_in_the_field_needs_a_manual_centre(upright):
    geo = replace(locate_block(upright.image), face_ends=(np.nan, np.nan))
    with pytest.raises(ValueError, match="no end"):
        locate_face_centre(geo, face_mm=(9.0, 14.0))


def test_without_the_far_edge_the_scale_comes_from_the_ends_or_the_pixel_size(upright):
    geo = locate_block(upright.image)
    y, x = _true_centre(geo)
    no_far = replace(geo, far_edge_x=np.nan, far_edge_source="none")
    face = locate_face_centre(no_far, face_mm=(9.0, 14.0))
    assert face.scale_source == "face length" and abs(face.x - x) < 3.0
    one_end = replace(no_far, face_ends=(geo.face_ends[0], np.nan))
    with pytest.raises(ValueError, match="pixel_mm"):
        locate_face_centre(one_end, face_mm=(9.0, 14.0))
    face = locate_face_centre(one_end, face_mm=(9.0, 14.0), pixel_mm=1 / PX_PER_MM)
    assert face.scale_source == "pixel size" and face.notes
    assert abs(face.x - x) < 1.5 and abs(face.y - y) < 3.0


def test_the_end_of_the_lit_field_is_a_far_edge_only_with_the_dimensions(upright):
    geo = replace(locate_block(upright.image), far_edge_source="field border")
    with pytest.raises(ValueError, match="far edge"):
        locate_face_centre(geo)
    # Without a pixel size it is not used: the scale comes from the pair of ends.
    face = locate_face_centre(geo, face_mm=(9.0, 14.0))
    assert face.source == "outline and dimensions" and face.scale_source == "face length"
    assert "far edge" not in face.features or any("not used" in note for note in face.notes)
    # With an agreeing pixel size it is used; a beam border inside the block gives a wrong width and is not.
    face = locate_face_centre(geo, face_mm=(9.0, 14.0), pixel_mm=1 / PX_PER_MM)
    assert face.scale_source == "face width" and any("used" in note for note in face.notes)
    clipped = replace(geo, far_edge_x=geo.edge_x + 0.7 * (geo.far_edge_x - geo.edge_x))
    face = locate_face_centre(clipped, face_mm=(9.0, 14.0), pixel_mm=1 / PX_PER_MM)
    assert face.scale_source == "face length" and any("not used" in note for note in face.notes)
    y, x = _true_centre(geo)
    assert abs(face.x - x) < 3.0


def test_a_stray_far_phase_step_is_dropped_when_the_pixel_size_disagrees(upright):
    geo = locate_block(upright.image)
    assert geo.far_edge_source == "phase step"
    stray = replace(geo, far_edge_x=geo.edge_x + 0.6 * (geo.far_edge_x - geo.edge_x))
    assert locate_face_centre(stray, face_mm=(9.0, 14.0)).scale_source == "face width"     # nothing to check it
    face = locate_face_centre(stray, face_mm=(9.0, 14.0), pixel_mm=1 / PX_PER_MM)
    assert face.scale_source == "face length" and any("not used" in note for note in face.notes)


def test_a_manual_centre_is_mapped_to_the_normalised_image(upright):
    geo = locate_block(upright.image[:, ::-1], platen_side="right")
    face = locate_face_centre(geo, face_mm=(9.0, 14.0), manual=(500.0, 600.0))
    assert face.source == "manual"
    assert (face.y, face.x) == pytest.approx((500.0, 1223 - 600.0))
    assert np.isnan(face.u_x)
    assert locate_face_centre(geo, manual=(500.0, 600.0), u_manual_px=5.0).u_y == 5.0


def test_the_far_dark_line_is_the_far_edge_and_the_beam_border_is_no_end():
    # As on the bench: no phase step at the far edge, only the dark chamfer line. The lit
    # window ends at rows 60 and 964 across the platen too: those are not block ends.
    img = _stepped_field({350: 0.25}, dark_lines=(350, 800))
    geo = locate_block(img, platen_side="left", eval_row=512)
    assert geo.far_edge_source == "dark line"
    assert abs(geo.far_edge_x - 800) < 1.0
    assert np.all(np.isnan(geo.face_ends))
