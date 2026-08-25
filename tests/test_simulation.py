"""Tests for the layered Twyman-Green interferogram synthesizer.
"""
import numpy as np
import pytest

from interflib.fringe.profile_fit import FringeFractionMeasurement
from interflib.simulation import (
    Illumination,
    PhaseModel,
    SceneGeometry,
    SensorModel,
    fringe_fraction_of_length,
    synthesize_interferogram,
)


def test_default_image_shape_dtype_and_range():
    result = synthesize_interferogram(seed=0)
    assert result.image.shape == (512, 512)
    assert result.image.dtype == np.uint8
    assert result.image.min() >= 0
    assert result.image.max() <= 255


def test_seed_is_reproducible():
    a = synthesize_interferogram(seed=42)
    b = synthesize_interferogram(seed=42)
    assert np.array_equal(a.image, b.image)


def test_different_seeds_differ_when_noise_is_on():
    a = synthesize_interferogram(seed=1)
    b = synthesize_interferogram(seed=2)
    assert not np.array_equal(a.image, b.image)


def test_fraction_truth_matches_reduced_length():
    lam = 632.8
    length = 158.2  # 2 * 158.2 / 632.8 = 0.5 exactly
    result = synthesize_interferogram(
        phase=PhaseModel(wavelength_nm=lam, central_length_nm=length),
        seed=0,
    )
    assert result.fraction_truth == pytest.approx(0.5, abs=1e-9)
    assert fringe_fraction_of_length(length, lam) == pytest.approx(0.5, abs=1e-9)


def test_tilt_produces_the_requested_fringe_count():
    """A vertical tilt of N fringes must show N cos periods across the field.

    This guards the double-pass convention: the earlier prototype was a factor of two
    off in the fringe count.
    """
    n_fringes = 8
    result = synthesize_interferogram(
        geometry=SceneGeometry(resolution=512, block_width_mm=0.0, block_height_mm=0.0),
        phase=PhaseModel(fringes_x=0.0, fringes_y=n_fringes),
        illumination=Illumination(psf_sigma_px=0.0),
        sensor=SensorModel(shot_noise=False, read_noise_e=0.0),
        seed=0,
    )
    # Count zero crossings of the clean cosine along a central column.
    column = np.cos(result.phase[:, 256])
    zero_crossings = np.sum(np.diff(np.signbit(column)))
    assert zero_crossings == pytest.approx(2 * n_fringes, abs=2)


def _clean_recovery_scene(length_nm, lam=632.8):
    """A low-noise, uniform scene with a central vertical-strip block for recovery."""
    geometry = SceneGeometry(resolution=512, fov_mm=25.0,
                             block_width_mm=9.0, block_height_mm=40.0)
    phase = PhaseModel(wavelength_nm=lam, central_length_nm=length_nm,
                       fringes_x=0.0, fringes_y=8.0)
    illumination = Illumination(background=0.5, beam_waist_frac=1e6, vignette=0.0,
                                visibility=0.9, visibility_falloff=0.0, psf_sigma_px=0.5)
    sensor = SensorModel(bit_depth=16, shot_noise=False, read_noise_e=0.0,
                         hot_pixel_fraction=0.0, dust_specks=0)
    return geometry, phase, illumination, sensor


def test_end_to_end_fraction_is_recovered_by_profile_fit():
    lam = 632.8
    length = 79.1  # 2 * 79.1 / 632.8 = 0.25 exactly -> unambiguous fraction
    geometry, phase, illumination, sensor = _clean_recovery_scene(length, lam)
    result = synthesize_interferogram(geometry, phase, illumination, sensor, seed=0)

    # The block is the central vertical strip; platen ROI sits to its left.
    # Block columns ~ [166, 346]; use interiors, matched rows for both ROIs.
    roi_platen = (60, 452, 40, 130)
    roi_block = (60, 452, 220, 300)
    measurement = FringeFractionMeasurement(result.image, roi_platen, roi_block)
    measured = measurement.measure(pixels_between_fringes=512 / 8).fringe_fraction

    assert measured == pytest.approx(result.fraction_truth, abs=0.03)


def test_bit_depth_sets_the_full_scale():
    result = synthesize_interferogram(sensor=SensorModel(bit_depth=12), seed=0)
    assert result.image.dtype == np.uint16
    assert result.image.max() <= 4095


def test_cs505mu_preset_matches_datasheet():
    """The CS505MU preset carries the datasheet noise values (Thorlabs ITN002518-D02)."""
    cam = SensorModel.cs505mu()
    assert cam.bit_depth == 12
    assert cam.full_well_e == 10000.0
    assert cam.read_noise_e == 2.5
    assert cam.shot_noise is True

    result = synthesize_interferogram(sensor=cam, seed=0)
    assert result.image.dtype == np.uint16
    assert result.image.max() <= 4095  # 12-bit full scale


def test_nonsquare_resolution_gives_matching_image_shape():
    result = synthesize_interferogram(
        geometry=SceneGeometry(resolution=(400, 600)), seed=0)
    assert result.image.shape == (400, 600)          # (height, width)
    assert result.opd_nm.shape == (400, 600)
    assert result.block_mask.shape == (400, 600)


def test_square_int_resolution_is_still_supported():
    result = synthesize_interferogram(geometry=SceneGeometry(resolution=256), seed=0)
    assert result.image.shape == (256, 256)


def test_cs505mu_geometry_carries_the_sensor_aspect():
    geom = SceneGeometry.cs505mu(scale=0.25)
    assert geom.resolution == (512, 612)             # round(2048*0.25), round(2448*0.25)
    result = synthesize_interferogram(geometry=geom, seed=0)
    assert result.image.shape == (512, 612)


def test_illumination_is_physically_isotropic_on_a_nonsquare_frame():
    """Square pixels: equal pixel offsets are equal physical distances, so a circular
    beam gives equal I0 along rows and columns."""
    from interflib.simulation.interferogram import _coordinate_grids, _illumination

    geom = SceneGeometry(resolution=(301, 501))       # odd dims -> exact centre pixel
    _, _, xr, yr = _coordinate_grids(geom)
    i0 = _illumination(xr, yr, Illumination(vignette=0.3, beam_waist_frac=0.9))
    cy, cx, k = 150, 250, 100
    assert i0[cy, cx + k] == pytest.approx(i0[cy + k, cx], rel=1e-9)


def test_block_mask_marks_a_plausible_area():
    result = synthesize_interferogram(
        geometry=SceneGeometry(fov_mm=25.0, block_width_mm=9.0, block_height_mm=35.0),
        seed=0,
    )
    coverage = result.block_mask.mean()
    # 9 x 35 mm block in a 25 x 25 mm field: about (9/25)*(25/25) ~ 0.36 of the area.
    assert 0.25 < coverage < 0.45
