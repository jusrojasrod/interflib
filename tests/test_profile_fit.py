"""Unit tests for the fringe-profile fit helpers (synthetic, no image needed)."""
from pathlib import Path

import numpy as np

from interflib.fringe.profile_fit import FringeFractionMeasurement
from interflib.fringe.models import interference_model

DATA = Path(__file__).parent / "data"


def _measure_profiles(platen, block, pixels_between_fringes, centre_offset=0.0):
    """Run the fraction measurement on two 1-D profiles, one image column each.

    ``centre_offset`` shifts the fit coordinate so that ``x = 0`` falls on that sample,
    as an analysis does when it evaluates the fraction at a fixed row.
    """
    img = np.column_stack([platen, block]).astype(float)
    n = len(platen)
    m = FringeFractionMeasurement(img, (0, n, 0, 1), (0, n, 1, 2))
    m.get_fringe_profiles()
    m.pixel_indices = m.pixel_indices - centre_offset
    return m, m.measure(pixels_between_fringes=pixels_between_fringes)


def test_initial_guess_recovers_amplitude_phase_offset_at_known_frequency():
    """At the true frequency the seed is exact: the linear solve recovers A, phi and C."""
    x = np.arange(-250, 250, dtype=float)
    true = (3.0, 2 * np.pi / 36.4, 2.1, 10.0)
    seed = FringeFractionMeasurement._initial_parameter_guess(interference_model(x, *true), x, true[1])
    np.testing.assert_allclose(seed, true, atol=1e-9)


def test_fit_converges_for_any_true_phase_with_a_wedge():
    """Block phase swept over a whole fringe (5 degree steps); block period 3 % shorter
    than the platen's (non-parallel surfaces) while the block fit is seeded at the
    platen frequency.

    With a fixed zero phase seed the fit converged to a wrong period whenever the true
    block phase lay about a quarter period from zero (95-110 and 250-265 degrees). Every
    fit must now converge to the true period and return the fraction at the reference
    row (x = 0).
    """
    rows = 500
    y = np.arange(rows) - rows / 2
    platen = 1 + 0.8 * np.sin(2 * np.pi / 37.6 * y + 0.3)
    for phi in np.linspace(0, 2 * np.pi, 72, endpoint=False):
        block = 1 + 0.8 * np.sin(2 * np.pi / 36.4 * y + phi)
        m, res = _measure_profiles(platen, block, 37.6, centre_offset=rows / 2)
        expected = ((phi - 0.3) / (2 * np.pi)) % 1.0
        assert abs(2 * np.pi / m.block_params[1] - 36.4) < 0.01
        assert res.block_quality.r2 > 0.999
        assert abs((res.fringe_fraction - expected + 0.5) % 1.0 - 0.5) < 1e-3


def test_real_wedge_profile_seeded_in_quadrature_converges():
    """Regression on real data: profiles averaged from a Twyman-Green interferogram of a
    2 mm gauge block (500 rows, fit coordinate centred on the evaluation row).

    The block period is ~3 % shorter than the platen's, so the block fit starts at a
    slightly wrong frequency, and the block's true phase lies about a quarter period
    from zero. With a fixed zero phase seed the optimiser traded phase for frequency and
    converged to a spurious sinusoid (R^2 ~ 0.03, period ~41 px). A data-driven phase
    seed must converge to the real fringes (period ~36.4 px).
    """
    d = np.load(DATA / "real_profiles_wedge_quadrature.npz")
    m, res = _measure_profiles(d["platen"], d["block"], float(d["pixels_between_fringes"]),
                               centre_offset=float(d["centre_offset"]))
    assert res.platen_quality.r2 > 0.8
    assert res.block_quality.r2 > 0.8
    assert abs(2 * np.pi / m.block_params[1] - 36.4) < 0.4


def test_canonicalize_flips_sign_and_preserves_curve():
    """A negative amplitude is flipped to positive, absorbing the sign as a pi shift.

    Guards the fix for the half-fringe (pi) ambiguity: the returned curve must be
    identical, so the fraction downstream is unaffected except for the sign branch.
    """
    x = np.arange(0, 200)
    negative = np.array([-5.0, 0.05, 0.4, 12.0])  # A < 0
    canonical = FringeFractionMeasurement._canonicalize(negative)

    assert canonical[0] > 0
    np.testing.assert_allclose(
        interference_model(x, *negative), interference_model(x, *canonical), atol=1e-9
    )


def test_canonicalize_leaves_positive_amplitude_untouched():
    positive = np.array([5.0, 0.05, 0.4, 12.0])  # A > 0
    np.testing.assert_allclose(
        FringeFractionMeasurement._canonicalize(positive), positive
    )


def test_canonicalize_folds_a_negative_frequency_and_preserves_curve():
    """On the w < 0 branch the phase runs backwards (F -> 1 - F); folding onto w > 0
    must leave the curve unchanged."""
    x = np.arange(0, 200)
    for params in ([5.0, -0.05, 0.4, 12.0], [-5.0, -0.05, 0.4, 12.0]):
        params = np.array(params)
        canonical = FringeFractionMeasurement._canonicalize(params)
        assert canonical[0] > 0 and canonical[1] > 0
        np.testing.assert_allclose(
            interference_model(x, *params), interference_model(x, *canonical), atol=1e-9
        )


def test_signed_fraction_reverses_the_order_and_validates_the_sign():
    import pytest
    from interflib.fringe.models import signed_fraction

    assert signed_fraction(2 * np.pi * 0.3) == pytest.approx(0.3)
    assert signed_fraction(2 * np.pi * 0.3, order_sign=-1) == pytest.approx(0.7)
    assert signed_fraction(2 * np.pi * 1.3) == pytest.approx(0.3)
    with pytest.raises(ValueError, match="order_sign"):
        signed_fraction(1.0, order_sign=0)
