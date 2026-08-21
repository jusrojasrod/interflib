"""Unit tests for the fringe-profile fit helpers (synthetic, no image needed)."""
import numpy as np

from interflib.fringe.profile_fit import FringeFractionMeasurement
from interflib.fringe.models import interference_model


def test_canonicalize_amplitude_flips_sign_and_preserves_curve():
    """A negative amplitude is flipped to positive, absorbing the sign as a pi shift.

    Guards the fix for the half-fringe (pi) ambiguity: the returned curve must be
    identical, so the fraction downstream is unaffected except for the sign branch.
    """
    x = np.arange(0, 200)
    negative = np.array([-5.0, 0.05, 0.4, 12.0])  # A < 0
    canonical = FringeFractionMeasurement._canonicalize_amplitude(negative)

    assert canonical[0] > 0
    np.testing.assert_allclose(
        interference_model(x, *negative), interference_model(x, *canonical), atol=1e-9
    )


def test_canonicalize_amplitude_leaves_positive_amplitude_untouched():
    positive = np.array([5.0, 0.05, 0.4, 12.0])  # A > 0
    np.testing.assert_allclose(
        FringeFractionMeasurement._canonicalize_amplitude(positive), positive
    )
