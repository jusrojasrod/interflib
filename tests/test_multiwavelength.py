"""Tests for the classical multi-wavelength method of exact fractions.

Validation strategy
-------------------
The reference data is the worked example of Liu et al. (2022): the Aoki 1400 um gauge
block, four visible wavelengths (Table I) and a nominal error range of +/- 1 um.

A subtlety, and the point of the method: with only *two* wavelengths and fringe
fractions given to two decimals, the integer-order choice is genuinely ambiguous.
Liu's Table II shows this on purpose -- the three wavelength pairs scatter by up to
0.668 um because two of them latch onto the wrong integer order. So the published
two-decimal fractions cannot reproduce Liu's Table II pair-by-pair (his figures come
from higher-precision fractions, and the whole paper is about that sensitivity).

What *is* unambiguous, and what this solver targets, is the joint solve over all four
wavelengths: only one length reproduces every measured fraction at once. Here that is
1399.8409 um (all three selecting residuals below 0.07 fringe). The l1 & l2 pair also
lands there and matches Liu's Table II value exactly, so it is asserted directly.
"""
import math

import numpy as np
import pytest

from interflib.dimensional.multiwavelength import (
    MultiwavelengthEFSolver,
    _circular_distance,
)

# Liu et al. (2022), Table I: employed wavelengths (nm) and measured excess fractions.
LIU_WAVELENGTHS = [667.8184, 587.5649, 501.5702, 447.1501]
LIU_FRACTIONS = [0.28, 0.85, 0.90, 0.23]
LIU_NOMINAL_NM = 1_400_000.0     # L' = 1400 um
LIU_TOLERANCE_NM = 1_000.0       # error range +/- 1 um
LIU_JOINT_LENGTH_UM = 1399.840861  # length consistent with all four fractions


def _frac(x):
    """Positive fractional part, for building synthetic fractions in the tests."""
    return x - np.floor(x)


def test_l1_l2_pair_matches_liu_table_2():
    """The l1 & l2 pair reproduces Liu's Table II length (1399.840861 um) to sub-nm."""
    solver = MultiwavelengthEFSolver(wavelengths=LIU_WAVELENGTHS[:2])
    solution = solver.solve(LIU_NOMINAL_NM, LIU_TOLERANCE_NM, LIU_FRACTIONS[:2])
    assert solution.length / 1000.0 == pytest.approx(1399.840861, abs=1e-3)


def test_joint_four_wavelength_solve_resolves_the_block():
    """All four wavelengths jointly pin the length that fits every fraction.

    This is the ambiguity that no single pair can resolve: the winning candidate has a
    small residual (all selecting fractions consistent), while any other integer order
    leaves at least one selecting wavelength off by a large fraction of a fringe.
    """
    solver = MultiwavelengthEFSolver(wavelengths=LIU_WAVELENGTHS)
    solution = solver.solve(LIU_NOMINAL_NM, LIU_TOLERANCE_NM, LIU_FRACTIONS)

    assert solution.length / 1000.0 == pytest.approx(LIU_JOINT_LENGTH_UM, abs=1e-3)
    assert solution.residual < 0.07          # every selecting fraction is consistent
    assert solution.margin > 0.1             # clearly separated from the runner-up


def test_recovers_synthetic_length_exactly():
    """With exact fractions the solver returns the true length to machine precision."""
    wavelengths = np.array([632.9912, 643.8469, 508.5824])   # He-Ne red + two Cd lines
    true_length = 25_000_137.4                                # ~25 mm block, in nm
    fractions = _frac(2.0 * true_length / wavelengths)

    solver = MultiwavelengthEFSolver(wavelengths=wavelengths)
    solution = solver.solve(nominal_length=25_000_000.0, nominal_tolerance=1_000.0,
                            fractions=fractions)

    assert solution.length == pytest.approx(true_length, abs=1e-6)
    expected_orders = [int(round(2.0 * true_length / lam - f))
                       for lam, f in zip(wavelengths, fractions)]
    assert solution.integer_orders == expected_orders
    assert solution.residual == pytest.approx(0.0, abs=1e-9)


def test_robust_to_small_fraction_noise():
    """Small fraction errors keep the correct integer order (length within one fringe)."""
    wavelengths = np.array([632.9912, 643.8469, 508.5824])
    true_length = 25_000_137.4
    fractions = _frac(2.0 * true_length / wavelengths)
    rng = np.random.default_rng(0)
    noisy = fractions + rng.normal(0.0, 0.01, size=fractions.shape)   # ~1/100 fringe

    solver = MultiwavelengthEFSolver(wavelengths=wavelengths)
    solution = solver.solve(nominal_length=25_000_000.0, nominal_tolerance=1_000.0,
                            fractions=noisy)

    # Correct integer order recovered => length within half the anchor wavelength.
    assert abs(solution.length - true_length) < wavelengths[0] / 2.0


def test_solver_is_agnostic_to_wavelength_count():
    """Two, three or four wavelengths all recover the same exact length.

    Note the margin (separation from the runner-up) depends on *which* wavelengths are
    used, not merely how many: an ill-chosen extra wavelength can even reduce it. That
    is the wavelength-selection problem (Falaggis 2011), out of scope for this solver.
    """
    true_length = 25_000_137.4
    for count in (2, 3, 4):
        wavelengths = np.array([632.9912, 643.8469, 508.5824, 479.9914])[:count]
        solution = MultiwavelengthEFSolver(wavelengths).solve(
            25_000_000.0, 1_000.0, _frac(2.0 * true_length / wavelengths))
        assert solution.length == pytest.approx(true_length, abs=1e-6)
        assert solution.margin > 0.0


def test_requires_at_least_two_wavelengths():
    with pytest.raises(ValueError, match="at least two wavelengths"):
        MultiwavelengthEFSolver(wavelengths=[632.9912])


def test_rejects_mismatched_fraction_count():
    solver = MultiwavelengthEFSolver(wavelengths=[632.9912, 543.5160])
    with pytest.raises(ValueError, match="same length"):
        solver.solve(nominal_length=1e6, nominal_tolerance=1e3, fractions=[0.1])


def test_raises_when_tolerance_excludes_all_candidates():
    solver = MultiwavelengthEFSolver(wavelengths=[632.9912, 543.5160])
    # Nominal sits at an integer order (fraction ~0); the measured 0.5-fraction lengths
    # are ~lambda/4 away, so a 1 nm tolerance leaves no candidate inside the window.
    nominal = 10_000 * 632.9912 / 2.0
    with pytest.raises(RuntimeError, match="within the nominal tolerance"):
        solver.solve(nominal_length=nominal, nominal_tolerance=1.0, fractions=[0.5, 0.5])


def test_circular_distance_wraps():
    assert _circular_distance(0.99, 0.01) == pytest.approx(0.02)
    assert _circular_distance(0.10, 0.60) == pytest.approx(0.5)
    assert _circular_distance(0.30, 0.30) == pytest.approx(0.0)


def test_anchor_fraction_is_reproduced_exactly():
    """The first wavelength anchors the sub-fringe: its fraction comes back exactly."""
    solver = MultiwavelengthEFSolver(wavelengths=LIU_WAVELENGTHS)
    solution = solver.solve(LIU_NOMINAL_NM, LIU_TOLERANCE_NM, LIU_FRACTIONS)
    anchor_fraction = _frac(2.0 * solution.length / LIU_WAVELENGTHS[0])
    assert anchor_fraction == pytest.approx(LIU_FRACTIONS[0], abs=1e-9)
