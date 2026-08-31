"""Tests for the brute-force coincidence method of exact fractions (Doiron & Beers).

Besides its own correctness, this solver is an *independent* implementation of the same
physics as the correction-to-nominal solver, so the key test is that the two agree on
clean data (test_agrees_with_multiwavelength_on_clean_data). They are built on different
principles -- symmetric comb coincidence versus anchored residual minimisation -- so
agreement is real corroboration, not a tautology.
"""
import numpy as np
import pytest

from interflib.dimensional.coincidence import CoincidenceEFSolver
from interflib.dimensional.multiwavelength import MultiwavelengthEFSolver

# Liu et al. (2022), Table I, reused here as a documented multi-colour dataset.
LIU_WAVELENGTHS = [667.8184, 587.5649, 501.5702, 447.1501]
LIU_FRACTIONS = [0.28, 0.85, 0.90, 0.23]
LIU_NOMINAL_NM = 1_400_000.0
LIU_TOLERANCE_NM = 1_000.0


def _frac(x):
    return x - np.floor(x)


def test_four_colours_give_a_unique_coincidence():
    """All four Liu colours coincide at exactly one length near 1399.841 um."""
    solver = CoincidenceEFSolver(LIU_WAVELENGTHS)
    solution = solver.solve(LIU_NOMINAL_NM, LIU_TOLERANCE_NM, LIU_FRACTIONS)

    assert solution.is_unique
    assert solution.length / 1000.0 == pytest.approx(1399.840861, abs=1e-2)
    assert solution.residual < 0.07


def test_two_colours_leave_the_ambiguity_explicit():
    """A single colour pair over a wide window leaves several coincidences.

    This is the distinctive behaviour: where the anchored solver silently returns one
    best length, the coincidence solver exposes that two colours cannot resolve it.
    """
    solver = CoincidenceEFSolver([LIU_WAVELENGTHS[0], LIU_WAVELENGTHS[3]])
    solution = solver.solve(LIU_NOMINAL_NM, LIU_TOLERANCE_NM,
                            [LIU_FRACTIONS[0], LIU_FRACTIONS[3]])

    assert not solution.is_unique
    assert len(solution.coincidence_lengths) >= 2


def test_agrees_with_multiwavelength_on_clean_data():
    """Independent solvers land on the same length (to machine precision) when exact."""
    wavelengths = np.array([632.9912, 643.8469, 508.5824, 479.9914])
    true_length = 25_000_137.4
    fractions = _frac(2.0 * true_length / wavelengths)

    a1 = MultiwavelengthEFSolver(wavelengths).solve(
        25_000_000.0, 1_000.0, fractions)
    a2 = CoincidenceEFSolver(wavelengths).solve(
        25_000_000.0, 1_000.0, fractions)

    assert a2.is_unique
    assert a2.length == pytest.approx(true_length, abs=1e-6)
    assert a2.length == pytest.approx(a1.length, abs=1e-6)
    assert a2.integer_orders == a1.integer_orders


def test_recovers_synthetic_length_exactly():
    wavelengths = np.array([632.9912, 643.8469, 508.5824])
    true_length = 25_000_137.4
    fractions = _frac(2.0 * true_length / wavelengths)

    solution = CoincidenceEFSolver(wavelengths).solve(
        25_000_000.0, 1_000.0, fractions)

    assert solution.length == pytest.approx(true_length, abs=1e-6)
    assert solution.residual == pytest.approx(0.0, abs=1e-9)


def test_robust_to_small_fraction_noise():
    """Noise below the reading uncertainty keeps the correct integer order."""
    wavelengths = np.array([632.9912, 643.8469, 508.5824])
    true_length = 25_000_137.4
    fractions = _frac(2.0 * true_length / wavelengths)
    rng = np.random.default_rng(1)
    noisy = fractions + rng.normal(0.0, 0.01, size=fractions.shape)

    solution = CoincidenceEFSolver(wavelengths, fraction_uncertainty=0.05).solve(
        25_000_000.0, 1_000.0, noisy)

    assert abs(solution.length - true_length) < wavelengths[0] / 2.0


def test_refined_length_is_symmetric_not_anchored():
    """Under noise the estimate balances all colours instead of pinning colour 0.

    The comb of colour 0 only passes through anchor-exact lengths; the refined estimate
    should move off that comb because it averages over the other colours too.
    """
    wavelengths = np.array([632.9912, 643.8469, 508.5824])
    true_length = 25_000_137.4
    exact = _frac(2.0 * true_length / wavelengths)
    noisy = exact.copy()
    noisy[0] += 0.02                              # perturb only colour 0's fraction

    solution = CoincidenceEFSolver(wavelengths).solve(
        25_000_000.0, 1_000.0, noisy)

    anchor_exact = _frac(2.0 * solution.length / wavelengths[0])
    assert anchor_exact != pytest.approx(noisy[0], abs=1e-6)


def test_requires_at_least_two_wavelengths():
    with pytest.raises(ValueError, match="at least two wavelengths"):
        CoincidenceEFSolver(wavelengths=[632.9912])


def test_rejects_mismatched_fraction_count():
    solver = CoincidenceEFSolver(wavelengths=[632.9912, 543.5160])
    with pytest.raises(ValueError, match="same length"):
        solver.solve(nominal_length=1e6, nominal_tolerance=1e3, fractions=[0.1])


def test_raises_when_window_has_no_candidate():
    solver = CoincidenceEFSolver(wavelengths=[632.9912, 543.5160])
    nominal = 10_000 * 632.9912 / 2.0             # sits at an integer order (fraction ~0)
    with pytest.raises(RuntimeError, match="within the search window"):
        solver.solve(nominal_length=nominal, nominal_tolerance=1.0, fractions=[0.5, 0.5])
