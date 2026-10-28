"""The flipped reading 1 - F and its test against a prior, with analytic values."""
import math

import pytest

from interflib.dimensional.parity import check_fraction_parity

H = 271.7   # half-wave, nm


def test_flipped_comb_is_shifted_by_one_minus_two_f():
    check = check_fraction_parity(H, 51.0, 0.18)
    assert check.flipped_fraction == pytest.approx(0.82)
    assert check.length_dev == pytest.approx(51.0)
    # (1 - 2 F) H = 0.64 H, folded to the member nearest the nominal: 51 + 0.64 H - H
    assert check.flipped_length_dev == pytest.approx(51.0 - 0.36 * H)
    assert check.separation == pytest.approx(0.36 * H)
    assert check.verdict == "unverified"
    assert math.isnan(check.z) and math.isnan(check.log10_likelihood_ratio)


@pytest.mark.parametrize("fraction, separation", [(0.25, H / 2), (0.75, H / 2), (0.5, 0.0), (0.0, 0.0)])
def test_separation_is_largest_at_a_quarter_and_vanishes_at_zero_and_half(fraction, separation):
    check = check_fraction_parity(H, 0.0, fraction)
    assert check.separation == pytest.approx(separation, abs=1e-9)
    if separation == 0.0:
        assert check.flipped_length_dev == pytest.approx(check.length_dev, abs=1e-9)


def test_the_order_of_the_given_deviation_does_not_matter():
    a = check_fraction_parity(H, 51.0, 0.18, prior_length_dev=40.0, prior_u=10.0)
    b = check_fraction_parity(H, 51.0 + 3 * H, 0.18, prior_length_dev=40.0, prior_u=10.0)
    assert (a.length_dev, a.flipped_length_dev) == pytest.approx((b.length_dev, b.flipped_length_dev))


def test_verdicts_against_the_prior():
    d_flip = 51.0 - 0.36 * H
    confirmed = check_fraction_parity(H, 51.0, 0.18, prior_length_dev=51.0, prior_u=10.0)
    assert confirmed.verdict == "confirmed"
    assert confirmed.z == pytest.approx(0.0)
    assert confirmed.flipped_z == pytest.approx(-0.36 * H / 10.0)
    assert confirmed.log10_likelihood_ratio > 10
    contradicted = check_fraction_parity(H, 51.0, 0.18, prior_length_dev=d_flip, prior_u=10.0)
    assert contradicted.verdict == "contradicted"
    assert contradicted.log10_likelihood_ratio < -10
    undecided = check_fraction_parity(H, 51.0, 0.18, prior_length_dev=0.0, prior_u=60.0)
    assert undecided.verdict == "undecided"
    # half-way between the combs, with a tight prior, neither candidate fits
    inconsistent = check_fraction_parity(H, 51.0, 0.18, prior_length_dev=51.0 + H / 2, prior_u=5.0)
    assert inconsistent.verdict == "inconsistent"


def test_the_optical_uncertainty_adds_to_the_prior():
    check = check_fraction_parity(H, 51.0, 0.18, prior_length_dev=21.0, prior_u=6.0, u_length=8.0)
    assert check.z == pytest.approx(30.0 / 10.0)


def test_a_prior_without_uncertainty_is_rejected():
    with pytest.raises(ValueError):
        check_fraction_parity(H, 51.0, 0.18, prior_length_dev=51.0, prior_u=0.0)
