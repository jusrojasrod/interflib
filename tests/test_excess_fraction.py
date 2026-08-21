import math

import pytest

from interflib.dimensional.excess_fraction import GaugeBlockLength
from interflib.dimensional.refraction import EdlenRefractiveIndex


def test_recovers_length_from_air_wavelength():
    """With the exact fringe fraction, the resolver returns the nominal length."""
    lambda_air_um = 0.6329905846
    Lp = 10.0
    order = 2 * Lp / (lambda_air_um / 1000.0)      # half-wave count 2 L / lambda
    obs_fraction = order - math.floor(order)
    gb = GaugeBlockLength(lambda_air=lambda_air_um, Lp=Lp, C=11.5e-6,
                          t_block=20.0, obs_fraction=obs_fraction)
    assert gb.block_length() == pytest.approx(Lp, abs=1e-9)


def test_from_environment_delegates_air_index_to_refraction():
    """``from_environment`` computes n and lambda_air via the Edlen module."""
    lambda_vacuum = 0.633
    gb = GaugeBlockLength.from_environment(
        lambda_vacuum=lambda_vacuum, Lp=10.0, C=11.5e-6, t_block=20.0,
        obs_fraction=0.0, temperature=20.0, pressure=101325.0,
        relative_humidity=50.0,
    )
    env = EdlenRefractiveIndex(20.0, 101325.0, relative_humidity=50.0)
    assert gb.refractive_index > 1.0
    assert gb.lambda_air < lambda_vacuum                       # air shortens the wave
    assert gb.refractive_index == pytest.approx(env.index_refraction(lambda_vacuum))
    assert gb.lambda_air == pytest.approx(env.wavelength_in_air(lambda_vacuum))


def test_reduces_to_reference_temperature():
    """A block above 20 C reduces to a shorter length at the 20 C reference."""
    lambda_air_um = 0.6329905846
    Lp = 10.0
    order = 2 * Lp / (lambda_air_um / 1000.0)
    obs_fraction = order - math.floor(order)
    at_20 = GaugeBlockLength(lambda_air=lambda_air_um, Lp=Lp, C=11.5e-6,
                             t_block=20.0, obs_fraction=obs_fraction).block_length()
    warmer = GaugeBlockLength(lambda_air=lambda_air_um, Lp=Lp, C=11.5e-6,
                              t_block=20.5, obs_fraction=obs_fraction).block_length()
    assert at_20 == pytest.approx(Lp, abs=1e-9)
    assert warmer < at_20
