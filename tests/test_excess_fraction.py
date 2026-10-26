import math

import pytest

from interflib.dimensional.excess_fraction import GaugeBlockLength
from interflib.dimensional.refraction import EdlenRefractiveIndex

WAVELENGTH = 632.9905846     # nm, in air
NOMINAL = 10e6               # nm (10 mm)


def _exact_fraction(length_nm, wavelength=WAVELENGTH):
    order = 2 * length_nm / wavelength          # half-wave count 2 L / lambda
    return order - math.floor(order)


def test_recovers_length_from_air_wavelength():
    """With the exact fringe fraction, the resolver returns the nominal length."""
    gb = GaugeBlockLength(wavelength=WAVELENGTH, nominal_length=NOMINAL, cte=11.5e-6, temp_c=20.0,
                          fringe_fraction=_exact_fraction(NOMINAL))
    solution = gb.solve()
    assert solution.length == pytest.approx(NOMINAL, abs=1e-6)
    assert solution.length_dev == pytest.approx(0.0, abs=1e-6)


def test_recovers_a_deviation_within_a_quarter_wave():
    """A block 120 nm long is resolved to the right order from the nominal prior."""
    true_length = NOMINAL + 120.0
    solution = GaugeBlockLength(wavelength=WAVELENGTH, nominal_length=NOMINAL, cte=11.5e-6, temp_c=20.0,
                                fringe_fraction=_exact_fraction(true_length)).solve()
    assert solution.length_dev == pytest.approx(120.0, abs=1e-6)


def test_from_environment_delegates_air_index_to_refraction():
    """``from_environment`` computes n and the wavelength in air via the Edlen module."""
    wavelength_vac = 633.0
    gb = GaugeBlockLength.from_environment(
        wavelength_vac=wavelength_vac, nominal_length=NOMINAL, cte=11.5e-6, temp_c=20.0,
        fringe_fraction=0.0, temperature=20.0, pressure=101325.0, relative_humidity=50.0,
    )
    env = EdlenRefractiveIndex(20.0, 101325.0, relative_humidity=50.0)
    assert gb.n_air > 1.0
    assert gb.wavelength < wavelength_vac                         # air shortens the wave
    assert gb.n_air == pytest.approx(env.index_refraction(wavelength_vac))
    assert gb.wavelength == pytest.approx(env.wavelength_in_air(wavelength_vac))


def test_reduces_to_reference_temperature():
    """A block above 20 C reduces to a shorter length at the 20 C reference."""
    fraction = _exact_fraction(NOMINAL)
    kwargs = dict(wavelength=WAVELENGTH, nominal_length=NOMINAL, cte=11.5e-6, fringe_fraction=fraction)
    at_20 = GaugeBlockLength(temp_c=20.0, **kwargs).solve().length
    warmer = GaugeBlockLength(temp_c=20.5, **kwargs).solve().length
    assert at_20 == pytest.approx(NOMINAL, abs=1e-6)
    assert warmer < at_20


def test_former_signature_warns_and_keeps_millimetres():
    """The positional (lambda_air, Lp [mm], C, t_block, obs_fraction) call and block_length()
    (mm) still work during the transition, with a DeprecationWarning."""
    fraction = _exact_fraction(NOMINAL)
    with pytest.warns(DeprecationWarning, match="nominal_length in nm"):
        old = GaugeBlockLength(WAVELENGTH, 10.0, 11.5e-6, 20.0, fraction)
    with pytest.warns(DeprecationWarning, match="nominal_length in nm"):
        old_kw = GaugeBlockLength(lambda_air=WAVELENGTH, Lp=10.0, C=11.5e-6, t_block=20.0, obs_fraction=fraction)
    assert old.nominal_length == old_kw.nominal_length == NOMINAL
    with pytest.warns(DeprecationWarning, match="solve"):
        assert old.block_length() == pytest.approx(10.0, abs=1e-12)
    with pytest.warns(DeprecationWarning, match="nominal_length"):
        assert old.Lp == 10.0
    with pytest.warns(DeprecationWarning, match="nominal_length in nm"):
        env = GaugeBlockLength.from_environment(633.0, 10.0, 11.5e-6, 20.0, 0.0, temperature=20.0,
                                                pressure=101325.0, relative_humidity=50.0)
    with pytest.warns(DeprecationWarning, match="n_air"):
        assert env.refractive_index == env.n_air


def test_mixing_old_and_new_names_is_refused():
    with pytest.raises(TypeError, match="both"), pytest.warns(DeprecationWarning):
        GaugeBlockLength(Lp=10.0, nominal_length=NOMINAL, wavelength=WAVELENGTH, cte=0.0, temp_c=20.0,
                         fringe_fraction=0.0)
    with pytest.raises(TypeError, match="missing"):
        GaugeBlockLength(wavelength=WAVELENGTH, nominal_length=NOMINAL)
