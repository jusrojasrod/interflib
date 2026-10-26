import pytest
from interflib.dimensional.refraction import EdlenRefractiveIndex, EdlenInputBoundsError

class TestEdlenCalculator:
    def test_nist_validation_happy_path(self):
        """Validates the refractive index calculation against typical laboratory conditions."""
        # Standard conditions: 20 °C, 101325 Pa, 50% humidity, 633 nm wavelength
        calc = EdlenRefractiveIndex(
            temperature=20.0,
            pressure=101325.0,
            relative_humidity=50.0
        )
        wavelength = 633.0  # nanometers
        n_calculated = calc.index_refraction(wavelength)
        expected_n =  1.00027134
        assert n_calculated == pytest.approx(expected_n, rel=1e-7)

    def test_dew_point_logic(self):
        """
        Validates that providing a dew point correctly influences the refractive index calculation.
        """
        calc = EdlenRefractiveIndex(
            temperature=20.0,
            pressure=101325.0,
            dew_point=9.2
        )
        n = calc.index_refraction(632.8)
        assert n > 1.0

    def test_fail_fast_incomplete_state(self):
        """
        Prueba que la clase lance ValueError si no se da ningún dato de humedad.
        """
        with pytest.raises(ValueError, match="Incomplete environment state"):
            EdlenRefractiveIndex(temperature=20.0, pressure=101325.0)

    def test_physical_bounds_exceptions(self):
        """
        Prueba que las excepciones personalizadas salten con valores imposibles.
        """
        # Temperatura extrema (-50 °C está fuera del rango -40 a 100)
        with pytest.raises(EdlenInputBoundsError, match="Temperature"):
            EdlenRefractiveIndex(temperature=-50.0, pressure=101325.0, relative_humidity=50.0)

        # Longitud de onda negativa o fuera de rango
        calc = EdlenRefractiveIndex(temperature=20.0, pressure=101325.0, relative_humidity=50.0)
        with pytest.raises(EdlenInputBoundsError, match="Wavelength"):
            calc.index_refraction(100.0) # 100 nm es menor que el límite de 300 nm

        # Punto de rocío mayor a temperatura ambiente
        with pytest.raises(ValueError, match="Dew point cannot be higher"):
            EdlenRefractiveIndex(temperature=20.0, pressure=101325.0, dew_point=25.0)

    def test_pressure_warning(self):
        """Ensures a warning is issued when pressure is outside the optimal range."""
        with pytest.warns(UserWarning, match="outside the optimal recommended range"):
            EdlenRefractiveIndex(temperature=20.0, pressure=50000.0, relative_humidity=50.0)

    def test_vapor_pressure_from_dew_point(self):
        """Validates that providing a dew point correctly calculates vapor pressure."""
        calc = EdlenRefractiveIndex(temperature=20.0, pressure=101325.0, dew_point=10.0)
        # Vapor pressure at 10C dew point should be positive
        assert calc.vapor_pressure > 0
        assert isinstance(calc.vapor_pressure, float)

    def test_wavelength_in_air(self):
        """Checks the conversion from vacuum wavelength to air wavelength."""
        vacuum_lambda = 632.991
        calc = EdlenRefractiveIndex(temperature=20.0, pressure=101325.0, relative_humidity=50.0)
        air_lambda = calc.wavelength_in_air(vacuum_lambda)

        # Air wavelength must be shorter than vacuum wavelength (n > 1)
        assert air_lambda < vacuum_lambda
        # n = lambda_vac / lambda_air
        n = calc.index_refraction(vacuum_lambda)
        assert pytest.approx(air_lambda * n) == vacuum_lambda

# Modified-Edlen column of Table 1 of J. A. Stone and J. H. Zimmerman, "Index of Refraction
# of Air", NIST Engineering Metrology Toolbox, section VII: temperature (C), relative
# humidity (%), pressure (kPa), vacuum wavelength (nm), index (rounded to 1e-9).
NIST_TABLE_1 = [
    (20, 0, 101.325, 633, 1.000271799),
    (20, 0, 60, 633, 1.000160920),
    (20, 0, 120, 633, 1.000321918),
    (50, 0, 100, 633, 1.000243270),
    (5, 0, 100, 633, 1.000282750),
    (-40, 0, 100, 633, 1.000337471),
    (50, 100, 120, 633, 1.000287864),
    (40, 75, 120, 633, 1.000299406),
    (20, 100, 100, 633, 1.000267394),
    (40, 100, 110, 1700, 1.000270237),
    (20, 0, 101.325, 1700, 1.000268483),
    (40, 100, 110, 300, 1.000288922),
    (20, 0, 101.325, 300, 1.000286579),
    (-40, 0, 120, 300, 1.000427072),
]


@pytest.mark.filterwarnings("ignore::UserWarning")
@pytest.mark.parametrize("t, rh, p_kpa, wavelength, expected", NIST_TABLE_1)
def test_matches_nist_table_1(t, rh, p_kpa, wavelength, expected):
    """Reproduces the NIST reference values to their rounding (5e-10); an error of 1e-9
    in n is 0.1 nm over 100 mm."""
    n = EdlenRefractiveIndex(t, p_kpa * 1e3, relative_humidity=rh).index_refraction(wavelength)
    assert n == pytest.approx(expected, abs=6e-10)


def test_more_than_one_humidity_state_is_rejected():
    with pytest.raises(ValueError, match="Ambiguous environment state"):
        EdlenRefractiveIndex(temperature=20.0, pressure=101325.0, relative_humidity=50.0, dew_point=9.0)
