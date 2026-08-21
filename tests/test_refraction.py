import pytest
from interflib.dimensional.refraction import EdlenRefractiveIndex, EdlenInputBoundsError

class TestEdlenCalculator:
    def test_nist_validation_happy_path(self):
        """Validates the refractive index calculation against typical laboratory conditions."""
        # Standard conditions: 20 °C, 101325 Pa, 50% humidity, 0.633 um wavelength
        calc = EdlenRefractiveIndex(
            temperature=20.0, 
            pressure=101325.0, 
            relative_humidity=50.0
        )
        wavelength = 0.633  # micrometers
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
        n = calc.index_refraction(0.6328)
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
            calc.index_refraction(0.1) # 0.1 um es menor que el límite de 0.3 um

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
        vacuum_lambda = 0.632991
        calc = EdlenRefractiveIndex(temperature=20.0, pressure=101325.0, relative_humidity=50.0)
        air_lambda = calc.wavelength_in_air(vacuum_lambda)
        
        # Air wavelength must be shorter than vacuum wavelength (n > 1)
        assert air_lambda < vacuum_lambda
        # n = lambda_vac / lambda_air
        n = calc.index_refraction(vacuum_lambda)
        assert pytest.approx(air_lambda * n) == vacuum_lambda