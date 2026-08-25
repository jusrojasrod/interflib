"""Refractive index of air and wavelength in air (modified Edlen, Birch & Downs).
"""
import math
import warnings
from typing import Optional

class EdlenInputBoundsError(ValueError):
    """Custom exception raised when inputs are physically out of valid bounds."""
    pass

class EdlenRefractiveIndex:
    """
    Calculator for the refractive index of air and wavelength in air
    based on the modified Edlen equation and Huang / IAPWS equations.
    """

    # Edlen Equation Constants
    _EDLEN_A = 8342.54
    _EDLEN_B = 2406147.0
    _EDLEN_C = 15998.0
    _EDLEN_D = 96095.43
    _EDLEN_E = 0.601
    _EDLEN_F = 0.00972
    _EDLEN_G = 0.003661
    
    # Physical Constants
    _CELSIUS_TO_KELVIN = 273.15
    _WATER_TRIPLE_POINT_K = 273.16

    def __init__(
        self, 
        temperature: float, 
        pressure: float, 
        relative_humidity: Optional[float] = None, 
        dew_point: Optional[float] = None, 
        frost_point: Optional[float] = None
    ):
        """
        Initializes the environmental conditions for the measurements.

        Args:
            temperature: Air temperature in degrees Celsius (°C).
            pressure: Air pressure in Pascals (Pa).
            relative_humidity: Fractional relative humidity (0 to 100).
            dew_point: Set to the dew point temperature if the conditions represent the dew point.
            frost_point: Set to the frost point temperature if the conditions represent the frost point.
        """
        if relative_humidity is None and dew_point is None and frost_point is None:
            raise ValueError(
                "Incomplete environment state: provide exactly one of 'relative_humidity', "
                "'dew_point', or 'frost_point'."
            )
        if not (-40.0 <= temperature <= 100.0): raise EdlenInputBoundsError(f"Temperature {temperature} °C is out of bounds. Must be between -40 °C and 100 °C.")
        if not (10000.0 <= pressure <= 140000.0): raise EdlenInputBoundsError(f"Pressure {pressure} Pa is out of bounds. Must be between 10,000 Pa and 140,000 Pa.")
        if not (60000.0 <= pressure <= 120000.0): warnings.warn(f"Pressure {pressure} Pa is outside the optimal recommended range (60,000 to 120,000 Pa).", UserWarning)
        if relative_humidity is not None:
            if not (0.0 <= relative_humidity <= 100.0): raise EdlenInputBoundsError(f"Relative humidity {relative_humidity} % is invalid. Must be between 0 and 100.")
            if relative_humidity > 85.0: warnings.warn(f"Relative humidity is {relative_humidity}%. Values above 85% pose a risk of condensation.", UserWarning)
        if dew_point is not None and dew_point > temperature: raise ValueError("Dew point cannot be higher than ambient temperature (RH > 100%).")
        if frost_point is not None and frost_point > temperature: raise ValueError("Frost point cannot be higher than ambient temperature (RH > 100%).")

        self.temperature = temperature
        self.pressure = pressure
        self.relative_humidity = relative_humidity
        self.dew_point = dew_point
        self.frost_point = frost_point

    @property
    def vapor_pressure(self) -> float:
        """
        Dynamically computes the vapor pressure (Pa) based on current conditions.
        Using a property ensures this value is always up to date if temperature 
        or humidity changes after instantiation.
        """
        if self.dew_point is not None: return self._saturation_vapor_pressure_over_water(self.dew_point)
        if self.frost_point is not None: return self._saturation_vapor_pressure_over_ice(self.frost_point) 
        if self.relative_humidity is not None: 
            if self.temperature >= 0.0: p_sv = self._saturation_vapor_pressure_over_water(self.temperature)
            else: p_sv = self._saturation_vapor_pressure_over_ice(self.temperature)
            return (self.relative_humidity / 100.0) * p_sv

    @classmethod
    def _saturation_vapor_pressure_over_water(cls,temperature: float) -> float:
        """
        Calculates saturation vapor pressure over water.
        Implemented as a static method to ensure pure function behavior for easy testing.
        """
        k1, k2, k3 = 1.16705214528e3, -7.24213167032e5, -1.70738469401e1
        k4, k5, k6 = 1.20208247025e4, -3.23255503223e6, 1.49151086135e1
        k7, k8, k9 = -4.82326573616e03, 4.05113405421e05, -2.38555575678e-1
        k10 = 6.50175348448e2
        
        t_kelvin = temperature + cls._CELSIUS_TO_KELVIN
        omega = t_kelvin + (k9 / (t_kelvin - k10))
        
        a = omega**2 + k1 * omega + k2
        b = k3 * omega**2 + k4 * omega + k5
        c = k6 * omega**2 + k7 * omega + k8
        x = -b + (b**2 - 4 * a * c)**0.5
        return 1e6 * (2 * c / x)**4

    @classmethod
    def _saturation_vapor_pressure_over_ice(cls, temperature: float) -> float:
        """
        Calculates saturation vapor pressure over ice.
        """
        a1, a2 = -13.928169, 34.7078238
        t_kelvin = temperature + cls._CELSIUS_TO_KELVIN
        theta = t_kelvin / cls._WATER_TRIPLE_POINT_K
        y = a1 * (1 - theta**(-1.5)) + a2 * (1 - theta**(-1.25))
        return 611.657 * math.exp(y)

    def index_refraction(self, wavelength: float) -> float:
        """
        Calculates the refractive index of air for a given wavelength.

        Args:
            wavelength: Wavelength in vacuum in micrometers (um).

        Returns:
            The refractive index of air.
        """
        if not (0.3 <= wavelength <= 1.7): raise EdlenInputBoundsError(f"Wavelength {wavelength} um is out of bounds. Must be between 0.3 and 1.7 um.")
        if not (0.35 <= wavelength <= 0.65): warnings.warn(f"Wavelength {wavelength} um is outside the strictly conservative Birch and Downs range (0.35 to 0.65 um).", UserWarning)
        # Aliases 
        t = self.temperature
        p = self.pressure
        p_v = self.vapor_pressure
        s = 1.0 / wavelength**2
        
        n_s = 1 + 10**(-8) * (self._EDLEN_A + self._EDLEN_B / (130.0 - s) + self._EDLEN_C / (38.9 - s))
        x = (1 + 10**(-8) * (self._EDLEN_E - self._EDLEN_F * t) * p) / (1 + self._EDLEN_G * t)
        n_tp = 1 + p * (n_s - 1) * x / self._EDLEN_D
        n = n_tp - 1e-10 * (292.75 / (t + self._CELSIUS_TO_KELVIN)) * (3.7345 - 0.0401 * s) * p_v
        return n

    def wavelength_in_air(self, wavelength: float) -> float:
        """
        Calculates the wavelength in air.

        Args:
            wavelength: Wavelength in vacuum in micrometers (um).

        Returns:
            The wavelength in air in micrometers (um).
        """
        n = self.index_refraction(wavelength)
        return wavelength / n