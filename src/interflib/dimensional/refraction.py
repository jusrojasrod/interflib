"""Refractive index of air and wavelength in air (modified Edlen, Birch & Downs).

The index is computed with the modified Edlen equation of Birch & Downs (Metrologia 30,
155-162, 1993, with the 1994 correction), in the form documented by J. A. Stone and
J. H. Zimmerman, "Index of Refraction of Air", NIST Engineering Metrology Toolbox,
Appendix A-IV, eq. (A42)-(A49). The saturation vapour pressure follows the IAPWS
formulas of the same document, Appendix A-I (over water, eq. A1-A7; over ice, eq.
A8-A13). The input limits are those of the NIST calculator (section IV). All constants
live in :mod:`interflib.constants`.
"""
import math
import warnings
from typing import Optional

from interflib import constants as k


class EdlenInputBoundsError(ValueError):
    """Raised when an input lies outside the range accepted by the calculation."""


class EdlenRefractiveIndex:
    """Refractive index of air and wavelength in air from the environmental state.

    Parameters
    ----------
    temperature : float
        Air temperature, degrees Celsius.
    pressure : float
        Air pressure, pascals.
    relative_humidity : float, optional
        Relative humidity, percent (0 to 100).
    dew_point : float, optional
        Dew-point temperature, degrees Celsius.
    frost_point : float, optional
        Frost-point temperature, degrees Celsius.

    Exactly one of ``relative_humidity``, ``dew_point`` and ``frost_point`` must be
    given.

    Raises
    ------
    ValueError
        If no humidity state or more than one is given, or if the dew or frost point
        exceeds the air temperature.
    EdlenInputBoundsError
        If the temperature, pressure or relative humidity lies outside the accepted
        range. A pressure outside 60-120 kPa or a humidity above 85 % only warns.
    """

    def __init__(
        self,
        temperature: float,
        pressure: float,
        relative_humidity: Optional[float] = None,
        dew_point: Optional[float] = None,
        frost_point: Optional[float] = None,
    ):
        given = sum(v is not None for v in (relative_humidity, dew_point, frost_point))
        if given == 0:
            raise ValueError(
                "Incomplete environment state: provide exactly one of 'relative_humidity', "
                "'dew_point', or 'frost_point'."
            )
        if given > 1:
            raise ValueError(
                "Ambiguous environment state: provide exactly one of 'relative_humidity', "
                "'dew_point', or 'frost_point'."
            )

        t_lo, t_hi = k.AIR_TEMPERATURE_RANGE_C
        if not t_lo <= temperature <= t_hi:
            raise EdlenInputBoundsError(
                f"Temperature {temperature} °C is out of bounds. Must be between {t_lo} °C and {t_hi} °C.")
        p_lo, p_hi = k.AIR_PRESSURE_RANGE_PA
        if not p_lo <= pressure <= p_hi:
            raise EdlenInputBoundsError(
                f"Pressure {pressure} Pa is out of bounds. Must be between {p_lo:,.0f} Pa and {p_hi:,.0f} Pa.")
        r_lo, r_hi = k.AIR_PRESSURE_REALISTIC_PA
        if not r_lo <= pressure <= r_hi:
            warnings.warn(f"Pressure {pressure} Pa is outside the optimal recommended range "
                          f"({r_lo:,.0f} to {r_hi:,.0f} Pa).", UserWarning, stacklevel=2)
        if relative_humidity is not None:
            if not 0.0 <= relative_humidity <= 100.0:
                raise EdlenInputBoundsError(
                    f"Relative humidity {relative_humidity} % is invalid. Must be between 0 and 100.")
            if relative_humidity > k.AIR_HUMIDITY_CONDENSATION_PCT:
                warnings.warn(f"Relative humidity is {relative_humidity}%. Values above "
                              f"{k.AIR_HUMIDITY_CONDENSATION_PCT:.0f}% pose a risk of condensation.",
                              UserWarning, stacklevel=2)
        if dew_point is not None and dew_point > temperature:
            raise ValueError("Dew point cannot be higher than ambient temperature (RH > 100%).")
        if frost_point is not None and frost_point > temperature:
            raise ValueError("Frost point cannot be higher than ambient temperature (RH > 100%).")

        self.temperature = temperature
        self.pressure = pressure
        self.relative_humidity = relative_humidity
        self.dew_point = dew_point
        self.frost_point = frost_point

    @property
    def vapor_pressure(self) -> float:
        """Partial pressure of water vapour, Pa, from the humidity state.

        Computed on access, so it follows later changes of the temperature or humidity.
        [NIST] eq. (A14)-(A15): the saturation pressure at the dew (or frost) point, or
        the relative humidity times the saturation pressure at the air temperature (over
        ice below 0 °C).
        """
        if self.dew_point is not None:
            return self._saturation_vapor_pressure_over_water(self.dew_point)
        if self.frost_point is not None:
            return self._saturation_vapor_pressure_over_ice(self.frost_point)
        if self.temperature >= 0.0:
            p_sv = self._saturation_vapor_pressure_over_water(self.temperature)
        else:
            p_sv = self._saturation_vapor_pressure_over_ice(self.temperature)
        return (self.relative_humidity / 100.0) * p_sv

    @staticmethod
    def _saturation_vapor_pressure_over_water(temperature: float) -> float:
        """Saturation vapour pressure over water, Pa, at ``temperature`` (°C); eq. (A1)-(A7)."""
        k1, k2, k3, k4, k5, k6, k7, k8, k9, k10 = k.IAPWS_WATER_K
        t_kelvin = temperature + k.CELSIUS_TO_KELVIN
        omega = t_kelvin + k9 / (t_kelvin - k10)
        a = omega ** 2 + k1 * omega + k2
        b = k3 * omega ** 2 + k4 * omega + k5
        c = k6 * omega ** 2 + k7 * omega + k8
        x = -b + (b ** 2 - 4 * a * c) ** 0.5
        return 1e6 * (2 * c / x) ** 4

    @staticmethod
    def _saturation_vapor_pressure_over_ice(temperature: float) -> float:
        """Saturation vapour pressure over ice, Pa, at ``temperature`` (°C); eq. (A8)-(A13)."""
        theta = (temperature + k.CELSIUS_TO_KELVIN) / k.WATER_TRIPLE_POINT_K
        y = k.IAPWS_ICE_A1 * (1 - theta ** -1.5) + k.IAPWS_ICE_A2 * (1 - theta ** -1.25)
        return k.WATER_TRIPLE_POINT_PRESSURE_PA * math.exp(y)

    def index_refraction(self, wavelength: float) -> float:
        """Refractive index of air at a vacuum wavelength.

        Parameters
        ----------
        wavelength : float
            Vacuum wavelength, nanometres. Accepted from 300 to 1700 nm; outside the
            350-650 nm range claimed by Birch & Downs a warning is issued.

        Returns
        -------
        float
            Refractive index ``n`` of air, [NIST] eq. (A42)-(A49).
        """
        w_lo, w_hi = k.VACUUM_WAVELENGTH_RANGE_NM
        if not w_lo <= wavelength <= w_hi:
            raise EdlenInputBoundsError(
                f"Wavelength {wavelength} nm is out of bounds. Must be between {w_lo:.0f} and {w_hi:.0f} nm.")
        e_lo, e_hi = k.EDLEN_WAVELENGTH_RANGE_NM
        if not e_lo <= wavelength <= e_hi:
            warnings.warn(f"Wavelength {wavelength} nm is outside the strictly conservative Birch and "
                          f"Downs range ({e_lo:.0f} to {e_hi:.0f} nm).", UserWarning, stacklevel=2)
        t, p, p_v = self.temperature, self.pressure, self.vapor_pressure
        sigma2 = 1.0 / (wavelength / 1000.0) ** 2      # the dispersion formula is written in um^-2

        n_s = 1 + 1e-8 * (k.EDLEN_A + k.EDLEN_B / (k.EDLEN_DISPERSION_POLE_1 - sigma2)
                          + k.EDLEN_C / (k.EDLEN_DISPERSION_POLE_2 - sigma2))
        x = (1 + 1e-8 * (k.EDLEN_E - k.EDLEN_F * t) * p) / (1 + k.EDLEN_G * t)
        n_tp = 1 + p * (n_s - 1) * x / k.EDLEN_D
        vapour = (k.EDLEN_VAPOUR_T_REF / (t + k.CELSIUS_TO_KELVIN)) * (k.EDLEN_VAPOUR_H1 - k.EDLEN_VAPOUR_H2 * sigma2)
        return n_tp - 1e-10 * vapour * p_v

    def wavelength_in_air(self, wavelength: float) -> float:
        """Wavelength in air, nanometres, for a vacuum ``wavelength`` in nanometres."""
        return wavelength / self.index_refraction(wavelength)
