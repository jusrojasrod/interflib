"""Absolute length of a gauge block by the excess fraction method (single wavelength).

Given the wavelength in air, the nominal length and the measured fringe fraction, the
integer fringe order is resolved from the nominal length and the absolute length is
reduced to the 20 C reference temperature.

The air refractive index is **not** computed here. Either pass the wavelength in air
directly (:meth:`__init__`), or build the object from a vacuum wavelength plus the
environmental state (:meth:`from_environment`), which delegates the air index to the
modified Edlen equation (Birch & Downs) in ``interflib.dimensional.refraction``. This
keeps a single air-index implementation in the package.
"""
import math

from interflib.constants import REFERENCE_TEMPERATURE_C
from interflib.dimensional.refraction import EdlenRefractiveIndex


class GaugeBlockLength:
    """Excess-fraction length resolver for a single wavelength."""

    def __init__(self, lambda_air: float, Lp: float, C: float, t_block: float,
                 obs_fraction: float):
        """
        Parameters
        ----------
        lambda_air : float
            Wavelength in air, in micrometres (um).
        Lp : float
            Nominal length of the block, in millimetres (mm).
        C : float
            Thermal expansion coefficient of the block material, in K^-1.
        t_block : float
            Block temperature, in degrees Celsius.
        obs_fraction : float
            Measured optical fringe fraction, in [0, 1).
        """
        self.lambda_air = lambda_air        # um
        self.Lp = Lp                        # mm
        self.C = C
        self.t_block = t_block
        self.obs_fraction = obs_fraction

    @classmethod
    def from_environment(cls, lambda_vacuum: float, Lp: float, C: float,
                         t_block: float, obs_fraction: float, *,
                         temperature: float, pressure: float,
                         relative_humidity: float = None,
                         dew_point: float = None,
                         frost_point: float = None) -> "GaugeBlockLength":
        """Build from a vacuum wavelength and the environmental state.

        The air refractive index and the wavelength in air are computed with the
        modified Edlen equation (Birch & Downs) in
        :class:`interflib.dimensional.refraction.EdlenRefractiveIndex`.

        Parameters
        ----------
        lambda_vacuum : float
            Vacuum wavelength, in micrometres (um).
        temperature : float
            Air temperature, in degrees Celsius.
        pressure : float
            Air pressure, in pascals (Pa).
        relative_humidity, dew_point, frost_point : float, optional
            Humidity state; provide exactly one (see ``EdlenRefractiveIndex``).
        Lp, C, t_block, obs_fraction :
            As in :meth:`__init__`.
        """
        env = EdlenRefractiveIndex(temperature=temperature, pressure=pressure,
                                   relative_humidity=relative_humidity,
                                   dew_point=dew_point, frost_point=frost_point)
        lambda_air = env.wavelength_in_air(lambda_vacuum)  # um
        instance = cls(lambda_air, Lp, C, t_block, obs_fraction)
        # Kept for reporting/traceability of the environmental correction.
        instance.wavelength_vacuum = lambda_vacuum
        instance.refractive_index = env.index_refraction(lambda_vacuum)
        return instance

    def estimate_integer_fringes(self) -> float:
        """Theoretical half-wave count 2 L / lambda_air for the nominal length."""
        lambda_air_mm = self.lambda_air / 1000.0
        expansion = 1 + self.C * (self.t_block - REFERENCE_TEMPERATURE_C)
        return (2 * self.Lp * expansion) / lambda_air_mm

    def block_length(self) -> float:
        """Absolute length reduced to the reference temperature, in millimetres."""
        f_theoretical = self.estimate_integer_fringes()
        order = math.floor(f_theoretical)
        fraction_theoretical = f_theoretical - order

        # Resolve the integer order: the observed fraction may sit one fringe across
        # the theoretical one.
        if (self.obs_fraction - fraction_theoretical) > 0.5:
            order -= 1
        elif (fraction_theoretical - self.obs_fraction) > 0.5:
            order += 1

        f_prime = order + self.obs_fraction
        lambda_air_mm = self.lambda_air / 1000.0
        reduction = 1 + self.C * (REFERENCE_TEMPERATURE_C - self.t_block)
        return (lambda_air_mm / 2.0) * f_prime * reduction


if __name__ == "__main__":
    # Real-data path: vacuum wavelength + environmental state -> air index.
    block = GaugeBlockLength.from_environment(
        lambda_vacuum=0.63299139,          # um (iodine-stabilised He-Ne)
        Lp=10.0, C=11.5e-6, t_block=20.0, obs_fraction=0.05,
        temperature=20.0, pressure=101325.0, relative_humidity=50.0,
    )
    print(f"n (air)          : {block.refractive_index:.9f}")
    print(f"wavelength (air) : {block.lambda_air * 1e3:.6f} nm")
    print(f"integer order E  : {math.floor(block.estimate_integer_fringes())}")
    print(f"L20              : {block.block_length():.8f} mm")
