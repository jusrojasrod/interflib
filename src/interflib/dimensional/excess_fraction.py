"""Absolute length of a gauge block by the excess fraction method (single wavelength).

Given the wavelength in air, the nominal length and the measured fringe fraction, the
integer fringe order is resolved from the nominal length and the absolute length is
reduced to the 20 C reference temperature (T. Doiron, J. Beers, "The Gage Block
Handbook", NIST Monograph 180, 1995, section 6.5, eqs. 6.15-6.17).

The air refractive index is **not** computed here. Either pass the wavelength in air
directly, or build the object from a vacuum wavelength plus the environmental state
(:meth:`GaugeBlockLength.from_environment`), which delegates the air index to the
modified Edlen equation (Birch & Downs) in ``interflib.dimensional.refraction``. This
keeps a single air-index implementation in the package.

All lengths and wavelengths are in nanometres, as in the other solvers of
:mod:`interflib.dimensional`. The former positional signature
``(lambda_air, Lp [mm], C, t_block, obs_fraction)`` and :meth:`block_length` (mm) still
work during a transition, with a :class:`DeprecationWarning`.
"""
import math
from dataclasses import dataclass

from interflib._deprecation import warn_deprecated
from interflib.constants import REFERENCE_TEMPERATURE_C
from interflib.dimensional.refraction import EdlenRefractiveIndex

_LEGACY_ARGUMENTS = ("lambda_air", "Lp", "C", "t_block", "obs_fraction")
_LEGACY_TO_NEW = {"lambda_air": "wavelength", "Lp": "nominal_length", "C": "cte", "t_block": "temp_c",
                  "obs_fraction": "fringe_fraction", "lambda_vacuum": "wavelength_vac"}


@dataclass
class SingleWavelengthSolution:
    """Absolute length from one wavelength, with the nominal length as prior.

    Attributes
    ----------
    length : float
        Length ``L`` reduced to 20 C, nm.
    length_dev : float
        Deviation from nominal ``L - L'``, nm.
    integer_order : int
        Resolved integer order ``kappa`` (half-waves in the block at its temperature).
    nominal_fraction : float
        Fraction ``F'`` predicted from the nominal length at the block temperature.
    """

    length: float
    length_dev: float
    integer_order: int
    nominal_fraction: float


def _legacy_values(owner, args, legacy, names):
    """Map the deprecated positional/keyword arguments onto the new names (``Lp`` mm -> nm)."""
    unknown = set(legacy) - set(names)
    if unknown:
        raise TypeError(f"{owner}() got unexpected keyword arguments {sorted(unknown)}")
    if len(args) > len(names):
        raise TypeError(f"{owner}() takes at most {len(names)} positional arguments")
    values = dict(zip(names, args, strict=False))   # fewer positional arguments than names is allowed
    repeated = set(values) & set(legacy)
    if repeated:
        raise TypeError(f"{owner}() got multiple values for {sorted(repeated)}")
    values.update(legacy)
    warn_deprecated(f"{owner}({', '.join(names)}) with Lp in mm is deprecated; use the keywords "
                    f"{', '.join(_LEGACY_TO_NEW[n] for n in names)}, with nominal_length in nm",
                    stacklevel=4)
    new = {_LEGACY_TO_NEW[k]: v for k, v in values.items()}
    if "nominal_length" in new:
        new["nominal_length"] = new["nominal_length"] * 1e6          # mm -> nm
    return new


def _merge(owner, legacy_new, given):
    """Combine values from the deprecated path with the new keywords, refusing duplicates."""
    for key, value in legacy_new.items():
        if given.get(key) is not None:
            raise TypeError(f"{owner}() got {key!r} both under its old and its new name")
        given[key] = value
    missing = [k for k, v in given.items() if v is None]
    if missing:
        raise TypeError(f"{owner}() missing required arguments {missing}")
    return given


class GaugeBlockLength:
    """Excess-fraction length resolver for a single wavelength.

    Parameters
    ----------
    wavelength : float
        Wavelength in air, nm.
    nominal_length : float
        Nominal length ``L'`` of the block, nm.
    cte : float
        Linear thermal expansion coefficient ``alpha`` of the block material, 1/K.
    temp_c : float
        Block temperature, degrees Celsius.
    fringe_fraction : float
        Measured fringe fraction ``F``, in ``[0, 1)``.
    """

    def __init__(self, *args, wavelength=None, nominal_length=None, cte=None, temp_c=None,
                 fringe_fraction=None, **legacy):
        given = dict(wavelength=wavelength, nominal_length=nominal_length, cte=cte, temp_c=temp_c,
                     fringe_fraction=fringe_fraction)
        old = _legacy_values("GaugeBlockLength", args, legacy, _LEGACY_ARGUMENTS) if (args or legacy) else {}
        values = _merge("GaugeBlockLength", old, given)
        self.wavelength = values["wavelength"]
        self.nominal_length = values["nominal_length"]
        self.cte = values["cte"]
        self.temp_c = values["temp_c"]
        self.fringe_fraction = values["fringe_fraction"]

    @classmethod
    def from_environment(cls, *args, wavelength_vac=None, nominal_length=None, cte=None, temp_c=None,
                         fringe_fraction=None, temperature, pressure, relative_humidity=None,
                         dew_point=None, frost_point=None, **legacy) -> "GaugeBlockLength":
        """Build from a vacuum wavelength and the environmental state.

        The air refractive index and the wavelength in air are computed with the
        modified Edlen equation (Birch & Downs) in
        :class:`interflib.dimensional.refraction.EdlenRefractiveIndex`.

        Parameters
        ----------
        wavelength_vac : float
            Vacuum wavelength, nm.
        nominal_length, cte, temp_c, fringe_fraction
            As in :class:`GaugeBlockLength` (``temp_c`` is the block temperature).
        temperature : float
            Air temperature, degrees Celsius.
        pressure : float
            Air pressure, Pa.
        relative_humidity, dew_point, frost_point : float, optional
            Humidity state; provide exactly one (see ``EdlenRefractiveIndex``).
        """
        given = dict(wavelength_vac=wavelength_vac, nominal_length=nominal_length, cte=cte, temp_c=temp_c,
                     fringe_fraction=fringe_fraction)
        names = ("lambda_vacuum",) + _LEGACY_ARGUMENTS[1:]
        old = _legacy_values("GaugeBlockLength.from_environment", args, legacy, names) if (args or legacy) else {}
        values = _merge("GaugeBlockLength.from_environment", old, given)
        env = EdlenRefractiveIndex(temperature=temperature, pressure=pressure,
                                   relative_humidity=relative_humidity,
                                   dew_point=dew_point, frost_point=frost_point)
        wavelength_vac = values.pop("wavelength_vac")
        instance = cls(wavelength=env.wavelength_in_air(wavelength_vac), **values)
        # Kept for reporting/traceability of the environmental correction.
        instance.wavelength_vac = wavelength_vac
        instance.n_air = env.index_refraction(wavelength_vac)
        return instance

    def estimate_integer_fringes(self) -> float:
        """Half-wave count ``2 L' (1 + alpha (t - 20)) / lambda`` of the nominal length."""
        expansion = 1 + self.cte * (self.temp_c - REFERENCE_TEMPERATURE_C)
        return 2 * self.nominal_length * expansion / self.wavelength

    def solve(self) -> SingleWavelengthSolution:
        """Resolve the integer order from the nominal length and reduce the length to 20 C."""
        f_theoretical = self.estimate_integer_fringes()
        order = math.floor(f_theoretical)
        fraction_theoretical = f_theoretical - order

        # Resolve the integer order: the observed fraction may sit one fringe across
        # the theoretical one.
        if (self.fringe_fraction - fraction_theoretical) > 0.5:
            order -= 1
        elif (fraction_theoretical - self.fringe_fraction) > 0.5:
            order += 1

        reduction = 1 + self.cte * (REFERENCE_TEMPERATURE_C - self.temp_c)
        length = (self.wavelength / 2.0) * (order + self.fringe_fraction) * reduction
        return SingleWavelengthSolution(length=length, length_dev=length - self.nominal_length,
                                        integer_order=order, nominal_fraction=fraction_theoretical)

    def block_length(self) -> float:
        """Deprecated: absolute length at 20 C in millimetres; use ``solve().length`` (nm)."""
        warn_deprecated("GaugeBlockLength.block_length() (mm) is deprecated; use solve().length (nm)")
        return self.solve().length / 1e6

    # Deprecated attribute names (and units) of the former signature
    @property
    def lambda_air(self):
        warn_deprecated("GaugeBlockLength.lambda_air is deprecated; use wavelength")
        return self.wavelength

    @property
    def Lp(self):  # noqa: N802
        warn_deprecated("GaugeBlockLength.Lp (mm) is deprecated; use nominal_length (nm)")
        return self.nominal_length / 1e6

    @property
    def C(self):  # noqa: N802
        warn_deprecated("GaugeBlockLength.C is deprecated; use cte")
        return self.cte

    @property
    def t_block(self):
        warn_deprecated("GaugeBlockLength.t_block is deprecated; use temp_c")
        return self.temp_c

    @property
    def obs_fraction(self):
        warn_deprecated("GaugeBlockLength.obs_fraction is deprecated; use fringe_fraction")
        return self.fringe_fraction

    @property
    def refractive_index(self):
        warn_deprecated("GaugeBlockLength.refractive_index is deprecated; use n_air")
        return self.n_air

    @property
    def wavelength_vacuum(self):
        warn_deprecated("GaugeBlockLength.wavelength_vacuum is deprecated; use wavelength_vac")
        return self.wavelength_vac


if __name__ == "__main__":
    # Real-data path: vacuum wavelength + environmental state -> air index.
    block = GaugeBlockLength.from_environment(
        wavelength_vac=632.99139,          # nm (iodine-stabilised He-Ne)
        nominal_length=10e6, cte=11.5e-6, temp_c=20.0, fringe_fraction=0.05,
        temperature=20.0, pressure=101325.0, relative_humidity=50.0,
    )
    solution = block.solve()
    print(f"n (air)          : {block.n_air:.9f}")
    print(f"wavelength (air) : {block.wavelength:.6f} nm")
    print(f"integer order    : {solution.integer_order}")
    print(f"L20              : {solution.length:.3f} nm ({solution.length_dev:+.1f} nm from nominal)")
