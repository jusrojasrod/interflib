"""Monochromatic (single-wavelength) excess fraction method solver (MEFM).

One wavelength gives only the fringe fraction; the integer order is resolved with a
prior mechanical measurement of the deviation from nominal. The candidate deviations
``c = (lambda / 2) (m + F - F')`` are spaced ``lambda / 2`` apart, so exactly one falls in
the acceptance interval ``delta +- U`` when ``U < lambda / 4``. Reference: J. S.
Rojas-Rodriguez, J. L. Galvis Arroyave and H. Vinck-Posada, "Absolute Gauge Block
Calibration: A Single-Wavelength Excess Fraction Method Assisted by High-Precision
Mechanical Pre-measurement", preprint, INM (2026), sec. III.B-C (formulation and
uniqueness condition) and sec. III.D (wavelength stability limit).

Thermal reference. The fraction is measured on the block at its temperature ``t``, while
a calibration certificate gives the nominal length and the prior deviation at 20 C. When
the expansion coefficient and the block temperature are given, the candidates are formed
from the nominal length expanded to ``t`` and each one is reduced to 20 C before it is
tested against the prior (T. Doiron, J. Beers, "The Gage Block Handbook", NIST Monograph
180, 1995, sec. 6.5, eqs. 6.15-6.17, the same expansion and reduction factors as
:class:`interflib.dimensional.excess_fraction.GaugeBlockLength`). Without them every
length is taken at the temperature of the measurement, which is only right at 20 C: for a
steel block (``alpha = 11.5e-6 / K``) the candidates move ``alpha L' (t - 20)``, 115 nm per
0.1 K on a 100 mm block, more than the acceptance window of a typical certificate.
"""
import math
from dataclasses import dataclass
from typing import Optional

import numpy as np

from interflib._deprecation import warn_deprecated
from interflib.constants import REFERENCE_TEMPERATURE_C


@dataclass
class MEFMSolution:
    """Absolute length resolved from one wavelength and a mechanical prior.

    Attributes
    ----------
    length : float
        Absolute length ``L = L' + d``, nm; reduced to 20 C when the solver was given the
        block temperature, otherwise at the temperature of the measurement.
    length_dev : float
        Accepted deviation from nominal ``d`` (the optical candidate inside the prior), nm,
        in the same thermal reference as ``length``.
    nominal_fraction : float
        Fringe fraction ``F' = frac(2 L' (1 + alpha (t - 20)) / lambda)`` of the nominal
        length at the block temperature (``frac(2 L' / lambda)`` without it).
    acceptance_interval : tuple of float
        Prior interval ``(d_prior - U, d_prior + U)``, nm.
    integer_order : int
        Resolved integer order ``kappa`` (half-waves in the block at its temperature).
    """

    length: float
    length_dev: float
    nominal_fraction: float
    acceptance_interval: tuple
    integer_order: int


class MonochromaticEFMSolver:
    """Absolute gauge-block length from one wavelength and a mechanical prior.

    Parameters
    ----------
    wavelength : float
        Laser wavelength in air (the medium of the measurement), nanometres.
    """

    def __init__(self, wavelength: float):
        self.wavelength = wavelength
        self.lambda_over_4 = wavelength / 4.0
        self.lambda_over_2 = wavelength / 2.0

    def stability_limit(self, nominal_length: float) -> float:
        """Largest wavelength instability ``lambda^2 / (4 L')`` that keeps the order, nm.

        Half the wavelength change that moves the nominal half-wave count ``2 L' / lambda``
        by one fringe (preprint sec. III.D). ``nominal_length`` in nm.
        """
        return self.wavelength ** 2 / (4.0 * nominal_length)

    def solve(self, nominal_length: float, prior_length_dev: float, prior_U: float,  # noqa: N803
              fringe_fraction: float, cte: Optional[float] = None,
              temp_c: Optional[float] = None) -> MEFMSolution:
        """Absolute length of one gauge block.

        Parameters
        ----------
        nominal_length : float
            Nominal length ``L'`` of the block, nm (e.g. ``10e6`` for 10 mm).
        prior_length_dev : float
            Deviation from nominal measured mechanically, nm (at 20 C when ``cte`` and
            ``temp_c`` are given).
        prior_U : float
            Expanded uncertainty ``U`` of that mechanical deviation, nm.
        fringe_fraction : float
            Measured fringe fraction ``F``, in ``[0, 1)``.
        cte : float, optional
            Linear thermal expansion coefficient ``alpha`` of the block, 1/K.
        temp_c : float, optional
            Block temperature during the optical measurement, degrees Celsius. Give both
            ``cte`` and ``temp_c`` or neither.

        Returns
        -------
        MEFMSolution

        Raises
        ------
        ValueError
            If only one of ``cte`` and ``temp_c`` is given, or if the uncertainty reaches
            the uniqueness limit, a quarter of the candidate spacing (``lambda / 4``).
        RuntimeError
            If no candidate, or more than one, falls in the acceptance interval.
        """
        if (cte is None) != (temp_c is None):
            raise ValueError("give both 'cte' and 'temp_c' (thermal reference) or neither")
        if cte is None:
            expansion = reduction = 1.0
        else:
            expansion = 1 + cte * (temp_c - REFERENCE_TEMPERATURE_C)
            reduction = 1 + cte * (REFERENCE_TEMPERATURE_C - temp_c)
        # Candidates reduced to 20 C are spaced (lambda / 2) * reduction apart.
        uniqueness_limit = self.lambda_over_4 * reduction
        if prior_U >= uniqueness_limit:
            raise ValueError(
                f"Mechanical uncertainty ({prior_U} nm) exceeds the strict "
                f"uniqueness limit (λ/4 = {uniqueness_limit:.2f} nm). "
                "The classic multi-wavelength method is required."
            )

        half_waves = (2 * nominal_length * expansion) / self.wavelength
        nominal_order = math.floor(half_waves)
        nominal_fraction = half_waves - nominal_order
        lower_bound = prior_length_dev - prior_U
        upper_bound = prior_length_dev + prior_U
        # A candidate c_t (deviation from the expanded nominal, at t) reduces to 20 C as
        # c = offset + c_t * reduction, with offset = L' (expansion * reduction - 1), a
        # second-order term (-L' alpha^2 (t - 20)^2). Both are exact zeros without cte.
        offset = nominal_length * (expansion * reduction - 1)
        target = (prior_length_dev - offset) / reduction
        # Candidates are spaced lambda/2 apart; centre the integer search on the order that
        # puts c closest to the mechanical deviation, so any deviation size is covered.
        m_centre = np.round(target / self.lambda_over_2 - fringe_fraction + nominal_fraction)
        m_j_values = m_centre + np.arange(-2, 3)
        c_values = offset + reduction * self.lambda_over_2 * (m_j_values + fringe_fraction - nominal_fraction)
        valid = [(c, m) for c, m in zip(c_values, m_j_values, strict=True) if lower_bound <= c <= upper_bound]

        if len(valid) == 0:
            closest_c, min_error = self._run_diagnostics(c_values, prior_length_dev)
            missed_by = min_error - prior_U
            raise RuntimeError(
                f"Convergence failure: No optical value matches the mechanical measurement. "
                f"Closest c_value is {closest_c:.2f} nm. "
                f"Allowed bounds: ({lower_bound:.2f}, {upper_bound:.2f}). "
                f"Tolerance missed by {missed_by:.2f} nm."
            )
        elif len(valid) > 1:
            raise RuntimeError(f"Residual ambiguity: Multiple valid values found: {[c for c, _ in valid]}")

        accepted, m = float(valid[0][0]), int(valid[0][1])
        return MEFMSolution(length=nominal_length + accepted, length_dev=accepted,
                            nominal_fraction=float(nominal_fraction),
                            acceptance_interval=(lower_bound, upper_bound),
                            integer_order=nominal_order + m)

    def evaluate_block(self, nominal_length: float, delta_nominal_length: float,
                       uncertainty: float, measured_fraction: float) -> dict:
        """Deprecated: use :meth:`solve`, which returns a :class:`MEFMSolution`.

        Returns the former dictionary: ``real_length_nm``, ``c_difference_nm``,
        ``fractional_prime`` and ``acceptance_interval``.
        """
        warn_deprecated("MonochromaticEFMSolver.evaluate_block() is deprecated; use solve(nominal_length, "
                        "prior_length_dev, prior_U, fringe_fraction), which returns an MEFMSolution")
        result = self.solve(nominal_length, delta_nominal_length, uncertainty, measured_fraction)
        return {
            "real_length_nm": result.length,
            "c_difference_nm": result.length_dev,
            "fractional_prime": result.nominal_fraction,
            "acceptance_interval": result.acceptance_interval,
        }

    def _run_diagnostics(self, c_values: np.ndarray, delta_nominal_length: float) -> tuple:
        """
        Analyzes the deviation between the calculated optical differences
        and the mechanical deviation when convergence fails.

        Parameters
        ----------
        c_values: np.ndarray
            Array of calculated optical length differences in nm.
        delta_nominal_length: float
            Mechanical deviation in nm.

        Returns
        -------
        tuple
            Closest optical difference and the minimum absolute error in nm.
        """
        differences = np.abs(c_values - delta_nominal_length)
        min_index = np.argmin(differences)

        closest_c = c_values[min_index]
        min_error = differences[min_index]

        return closest_c, min_error


if __name__ == "__main__":
    he_ne_laser = MonochromaticEFMSolver(wavelength=632.9912)

    # Evaluate the 10 mm block with the mechanical and optical data
    try:
        result = he_ne_laser.solve(nominal_length=10e6, prior_length_dev=-150.0, prior_U=36.0,
                                   fringe_fraction=0.573)
        print("Calibration Results:")
        print("-" * 20)
        print(f"Length:             {result.length:.3f} nm")
        print(f"Deviation accepted: {result.length_dev:.3f} nm")
        print(f"Integer order:      {result.integer_order}")
        print(f"Acceptance bounds:  {result.acceptance_interval}")
    except (ValueError, RuntimeError) as e:
        print(f"MEFM error: {e}")
