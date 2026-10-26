"""Monochromatic (single-wavelength) excess fraction method solver (MEFM).

One wavelength gives only the fringe fraction; the integer order is resolved with a
prior mechanical measurement of the deviation from nominal. The candidate deviations
``c = (lambda / 2) (m + F - F')`` are spaced ``lambda / 2`` apart, so exactly one falls in
the acceptance interval ``delta +- U`` when ``U < lambda / 4``. Reference: J. S.
Rojas-Rodriguez, J. L. Galvis Arroyave and H. Vinck-Posada, "Absolute Gauge Block
Calibration: A Single-Wavelength Excess Fraction Method Assisted by High-Precision
Mechanical Pre-measurement", preprint, INM (2026), sec. III.C (uniqueness condition).
"""
from dataclasses import dataclass

import numpy as np

from interflib._deprecation import warn_deprecated


@dataclass
class MEFMSolution:
    """Absolute length resolved from one wavelength and a mechanical prior.

    Attributes
    ----------
    length : float
        Absolute length ``L = L' + d``, nm.
    length_dev : float
        Accepted deviation from nominal ``d`` (the optical candidate inside the prior), nm.
    nominal_fraction : float
        Fringe fraction ``F' = frac(2 L' / lambda)`` of the nominal length.
    acceptance_interval : tuple of float
        Prior interval ``(d_prior - U, d_prior + U)``, nm.
    """

    length: float
    length_dev: float
    nominal_fraction: float
    acceptance_interval: tuple


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

    def solve(self, nominal_length: float, prior_length_dev: float, prior_U: float,  # noqa: N803
              fringe_fraction: float) -> MEFMSolution:
        """Absolute length of one gauge block.

        Parameters
        ----------
        nominal_length : float
            Nominal length ``L'`` of the block, nm (e.g. ``10e6`` for 10 mm).
        prior_length_dev : float
            Deviation from nominal measured mechanically, nm.
        prior_U : float
            Expanded uncertainty ``U`` of that mechanical deviation, nm.
        fringe_fraction : float
            Measured fringe fraction ``F``, in ``[0, 1)``.

        Returns
        -------
        MEFMSolution

        Raises
        ------
        ValueError
            If the uncertainty reaches the uniqueness limit ``lambda / 4``.
        RuntimeError
            If no candidate, or more than one, falls in the acceptance interval.
        """
        if prior_U >= self.lambda_over_4:
            raise ValueError(
                f"Mechanical uncertainty ({prior_U} nm) exceeds the strict "
                f"uniqueness limit (λ/4 = {self.lambda_over_4:.2f} nm). "
                "The classic multi-wavelength method is required."
            )

        nominal_fraction, _ = np.modf((2 * nominal_length) / self.wavelength)
        lower_bound = prior_length_dev - prior_U
        upper_bound = prior_length_dev + prior_U
        # Candidates are spaced lambda/2 apart; centre the integer search on the order that
        # puts c closest to the mechanical deviation, so any deviation size is covered.
        m_centre = np.round(prior_length_dev / self.lambda_over_2 - fringe_fraction + nominal_fraction)
        m_j_values = m_centre + np.arange(-2, 3)
        c_values = self.lambda_over_2 * (m_j_values + fringe_fraction - nominal_fraction)
        valid_c = [c for c in c_values if lower_bound <= c <= upper_bound]

        if len(valid_c) == 0:
            closest_c, min_error = self._run_diagnostics(c_values, prior_length_dev)
            missed_by = min_error - prior_U
            raise RuntimeError(
                f"Convergence failure: No optical value matches the mechanical measurement. "
                f"Closest c_value is {closest_c:.2f} nm. "
                f"Allowed bounds: ({lower_bound:.2f}, {upper_bound:.2f}). "
                f"Tolerance missed by {missed_by:.2f} nm."
            )
        elif len(valid_c) > 1:
            raise RuntimeError(f"Residual ambiguity: Multiple valid values found: {valid_c}")

        accepted = float(valid_c[0])
        return MEFMSolution(length=nominal_length + accepted, length_dev=accepted,
                            nominal_fraction=float(nominal_fraction),
                            acceptance_interval=(lower_bound, upper_bound))

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
        print(f"Acceptance bounds:  {result.acceptance_interval}")
    except (ValueError, RuntimeError) as e:
        print(f"MEFM error: {e}")
