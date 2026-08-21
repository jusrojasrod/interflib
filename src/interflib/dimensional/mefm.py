"""Monochromatic (single-wavelength) excess fraction method solver.

Wiki: wiki/metodos/exceso-fraccional.md
"""
import numpy as np


class MonochromaticEFMSolver:
    """
    Solver for the Monochromatic Excess Fraction Method (MEFM).
    Calculates the absolute length of gauge blocks by resolving 
    fringe order ambiguity using mechanical preconditioning.
    """

    def __init__(self, wavelength: float):
        """
        Parameters
        ----------
        wavelength: float 
            Laser wavelength in air/vacuum in nanometers (nm).
        """
        self.wavelength = wavelength
        self.lambda_over_4 = wavelength / 4.0
        self.lambda_over_2 = wavelength / 2.0

    def evaluate_block(self, nominal_length: float, delta_nominal_length: float, 
                       uncertainty: float, measured_fraction: float) -> dict:
        """
        Evaluates an individual gauge block and returns its real absolute length.

        Parameters
        ----------
        nominal_length: float
            Nominal length of the block (e.g., 10e6 nm).
        delta_nominal_length: float
            Mechanical deviation (L'_error) in nm.
        uncertainty: float
            Expanded uncertainty of the prior mechanical measurement in nm.
        measured_fraction: float
            Measured optical excess fraction [0, 1).

        Returns:
        dict: Detailed results of the absolute calibration.
        
        Raises:
        ValueError: If mechanical uncertainty exceeds the lambda/4 uniqueness limit.
        RuntimeError: If zero or multiple valid optical solutions are found.
        """
        if uncertainty >= self.lambda_over_4:
            raise ValueError(
                f"Mechanical uncertainty ({uncertainty} nm) exceeds the strict "
                f"uniqueness limit (\u03bb/4 = {self.lambda_over_4:.2f} nm). "
                "The classic multi-wavelength method is required."
            )

        fractional_prime, _ = np.modf((2 * nominal_length) / self.wavelength)
        lower_bound = delta_nominal_length - uncertainty
        upper_bound = delta_nominal_length + uncertainty
        m_j_values = np.arange(-5, 6)
        c_values = self.lambda_over_2 * (m_j_values + measured_fraction - fractional_prime)
        valid_c = [c for c in c_values if lower_bound <= c <= upper_bound]

        if len(valid_c) == 0:
            # print(f"c_difference_nm: {c_values} - Bounds ({lower_bound}, {upper_bound})" )
            # return 0
            closest_c, min_error = self._run_diagnostics(c_values, delta_nominal_length)
            missed_by = min_error - uncertainty
            raise RuntimeError(
                f"Convergence failure: No optical value matches the mechanical measurement. "
                f"Closest c_value is {closest_c:.2f} nm. "
                f"Allowed bounds: ({lower_bound:.2f}, {upper_bound:.2f}). "
                f"Tolerance missed by {missed_by:.2f} nm."
            )
        elif len(valid_c) > 1:
            raise RuntimeError(f"Residual ambiguity: Multiple valid values found: {valid_c}")

        c_accepted = valid_c[0]
        real_length = nominal_length + c_accepted

        return {
            "real_length_nm": real_length,
            "c_difference_nm": c_accepted,
            "fractional_prime": fractional_prime,
            "acceptance_interval": (lower_bound, upper_bound)
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
        result = he_ne_laser.evaluate_block(
            nominal_length=10e6,           
            delta_nominal_length=-150.0,   
            uncertainty=36.0,              
            measured_fraction=0.573        
        )
        
        print("Calibration Results:")
        print("-" * 20)
        print(f"Real Length:        {result['real_length_nm']:.3f} nm")
        print(f"Accepted c_j value: {result['c_difference_nm']:.3f} nm")
        print(f"Acceptance bounds:  {result['acceptance_interval']}")

    except ValueError as e:
        print(f"Method Error: {e}")
    except RuntimeError as e:
        print(f"Logic Error: {e}")