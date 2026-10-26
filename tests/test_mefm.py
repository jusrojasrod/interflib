import pytest
from interflib.dimensional.mefm import MonochromaticEFMSolver

class TestMonochromaticEFMSolver:
    
    @pytest.fixture
    def solver(self):
        return MonochromaticEFMSolver(wavelength=632.9912)

    def test_evaluate_block_success(self, solver):
        """Tests a standard successful resolution of fringe ambiguity."""
        result = solver.evaluate_block(
            nominal_length=10e6,           # 10 mm in nm
            delta_nominal_length=-150.0,   # -150 nm mechanical error
            uncertainty=36.0,              # +/- 36 nm uncertainty
            measured_fraction=0.573        # Optical fraction
        )
        
        assert "real_length_nm" in result
        assert "c_difference_nm" in result
        # The accepted c value must be within the uncertainty bounds
        lower, upper = result["acceptance_interval"]
        assert lower <= result["c_difference_nm"] <= upper
        # Based on sample data, expected c_difference is approx -140.16602119940111 nm
        assert result["c_difference_nm"] == pytest.approx(-140.16602119940111, abs=1e-14)

    def test_uncertainty_limit_error(self, solver):
        """Ensures the solver raises an error if mechanical uncertainty is too high."""
        # lambda/4 is approx 158.25 nm for HeNe
        high_uncertainty = 160.0
        with pytest.raises(ValueError, match="exceeds the strict uniqueness limit"):
            solver.evaluate_block(10e6, 0.0, high_uncertainty, 0.5)

    def test_convergence_failure(self, solver):
        """Ensures the solver identifies when no optical solution fits the mechanical bounds."""
        # Using a very small uncertainty and a fraction that is physically impossible 
        # to reconcile with the mechanical offset.
        with pytest.raises(RuntimeError, match="Convergence failure"):
            solver.evaluate_block(
                nominal_length=10e6, 
                delta_nominal_length=0.0, 
                uncertainty=5.0, 
                measured_fraction=0.9 # This will likely be outside +/- 5nm of nominal
            )

    @pytest.mark.parametrize("true_deviation", [2000.0, -2000.0, 15000.0])
    def test_large_mechanical_deviation(self, solver, true_deviation):
        """Recovers the length when the deviation spans many lambda/2 orders."""
        nominal = 100e6                              # 100 mm in nm
        true_length = nominal + true_deviation
        fraction = (2 * true_length / solver.wavelength) % 1.0
        result = solver.evaluate_block(
            nominal_length=nominal,
            delta_nominal_length=true_deviation + 10.0,   # prior off by 10 nm
            uncertainty=30.0,
            measured_fraction=fraction,
        )
        assert result["c_difference_nm"] == pytest.approx(true_deviation, abs=1e-6)

    def test_wavelength_properties(self, solver):
        assert solver.lambda_over_2 == 632.9912 / 2.0
        assert solver.lambda_over_4 == 632.9912 / 4.0