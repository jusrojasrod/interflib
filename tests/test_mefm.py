import pytest

from interflib.dimensional.mefm import MEFMSolution, MonochromaticEFMSolver


class TestMonochromaticEFMSolver:

    @pytest.fixture
    def solver(self):
        return MonochromaticEFMSolver(wavelength=632.9912)

    def test_solve_success(self, solver):
        """Tests a standard successful resolution of fringe ambiguity."""
        result = solver.solve(
            nominal_length=10e6,           # 10 mm in nm
            prior_length_dev=-150.0,       # -150 nm mechanical deviation
            prior_U=36.0,                  # +/- 36 nm expanded uncertainty
            fringe_fraction=0.573,         # optical fraction
        )
        assert isinstance(result, MEFMSolution)
        # The accepted deviation must lie within the acceptance interval
        lower, upper = result.acceptance_interval
        assert lower <= result.length_dev <= upper
        # Based on sample data, the expected deviation is approx -140.16602119940111 nm
        assert result.length_dev == pytest.approx(-140.16602119940111, abs=1e-9)
        assert result.length == pytest.approx(10e6 + result.length_dev, abs=1e-9)

    def test_uncertainty_limit_error(self, solver):
        """Ensures the solver raises an error if mechanical uncertainty is too high."""
        # lambda/4 is approx 158.25 nm for HeNe
        with pytest.raises(ValueError, match="exceeds the strict uniqueness limit"):
            solver.solve(10e6, 0.0, 160.0, 0.5)

    def test_convergence_failure(self, solver):
        """Ensures the solver identifies when no optical solution fits the mechanical bounds."""
        with pytest.raises(RuntimeError, match="Convergence failure"):
            solver.solve(nominal_length=10e6, prior_length_dev=0.0, prior_U=5.0, fringe_fraction=0.9)

    @pytest.mark.parametrize("true_deviation", [2000.0, -2000.0, 15000.0])
    def test_large_mechanical_deviation(self, solver, true_deviation):
        """Recovers the length when the deviation spans many lambda/2 orders."""
        nominal = 100e6                              # 100 mm in nm
        true_length = nominal + true_deviation
        fraction = (2 * true_length / solver.wavelength) % 1.0
        result = solver.solve(
            nominal_length=nominal,
            prior_length_dev=true_deviation + 10.0,  # prior off by 10 nm
            prior_U=30.0,
            fringe_fraction=fraction,
        )
        assert result.length_dev == pytest.approx(true_deviation, abs=1e-6)

    def test_evaluate_block_is_deprecated_and_keeps_its_dictionary(self, solver):
        with pytest.warns(DeprecationWarning, match="solve"):
            result = solver.evaluate_block(nominal_length=10e6, delta_nominal_length=-150.0,
                                           uncertainty=36.0, measured_fraction=0.573)
        assert set(result) == {"real_length_nm", "c_difference_nm", "fractional_prime", "acceptance_interval"}
        assert result["c_difference_nm"] == pytest.approx(-140.16602119940111, abs=1e-9)

    def test_wavelength_properties(self, solver):
        assert solver.lambda_over_2 == 632.9912 / 2.0
        assert solver.lambda_over_4 == 632.9912 / 4.0
