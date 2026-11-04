import numpy as np
import pytest

from interflib.dimensional.excess_fraction import GaugeBlockLength
from interflib.dimensional.mefm import MEFMSolution, MonochromaticEFMSolver


@pytest.fixture
def solver():
    return MonochromaticEFMSolver(wavelength=632.9912)


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


class TestThermalReference:
    """The prior is at 20 C, the fraction at the block temperature (Doiron & Beers 1995, sec. 6.5)."""

    CTE = 11.5e-6

    @pytest.mark.parametrize("nominal, temp_c", [(2e6, 21.0), (10e6, 20.3), (100e6, 20.5), (100e6, 19.2)])
    def test_recovers_the_length_at_20c(self, solver, nominal, temp_c):
        true_dev = -60.0
        length_at_t = (nominal + true_dev) * (1 + self.CTE * (temp_c - 20.0))
        fraction = (2 * length_at_t / solver.wavelength) % 1.0
        result = solver.solve(nominal, prior_length_dev=-50.0, prior_U=30.0, fringe_fraction=fraction,
                              cte=self.CTE, temp_c=temp_c)
        # exact up to the second-order difference between the reduction factor and 1 / expansion
        assert result.length_dev == pytest.approx(true_dev, abs=nominal * (self.CTE * (temp_c - 20.0)) ** 2 + 1e-6)

    def test_without_the_temperature_a_warm_block_fails(self, solver):
        # 100 mm at 20.2 C: the candidates move 230 nm, far outside a +-30 nm window
        fraction = (2 * (100e6 - 60.0) * (1 + self.CTE * 0.2) / solver.wavelength) % 1.0
        with pytest.raises(RuntimeError, match="Convergence failure"):
            solver.solve(100e6, -60.0, 30.0, fraction)

    @pytest.mark.parametrize("temp_c", [19.6, 20.0, 20.8])
    def test_agrees_with_the_nominal_resolver(self, solver, temp_c):
        fraction = 0.3172
        nominal = GaugeBlockLength(wavelength=solver.wavelength, nominal_length=25e6, cte=self.CTE,
                                   temp_c=temp_c, fringe_fraction=fraction).solve()
        result = solver.solve(25e6, nominal.length_dev, 20.0, fraction, cte=self.CTE, temp_c=temp_c)
        assert result.integer_order == nominal.integer_order
        assert result.length_dev == pytest.approx(nominal.length_dev, abs=1e-6)

    def test_at_20c_the_thermal_path_equals_the_plain_one(self, solver):
        plain = solver.solve(10e6, -150.0, 36.0, 0.573)
        thermal = solver.solve(10e6, -150.0, 36.0, 0.573, cte=self.CTE, temp_c=20.0)
        assert thermal == plain

    def test_cte_and_temperature_go_together(self, solver):
        with pytest.raises(ValueError, match="both"):
            solver.solve(10e6, 0.0, 30.0, 0.5, cte=self.CTE)


class TestPreprintRegression:
    """Numbers published in the MEFM preprint (Rojas-Rodriguez, Galvis Arroyave and Vinck-Posada,
    INM 2026): Tables I and IV, and the wavelength sweeps of sec. IV.C."""

    RED, GREEN = 632.9905846, 543.5

    @pytest.mark.parametrize("nominal, wavelength, prior, U, fraction, nominal_fraction, dev", [
        (10e6, RED, -150.0, 36.0, 0.6114, 0.0466, -137.735),
        (100e6, RED, -60.0, 90.0, 0.1762, 0.4659, -91.678),
        (10e6, GREEN, -150.0, 36.0, 0.0205, 0.5281, -137.929),
        (100e6, GREEN, -60.0, 90.0, 0.9425, 0.2806, -91.876),
    ])
    def test_table_iv(self, nominal, wavelength, prior, U, fraction, nominal_fraction, dev):
        result = MonochromaticEFMSolver(wavelength).solve(nominal, prior, U, fraction)
        assert result.nominal_fraction == pytest.approx(nominal_fraction, abs=5e-5)
        assert result.length_dev == pytest.approx(dev, abs=5e-4)
        assert result.length == pytest.approx(nominal + dev, abs=5e-4)

    @pytest.mark.parametrize("nominal_mm, green_pm, red_pm",
                             [(10, 7.38, 10.02), (25, 2.95, 4.01), (50, 1.48, 2.00), (100, 0.74, 1.00)])
    def test_table_i_stability_limit(self, nominal_mm, green_pm, red_pm):
        assert MonochromaticEFMSolver(self.GREEN).stability_limit(nominal_mm * 1e6) * 1e3 == pytest.approx(
            green_pm, abs=5e-3)
        assert MonochromaticEFMSolver(632.99).stability_limit(nominal_mm * 1e6) * 1e3 == pytest.approx(
            red_pm, abs=5e-3)

    @staticmethod
    def _sweep(centre, half_width, fraction):
        devs = []
        for wavelength in np.linspace(centre - half_width, centre + half_width, 7):
            try:
                devs.append(MonochromaticEFMSolver(wavelength).solve(100e6, -60.0, 90.0, fraction).length_dev)
            except RuntimeError:
                pass
        return np.array(devs)

    def test_green_sweep(self):
        devs = self._sweep(self.GREEN, MonochromaticEFMSolver(self.GREEN).stability_limit(100e6), 0.9425)
        assert devs.size == 4
        assert np.ptp(devs) == pytest.approx(135.875, abs=1e-3)

    def test_red_sweep(self):
        devs = self._sweep(self.RED, 0.060e-6, 0.1762)
        assert devs.size == 7
        assert np.ptp(devs) == pytest.approx(0.019, abs=5e-4)
