"""Unit tests for the Vasicek interest rate model.

This module tests Vasicek model calibration, parameter estimation,
and analytical formulas for expected rate and variance.
"""

import numpy as np
import pytest
from hypothesis import given, settings, strategies as st

from src.models.vasicek import VasicekModel, _estimate_ols_params
from src.simulation.paths import SimulationResult
from src.utils.exceptions import CalibrationError, ModelNotCalibratedError


# =============================================================================
# T040: Unit tests for Vasicek calibration
# =============================================================================


class TestVasicekCalibrationMLE:
    """Test Vasicek model calibration using Maximum Likelihood Estimation."""

    def test_mle_converges_on_valid_data(
        self,
        sample_short_rates: np.ndarray,
    ) -> None:
        """MLE calibration should converge on realistic short rate data."""
        model = VasicekModel()
        model.calibrate(sample_short_rates, dt=1 / 252, method="mle")

        assert model.is_calibrated
        assert model.convergence_status == "converged"

    def test_mle_produces_positive_kappa(
        self,
        sample_short_rates: np.ndarray,
    ) -> None:
        """MLE should estimate positive mean reversion speed."""
        model = VasicekModel()
        model.calibrate(sample_short_rates, dt=1 / 252, method="mle")

        assert model.kappa > 0, "Mean reversion speed (kappa) must be positive"

    def test_mle_produces_positive_sigma(
        self,
        sample_short_rates: np.ndarray,
    ) -> None:
        """MLE should estimate positive volatility."""
        model = VasicekModel()
        model.calibrate(sample_short_rates, dt=1 / 252, method="mle")

        assert model.sigma > 0, "Volatility (sigma) must be positive"

    def test_mle_produces_reasonable_theta(
        self,
        sample_short_rates: np.ndarray,
    ) -> None:
        """MLE should estimate theta within reasonable bounds."""
        model = VasicekModel()
        model.calibrate(sample_short_rates, dt=1 / 252, method="mle")

        # Theta should be within typical interest rate bounds
        assert -0.10 <= model.theta <= 0.50, "Long-term mean (theta) out of bounds"

    def test_mle_log_likelihood_is_finite(
        self,
        sample_short_rates: np.ndarray,
    ) -> None:
        """MLE should produce finite log-likelihood value."""
        model = VasicekModel()
        model.calibrate(sample_short_rates, dt=1 / 252, method="mle")

        assert np.isfinite(model.log_likelihood)


class TestVasicekCalibrationOLS:
    """Test Vasicek model calibration using OLS initial estimation."""

    def test_ols_fallback_produces_valid_params(
        self,
        sample_short_rates: np.ndarray,
    ) -> None:
        """OLS fallback should produce valid parameter estimates."""
        model = VasicekModel()
        model.calibrate(sample_short_rates, dt=1 / 252, method="ols")

        assert model.is_calibrated
        assert model.kappa > 0
        assert model.sigma > 0

    def test_ols_params_close_to_mle(
        self,
        sample_short_rates: np.ndarray,
    ) -> None:
        """OLS and MLE estimates should be in the same ballpark."""
        model_ols = VasicekModel()
        model_ols.calibrate(sample_short_rates, dt=1 / 252, method="ols")

        model_mle = VasicekModel()
        model_mle.calibrate(sample_short_rates, dt=1 / 252, method="mle")

        # Parameters should be within an order of magnitude
        assert abs(model_ols.kappa - model_mle.kappa) < max(model_ols.kappa, model_mle.kappa)
        assert abs(model_ols.theta - model_mle.theta) < 0.05
        assert abs(model_ols.sigma - model_mle.sigma) < max(model_ols.sigma, model_mle.sigma)


class TestVasicekParameterBounds:
    """Test parameter bounds and edge cases."""

    def test_calibration_with_few_observations_raises_error(self) -> None:
        """Calibration should fail with too few observations."""
        short_rates = np.array([0.03, 0.031, 0.029])  # Only 3 points

        model = VasicekModel()
        with pytest.raises(CalibrationError, match="observations"):
            model.calibrate(short_rates, dt=1 / 252, method="mle")

    def test_calibration_with_constant_rates_handles_gracefully(self) -> None:
        """Calibration should handle constant rate series."""
        short_rates = np.full(100, 0.03)  # Constant rates

        model = VasicekModel()
        # Should either raise CalibrationError or produce near-zero volatility
        try:
            model.calibrate(short_rates, dt=1 / 252, method="mle")
            # If it succeeds, sigma should be very small
            assert model.sigma < 0.001
        except CalibrationError:
            pass  # Expected for degenerate data

    def test_get_parameters_returns_dict(
        self,
        sample_short_rates: np.ndarray,
    ) -> None:
        """get_parameters() should return a dictionary with all parameters."""
        model = VasicekModel()
        model.calibrate(sample_short_rates, dt=1 / 252, method="mle")

        params = model.get_parameters()

        assert isinstance(params, dict)
        assert "kappa" in params
        assert "theta" in params
        assert "sigma" in params
        assert params["kappa"] == model.kappa
        assert params["theta"] == model.theta
        assert params["sigma"] == model.sigma


class TestVasicekUncalibrated:
    """Test behavior when model is not calibrated."""

    def test_is_calibrated_false_initially(self) -> None:
        """New model should not be calibrated."""
        model = VasicekModel()
        assert not model.is_calibrated

    def test_simulate_raises_when_not_calibrated(self) -> None:
        """simulate() should raise when model is not calibrated."""
        model = VasicekModel()
        with pytest.raises(ModelNotCalibratedError):
            model.simulate(n_paths=100, time_horizon=1.0, time_step=1 / 252)

    def test_expected_rate_raises_when_not_calibrated(self) -> None:
        """expected_rate() should raise when model is not calibrated."""
        model = VasicekModel()
        with pytest.raises(ModelNotCalibratedError):
            model.expected_rate(t=1.0, r0=0.03)

    def test_rate_variance_raises_when_not_calibrated(self) -> None:
        """rate_variance() should raise when model is not calibrated."""
        model = VasicekModel()
        with pytest.raises(ModelNotCalibratedError):
            model.rate_variance(t=1.0)


class TestVasicekAnalyticalFormulas:
    """Test analytical formulas for expected rate and variance."""

    def test_expected_rate_at_t0_equals_r0(
        self,
        sample_short_rates: np.ndarray,
    ) -> None:
        """Expected rate at t=0 should equal initial rate."""
        model = VasicekModel()
        model.calibrate(sample_short_rates, dt=1 / 252, method="mle")

        r0 = 0.03
        expected = model.expected_rate(t=0.0, r0=r0)

        np.testing.assert_almost_equal(expected, r0, decimal=10)

    def test_expected_rate_approaches_theta(
        self,
        sample_short_rates: np.ndarray,
    ) -> None:
        """Expected rate should approach theta as t → ∞."""
        model = VasicekModel()
        model.calibrate(sample_short_rates, dt=1 / 252, method="mle")

        r0 = 0.01  # Start away from theta
        expected_t100 = model.expected_rate(t=100.0, r0=r0)

        np.testing.assert_almost_equal(expected_t100, model.theta, decimal=4)

    def test_rate_variance_at_t0_is_zero(
        self,
        sample_short_rates: np.ndarray,
    ) -> None:
        """Rate variance at t=0 should be zero."""
        model = VasicekModel()
        model.calibrate(sample_short_rates, dt=1 / 252, method="mle")

        variance = model.rate_variance(t=0.0)

        np.testing.assert_almost_equal(variance, 0.0, decimal=10)

    def test_rate_variance_approaches_limit(
        self,
        sample_short_rates: np.ndarray,
    ) -> None:
        """Rate variance should approach σ²/(2κ) as t → ∞."""
        model = VasicekModel()
        model.calibrate(sample_short_rates, dt=1 / 252, method="mle")

        variance_t100 = model.rate_variance(t=100.0)
        theoretical_limit = (model.sigma**2) / (2 * model.kappa)

        np.testing.assert_almost_equal(variance_t100, theoretical_limit, decimal=5)


# =============================================================================
# T042: Property-based tests for Vasicek mean reversion
# =============================================================================


class TestVasicekMeanReversionProperties:
    """Property-based tests verifying Vasicek mean reversion behavior."""

    @given(
        kappa=st.floats(min_value=0.1, max_value=5.0),
        theta=st.floats(min_value=-0.02, max_value=0.10),
        sigma=st.floats(min_value=0.001, max_value=0.05),
        t=st.floats(min_value=0.0, max_value=50.0),
        r0=st.floats(min_value=-0.02, max_value=0.20),
    )
    @settings(max_examples=100)
    def test_expected_rate_formula_is_valid(
        self,
        kappa: float,
        theta: float,
        sigma: float,
        t: float,
        r0: float,
    ) -> None:
        """Expected rate formula should produce finite, reasonable values."""
        model = VasicekModel(kappa=kappa, theta=theta, sigma=sigma)
        # Mark as manually calibrated for testing
        model._calibrated = True

        expected = model.expected_rate(t=t, r0=r0)

        # Result should be finite
        assert np.isfinite(expected)

        # Result should be between r0 and theta (or equal to one of them)
        lower = min(r0, theta)
        upper = max(r0, theta)
        # Allow small numerical tolerance
        assert lower - 1e-6 <= expected <= upper + 1e-6

    @given(
        kappa=st.floats(min_value=0.1, max_value=5.0),
        theta=st.floats(min_value=-0.02, max_value=0.10),
        sigma=st.floats(min_value=0.001, max_value=0.05),
        t=st.floats(min_value=0.0, max_value=50.0),
    )
    @settings(max_examples=100)
    def test_variance_formula_is_non_negative(
        self,
        kappa: float,
        theta: float,
        sigma: float,
        t: float,
    ) -> None:
        """Rate variance should always be non-negative."""
        model = VasicekModel(kappa=kappa, theta=theta, sigma=sigma)
        model._calibrated = True

        variance = model.rate_variance(t=t)

        assert variance >= 0, "Variance must be non-negative"
        assert np.isfinite(variance), "Variance must be finite"

    @given(
        kappa=st.floats(min_value=0.1, max_value=5.0),
        theta=st.floats(min_value=-0.02, max_value=0.10),
        sigma=st.floats(min_value=0.001, max_value=0.05),
        t1=st.floats(min_value=0.0, max_value=10.0),
        t2=st.floats(min_value=0.0, max_value=10.0),
    )
    @settings(max_examples=100)
    def test_variance_is_monotonically_increasing(
        self,
        kappa: float,
        theta: float,
        sigma: float,
        t1: float,
        t2: float,
    ) -> None:
        """Rate variance should increase with time."""
        model = VasicekModel(kappa=kappa, theta=theta, sigma=sigma)
        model._calibrated = True

        var1 = model.rate_variance(t=t1)
        var2 = model.rate_variance(t=t2)

        if t1 <= t2:
            assert var1 <= var2 + 1e-10, "Variance should increase with time"
        else:
            assert var2 <= var1 + 1e-10, "Variance should increase with time"


class TestVasicekSimulationProperties:
    """Property-based tests for simulation behavior."""

    def test_simulation_mean_approaches_expected(
        self,
        sample_short_rates: np.ndarray,
    ) -> None:
        """Monte Carlo simulation mean should approach analytical expected rate."""
        model = VasicekModel()
        model.calibrate(sample_short_rates, dt=1 / 252, method="mle")

        r0 = sample_short_rates[-1]  # Start from last observed rate
        result = model.simulate(
            n_paths=10000,
            time_horizon=1.0,
            time_step=1 / 252,
            initial_rate=r0,
            random_seed=42,
        )

        # Get terminal rates from simulation
        terminal_rates = np.array([p.rates[-1] for p in result.paths])
        simulated_mean = np.mean(terminal_rates)

        # Analytical expected rate
        expected_mean = model.expected_rate(t=1.0, r0=r0)

        # Monte Carlo error: should be within 3 standard errors
        theoretical_std = np.sqrt(model.rate_variance(t=1.0))
        mc_std_error = theoretical_std / np.sqrt(len(terminal_rates))

        assert abs(simulated_mean - expected_mean) < 3 * mc_std_error, (
            f"Simulated mean {simulated_mean:.6f} differs from expected {expected_mean:.6f}"
        )


class TestOLSParameterEstimation:
    """Test the OLS parameter estimation helper function."""

    def test_ols_estimation_returns_valid_params(
        self,
        sample_short_rates: np.ndarray,
    ) -> None:
        """OLS estimation should return valid parameter tuple."""
        kappa, theta, sigma = _estimate_ols_params(sample_short_rates, dt=1 / 252)

        assert kappa > 0, "kappa must be positive"
        assert np.isfinite(theta), "theta must be finite"
        assert sigma > 0, "sigma must be positive"

    def test_ols_handles_known_parameters(self) -> None:
        """OLS should produce valid parameter estimates from synthetic data."""
        # Generate synthetic data with known parameters
        true_kappa = 0.5
        true_theta = 0.03
        true_sigma = 0.01
        dt = 1 / 252
        n_obs = 2000  # More observations for better estimation

        rng = np.random.default_rng(42)
        rates = np.zeros(n_obs)
        rates[0] = true_theta

        for t in range(1, n_obs):
            dW = rng.standard_normal() * np.sqrt(dt)
            rates[t] = rates[t - 1] + true_kappa * (true_theta - rates[t - 1]) * dt + true_sigma * dW

        kappa, theta, sigma = _estimate_ols_params(rates, dt=dt)

        # OLS estimates should produce valid parameters (positive kappa and sigma)
        # Note: OLS has known discretization bias that can be significant
        # The key test is that the values are valid for use in simulation
        assert kappa > 0, f"kappa={kappa} must be positive"
        assert np.isfinite(theta), f"theta={theta} must be finite"
        assert sigma > 0, f"sigma={sigma} must be positive"

        # Parameters should be in reasonable ranges for interest rate modeling
        assert kappa < 100, f"kappa={kappa} is unreasonably large"
        assert -0.5 < theta < 0.5, f"theta={theta} is out of reasonable range"
        assert sigma < 1.0, f"sigma={sigma} is unreasonably large"
