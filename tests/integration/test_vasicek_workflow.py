"""Integration tests for the complete Vasicek model workflow.

This module tests the end-to-end workflow:
load data → calibrate → simulate → visualize
"""

import tempfile
from pathlib import Path

import matplotlib
import numpy as np
import pytest

matplotlib.use("Agg")

from src.data.loader import InterestRateDataset, load_interest_rate_data
from src.models.vasicek import VasicekModel
from src.simulation.paths import SimulationResult
from src.visualization.simulation_plots import (
    plot_simulation_paths,
    plot_terminal_distribution,
)


# =============================================================================
# T043: Integration test for Vasicek workflow
# =============================================================================


class TestVasicekWorkflowFromCSV:
    """Test complete workflow starting from CSV file."""

    def test_full_workflow_from_sample_data(
        self,
        sample_observation_dates: list,
        sample_maturities: np.ndarray,
        sample_rate_matrix: np.ndarray,
    ) -> None:
        """Test: load data → calibrate → simulate → validate results."""
        # Create temporary CSV file
        with tempfile.NamedTemporaryFile(
            mode="w",
            suffix=".csv",
            delete=False,
        ) as f:
            # Write header with maturity columns
            maturity_cols = ["3M", "6M", "1Y", "2Y", "5Y", "10Y", "30Y"]
            f.write("date," + ",".join(maturity_cols) + "\n")

            # Write data rows
            for i, obs_date in enumerate(sample_observation_dates):
                rates = ",".join(f"{r:.6f}" for r in sample_rate_matrix[i])
                f.write(f"{obs_date.isoformat()},{rates}\n")

            csv_path = f.name

        try:
            # Step 1: Load data
            dataset = load_interest_rate_data(
                Path(csv_path),
                date_column="date",
            )

            assert isinstance(dataset, InterestRateDataset)
            assert len(dataset.observation_dates) == len(sample_observation_dates)
            assert len(dataset.maturities) == 7

            # Step 2: Calibrate Vasicek model
            model = VasicekModel()
            model.calibrate(dataset, dt=1 / 252, method="mle")

            assert model.is_calibrated
            assert model.kappa > 0
            assert model.sigma > 0
            assert model.convergence_status == "converged"

            # Step 3: Simulate paths
            initial_rate = dataset.rates[-1, 0]  # Last observed short rate
            result = model.simulate(
                n_paths=1000,
                time_horizon=1.0,
                time_step=1 / 252,
                initial_rate=initial_rate,
                random_seed=42,
            )

            assert isinstance(result, SimulationResult)
            assert result.path_count == 1000
            assert result.model_type == "vasicek"

            # Step 4: Validate simulation statistics
            terminal_mean = result.terminal_mean()
            terminal_std = result.terminal_std()

            # Mean should be between initial rate and theta
            lower = min(initial_rate, model.theta)
            upper = max(initial_rate, model.theta)
            assert lower - 0.02 <= terminal_mean <= upper + 0.02

            # Std should be positive and reasonable
            assert terminal_std > 0
            assert terminal_std < 0.10  # Less than 10%

            # Step 5: Generate visualizations (verify no errors)
            fig1 = plot_simulation_paths(result, n_paths=50)
            assert fig1 is not None

            fig2 = plot_terminal_distribution(result)
            assert fig2 is not None

        finally:
            Path(csv_path).unlink()  # Clean up temp file


class TestVasicekWorkflowFromArray:
    """Test complete workflow starting from NumPy array."""

    def test_calibrate_simulate_from_array(
        self,
        sample_short_rates: np.ndarray,
    ) -> None:
        """Test workflow using raw short rate array."""
        # Step 1: Calibrate directly from array
        model = VasicekModel()
        model.calibrate(sample_short_rates, dt=1 / 252, method="mle")

        assert model.is_calibrated

        # Step 2: Simulate
        result = model.simulate(
            n_paths=500,
            time_horizon=0.5,
            time_step=1 / 252,
            initial_rate=sample_short_rates[-1],
            random_seed=123,
        )

        assert result.path_count == 500

        # Step 3: Verify path properties
        rates_matrix = result.get_rates_matrix()

        # All paths should start at initial rate
        np.testing.assert_array_almost_equal(
            rates_matrix[:, 0],
            np.full(500, sample_short_rates[-1]),
        )

        # No NaN or Inf values
        assert not np.any(np.isnan(rates_matrix))
        assert not np.any(np.isinf(rates_matrix))


class TestVasicekReproducibility:
    """Test simulation reproducibility with random seeds."""

    def test_same_seed_produces_identical_results(
        self,
        sample_short_rates: np.ndarray,
    ) -> None:
        """Same random seed should produce identical simulation paths."""
        model = VasicekModel()
        model.calibrate(sample_short_rates, dt=1 / 252, method="mle")

        result1 = model.simulate(
            n_paths=100,
            time_horizon=1.0,
            time_step=1 / 252,
            random_seed=42,
        )

        result2 = model.simulate(
            n_paths=100,
            time_horizon=1.0,
            time_step=1 / 252,
            random_seed=42,
        )

        np.testing.assert_array_equal(
            result1.get_rates_matrix(),
            result2.get_rates_matrix(),
        )

    def test_different_seeds_produce_different_results(
        self,
        sample_short_rates: np.ndarray,
    ) -> None:
        """Different random seeds should produce different paths."""
        model = VasicekModel()
        model.calibrate(sample_short_rates, dt=1 / 252, method="mle")

        result1 = model.simulate(
            n_paths=100,
            time_horizon=1.0,
            time_step=1 / 252,
            random_seed=42,
        )

        result2 = model.simulate(
            n_paths=100,
            time_horizon=1.0,
            time_step=1 / 252,
            random_seed=99,
        )

        assert not np.allclose(
            result1.get_rates_matrix(),
            result2.get_rates_matrix(),
        )


class TestVasicekCalibrationMethods:
    """Test different calibration methods produce consistent results."""

    def test_ols_vs_mle_consistency(
        self,
        sample_short_rates: np.ndarray,
    ) -> None:
        """OLS and MLE should produce similar (not identical) parameters."""
        model_ols = VasicekModel()
        model_ols.calibrate(sample_short_rates, dt=1 / 252, method="ols")

        model_mle = VasicekModel()
        model_mle.calibrate(sample_short_rates, dt=1 / 252, method="mle")

        # Both should calibrate successfully
        assert model_ols.is_calibrated
        assert model_mle.is_calibrated

        # Parameters should be in similar range (within order of magnitude)
        assert 0.1 * model_mle.kappa < model_ols.kappa < 10 * model_mle.kappa
        assert abs(model_ols.theta - model_mle.theta) < 0.05
        assert 0.1 * model_mle.sigma < model_ols.sigma < 10 * model_mle.sigma


class TestVasicekParameterInterpretations:
    """Test parameter interpretation functionality."""

    def test_interpretations_are_informative(
        self,
        sample_short_rates: np.ndarray,
    ) -> None:
        """Parameter interpretations should be informative strings."""
        model = VasicekModel()
        model.calibrate(sample_short_rates, dt=1 / 252, method="mle")

        interpretations = model.get_parameter_interpretations()

        assert "kappa" in interpretations
        assert "theta" in interpretations
        assert "sigma" in interpretations

        # Each interpretation should be a non-empty string
        for key, text in interpretations.items():
            assert isinstance(text, str)
            assert len(text) > 20, f"Interpretation for {key} is too short"
            # Text should contain relevant financial terminology
            text_lower = text.lower()
            assert any(word in text_lower for word in [
                "reversion", "mean", "rate", "volatility", "speed", "equilibrium",
                "long-term", "standard deviation"
            ]), f"Interpretation for {key} lacks informative content: {text}"


class TestVasicekAnalyticalVsSimulated:
    """Test that simulated results match analytical formulas."""

    def test_simulated_mean_matches_analytical(
        self,
        sample_short_rates: np.ndarray,
    ) -> None:
        """Simulated mean should match E[r(t)] = r0*e^(-κt) + θ(1-e^(-κt))."""
        model = VasicekModel()
        model.calibrate(sample_short_rates, dt=1 / 252, method="mle")

        r0 = sample_short_rates[-1]
        t = 1.0

        # Analytical expected rate
        analytical_mean = model.expected_rate(t=t, r0=r0)

        # Simulated mean (large sample for accuracy)
        result = model.simulate(
            n_paths=50000,
            time_horizon=t,
            time_step=1 / 252,
            initial_rate=r0,
            random_seed=42,
        )

        simulated_mean = result.terminal_mean()

        # Monte Carlo error tolerance (3 standard errors)
        analytical_std = np.sqrt(model.rate_variance(t=t))
        mc_error = 3 * analytical_std / np.sqrt(50000)

        np.testing.assert_almost_equal(
            simulated_mean,
            analytical_mean,
            decimal=3,
            err_msg=f"Simulated mean differs by more than {mc_error:.6f}",
        )

    def test_simulated_variance_matches_analytical(
        self,
        sample_short_rates: np.ndarray,
    ) -> None:
        """Simulated variance should match Var[r(t)] = (σ²/2κ)(1-e^(-2κt))."""
        model = VasicekModel()
        model.calibrate(sample_short_rates, dt=1 / 252, method="mle")

        r0 = sample_short_rates[-1]
        t = 1.0

        # Analytical variance
        analytical_var = model.rate_variance(t=t)

        # Simulated variance (large sample for accuracy)
        result = model.simulate(
            n_paths=50000,
            time_horizon=t,
            time_step=1 / 252,
            initial_rate=r0,
            random_seed=42,
        )

        simulated_var = result.terminal_std() ** 2

        # Variance estimation has higher error than mean
        # Use relative tolerance
        relative_error = abs(simulated_var - analytical_var) / analytical_var

        assert relative_error < 0.10, (
            f"Simulated variance {simulated_var:.8f} differs from "
            f"analytical {analytical_var:.8f} by {relative_error:.1%}"
        )


class TestVasicekEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_zero_time_horizon_simulation(
        self,
        sample_short_rates: np.ndarray,
    ) -> None:
        """Simulation with very short horizon should work."""
        model = VasicekModel()
        model.calibrate(sample_short_rates, dt=1 / 252, method="mle")

        result = model.simulate(
            n_paths=10,
            time_horizon=0.001,  # ~0.25 days
            time_step=0.001,
            random_seed=42,
        )

        assert result.path_count == 10
        # Paths should be very close to initial value
        terminal_rates = result.terminal_distribution()
        np.testing.assert_array_almost_equal(
            terminal_rates,
            np.full(10, model.theta),
            decimal=2,
        )

    def test_long_time_horizon_simulation(
        self,
        sample_short_rates: np.ndarray,
    ) -> None:
        """Simulation with long horizon should converge to equilibrium."""
        model = VasicekModel()
        model.calibrate(sample_short_rates, dt=1 / 252, method="mle")

        result = model.simulate(
            n_paths=1000,
            time_horizon=10.0,  # 10 years
            time_step=1 / 52,  # Weekly steps
            initial_rate=0.10,  # Start far from equilibrium
            random_seed=42,
        )

        # Terminal mean should be close to theta
        terminal_mean = result.terminal_mean()
        np.testing.assert_almost_equal(
            terminal_mean,
            model.theta,
            decimal=2,
            err_msg="Long-run mean should approach theta",
        )
