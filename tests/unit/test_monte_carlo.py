"""Unit tests for the Monte Carlo simulation engine.

This module tests shock generation, Vasicek dynamics application,
and simulation reproducibility.
"""

import numpy as np
import pytest

from src.simulation.monte_carlo import (
    apply_vasicek_dynamics,
    generate_normal_shocks,
)


# =============================================================================
# T041: Unit tests for Monte Carlo engine
# =============================================================================


class TestGenerateNormalShocks:
    """Test the generate_normal_shocks function."""

    def test_returns_correct_shape(self) -> None:
        """Generated shocks should have shape (n_paths, n_factors, n_steps)."""
        shocks = generate_normal_shocks(
            n_paths=100,
            n_steps=252,
            n_factors=1,
        )

        assert shocks.shape == (100, 1, 252)

    def test_multifactor_shape(self) -> None:
        """Multi-factor shocks should have correct shape."""
        shocks = generate_normal_shocks(
            n_paths=50,
            n_steps=100,
            n_factors=3,
        )

        assert shocks.shape == (50, 3, 100)

    def test_shocks_have_unit_variance(self) -> None:
        """Generated shocks should have approximately unit variance."""
        shocks = generate_normal_shocks(
            n_paths=10000,
            n_steps=100,
            n_factors=1,
            random_seed=42,
        )

        variance = np.var(shocks)
        np.testing.assert_almost_equal(variance, 1.0, decimal=1)

    def test_shocks_have_zero_mean(self) -> None:
        """Generated shocks should have approximately zero mean."""
        shocks = generate_normal_shocks(
            n_paths=10000,
            n_steps=100,
            n_factors=1,
            random_seed=42,
        )

        mean = np.mean(shocks)
        np.testing.assert_almost_equal(mean, 0.0, decimal=1)

    def test_reproducibility_with_seed(self) -> None:
        """Same seed should produce identical shocks."""
        shocks1 = generate_normal_shocks(
            n_paths=100,
            n_steps=50,
            n_factors=1,
            random_seed=12345,
        )

        shocks2 = generate_normal_shocks(
            n_paths=100,
            n_steps=50,
            n_factors=1,
            random_seed=12345,
        )

        np.testing.assert_array_equal(shocks1, shocks2)

    def test_different_seeds_produce_different_shocks(self) -> None:
        """Different seeds should produce different shocks."""
        shocks1 = generate_normal_shocks(
            n_paths=100,
            n_steps=50,
            n_factors=1,
            random_seed=12345,
        )

        shocks2 = generate_normal_shocks(
            n_paths=100,
            n_steps=50,
            n_factors=1,
            random_seed=54321,
        )

        assert not np.allclose(shocks1, shocks2)

    def test_no_seed_produces_different_each_call(self) -> None:
        """Without seed, each call should produce different shocks."""
        shocks1 = generate_normal_shocks(
            n_paths=100,
            n_steps=50,
            n_factors=1,
            random_seed=None,
        )

        shocks2 = generate_normal_shocks(
            n_paths=100,
            n_steps=50,
            n_factors=1,
            random_seed=None,
        )

        # Very unlikely to be equal by chance
        assert not np.allclose(shocks1, shocks2)

    def test_factors_are_independent(self) -> None:
        """Shocks across factors should be independent (uncorrelated)."""
        shocks = generate_normal_shocks(
            n_paths=10000,
            n_steps=100,
            n_factors=3,
            random_seed=42,
        )

        # Flatten to get all shocks for each factor
        factor_shocks = [shocks[:, i, :].flatten() for i in range(3)]

        # Compute correlation matrix
        correlation = np.corrcoef(factor_shocks)

        # Off-diagonal elements should be close to zero
        np.testing.assert_array_less(
            np.abs(correlation - np.eye(3)),
            0.1,
            err_msg="Factors should be uncorrelated",
        )


class TestApplyVasicekDynamics:
    """Test the apply_vasicek_dynamics function."""

    def test_returns_correct_shape(self) -> None:
        """Result should have shape (n_paths, n_steps + 1)."""
        shocks = generate_normal_shocks(
            n_paths=100,
            n_steps=252,
            n_factors=1,
            random_seed=42,
        )
        # Squeeze to (n_paths, n_steps)
        shocks = shocks.squeeze(axis=1)

        rates = apply_vasicek_dynamics(
            initial_rate=0.03,
            kappa=0.5,
            theta=0.03,
            sigma=0.01,
            shocks=shocks,
            dt=1 / 252,
        )

        assert rates.shape == (100, 253)  # n_steps + 1 for initial value

    def test_initial_rate_preserved(self) -> None:
        """First column should be the initial rate."""
        shocks = generate_normal_shocks(
            n_paths=100,
            n_steps=50,
            n_factors=1,
            random_seed=42,
        ).squeeze(axis=1)

        initial_rate = 0.05

        rates = apply_vasicek_dynamics(
            initial_rate=initial_rate,
            kappa=0.5,
            theta=0.03,
            sigma=0.01,
            shocks=shocks,
            dt=1 / 252,
        )

        np.testing.assert_array_almost_equal(
            rates[:, 0],
            np.full(100, initial_rate),
        )

    def test_mean_reversion_behavior(self) -> None:
        """Paths should revert toward theta over time."""
        n_paths = 1000
        n_steps = 252 * 5  # 5 years
        dt = 1 / 252

        shocks = generate_normal_shocks(
            n_paths=n_paths,
            n_steps=n_steps,
            n_factors=1,
            random_seed=42,
        ).squeeze(axis=1)

        theta = 0.03
        initial_rate = 0.08  # Start far from theta

        rates = apply_vasicek_dynamics(
            initial_rate=initial_rate,
            kappa=1.0,  # Faster mean reversion
            theta=theta,
            sigma=0.005,
            shocks=shocks,
            dt=dt,
        )

        # Mean of final rates should be close to theta
        final_mean = np.mean(rates[:, -1])
        np.testing.assert_almost_equal(final_mean, theta, decimal=2)

    def test_volatility_affects_spread(self) -> None:
        """Higher sigma should produce wider distribution."""
        n_paths = 1000
        n_steps = 252
        dt = 1 / 252

        shocks = generate_normal_shocks(
            n_paths=n_paths,
            n_steps=n_steps,
            n_factors=1,
            random_seed=42,
        ).squeeze(axis=1)

        rates_low_vol = apply_vasicek_dynamics(
            initial_rate=0.03,
            kappa=0.5,
            theta=0.03,
            sigma=0.005,
            shocks=shocks,
            dt=dt,
        )

        rates_high_vol = apply_vasicek_dynamics(
            initial_rate=0.03,
            kappa=0.5,
            theta=0.03,
            sigma=0.02,
            shocks=shocks,
            dt=dt,
        )

        std_low = np.std(rates_low_vol[:, -1])
        std_high = np.std(rates_high_vol[:, -1])

        assert std_high > std_low, "Higher volatility should produce wider spread"

    def test_kappa_affects_reversion_speed(self) -> None:
        """Higher kappa should produce faster reversion."""
        n_paths = 1000
        n_steps = 252
        dt = 1 / 252
        theta = 0.03
        initial_rate = 0.08

        shocks = generate_normal_shocks(
            n_paths=n_paths,
            n_steps=n_steps,
            n_factors=1,
            random_seed=42,
        ).squeeze(axis=1)

        rates_slow = apply_vasicek_dynamics(
            initial_rate=initial_rate,
            kappa=0.1,
            theta=theta,
            sigma=0.005,
            shocks=shocks,
            dt=dt,
        )

        rates_fast = apply_vasicek_dynamics(
            initial_rate=initial_rate,
            kappa=2.0,
            theta=theta,
            sigma=0.005,
            shocks=shocks,
            dt=dt,
        )

        # Fast reversion should be closer to theta
        distance_slow = abs(np.mean(rates_slow[:, -1]) - theta)
        distance_fast = abs(np.mean(rates_fast[:, -1]) - theta)

        assert distance_fast < distance_slow, "Higher kappa should revert faster"

    def test_vectorized_output_matches_loop(self) -> None:
        """Vectorized implementation should match explicit loop."""
        n_paths = 10
        n_steps = 50
        dt = 1 / 252
        kappa = 0.5
        theta = 0.03
        sigma = 0.01
        initial_rate = 0.04

        shocks = generate_normal_shocks(
            n_paths=n_paths,
            n_steps=n_steps,
            n_factors=1,
            random_seed=42,
        ).squeeze(axis=1)

        # Vectorized version
        rates_vectorized = apply_vasicek_dynamics(
            initial_rate=initial_rate,
            kappa=kappa,
            theta=theta,
            sigma=sigma,
            shocks=shocks,
            dt=dt,
        )

        # Loop version for reference
        rates_loop = np.zeros((n_paths, n_steps + 1))
        rates_loop[:, 0] = initial_rate
        sqrt_dt = np.sqrt(dt)

        for t in range(n_steps):
            drift = kappa * (theta - rates_loop[:, t]) * dt
            diffusion = sigma * sqrt_dt * shocks[:, t]
            rates_loop[:, t + 1] = rates_loop[:, t] + drift + diffusion

        np.testing.assert_array_almost_equal(rates_vectorized, rates_loop, decimal=10)


class TestMonteCarloNumericalStability:
    """Test numerical stability of Monte Carlo simulation."""

    def test_no_nan_values_in_output(self) -> None:
        """Output should not contain NaN values."""
        shocks = generate_normal_shocks(
            n_paths=100,
            n_steps=1000,
            n_factors=1,
            random_seed=42,
        ).squeeze(axis=1)

        rates = apply_vasicek_dynamics(
            initial_rate=0.03,
            kappa=0.5,
            theta=0.03,
            sigma=0.02,
            shocks=shocks,
            dt=1 / 252,
        )

        assert not np.any(np.isnan(rates)), "Output contains NaN values"

    def test_no_inf_values_in_output(self) -> None:
        """Output should not contain infinite values."""
        shocks = generate_normal_shocks(
            n_paths=100,
            n_steps=1000,
            n_factors=1,
            random_seed=42,
        ).squeeze(axis=1)

        rates = apply_vasicek_dynamics(
            initial_rate=0.03,
            kappa=0.5,
            theta=0.03,
            sigma=0.02,
            shocks=shocks,
            dt=1 / 252,
        )

        assert not np.any(np.isinf(rates)), "Output contains infinite values"

    def test_handles_large_number_of_paths(self) -> None:
        """Should handle large number of simulation paths."""
        n_paths = 10000

        shocks = generate_normal_shocks(
            n_paths=n_paths,
            n_steps=252,
            n_factors=1,
            random_seed=42,
        ).squeeze(axis=1)

        rates = apply_vasicek_dynamics(
            initial_rate=0.03,
            kappa=0.5,
            theta=0.03,
            sigma=0.01,
            shocks=shocks,
            dt=1 / 252,
        )

        assert rates.shape[0] == n_paths
        assert not np.any(np.isnan(rates))

    def test_handles_small_time_step(self) -> None:
        """Should handle very small time steps."""
        dt = 1 / 2520  # 10x smaller than daily

        shocks = generate_normal_shocks(
            n_paths=100,
            n_steps=252,
            n_factors=1,
            random_seed=42,
        ).squeeze(axis=1)

        rates = apply_vasicek_dynamics(
            initial_rate=0.03,
            kappa=0.5,
            theta=0.03,
            sigma=0.01,
            shocks=shocks,
            dt=dt,
        )

        assert not np.any(np.isnan(rates))
        assert not np.any(np.isinf(rates))
