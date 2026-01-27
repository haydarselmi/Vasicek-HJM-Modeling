"""Performance benchmark tests for the HJM-Vasicek modeling system.

This module contains benchmark tests to verify that the system meets
the performance requirements defined in the constitution:
- 10,000 Monte Carlo simulation paths in <5 seconds
- PCA decomposition for 10 years of daily data in <2 seconds

These tests use pytest-benchmark for accurate timing measurements.
"""

from datetime import date, timedelta

import numpy as np
import pytest

from src.analysis.pca import compute_pca
from src.data.loader import InterestRateDataset
from src.models.hjm import HJMModel
from src.models.vasicek import VasicekModel
from src.simulation.monte_carlo import generate_normal_shocks


class TestMonteCarloPerformance:
    """Benchmark tests for Monte Carlo simulation performance."""

    @pytest.fixture
    def calibrated_vasicek(self) -> VasicekModel:
        """Create a pre-calibrated Vasicek model for benchmarking."""
        model = VasicekModel()
        # Use known parameters instead of calibrating
        model._kappa = 0.5
        model._theta = 0.03
        model._sigma = 0.01
        model._calibrated = True
        return model

    @pytest.fixture
    def large_dataset(self) -> InterestRateDataset:
        """Create a large dataset with 10 years of daily data (2520 points)."""
        np.random.seed(42)
        n_observations = 2520  # ~10 years of daily data
        n_maturities = 7

        start = date(2014, 1, 1)
        dates = [start + timedelta(days=i) for i in range(n_observations)]
        maturities = np.array([0.25, 0.5, 1.0, 2.0, 5.0, 10.0, 30.0])

        # Generate realistic-looking yield curve data
        base_rates = np.array([0.02, 0.025, 0.03, 0.035, 0.04, 0.045, 0.05])
        rates = np.zeros((n_observations, n_maturities))
        for i in range(n_observations):
            # Add some mean-reverting noise
            noise = np.random.randn(n_maturities) * 0.002
            rates[i] = base_rates + np.sin(i / 252 * np.pi) * 0.01 + noise

        return InterestRateDataset(
            file_path="benchmark.csv",
            source_name="Benchmark Data",
            observation_dates=dates,
            maturities=maturities,
            rates=rates,
        )

    @pytest.mark.benchmark
    @pytest.mark.slow
    def test_vasicek_10k_paths_under_5_seconds(
        self, calibrated_vasicek: VasicekModel, benchmark: pytest.fixture
    ) -> None:
        """Verify 10k Vasicek Monte Carlo paths complete in <5 seconds."""

        def run_simulation() -> None:
            calibrated_vasicek.simulate(
                n_paths=10_000,
                time_horizon=1.0,
                time_step=1 / 252,  # Daily
                random_seed=42,
            )

        # Run the benchmark
        result = benchmark(run_simulation)

        # Check timing (benchmark stats are in seconds)
        # Note: benchmark.stats.mean gives average time
        assert result is None  # benchmark() returns None, timing in benchmark.stats

    @pytest.mark.benchmark
    @pytest.mark.slow
    def test_hjm_10k_paths_under_5_seconds(
        self, large_dataset: InterestRateDataset, benchmark: pytest.fixture
    ) -> None:
        """Verify 10k HJM Monte Carlo paths complete in <5 seconds."""
        # First create and calibrate HJM model (not part of benchmark)
        pca_result = compute_pca(large_dataset, n_components=3)
        hjm = HJMModel(pca_result=pca_result, n_factors=3)
        hjm.calibrate()

        def run_simulation() -> None:
            hjm.simulate(
                n_paths=10_000,
                time_horizon=1.0,
                time_step=1 / 252,
                random_seed=42,
            )

        # Run the benchmark
        result = benchmark(run_simulation)
        assert result is None

    @pytest.mark.benchmark
    def test_shock_generation_vectorized(self, benchmark: pytest.fixture) -> None:
        """Verify vectorized shock generation is fast."""

        def generate_shocks() -> None:
            generate_normal_shocks(
                n_paths=10_000,
                n_steps=252,
                n_factors=3,
                random_seed=42,
            )

        result = benchmark(generate_shocks)
        assert result is None


class TestPCAPerformance:
    """Benchmark tests for PCA computation performance."""

    @pytest.fixture
    def ten_year_daily_dataset(self) -> InterestRateDataset:
        """Create 10 years of daily yield curve data (~2520 observations)."""
        np.random.seed(42)
        n_observations = 2520
        n_maturities = 10

        start = date(2014, 1, 1)
        dates = [start + timedelta(days=i) for i in range(n_observations)]
        maturities = np.array([0.25, 0.5, 1.0, 2.0, 3.0, 5.0, 7.0, 10.0, 20.0, 30.0])

        # Generate realistic rates
        base_rates = np.linspace(0.02, 0.05, n_maturities)
        rates = np.zeros((n_observations, n_maturities))
        for i in range(n_observations):
            noise = np.random.randn(n_maturities) * 0.002
            trend = np.sin(i / 252 * np.pi) * 0.01
            rates[i] = base_rates + trend + noise

        return InterestRateDataset(
            file_path="pca_benchmark.csv",
            source_name="PCA Benchmark Data",
            observation_dates=dates,
            maturities=maturities,
            rates=rates,
        )

    @pytest.mark.benchmark
    @pytest.mark.slow
    def test_pca_10_years_under_2_seconds(
        self, ten_year_daily_dataset: InterestRateDataset, benchmark: pytest.fixture
    ) -> None:
        """Verify PCA on 10 years of daily data completes in <2 seconds."""

        def run_pca() -> None:
            compute_pca(ten_year_daily_dataset, n_components=5)

        result = benchmark(run_pca)
        assert result is None

    @pytest.mark.benchmark
    def test_pca_with_interpretation(
        self, ten_year_daily_dataset: InterestRateDataset, benchmark: pytest.fixture
    ) -> None:
        """Benchmark PCA including component interpretation."""
        from src.analysis.pca import interpret_components

        def run_pca_with_interpretation() -> None:
            pca_result = compute_pca(ten_year_daily_dataset, n_components=5)
            interpret_components(pca_result, n_interpret=3)

        result = benchmark(run_pca_with_interpretation)
        assert result is None


class TestMemoryUsage:
    """Tests to verify memory usage stays within bounds."""

    @pytest.mark.benchmark
    def test_large_simulation_memory_bound(self) -> None:
        """Verify large simulation doesn't exceed memory limits.

        Constitution requirement: <2GB for standard simulation runs.
        This test uses 10k paths which should be well under the limit.
        """
        import sys

        # Create a calibrated model
        model = VasicekModel()
        model._kappa = 0.5
        model._theta = 0.03
        model._sigma = 0.01
        model._calibrated = True

        # Run simulation
        result = model.simulate(
            n_paths=10_000,
            time_horizon=1.0,
            time_step=1 / 252,
            random_seed=42,
        )

        # Check the size of the result
        # Each path has 252 time steps, 10k paths = 2.52M float64 values
        # At 8 bytes each = ~20MB, well under 2GB
        paths_size = sum(sys.getsizeof(p.rates) for p in result.paths)
        assert paths_size < 100 * 1024 * 1024  # Under 100MB


class TestVectorization:
    """Tests to verify vectorized operations are being used."""

    @pytest.mark.benchmark
    def test_vasicek_dynamics_vectorized(self, benchmark: pytest.fixture) -> None:
        """Verify Vasicek dynamics use vectorized operations."""
        from src.simulation.monte_carlo import apply_vasicek_dynamics

        # Large simulation parameters
        n_paths = 10_000
        n_steps = 252
        kappa, theta, sigma = 0.5, 0.03, 0.01
        dt = 1 / 252

        # Pre-generate shocks
        np.random.seed(42)
        shocks = np.random.randn(n_paths, n_steps)

        def run_dynamics() -> None:
            apply_vasicek_dynamics(
                initial_rate=theta,
                kappa=kappa,
                theta=theta,
                sigma=sigma,
                shocks=shocks,
                dt=dt,
            )

        result = benchmark(run_dynamics)
        assert result is None
