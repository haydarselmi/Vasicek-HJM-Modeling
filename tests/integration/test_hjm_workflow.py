"""Integration tests for the HJM model workflow.

This module tests the complete HJM workflow:
PCA → HJM setup → calibrate → simulate → factor contributions.
"""

from datetime import date
from pathlib import Path

import numpy as np
import pytest

from src.analysis.pca import PCAResult, compute_pca
from src.data.loader import InterestRateDataset, load_interest_rate_data
from src.models.hjm import FactorContribution, HJMModel
from src.simulation.paths import SimulationResult
from src.visualization.simulation_plots import (
    plot_forward_curve_evolution,
    plot_volatility_functions,
)

# =============================================================================
# T073: Integration test for HJM workflow
# =============================================================================


class TestHJMWorkflowFromPCA:
    """Test complete workflow: PCA → HJM → simulate → analyze."""

    @pytest.fixture
    def sample_dataset(
        self,
        sample_maturities: np.ndarray,
        sample_observation_dates: list[date],
        sample_rate_matrix: np.ndarray,
    ) -> InterestRateDataset:
        """Create a sample dataset for integration testing."""
        return InterestRateDataset(
            file_path=Path("test_data.csv"),
            source_name="Test Data",
            observation_dates=sample_observation_dates,
            maturities=sample_maturities,
            rates=sample_rate_matrix,
        )

    def test_full_hjm_workflow(
        self,
        sample_dataset: InterestRateDataset,
    ) -> None:
        """Test complete HJM workflow from data to simulation."""
        # Step 1: Compute PCA on yield curve changes
        pca_result = compute_pca(sample_dataset, n_components=3, use_changes=True)

        assert isinstance(pca_result, PCAResult)
        assert pca_result.n_components == 3
        assert len(pca_result.interpretations) == 3

        # Step 2: Initialize HJM model from PCA
        hjm_model = HJMModel(pca_result=pca_result, n_factors=3)

        assert hjm_model.n_factors == 3
        assert not hjm_model.is_calibrated

        # Step 3: Calibrate HJM model
        hjm_model.calibrate()

        assert hjm_model.is_calibrated
        assert hjm_model.volatility_functions is not None
        assert hjm_model.volatility_functions.shape == (3, len(sample_dataset.maturities))

        # Step 4: Simulate forward curves
        initial_curve = sample_dataset.rates[-1]  # Use last observed curve
        result = hjm_model.simulate(
            n_paths=500,
            time_horizon=1.0,
            time_step=1 / 252,
            initial_curve=initial_curve,
            random_seed=42,
        )

        assert isinstance(result, SimulationResult)
        assert result.model_type == "hjm"
        assert result.path_count == 500

        # Step 5: Analyze factor contributions
        contributions = hjm_model.get_factor_contributions(result)

        assert len(contributions) == 3
        assert all(isinstance(c, FactorContribution) for c in contributions)

        # Verify contribution percentages sum to approximately 100%
        total_pct = sum(c.percentage for c in contributions)
        assert 90.0 <= total_pct <= 110.0

    def test_hjm_with_two_factors(
        self,
        sample_dataset: InterestRateDataset,
    ) -> None:
        """Test HJM workflow with 2 factors (level + slope)."""
        # Compute PCA with all components
        pca_result = compute_pca(sample_dataset, n_components=None, use_changes=True)

        # Initialize HJM with only 2 factors
        hjm_model = HJMModel(pca_result=pca_result, n_factors=2)
        hjm_model.calibrate()

        assert hjm_model.n_factors == 2
        assert hjm_model.volatility_functions.shape[0] == 2

        # Simulate
        initial_curve = sample_dataset.rates[-1]
        result = hjm_model.simulate(
            n_paths=100,
            time_horizon=0.25,
            time_step=1 / 252,
            initial_curve=initial_curve,
            random_seed=42,
        )

        assert result.path_count == 100

        # Factor contributions should only have 2 factors
        contributions = hjm_model.get_factor_contributions(result)
        assert len(contributions) == 2

    def test_hjm_preserves_curve_structure(
        self,
        sample_dataset: InterestRateDataset,
    ) -> None:
        """Test that HJM simulation preserves reasonable curve structure."""
        pca_result = compute_pca(sample_dataset, n_components=3, use_changes=True)
        hjm_model = HJMModel(pca_result=pca_result)
        hjm_model.calibrate()

        initial_curve = sample_dataset.rates[-1]
        result = hjm_model.simulate(
            n_paths=100,
            time_horizon=1.0,
            time_step=1 / 252,
            initial_curve=initial_curve,
            random_seed=42,
        )

        # Check that terminal curves are still reasonable
        terminal_curves = np.array([p.rates[-1] for p in result.paths])

        # Rates should be within reasonable bounds (-10% to 50%)
        assert np.all(terminal_curves > -0.10)
        assert np.all(terminal_curves < 0.50)

        # Most curves should still be upward sloping (on average)
        mean_terminal_curve = np.mean(terminal_curves, axis=0)
        slope = mean_terminal_curve[-1] - mean_terminal_curve[0]
        # Allow for curve inversion but expect generally positive slope
        assert slope > -0.05  # At most 5% inverted

    def test_hjm_simulation_reproducibility(
        self,
        sample_dataset: InterestRateDataset,
    ) -> None:
        """Test that HJM simulations are reproducible with seed."""
        pca_result = compute_pca(sample_dataset, n_components=3, use_changes=True)

        # Two independent models
        model1 = HJMModel(pca_result=pca_result)
        model1.calibrate()

        model2 = HJMModel(pca_result=pca_result)
        model2.calibrate()

        initial_curve = sample_dataset.rates[-1]

        result1 = model1.simulate(
            n_paths=50,
            time_horizon=0.25,
            time_step=1 / 252,
            initial_curve=initial_curve,
            random_seed=123,
        )

        result2 = model2.simulate(
            n_paths=50,
            time_horizon=0.25,
            time_step=1 / 252,
            initial_curve=initial_curve,
            random_seed=123,
        )

        # Results should be identical
        for p1, p2 in zip(result1.paths, result2.paths, strict=False):
            np.testing.assert_array_equal(p1.rates, p2.rates)


class TestHJMWorkflowWithRealData:
    """Test HJM workflow with sample treasury rates data."""

    @pytest.fixture
    def treasury_dataset(self) -> InterestRateDataset:
        """Load sample treasury rates data."""
        data_path = Path("data/sample_treasury_rates.csv")
        if not data_path.exists():
            pytest.skip("Sample treasury data not available")

        return load_interest_rate_data(
            file_path=data_path,
            date_column="date",
            source_name="Sample Treasury Rates",
        )

    def test_hjm_workflow_real_data(
        self,
        treasury_dataset: InterestRateDataset,
    ) -> None:
        """Test HJM workflow with actual treasury rate data."""
        # PCA on real data
        pca_result = compute_pca(treasury_dataset, n_components=3, use_changes=True)

        # Typical PCA results on treasury data:
        # PC1 should explain majority of variance (allow for sample data)
        assert pca_result.explained_variance_ratio[0] > 0.50

        # First 3 components should explain most variance
        assert pca_result.cumulative_variance[2] > 0.80

        # HJM model
        hjm_model = HJMModel(pca_result=pca_result, n_factors=3)
        hjm_model.calibrate()

        # Simulate
        initial_curve = treasury_dataset.rates[-1]
        result = hjm_model.simulate(
            n_paths=500,
            time_horizon=1.0,
            time_step=1 / 252,
            initial_curve=initial_curve,
            random_seed=42,
        )

        # Verify simulation statistics
        mean_path = result.mean_path()
        assert mean_path.shape == (253, len(treasury_dataset.maturities))

        # Terminal distribution should be reasonable
        terminal_dist = result.terminal_distribution()
        assert terminal_dist.shape == (500, len(treasury_dataset.maturities))


class TestHJMVisualization:
    """Test HJM visualization functions."""

    @pytest.fixture
    def viz_dataset(
        self,
        sample_maturities: np.ndarray,
        sample_observation_dates: list[date],
        sample_rate_matrix: np.ndarray,
    ) -> InterestRateDataset:
        """Create a sample dataset for visualization testing."""
        return InterestRateDataset(
            file_path=Path("test_viz.csv"),
            source_name="Test Viz Data",
            observation_dates=sample_observation_dates,
            maturities=sample_maturities,
            rates=sample_rate_matrix,
        )

    def test_plot_volatility_functions(
        self,
        viz_dataset: InterestRateDataset,
    ) -> None:
        """Test volatility function plotting."""
        pca_result = compute_pca(viz_dataset, n_components=3, use_changes=True)
        hjm_model = HJMModel(pca_result=pca_result)
        hjm_model.calibrate()

        fig = plot_volatility_functions(hjm_model)

        assert fig is not None
        # Should have axes for the plot
        assert len(fig.axes) > 0

    def test_plot_forward_curve_evolution(
        self,
        viz_dataset: InterestRateDataset,
    ) -> None:
        """Test forward curve evolution plotting."""
        pca_result = compute_pca(viz_dataset, n_components=3, use_changes=True)
        hjm_model = HJMModel(pca_result=pca_result)
        hjm_model.calibrate()

        initial_curve = viz_dataset.rates[-1]
        result = hjm_model.simulate(
            n_paths=50,
            time_horizon=0.25,
            time_step=1 / 252,
            initial_curve=initial_curve,
            random_seed=42,
        )

        fig = plot_forward_curve_evolution(result)

        assert fig is not None
        assert len(fig.axes) > 0


# =============================================================================
# Performance test
# =============================================================================


class TestHJMPerformance:
    """Test HJM performance meets requirements."""

    def test_hjm_simulation_performance(
        self,
        sample_maturities: np.ndarray,
        sample_observation_dates: list[date],
        sample_rate_matrix: np.ndarray,
    ) -> None:
        """HJM simulation should complete within reasonable time."""
        import time

        # Create dataset
        dataset = InterestRateDataset(
            file_path=Path("test.csv"),
            source_name="Test",
            observation_dates=sample_observation_dates,
            maturities=sample_maturities,
            rates=sample_rate_matrix,
        )

        # PCA + HJM setup
        pca_result = compute_pca(dataset, n_components=3, use_changes=True)
        hjm_model = HJMModel(pca_result=pca_result)
        hjm_model.calibrate()

        # Time the simulation
        initial_curve = dataset.rates[-1]
        start = time.time()

        result = hjm_model.simulate(
            n_paths=1000,
            time_horizon=1.0,
            time_step=1 / 252,
            initial_curve=initial_curve,
            random_seed=42,
        )

        elapsed = time.time() - start

        # Should complete in under 10 seconds (generous limit for CI)
        assert elapsed < 10.0, f"HJM simulation took {elapsed:.2f}s, expected <10s"
        assert result.path_count == 1000
