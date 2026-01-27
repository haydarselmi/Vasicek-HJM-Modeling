"""Integration tests for model comparison workflow.

This module tests the complete workflow of comparing Vasicek and HJM
models, including:
- Training both models on training data
- Backtesting on held-out test data
- Comparing metrics and generating recommendations
"""

from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pytest

from src.analysis.pca import compute_pca
from src.data.loader import InterestRateDataset
from src.data.types import MaturityVector, RateMatrix
from src.models.hjm import HJMModel
from src.models.vasicek import VasicekModel
from src.validation.backtesting import ValidationResult, backtest_model
from src.validation.comparison import (
    ComparisonResult,
    compare_models,
    get_model_recommendation,
)
from src.visualization.simulation_plots import (
    plot_backtest_results,
    plot_model_comparison,
)

# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def full_dataset() -> InterestRateDataset:
    """Create a full dataset for train/test splitting."""
    n_obs = 300  # ~1.2 years of data
    n_mats = 7
    start_date = date(2024, 1, 2)

    dates = []
    current = start_date
    while len(dates) < n_obs:
        if current.weekday() < 5:
            dates.append(current)
        current += timedelta(days=1)

    maturities: MaturityVector = np.array(
        [0.25, 0.5, 1.0, 2.0, 5.0, 10.0, 30.0], dtype=np.float64
    )

    # Generate realistic rate data with mean reversion
    rng = np.random.default_rng(42)
    base_rates = np.array([0.02, 0.022, 0.025, 0.028, 0.032, 0.035, 0.038])

    # Create correlation matrix
    correlation = np.zeros((n_mats, n_mats))
    for i in range(n_mats):
        for j in range(n_mats):
            correlation[i, j] = np.exp(-0.3 * abs(i - j))

    cholesky = np.linalg.cholesky(correlation)

    rates: RateMatrix = np.zeros((n_obs, n_mats))
    rates[0] = base_rates

    for t in range(1, n_obs):
        mean_reversion = 0.02 * (base_rates - rates[t - 1])
        uncorrelated = rng.standard_normal(n_mats)
        correlated_shocks = cholesky @ uncorrelated * 0.001
        rates[t] = rates[t - 1] + mean_reversion + correlated_shocks

    return InterestRateDataset(
        file_path=Path("test_full.csv"),
        source_name="test_full",
        observation_dates=dates,
        maturities=maturities,
        rates=rates,
    )


@pytest.fixture
def train_dataset(full_dataset: InterestRateDataset) -> InterestRateDataset:
    """Create training dataset (first 80% of data)."""
    n_obs = full_dataset.n_observations
    split_idx = int(n_obs * 0.8)
    split_date = full_dataset.observation_dates[split_idx - 1]
    return full_dataset.filter_dates(end_date=split_date)


@pytest.fixture
def test_dataset(full_dataset: InterestRateDataset) -> InterestRateDataset:
    """Create test dataset (last 20% of data)."""
    n_obs = full_dataset.n_observations
    split_idx = int(n_obs * 0.8)
    split_date = full_dataset.observation_dates[split_idx]
    return full_dataset.filter_dates(start_date=split_date)


@pytest.fixture
def calibrated_vasicek(train_dataset: InterestRateDataset) -> VasicekModel:
    """Create a calibrated Vasicek model."""
    model = VasicekModel()
    model.calibrate(train_dataset)
    return model


@pytest.fixture
def calibrated_hjm(train_dataset: InterestRateDataset) -> HJMModel:
    """Create a calibrated HJM model."""
    pca_result = compute_pca(train_dataset, n_components=3)
    model = HJMModel(pca_result=pca_result, n_factors=3)
    model.calibrate()
    return model


# =============================================================================
# Integration Tests
# =============================================================================


class TestTrainTestSplit:
    """Tests for train/test data splitting."""

    def test_filter_dates_creates_train_split(
        self, train_dataset: InterestRateDataset, full_dataset: InterestRateDataset
    ) -> None:
        """Test that filter_dates correctly creates training data."""
        assert train_dataset.n_observations > 0
        assert train_dataset.n_observations < full_dataset.n_observations
        # Training should be ~80% of full data
        ratio = train_dataset.n_observations / full_dataset.n_observations
        assert 0.75 <= ratio <= 0.85

    def test_filter_dates_creates_test_split(
        self, test_dataset: InterestRateDataset, full_dataset: InterestRateDataset
    ) -> None:
        """Test that filter_dates correctly creates test data."""
        assert test_dataset.n_observations > 0
        assert test_dataset.n_observations < full_dataset.n_observations
        # Test should be ~20% of full data
        ratio = test_dataset.n_observations / full_dataset.n_observations
        assert 0.15 <= ratio <= 0.25

    def test_train_test_no_overlap(
        self,
        train_dataset: InterestRateDataset,
        test_dataset: InterestRateDataset,
    ) -> None:
        """Test that train and test sets have no overlapping dates."""
        train_dates = set(train_dataset.observation_dates)
        test_dates = set(test_dataset.observation_dates)

        # Allow at most 1 overlapping date (the split point)
        overlap = train_dates & test_dates
        assert len(overlap) <= 1


class TestModelCalibration:
    """Tests for model calibration on training data."""

    def test_vasicek_calibration(self, calibrated_vasicek: VasicekModel) -> None:
        """Test Vasicek model calibration succeeds."""
        assert calibrated_vasicek.is_calibrated
        assert calibrated_vasicek.kappa > 0
        assert calibrated_vasicek.sigma > 0

    def test_hjm_calibration(self, calibrated_hjm: HJMModel) -> None:
        """Test HJM model calibration succeeds."""
        assert calibrated_hjm.is_calibrated
        assert calibrated_hjm.n_factors == 3


class TestBacktesting:
    """Tests for model backtesting on test data."""

    def test_backtest_vasicek(
        self,
        calibrated_vasicek: VasicekModel,
        test_dataset: InterestRateDataset,
    ) -> None:
        """Test Vasicek backtesting produces valid results."""
        result = backtest_model(
            model=calibrated_vasicek,
            validation_data=test_dataset,
            prediction_horizon=1,
        )

        assert isinstance(result, ValidationResult)
        assert result.model_type == "vasicek"
        assert result.mae >= 0
        assert result.rmse >= 0
        assert 0 <= result.hit_ratio <= 1
        assert len(result.predicted_values) > 0

    def test_backtest_hjm(
        self,
        calibrated_hjm: HJMModel,
        test_dataset: InterestRateDataset,
    ) -> None:
        """Test HJM backtesting produces valid results."""
        result = backtest_model(
            model=calibrated_hjm,
            validation_data=test_dataset,
            prediction_horizon=1,
        )

        assert isinstance(result, ValidationResult)
        assert result.model_type == "hjm"
        assert result.mae >= 0
        assert result.rmse >= 0
        assert 0 <= result.hit_ratio <= 1


class TestModelComparison:
    """Tests for model comparison functionality."""

    def test_compare_vasicek_vs_hjm(
        self,
        calibrated_vasicek: VasicekModel,
        calibrated_hjm: HJMModel,
        test_dataset: InterestRateDataset,
    ) -> None:
        """Test comparing Vasicek and HJM models."""
        comparison = compare_models(
            models=[calibrated_vasicek, calibrated_hjm],
            validation_data=test_dataset,
            model_names=["Vasicek", "HJM"],
        )

        assert isinstance(comparison, ComparisonResult)
        assert len(comparison.models) == 2
        assert "Vasicek" in comparison.models
        assert "HJM" in comparison.models

        # Both models should have validation results
        assert "Vasicek" in comparison.validation_results
        assert "HJM" in comparison.validation_results

        # Should have recommendations for all use cases
        assert "short_rate" in comparison.recommendations
        assert "yield_curve" in comparison.recommendations
        assert "derivatives" in comparison.recommendations
        assert "general" in comparison.recommendations

    def test_comparison_best_model_by_metric(
        self,
        calibrated_vasicek: VasicekModel,
        calibrated_hjm: HJMModel,
        test_dataset: InterestRateDataset,
    ) -> None:
        """Test that best model by metric is computed correctly."""
        comparison = compare_models(
            models=[calibrated_vasicek, calibrated_hjm],
            validation_data=test_dataset,
            model_names=["Vasicek", "HJM"],
        )

        # Should have best model for each metric
        assert "mae" in comparison.best_model_by_metric
        assert "rmse" in comparison.best_model_by_metric
        assert "hit_ratio" in comparison.best_model_by_metric

        # Best models should be one of the compared models
        for _metric, best in comparison.best_model_by_metric.items():
            assert best in comparison.models

    def test_comparison_metrics_table(
        self,
        calibrated_vasicek: VasicekModel,
        calibrated_hjm: HJMModel,
        test_dataset: InterestRateDataset,
    ) -> None:
        """Test that metrics table is generated correctly."""
        comparison = compare_models(
            models=[calibrated_vasicek, calibrated_hjm],
            validation_data=test_dataset,
            model_names=["Vasicek", "HJM"],
        )

        table = comparison.get_metrics_table()

        assert "Vasicek" in table
        assert "HJM" in table
        assert "MAE" in table
        assert "RMSE" in table
        assert "Hit Ratio" in table


class TestModelRecommendation:
    """Tests for model recommendation functionality."""

    def test_recommendation_for_each_use_case(
        self,
        calibrated_vasicek: VasicekModel,
        calibrated_hjm: HJMModel,
        test_dataset: InterestRateDataset,
    ) -> None:
        """Test recommendations are generated for all use cases."""
        comparison = compare_models(
            models=[calibrated_vasicek, calibrated_hjm],
            validation_data=test_dataset,
            model_names=["Vasicek", "HJM"],
        )

        use_cases = ["short_rate", "yield_curve", "derivatives", "general"]

        for use_case in use_cases:
            recommendation = get_model_recommendation(comparison, use_case)

            # Recommendation should be a non-empty string
            assert isinstance(recommendation, str)
            assert len(recommendation) > 0

            # Should mention at least one of the models
            assert "Vasicek" in recommendation or "HJM" in recommendation

    def test_recommendation_has_justification(
        self,
        calibrated_vasicek: VasicekModel,
        calibrated_hjm: HJMModel,
        test_dataset: InterestRateDataset,
    ) -> None:
        """Test recommendations include justification text."""
        comparison = compare_models(
            models=[calibrated_vasicek, calibrated_hjm],
            validation_data=test_dataset,
        )

        recommendation = get_model_recommendation(comparison, "short_rate")

        # Should include justification
        assert "Justification" in recommendation


class TestVisualization:
    """Tests for comparison and backtest visualization."""

    def test_plot_model_comparison(
        self,
        calibrated_vasicek: VasicekModel,
        calibrated_hjm: HJMModel,
        test_dataset: InterestRateDataset,
    ) -> None:
        """Test model comparison plotting."""
        comparison = compare_models(
            models=[calibrated_vasicek, calibrated_hjm],
            validation_data=test_dataset,
            model_names=["Vasicek", "HJM"],
        )

        # Test all metrics plot
        fig = plot_model_comparison(comparison)
        assert fig is not None

        # Clean up
        import matplotlib.pyplot as plt
        plt.close(fig)

    def test_plot_backtest_results(
        self,
        calibrated_vasicek: VasicekModel,
        test_dataset: InterestRateDataset,
    ) -> None:
        """Test backtest results plotting."""
        result = backtest_model(
            model=calibrated_vasicek,
            validation_data=test_dataset,
        )

        fig = plot_backtest_results(result)
        assert fig is not None

        # Clean up
        import matplotlib.pyplot as plt
        plt.close(fig)


class TestEndToEndWorkflow:
    """End-to-end integration tests for the complete comparison workflow."""

    def test_full_comparison_workflow(
        self,
        full_dataset: InterestRateDataset,
    ) -> None:
        """Test the complete workflow from data split to recommendation."""
        # Step 1: Split data
        n_obs = full_dataset.n_observations
        split_idx = int(n_obs * 0.8)
        split_date = full_dataset.observation_dates[split_idx - 1]

        train_data = full_dataset.filter_dates(end_date=split_date)
        test_data = full_dataset.filter_dates(
            start_date=full_dataset.observation_dates[split_idx]
        )

        # Step 2: Calibrate Vasicek model
        vasicek = VasicekModel()
        vasicek.calibrate(train_data)
        assert vasicek.is_calibrated

        # Step 3: Perform PCA and calibrate HJM model
        pca_result = compute_pca(train_data, n_components=3)
        hjm = HJMModel(pca_result=pca_result, n_factors=3)
        hjm.calibrate()
        assert hjm.is_calibrated

        # Step 4: Compare models on test data
        comparison = compare_models(
            models=[vasicek, hjm],
            validation_data=test_data,
            model_names=["Vasicek", "HJM"],
        )

        # Step 5: Get recommendations
        short_rate_rec = get_model_recommendation(comparison, "short_rate")
        yield_curve_rec = get_model_recommendation(comparison, "yield_curve")
        get_model_recommendation(comparison, "derivatives")

        # Step 6: Validate outputs
        assert "Vasicek" in short_rate_rec or "HJM" in short_rate_rec
        assert "HJM" in yield_curve_rec  # HJM should be recommended for yield curves
        assert comparison.summary()  # Should generate summary text

        # Step 7: Display comparison summary
        print("\n" + "=" * 60)
        print(comparison.summary())
        print("=" * 60 + "\n")
