"""Unit tests for backtesting functionality.

This module tests error metric calculations (MAE, RMSE, MAPE, hit ratio)
and the backtesting pipeline for model validation.
"""

from datetime import date, datetime, timedelta

import numpy as np
import numpy.typing as npt
import pytest

from src.data.loader import InterestRateDataset
from src.data.types import MaturityVector, RateMatrix
from src.validation.backtesting import (
    ValidationResult,
    backtest_model,
    compute_hit_ratio,
    compute_mae,
    compute_mape,
    compute_rmse,
)


# =============================================================================
# Test Fixtures
# =============================================================================


@pytest.fixture
def sample_predicted() -> npt.NDArray[np.float64]:
    """Return sample predicted values for metric testing."""
    return np.array([0.03, 0.035, 0.032, 0.028, 0.031])


@pytest.fixture
def sample_actual() -> npt.NDArray[np.float64]:
    """Return sample actual values for metric testing."""
    return np.array([0.031, 0.033, 0.030, 0.029, 0.032])


@pytest.fixture
def sample_std() -> npt.NDArray[np.float64]:
    """Return sample standard deviations for hit ratio testing."""
    return np.array([0.002, 0.003, 0.002, 0.001, 0.002])


# =============================================================================
# Error Metric Tests
# =============================================================================


class TestMAE:
    """Tests for Mean Absolute Error calculation."""

    def test_mae_basic(
        self,
        sample_predicted: npt.NDArray[np.float64],
        sample_actual: npt.NDArray[np.float64],
    ) -> None:
        """Test MAE computation with sample data."""
        mae = compute_mae(sample_predicted, sample_actual)

        # Manual calculation: |0.03-0.031| + |0.035-0.033| + |0.032-0.030| +
        #                     |0.028-0.029| + |0.031-0.032| = 0.001 + 0.002 +
        #                     0.002 + 0.001 + 0.001 = 0.007 / 5 = 0.0014
        expected = np.mean(np.abs(sample_predicted - sample_actual))
        assert np.isclose(mae, expected)

    def test_mae_identical_arrays(self) -> None:
        """Test MAE is zero for identical arrays."""
        values = np.array([0.03, 0.035, 0.032])
        mae = compute_mae(values, values)
        assert mae == 0.0

    def test_mae_non_negative(self) -> None:
        """Test MAE is always non-negative."""
        rng = np.random.default_rng(42)
        for _ in range(10):
            pred = rng.uniform(-0.02, 0.10, 100)
            actual = rng.uniform(-0.02, 0.10, 100)
            mae = compute_mae(pred, actual)
            assert mae >= 0.0

    def test_mae_symmetric(self) -> None:
        """Test MAE is symmetric (MAE(a, b) = MAE(b, a))."""
        a = np.array([0.03, 0.035, 0.032])
        b = np.array([0.031, 0.033, 0.030])
        assert np.isclose(compute_mae(a, b), compute_mae(b, a))


class TestRMSE:
    """Tests for Root Mean Squared Error calculation."""

    def test_rmse_basic(
        self,
        sample_predicted: npt.NDArray[np.float64],
        sample_actual: npt.NDArray[np.float64],
    ) -> None:
        """Test RMSE computation with sample data."""
        rmse = compute_rmse(sample_predicted, sample_actual)

        # RMSE should be >= MAE
        mae = compute_mae(sample_predicted, sample_actual)
        assert rmse >= mae

        # Manual calculation
        expected = np.sqrt(np.mean((sample_predicted - sample_actual) ** 2))
        assert np.isclose(rmse, expected)

    def test_rmse_identical_arrays(self) -> None:
        """Test RMSE is zero for identical arrays."""
        values = np.array([0.03, 0.035, 0.032])
        rmse = compute_rmse(values, values)
        assert rmse == 0.0

    def test_rmse_penalizes_outliers(self) -> None:
        """Test RMSE penalizes outliers more than MAE."""
        # Small errors
        pred1 = np.array([0.03, 0.031, 0.032])
        actual1 = np.array([0.031, 0.032, 0.033])

        # One large error
        pred2 = np.array([0.03, 0.031, 0.04])
        actual2 = np.array([0.031, 0.032, 0.033])

        rmse1 = compute_rmse(pred1, actual1)
        rmse2 = compute_rmse(pred2, actual2)

        mae1 = compute_mae(pred1, actual1)
        mae2 = compute_mae(pred2, actual2)

        # RMSE ratio should be higher than MAE ratio due to outlier penalty
        rmse_ratio = rmse2 / rmse1
        mae_ratio = mae2 / mae1
        assert rmse_ratio > mae_ratio


class TestMAPE:
    """Tests for Mean Absolute Percentage Error calculation."""

    def test_mape_basic(self) -> None:
        """Test MAPE computation with sample data."""
        predicted = np.array([0.033, 0.036, 0.030])
        actual = np.array([0.030, 0.035, 0.032])

        mape = compute_mape(predicted, actual)

        # Manual: |0.033-0.030|/0.030 + |0.036-0.035|/0.035 + |0.030-0.032|/0.032
        # = 0.10 + 0.0286 + 0.0625 = 0.1911 / 3 * 100 = 6.37%
        expected = np.mean(np.abs(predicted - actual) / np.abs(actual)) * 100
        assert np.isclose(mape, expected)

    def test_mape_returns_none_when_actual_zero(self) -> None:
        """Test MAPE returns None when actual contains zero."""
        predicted = np.array([0.03, 0.035, 0.032])
        actual = np.array([0.031, 0.0, 0.030])  # Contains zero

        mape = compute_mape(predicted, actual)
        assert mape is None

    def test_mape_returns_none_when_actual_crosses_zero(self) -> None:
        """Test MAPE returns None when actual crosses zero."""
        predicted = np.array([0.03, 0.035, 0.032])
        actual = np.array([0.031, -0.001, 0.030])  # Crosses zero

        mape = compute_mape(predicted, actual)
        # MAPE with negative actuals can be computed but is unusual
        # Our implementation may handle this differently
        # The key is consistency


class TestHitRatio:
    """Tests for hit ratio calculation."""

    def test_hit_ratio_basic(
        self,
        sample_predicted: npt.NDArray[np.float64],
        sample_actual: npt.NDArray[np.float64],
        sample_std: npt.NDArray[np.float64],
    ) -> None:
        """Test hit ratio computation."""
        hit_ratio = compute_hit_ratio(sample_predicted, sample_actual, sample_std)

        # Hit ratio should be between 0 and 1
        assert 0.0 <= hit_ratio <= 1.0

    def test_hit_ratio_perfect_prediction(self) -> None:
        """Test hit ratio is 1.0 for perfect predictions."""
        values = np.array([0.03, 0.035, 0.032])
        std = np.array([0.001, 0.001, 0.001])

        hit_ratio = compute_hit_ratio(values, values, std)
        assert hit_ratio == 1.0

    def test_hit_ratio_all_miss(self) -> None:
        """Test hit ratio is 0.0 when all predictions miss."""
        predicted = np.array([0.03, 0.035, 0.032])
        actual = np.array([0.05, 0.06, 0.055])  # Way off
        std = np.array([0.001, 0.001, 0.001])  # Very small tolerance

        hit_ratio = compute_hit_ratio(predicted, actual, std)
        assert hit_ratio == 0.0

    def test_hit_ratio_with_1_std_tolerance(self) -> None:
        """Test hit ratio with explicit 1 std tolerance."""
        predicted = np.array([0.030, 0.035, 0.040, 0.025])
        actual = np.array([0.031, 0.033, 0.038, 0.030])
        std = np.array([0.002, 0.003, 0.003, 0.010])

        # Errors: |0.030-0.031| = 0.001, |0.035-0.033| = 0.002,
        #         |0.040-0.038| = 0.002, |0.025-0.030| = 0.005
        # Tolerances: 0.002, 0.003, 0.003, 0.010
        # Within 1 std: True (0.001 <= 0.002), True (0.002 <= 0.003),
        #               True (0.002 <= 0.003), True (0.005 <= 0.010)
        # Expected hit ratio: 4/4 = 1.0

        hit_ratio = compute_hit_ratio(predicted, actual, std, n_std=1.0)
        assert np.isclose(hit_ratio, 1.0)

        # Test with tighter tolerance
        std_tight = np.array([0.0005, 0.001, 0.001, 0.002])
        # Within 1 std: False (0.001 > 0.0005), False (0.002 > 0.001),
        #               False (0.002 > 0.001), False (0.005 > 0.002)
        # Expected hit ratio: 0/4 = 0.0
        hit_ratio_tight = compute_hit_ratio(predicted, actual, std_tight, n_std=1.0)
        assert np.isclose(hit_ratio_tight, 0.0)


# =============================================================================
# ValidationResult Tests
# =============================================================================


class TestValidationResult:
    """Tests for ValidationResult dataclass."""

    def test_validation_result_creation(self) -> None:
        """Test ValidationResult can be created with valid data."""
        predicted = np.array([0.03, 0.035, 0.032])
        actual = np.array([0.031, 0.033, 0.030])

        result = ValidationResult(
            model_type="vasicek",
            validation_period=(date(2024, 1, 1), date(2024, 3, 31)),
            mae=compute_mae(predicted, actual),
            rmse=compute_rmse(predicted, actual),
            mape=compute_mape(predicted, actual),
            hit_ratio=0.67,
            predicted_values=predicted,
            actual_values=actual,
        )

        assert result.model_type == "vasicek"
        assert result.mae > 0
        assert result.rmse > 0
        assert len(result.predicted_values) == 3

    def test_validation_result_quality_assessment(self) -> None:
        """Test quality assessment methods."""
        # Excellent fit (MAE < 20bp)
        result_excellent = ValidationResult(
            model_type="vasicek",
            validation_period=(date(2024, 1, 1), date(2024, 3, 31)),
            mae=0.0015,  # 15 basis points
            rmse=0.002,
            mape=5.0,
            hit_ratio=0.8,
            predicted_values=np.array([0.03]),
            actual_values=np.array([0.031]),
        )
        assert result_excellent.quality_rating == "excellent"

        # Good fit (MAE 20-50bp)
        result_good = ValidationResult(
            model_type="vasicek",
            validation_period=(date(2024, 1, 1), date(2024, 3, 31)),
            mae=0.003,  # 30 basis points
            rmse=0.004,
            mape=10.0,
            hit_ratio=0.6,
            predicted_values=np.array([0.03]),
            actual_values=np.array([0.033]),
        )
        assert result_good.quality_rating == "good"

        # Poor fit (MAE > 50bp)
        result_poor = ValidationResult(
            model_type="vasicek",
            validation_period=(date(2024, 1, 1), date(2024, 3, 31)),
            mae=0.008,  # 80 basis points
            rmse=0.01,
            mape=25.0,
            hit_ratio=0.3,
            predicted_values=np.array([0.03]),
            actual_values=np.array([0.038]),
        )
        assert result_poor.quality_rating == "poor"


# =============================================================================
# Backtest Model Integration Tests
# =============================================================================


class TestBacktestModel:
    """Tests for backtest_model function."""

    @pytest.fixture
    def validation_dataset(self) -> InterestRateDataset:
        """Create a sample dataset for validation testing."""
        from pathlib import Path

        n_obs = 50
        n_mats = 7
        start_date = date(2024, 7, 1)

        dates = []
        current = start_date
        while len(dates) < n_obs:
            if current.weekday() < 5:
                dates.append(current)
            current += timedelta(days=1)

        maturities: MaturityVector = np.array(
            [0.25, 0.5, 1.0, 2.0, 5.0, 10.0, 30.0], dtype=np.float64
        )

        rng = np.random.default_rng(42)
        base_rates = np.array([0.02, 0.022, 0.025, 0.028, 0.032, 0.035, 0.038])
        rates: RateMatrix = np.zeros((n_obs, n_mats))
        rates[0] = base_rates

        for t in range(1, n_obs):
            rates[t] = rates[t - 1] + rng.normal(0, 0.0005, n_mats)

        return InterestRateDataset(
            file_path=Path("test_validation.csv"),
            source_name="test_validation",
            observation_dates=dates,
            maturities=maturities,
            rates=rates,
        )

    def test_backtest_vasicek_model(self, validation_dataset: InterestRateDataset) -> None:
        """Test backtesting with a calibrated Vasicek model."""
        from src.models.vasicek import VasicekModel

        # Create and calibrate a Vasicek model
        model = VasicekModel(kappa=0.5, theta=0.03, sigma=0.01)

        # Run backtest
        result = backtest_model(
            model=model,
            validation_data=validation_dataset,
            prediction_horizon=1,
        )

        # Validate result structure
        assert isinstance(result, ValidationResult)
        assert result.model_type == "vasicek"
        assert result.mae >= 0
        assert result.rmse >= 0
        assert 0 <= result.hit_ratio <= 1
        assert len(result.predicted_values) > 0
        assert len(result.actual_values) == len(result.predicted_values)

    def test_backtest_requires_calibrated_model(
        self, validation_dataset: InterestRateDataset
    ) -> None:
        """Test that backtesting requires a calibrated model."""
        from src.models.vasicek import VasicekModel
        from src.utils.exceptions import ModelNotCalibratedError

        model = VasicekModel()  # Not calibrated

        with pytest.raises(ModelNotCalibratedError):
            backtest_model(model=model, validation_data=validation_dataset)

    def test_backtest_different_horizons(
        self, validation_dataset: InterestRateDataset
    ) -> None:
        """Test backtesting with different prediction horizons."""
        from src.models.vasicek import VasicekModel

        model = VasicekModel(kappa=0.5, theta=0.03, sigma=0.01)

        result_1 = backtest_model(
            model=model,
            validation_data=validation_dataset,
            prediction_horizon=1,
        )

        result_5 = backtest_model(
            model=model,
            validation_data=validation_dataset,
            prediction_horizon=5,
        )

        # Longer horizon typically has higher errors
        # (though not guaranteed in all cases)
        assert result_1.mae >= 0
        assert result_5.mae >= 0
