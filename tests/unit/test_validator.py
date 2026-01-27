"""Unit tests for data validation functionality.

Tests cover:
- Missing value detection
- Rate bounds checking
- Chronological order validation
- Missing value handling strategies
"""

from datetime import date
from pathlib import Path

import numpy as np
import pytest

from src.data.loader import InterestRateDataset
from src.data.validator import (
    ValidationReport,
    handle_missing_values,
    validate_dataset,
)
from src.utils.exceptions import DataValidationError


class TestValidationReport:
    """Tests for the ValidationReport dataclass."""

    def test_create_valid_report(self) -> None:
        """Test creating a valid report."""
        report = ValidationReport(
            warnings=["Minor issue found"],
            errors=[],
            is_valid=True,
        )

        assert report.is_valid is True
        assert len(report.warnings) == 1
        assert len(report.errors) == 0

    def test_create_invalid_report(self) -> None:
        """Test creating an invalid report."""
        report = ValidationReport(
            warnings=[],
            errors=["Critical error"],
            is_valid=False,
        )

        assert report.is_valid is False
        assert len(report.errors) == 1


class TestValidateDataset:
    """Tests for the validate_dataset function."""

    def test_validate_valid_dataset(
        self,
        sample_rate_matrix: np.ndarray,
        sample_maturities: np.ndarray,
        sample_observation_dates: list[date],
    ) -> None:
        """Test validating a valid dataset."""
        dataset = InterestRateDataset(
            file_path=Path("test.csv"),
            source_name="Test",
            observation_dates=sample_observation_dates,
            maturities=sample_maturities,
            rates=sample_rate_matrix,
        )

        report = validate_dataset(dataset)

        assert report.is_valid is True
        assert len(report.errors) == 0

    def test_validate_detects_nan_values(
        self,
        sample_maturities: np.ndarray,
        sample_observation_dates: list[date],
    ) -> None:
        """Test that NaN values are detected."""
        rates = np.random.randn(len(sample_observation_dates), len(sample_maturities)) * 0.01 + 0.03
        rates[10, 3] = np.nan  # Introduce a NaN

        dataset = InterestRateDataset(
            file_path=Path("test.csv"),
            source_name="Test",
            observation_dates=sample_observation_dates,
            maturities=sample_maturities,
            rates=rates,
        )

        report = validate_dataset(dataset)

        assert any("missing" in w.lower() or "nan" in w.lower() for w in report.warnings)

    def test_validate_detects_out_of_bounds_rates(
        self,
        sample_maturities: np.ndarray,
        sample_observation_dates: list[date],
    ) -> None:
        """Test that out-of-bounds rates generate warnings."""
        rates = np.random.randn(len(sample_observation_dates), len(sample_maturities)) * 0.01 + 0.03
        rates[5, 2] = 0.60  # 60% - out of typical bounds

        dataset = InterestRateDataset(
            file_path=Path("test.csv"),
            source_name="Test",
            observation_dates=sample_observation_dates,
            maturities=sample_maturities,
            rates=rates,
        )

        report = validate_dataset(dataset)

        # Should have a warning about out-of-bounds rate
        assert len(report.warnings) > 0

    def test_validate_detects_non_chronological_dates(
        self,
        sample_rate_matrix: np.ndarray,
        sample_maturities: np.ndarray,
    ) -> None:
        """Test that non-chronological dates are detected and raise an error."""
        # Create dates that are not in order
        dates = [
            date(2024, 1, 2),
            date(2024, 1, 5),
            date(2024, 1, 3),  # Out of order!
        ]
        rates = sample_rate_matrix[:3]

        dataset = InterestRateDataset(
            file_path=Path("test.csv"),
            source_name="Test",
            observation_dates=dates,
            maturities=sample_maturities,
            rates=rates,
        )

        with pytest.raises(DataValidationError) as exc_info:
            validate_dataset(dataset)

        error_msg = str(exc_info.value).lower()
        assert "chronological" in error_msg or "order" in error_msg

    def test_validate_strict_mode_raises_on_warnings(
        self,
        sample_maturities: np.ndarray,
        sample_observation_dates: list[date],
    ) -> None:
        """Test that strict mode raises on warnings."""
        rates = np.random.randn(len(sample_observation_dates), len(sample_maturities)) * 0.01 + 0.03
        rates[5, 2] = 0.60  # Out of typical bounds

        dataset = InterestRateDataset(
            file_path=Path("test.csv"),
            source_name="Test",
            observation_dates=sample_observation_dates,
            maturities=sample_maturities,
            rates=rates,
        )

        with pytest.raises(DataValidationError):
            validate_dataset(dataset, strict=True)


class TestHandleMissingValues:
    """Tests for the handle_missing_values function."""

    def test_interpolate_missing_values(
        self,
        sample_maturities: np.ndarray,
        sample_observation_dates: list[date],
    ) -> None:
        """Test interpolation of missing values."""
        rates = np.random.randn(len(sample_observation_dates), len(sample_maturities)) * 0.01 + 0.03
        rates[10, 3] = np.nan  # Introduce a NaN

        dataset = InterestRateDataset(
            file_path=Path("test.csv"),
            source_name="Test",
            observation_dates=sample_observation_dates,
            maturities=sample_maturities,
            rates=rates,
        )

        result = handle_missing_values(dataset, method="interpolate")

        # Check no NaN values remain
        assert not np.any(np.isnan(result.rates))

    def test_forward_fill_missing_values(
        self,
        sample_maturities: np.ndarray,
        sample_observation_dates: list[date],
    ) -> None:
        """Test forward fill of missing values."""
        rates = np.random.randn(len(sample_observation_dates), len(sample_maturities)) * 0.01 + 0.03
        rates[10, 3] = np.nan

        dataset = InterestRateDataset(
            file_path=Path("test.csv"),
            source_name="Test",
            observation_dates=sample_observation_dates,
            maturities=sample_maturities,
            rates=rates,
        )

        result = handle_missing_values(dataset, method="forward_fill")

        assert not np.any(np.isnan(result.rates))

    def test_drop_rows_with_missing_values(
        self,
        sample_maturities: np.ndarray,
        sample_observation_dates: list[date],
    ) -> None:
        """Test dropping rows with missing values."""
        rates = np.random.randn(len(sample_observation_dates), len(sample_maturities)) * 0.01 + 0.03
        rates[10, 3] = np.nan

        dataset = InterestRateDataset(
            file_path=Path("test.csv"),
            source_name="Test",
            observation_dates=sample_observation_dates,
            maturities=sample_maturities,
            rates=rates,
        )

        result = handle_missing_values(dataset, method="drop")

        assert result.n_observations == len(sample_observation_dates) - 1
        assert not np.any(np.isnan(result.rates))

    def test_too_many_missing_values_raises_error(
        self,
        sample_maturities: np.ndarray,
        sample_observation_dates: list[date],
    ) -> None:
        """Test that too many missing values raises an error."""
        rates = np.random.randn(len(sample_observation_dates), len(sample_maturities)) * 0.01 + 0.03

        # Make 10% of values NaN (exceeds 5% threshold)
        n_missing = int(0.10 * rates.size)
        flat_indices = np.random.choice(rates.size, n_missing, replace=False)
        rates.flat[flat_indices] = np.nan

        dataset = InterestRateDataset(
            file_path=Path("test.csv"),
            source_name="Test",
            observation_dates=sample_observation_dates,
            maturities=sample_maturities,
            rates=rates,
        )

        with pytest.raises(DataValidationError):
            handle_missing_values(dataset)

