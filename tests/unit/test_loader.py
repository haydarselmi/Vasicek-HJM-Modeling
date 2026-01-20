"""Unit tests for data loading functionality.

Tests cover:
- CSV file parsing
- Date format handling
- Maturity extraction
- Error handling for invalid files
"""

from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.data.loader import (
    InterestRateDataset,
    YieldCurve,
    load_interest_rate_data,
)
from src.utils.exceptions import DataValidationError


class TestInterestRateDataset:
    """Tests for the InterestRateDataset dataclass."""

    def test_create_dataset_with_valid_data(
        self,
        sample_rate_matrix: np.ndarray,
        sample_maturities: np.ndarray,
        sample_observation_dates: list[date],
    ) -> None:
        """Test creating a dataset with valid data."""
        dataset = InterestRateDataset(
            file_path=Path("test.csv"),
            source_name="Test Source",
            observation_dates=sample_observation_dates,
            maturities=sample_maturities,
            rates=sample_rate_matrix,
        )

        assert dataset.file_path == Path("test.csv")
        assert dataset.source_name == "Test Source"
        assert len(dataset.observation_dates) == len(sample_observation_dates)
        assert len(dataset.maturities) == len(sample_maturities)
        assert dataset.rates.shape == sample_rate_matrix.shape
        assert dataset.loaded_at is not None

    def test_dataset_n_observations(
        self,
        sample_rate_matrix: np.ndarray,
        sample_maturities: np.ndarray,
        sample_observation_dates: list[date],
    ) -> None:
        """Test n_observations property."""
        dataset = InterestRateDataset(
            file_path=Path("test.csv"),
            source_name="Test",
            observation_dates=sample_observation_dates,
            maturities=sample_maturities,
            rates=sample_rate_matrix,
        )

        assert dataset.n_observations == len(sample_observation_dates)

    def test_dataset_n_maturities(
        self,
        sample_rate_matrix: np.ndarray,
        sample_maturities: np.ndarray,
        sample_observation_dates: list[date],
    ) -> None:
        """Test n_maturities property."""
        dataset = InterestRateDataset(
            file_path=Path("test.csv"),
            source_name="Test",
            observation_dates=sample_observation_dates,
            maturities=sample_maturities,
            rates=sample_rate_matrix,
        )

        assert dataset.n_maturities == len(sample_maturities)

    def test_dataset_date_range(
        self,
        sample_rate_matrix: np.ndarray,
        sample_maturities: np.ndarray,
        sample_observation_dates: list[date],
    ) -> None:
        """Test date_range property."""
        dataset = InterestRateDataset(
            file_path=Path("test.csv"),
            source_name="Test",
            observation_dates=sample_observation_dates,
            maturities=sample_maturities,
            rates=sample_rate_matrix,
        )

        start, end = dataset.date_range
        assert start == sample_observation_dates[0]
        assert end == sample_observation_dates[-1]


class TestYieldCurve:
    """Tests for the YieldCurve dataclass."""

    def test_create_yield_curve(self, sample_maturities: np.ndarray) -> None:
        """Test creating a yield curve."""
        rates = np.array([0.02, 0.022, 0.025, 0.028, 0.032, 0.035, 0.038])
        curve = YieldCurve(
            observation_date=date(2024, 1, 2),
            maturities=sample_maturities,
            rates=rates,
        )

        assert curve.observation_date == date(2024, 1, 2)
        assert len(curve.maturities) == 7
        assert len(curve.rates) == 7

    def test_yield_curve_n_points(self, sample_maturities: np.ndarray) -> None:
        """Test n_points property."""
        rates = np.array([0.02, 0.022, 0.025, 0.028, 0.032, 0.035, 0.038])
        curve = YieldCurve(
            observation_date=date(2024, 1, 2),
            maturities=sample_maturities,
            rates=rates,
        )

        assert curve.n_points == 7


class TestLoadInterestRateData:
    """Tests for the load_interest_rate_data function."""

    def test_load_valid_csv(self, tmp_path: Path) -> None:
        """Test loading a valid CSV file."""
        # Create a test CSV file
        csv_path = tmp_path / "test_rates.csv"
        dates = pd.date_range("2024-01-02", periods=50, freq="B")
        maturities = ["3M", "6M", "1Y", "2Y", "5Y", "10Y", "30Y"]

        data = {"date": dates}
        for i, mat in enumerate(maturities):
            data[mat] = 0.02 + 0.005 * i + np.random.randn(50) * 0.001

        df = pd.DataFrame(data)
        df.to_csv(csv_path, index=False)

        # Load the data
        dataset = load_interest_rate_data(csv_path)

        assert dataset.n_observations == 50
        assert dataset.n_maturities == 7
        assert dataset.source_name is not None

    def test_load_nonexistent_file(self) -> None:
        """Test error when loading a non-existent file."""
        with pytest.raises(FileNotFoundError):
            load_interest_rate_data(Path("/nonexistent/path.csv"))

    def test_load_with_custom_date_format(self, tmp_path: Path) -> None:
        """Test loading CSV with custom date format."""
        csv_path = tmp_path / "test_rates.csv"
        dates = ["01/02/2024", "01/03/2024", "01/04/2024"]

        data = {
            "date": dates,
            "3M": [0.02, 0.021, 0.022],
            "6M": [0.025, 0.026, 0.027],
        }
        df = pd.DataFrame(data)
        df.to_csv(csv_path, index=False)

        dataset = load_interest_rate_data(
            csv_path,
            date_format="%m/%d/%Y",
        )

        assert dataset.n_observations == 3

    def test_load_with_custom_source_name(self, tmp_path: Path) -> None:
        """Test loading with custom source name."""
        csv_path = tmp_path / "test_rates.csv"
        dates = pd.date_range("2024-01-02", periods=10, freq="B")

        data = {"date": dates, "3M": [0.02] * 10, "6M": [0.025] * 10}
        df = pd.DataFrame(data)
        df.to_csv(csv_path, index=False)

        dataset = load_interest_rate_data(
            csv_path,
            source_name="Custom Source",
        )

        assert dataset.source_name == "Custom Source"

    def test_maturity_extraction_standard_formats(self, tmp_path: Path) -> None:
        """Test that standard maturity formats are correctly parsed."""
        csv_path = tmp_path / "test_rates.csv"
        dates = pd.date_range("2024-01-02", periods=10, freq="B")

        # Various maturity formats
        data = {
            "date": dates,
            "3M": [0.02] * 10,  # 3 months = 0.25 years
            "6M": [0.025] * 10,  # 6 months = 0.5 years
            "1Y": [0.03] * 10,  # 1 year
            "2Y": [0.035] * 10,  # 2 years
            "10Y": [0.04] * 10,  # 10 years
        }
        df = pd.DataFrame(data)
        df.to_csv(csv_path, index=False)

        dataset = load_interest_rate_data(csv_path)

        # Check maturities are correctly parsed
        expected = [0.25, 0.5, 1.0, 2.0, 10.0]
        np.testing.assert_array_almost_equal(dataset.maturities, expected)


class TestLoadInterestRateDataEdgeCases:
    """Edge case tests for data loading."""

    def test_empty_csv_raises_error(self, tmp_path: Path) -> None:
        """Test that an empty CSV raises an error."""
        csv_path = tmp_path / "empty.csv"
        csv_path.write_text("date,3M,6M\n")

        with pytest.raises(DataValidationError):
            load_interest_rate_data(csv_path)

    def test_missing_date_column_raises_error(self, tmp_path: Path) -> None:
        """Test that missing date column raises an error."""
        csv_path = tmp_path / "no_date.csv"
        csv_path.write_text("3M,6M\n0.02,0.03\n0.021,0.031\n")

        with pytest.raises(DataValidationError):
            load_interest_rate_data(csv_path)

    def test_single_maturity_column(self, tmp_path: Path) -> None:
        """Test loading with a single maturity column."""
        csv_path = tmp_path / "single_mat.csv"
        dates = pd.date_range("2024-01-02", periods=50, freq="B")

        data = {"date": dates, "3M": [0.02 + i * 0.0001 for i in range(50)]}
        df = pd.DataFrame(data)
        df.to_csv(csv_path, index=False)

        dataset = load_interest_rate_data(csv_path)
        assert dataset.n_maturities == 1

