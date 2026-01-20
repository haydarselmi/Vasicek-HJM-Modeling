"""Unit tests for statistics functionality.

Tests cover:
- Mean, std, min, max calculations
- Correlation matrix computation
- Summary statistics dataclass
"""

from datetime import date
from pathlib import Path

import numpy as np

from src.analysis.statistics import (
    SummaryStatistics,
    compute_correlation_matrix,
    compute_summary_statistics,
)
from src.data.loader import InterestRateDataset


class TestSummaryStatistics:
    """Tests for the SummaryStatistics dataclass."""

    def test_create_summary_statistics(self, sample_maturities: np.ndarray) -> None:
        """Test creating summary statistics."""
        n_maturities = len(sample_maturities)
        stats = SummaryStatistics(
            means=np.array([0.03] * n_maturities),
            stds=np.array([0.01] * n_maturities),
            mins=np.array([0.01] * n_maturities),
            maxs=np.array([0.05] * n_maturities),
            correlations=np.eye(n_maturities),
            maturities=sample_maturities,
        )

        assert len(stats.means) == n_maturities
        assert len(stats.stds) == n_maturities
        assert stats.correlations.shape == (n_maturities, n_maturities)


class TestComputeSummaryStatistics:
    """Tests for the compute_summary_statistics function."""

    def test_compute_stats_for_valid_dataset(
        self,
        sample_rate_matrix: np.ndarray,
        sample_maturities: np.ndarray,
        sample_observation_dates: list[date],
    ) -> None:
        """Test computing statistics for a valid dataset."""
        dataset = InterestRateDataset(
            file_path=Path("test.csv"),
            source_name="Test",
            observation_dates=sample_observation_dates,
            maturities=sample_maturities,
            rates=sample_rate_matrix,
        )

        stats = compute_summary_statistics(dataset)

        assert len(stats.means) == len(sample_maturities)
        assert len(stats.stds) == len(sample_maturities)
        assert len(stats.mins) == len(sample_maturities)
        assert len(stats.maxs) == len(sample_maturities)

    def test_means_are_correct(
        self,
        sample_rate_matrix: np.ndarray,
        sample_maturities: np.ndarray,
        sample_observation_dates: list[date],
    ) -> None:
        """Test that means are correctly computed."""
        dataset = InterestRateDataset(
            file_path=Path("test.csv"),
            source_name="Test",
            observation_dates=sample_observation_dates,
            maturities=sample_maturities,
            rates=sample_rate_matrix,
        )

        stats = compute_summary_statistics(dataset)

        expected_means = np.mean(sample_rate_matrix, axis=0)
        np.testing.assert_array_almost_equal(stats.means, expected_means)

    def test_stds_are_correct(
        self,
        sample_rate_matrix: np.ndarray,
        sample_maturities: np.ndarray,
        sample_observation_dates: list[date],
    ) -> None:
        """Test that standard deviations are correctly computed."""
        dataset = InterestRateDataset(
            file_path=Path("test.csv"),
            source_name="Test",
            observation_dates=sample_observation_dates,
            maturities=sample_maturities,
            rates=sample_rate_matrix,
        )

        stats = compute_summary_statistics(dataset)

        expected_stds = np.std(sample_rate_matrix, axis=0, ddof=1)
        np.testing.assert_array_almost_equal(stats.stds, expected_stds)

    def test_mins_are_correct(
        self,
        sample_rate_matrix: np.ndarray,
        sample_maturities: np.ndarray,
        sample_observation_dates: list[date],
    ) -> None:
        """Test that minimums are correctly computed."""
        dataset = InterestRateDataset(
            file_path=Path("test.csv"),
            source_name="Test",
            observation_dates=sample_observation_dates,
            maturities=sample_maturities,
            rates=sample_rate_matrix,
        )

        stats = compute_summary_statistics(dataset)

        expected_mins = np.min(sample_rate_matrix, axis=0)
        np.testing.assert_array_almost_equal(stats.mins, expected_mins)

    def test_maxs_are_correct(
        self,
        sample_rate_matrix: np.ndarray,
        sample_maturities: np.ndarray,
        sample_observation_dates: list[date],
    ) -> None:
        """Test that maximums are correctly computed."""
        dataset = InterestRateDataset(
            file_path=Path("test.csv"),
            source_name="Test",
            observation_dates=sample_observation_dates,
            maturities=sample_maturities,
            rates=sample_rate_matrix,
        )

        stats = compute_summary_statistics(dataset)

        expected_maxs = np.max(sample_rate_matrix, axis=0)
        np.testing.assert_array_almost_equal(stats.maxs, expected_maxs)


class TestComputeCorrelationMatrix:
    """Tests for the compute_correlation_matrix function."""

    def test_correlation_matrix_shape(
        self,
        sample_rate_matrix: np.ndarray,
        sample_maturities: np.ndarray,
        sample_observation_dates: list[date],
    ) -> None:
        """Test that correlation matrix has correct shape."""
        dataset = InterestRateDataset(
            file_path=Path("test.csv"),
            source_name="Test",
            observation_dates=sample_observation_dates,
            maturities=sample_maturities,
            rates=sample_rate_matrix,
        )

        corr = compute_correlation_matrix(dataset)

        n_mat = len(sample_maturities)
        assert corr.shape == (n_mat, n_mat)

    def test_correlation_matrix_diagonal_is_one(
        self,
        sample_rate_matrix: np.ndarray,
        sample_maturities: np.ndarray,
        sample_observation_dates: list[date],
    ) -> None:
        """Test that diagonal of correlation matrix is 1."""
        dataset = InterestRateDataset(
            file_path=Path("test.csv"),
            source_name="Test",
            observation_dates=sample_observation_dates,
            maturities=sample_maturities,
            rates=sample_rate_matrix,
        )

        corr = compute_correlation_matrix(dataset)

        np.testing.assert_array_almost_equal(np.diag(corr), np.ones(len(sample_maturities)))

    def test_correlation_matrix_is_symmetric(
        self,
        sample_rate_matrix: np.ndarray,
        sample_maturities: np.ndarray,
        sample_observation_dates: list[date],
    ) -> None:
        """Test that correlation matrix is symmetric."""
        dataset = InterestRateDataset(
            file_path=Path("test.csv"),
            source_name="Test",
            observation_dates=sample_observation_dates,
            maturities=sample_maturities,
            rates=sample_rate_matrix,
        )

        corr = compute_correlation_matrix(dataset)

        np.testing.assert_array_almost_equal(corr, corr.T)

    def test_correlation_values_in_valid_range(
        self,
        sample_rate_matrix: np.ndarray,
        sample_maturities: np.ndarray,
        sample_observation_dates: list[date],
    ) -> None:
        """Test that correlation values are between -1 and 1."""
        dataset = InterestRateDataset(
            file_path=Path("test.csv"),
            source_name="Test",
            observation_dates=sample_observation_dates,
            maturities=sample_maturities,
            rates=sample_rate_matrix,
        )

        corr = compute_correlation_matrix(dataset)

        assert np.all(corr >= -1.0)
        assert np.all(corr <= 1.0)

    def test_nearby_maturities_are_highly_correlated(
        self,
        sample_rate_matrix: np.ndarray,
        sample_maturities: np.ndarray,
        sample_observation_dates: list[date],
    ) -> None:
        """Test that nearby maturities have higher correlation than distant ones."""
        dataset = InterestRateDataset(
            file_path=Path("test.csv"),
            source_name="Test",
            observation_dates=sample_observation_dates,
            maturities=sample_maturities,
            rates=sample_rate_matrix,
        )

        corr = compute_correlation_matrix(dataset)

        # Adjacent maturities (e.g., 3M and 6M) should be more correlated
        # than distant ones (e.g., 3M and 30Y)
        assert corr[0, 1] > corr[0, -1]

