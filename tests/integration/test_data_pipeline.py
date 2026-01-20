"""Integration tests for the data loading pipeline.

Tests the complete workflow:
load CSV → validate → compute statistics → visualize
"""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest


@pytest.mark.integration
class TestDataPipeline:
    """Integration tests for the complete data pipeline."""

    def test_full_pipeline_with_valid_data(self, tmp_path: Path) -> None:
        """Test the complete pipeline: load → validate → statistics → visualize."""
        from src.analysis.statistics import compute_correlation_matrix, compute_summary_statistics
        from src.data.loader import load_interest_rate_data
        from src.data.validator import validate_dataset
        from src.visualization.yield_curves import plot_yield_curves

        # Create test CSV
        csv_path = tmp_path / "test_rates.csv"
        dates = pd.date_range("2024-01-02", periods=100, freq="B")
        maturities = ["3M", "6M", "1Y", "2Y", "5Y", "10Y", "30Y"]

        rng = np.random.default_rng(42)
        data = {"date": dates}
        for i, mat in enumerate(maturities):
            data[mat] = 0.02 + 0.005 * i + rng.standard_normal(100) * 0.001

        df = pd.DataFrame(data)
        df.to_csv(csv_path, index=False)

        # Step 1: Load data
        dataset = load_interest_rate_data(csv_path, source_name="Test Treasury Rates")
        assert dataset.n_observations == 100
        assert dataset.n_maturities == 7

        # Step 2: Validate data
        report = validate_dataset(dataset)
        assert report.is_valid is True

        # Step 3: Compute statistics
        stats = compute_summary_statistics(dataset)
        assert len(stats.means) == 7
        assert all(stats.stds > 0)

        # Step 4: Compute correlation matrix
        corr = compute_correlation_matrix(dataset)
        assert corr.shape == (7, 7)
        assert np.allclose(np.diag(corr), 1.0)

        # Step 5: Create visualization
        fig = plot_yield_curves(dataset, n_curves=5)
        assert fig is not None

        # Cleanup
        import matplotlib.pyplot as plt
        plt.close(fig)

    def test_pipeline_with_missing_values(self, tmp_path: Path) -> None:
        """Test pipeline handles missing values correctly."""
        from src.analysis.statistics import compute_summary_statistics
        from src.data.loader import load_interest_rate_data
        from src.data.validator import handle_missing_values, validate_dataset

        # Create test CSV with some missing values
        csv_path = tmp_path / "test_rates_missing.csv"
        dates = pd.date_range("2024-01-02", periods=100, freq="B")
        maturities = ["3M", "6M", "1Y", "2Y"]

        rng = np.random.default_rng(42)
        data = {"date": dates}
        for i, mat in enumerate(maturities):
            values = 0.02 + 0.005 * i + rng.standard_normal(100) * 0.001
            # Add some missing values (less than 5%)
            values[10] = np.nan
            values[50] = np.nan
            data[mat] = values

        df = pd.DataFrame(data)
        df.to_csv(csv_path, index=False)

        # Load and validate
        dataset = load_interest_rate_data(csv_path)
        report = validate_dataset(dataset)

        # Should have warnings about missing values
        assert any("missing" in w.lower() or "nan" in w.lower() for w in report.warnings)

        # Handle missing values
        clean_dataset = handle_missing_values(dataset, method="interpolate")
        assert not np.any(np.isnan(clean_dataset.rates))

        # Compute statistics on clean data
        stats = compute_summary_statistics(clean_dataset)
        assert all(np.isfinite(stats.means))

    def test_pipeline_preserves_data_integrity(self, tmp_path: Path) -> None:
        """Test that the pipeline doesn't modify original data."""
        from src.analysis.statistics import compute_summary_statistics
        from src.data.loader import load_interest_rate_data
        from src.data.validator import validate_dataset

        # Create test CSV
        csv_path = tmp_path / "test_rates.csv"
        dates = pd.date_range("2024-01-02", periods=50, freq="B")

        rng = np.random.default_rng(42)
        original_rates = {
            "3M": 0.02 + rng.standard_normal(50) * 0.001,
            "6M": 0.025 + rng.standard_normal(50) * 0.001,
        }

        data = {"date": dates, **original_rates}
        df = pd.DataFrame(data)
        df.to_csv(csv_path, index=False)

        # Process through pipeline
        dataset = load_interest_rate_data(csv_path)
        original_shape = dataset.rates.shape
        original_values = dataset.rates.copy()

        validate_dataset(dataset)
        compute_summary_statistics(dataset)

        # Verify data unchanged
        assert dataset.rates.shape == original_shape
        np.testing.assert_array_equal(dataset.rates, original_values)

    def test_pipeline_date_range_filtering(self, tmp_path: Path) -> None:
        """Test that we can work with date subsets."""
        from datetime import date

        from src.data.loader import load_interest_rate_data

        # Create test CSV
        csv_path = tmp_path / "test_rates.csv"
        dates = pd.date_range("2024-01-02", periods=100, freq="B")

        rng = np.random.default_rng(42)
        data = {
            "date": dates,
            "3M": 0.02 + rng.standard_normal(100) * 0.001,
            "6M": 0.025 + rng.standard_normal(100) * 0.001,
        }

        df = pd.DataFrame(data)
        df.to_csv(csv_path, index=False)

        # Load data
        dataset = load_interest_rate_data(csv_path)

        # Verify date range
        start, end = dataset.date_range
        assert isinstance(start, date)
        assert isinstance(end, date)
        assert start < end

