"""Integration tests for PCA workflow.

This module tests the complete PCA analysis workflow:
load data → compute changes → PCA → interpret → visualize
"""

from datetime import date
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

# Use non-interactive backend for tests
matplotlib.use("Agg")

from src.analysis.pca import (
    compute_pca,
    compute_yield_curve_changes,
    reconstruct_curve,
)
from src.data.loader import InterestRateDataset, load_interest_rate_data
from src.visualization.pca_plots import plot_component_loadings, plot_explained_variance


class TestPCAWorkflow:
    """Integration tests for the complete PCA analysis workflow."""

    def test_full_pca_workflow_with_sample_data(
        self,
        sample_rate_matrix: np.ndarray,
        sample_maturities: np.ndarray,
        sample_observation_dates: list[date],
    ) -> None:
        """Test complete workflow: load data → compute changes → PCA → interpret → visualize."""
        # Step 1: Create dataset (simulating load)
        dataset = InterestRateDataset(
            file_path=Path("test.csv"),
            source_name="test",
            observation_dates=sample_observation_dates,
            maturities=sample_maturities,
            rates=sample_rate_matrix,
        )

        # Step 2: Compute yield curve changes
        changes = compute_yield_curve_changes(dataset)
        assert len(changes) == len(sample_observation_dates) - 1
        assert len(changes) > 0

        # Step 3: Perform PCA
        pca_result = compute_pca(dataset, n_components=3, use_changes=True)

        # Verify PCA results
        assert pca_result.n_components == 3
        assert len(pca_result.explained_variance_ratio) == 3
        assert len(pca_result.eigenvalues) == 3
        assert pca_result.loadings.shape == (len(sample_maturities), 3)

        # PC1 should explain most variance (typically ~85%)
        assert pca_result.explained_variance_ratio[0] > 0.5, (
            f"PC1 explains only {pca_result.explained_variance_ratio[0] * 100:.1f}%"
        )

        # Step 4: Interpret components
        interpretations = pca_result.interpretations
        assert len(interpretations) == 3
        assert all(isinstance(interp, str) for interp in interpretations)
        assert all(len(interp) > 0 for interp in interpretations)

        # Step 5: Test reconstruction
        original_change = changes[0].changes
        reconstructed = reconstruct_curve(pca_result, n_components=3, curve_changes=original_change)
        assert reconstructed.shape == original_change.shape

        # Step 6: Visualize (verify functions don't raise)
        fig1 = plot_explained_variance(pca_result)
        assert isinstance(fig1, plt.Figure)
        plt.close(fig1)

        fig2 = plot_component_loadings(pca_result)
        assert isinstance(fig2, plt.Figure)
        plt.close(fig2)

    def test_pca_workflow_with_csv_file(
        self,
        tmp_path: Path,
        sample_rate_matrix: np.ndarray,
        sample_maturities: np.ndarray,
        sample_observation_dates: list[date],
    ) -> None:
        """Test workflow loading from actual CSV file."""
        import pandas as pd

        # Create sample CSV file
        csv_path = tmp_path / "test_rates.csv"

        # Create DataFrame with proper column names
        data = {"date": [d.strftime("%Y-%m-%d") for d in sample_observation_dates]}
        maturity_names = ["3M", "6M", "1Y", "2Y", "5Y", "10Y", "30Y"]
        for i, name in enumerate(maturity_names):
            data[name] = sample_rate_matrix[:, i]

        df = pd.DataFrame(data)
        df.to_csv(csv_path, index=False)

        # Load data from CSV
        dataset = load_interest_rate_data(csv_path)

        # Verify load
        assert dataset.n_observations == len(sample_observation_dates)
        assert dataset.n_maturities == len(sample_maturities)

        # Run PCA
        pca_result = compute_pca(dataset, n_components=None, use_changes=True)

        # Verify results
        assert pca_result.n_components == len(sample_maturities)
        assert np.sum(pca_result.explained_variance_ratio) > 0.99

    def test_expected_variance_distribution(
        self,
        sample_rate_matrix: np.ndarray,
        sample_maturities: np.ndarray,
        sample_observation_dates: list[date],
    ) -> None:
        """Test that variance distribution matches expected pattern for realistic data.

        For typical yield curve data:
        - PC1 (level) explains ~85% of variance
        - PC2 (slope) explains ~10% of variance
        - PC3 (curvature) explains ~3% of variance

        Note: Simulated data may vary; we use a relaxed threshold of 80%.
        """
        dataset = InterestRateDataset(
            file_path=Path("test.csv"),
            source_name="test",
            observation_dates=sample_observation_dates,
            maturities=sample_maturities,
            rates=sample_rate_matrix,
        )

        pca_result = compute_pca(dataset, n_components=3, use_changes=True)

        # Check that first 3 components explain most variance
        # Using 80% threshold as simulated data may have more noise than real data
        cumulative_3 = pca_result.cumulative_variance[2]
        assert cumulative_3 > 0.80, (
            f"First 3 PCs explain only {cumulative_3 * 100:.1f}% (expected >80%)"
        )

        # PC1 should dominate
        assert pca_result.explained_variance_ratio[0] > pca_result.explained_variance_ratio[1]
        assert pca_result.explained_variance_ratio[1] > pca_result.explained_variance_ratio[2]

    def test_reconstruction_quality_with_fewer_components(
        self,
        sample_rate_matrix: np.ndarray,
        sample_maturities: np.ndarray,
        sample_observation_dates: list[date],
    ) -> None:
        """Test that reconstruction quality improves with more components."""
        dataset = InterestRateDataset(
            file_path=Path("test.csv"),
            source_name="test",
            observation_dates=sample_observation_dates,
            maturities=sample_maturities,
            rates=sample_rate_matrix,
        )

        pca_result = compute_pca(dataset, n_components=None, use_changes=True)
        changes = compute_yield_curve_changes(dataset)

        # Calculate reconstruction error for different numbers of components
        original = changes[10].changes  # Use a middle observation
        errors = []

        for n_comp in range(1, pca_result.n_components + 1):
            reconstructed = reconstruct_curve(
                pca_result, n_components=n_comp, curve_changes=original
            )
            error = np.sqrt(np.mean((original - reconstructed) ** 2))
            errors.append(error)

        # Errors should be monotonically decreasing
        for i in range(len(errors) - 1):
            assert errors[i] >= errors[i + 1] - 1e-10, (
                f"Error increased from {errors[i]} to {errors[i + 1]}"
            )

        # Full reconstruction should have near-zero error
        assert errors[-1] < 1e-10, f"Full reconstruction error too large: {errors[-1]}"

    def test_pca_interpretation_output(
        self,
        sample_rate_matrix: np.ndarray,
        sample_maturities: np.ndarray,
        sample_observation_dates: list[date],
    ) -> None:
        """Test that component interpretations are meaningful."""
        dataset = InterestRateDataset(
            file_path=Path("test.csv"),
            source_name="test",
            observation_dates=sample_observation_dates,
            maturities=sample_maturities,
            rates=sample_rate_matrix,
        )

        pca_result = compute_pca(dataset, n_components=3, use_changes=True)

        # Should have 3 interpretations
        assert len(pca_result.interpretations) == 3

        # Each interpretation should mention the PC number
        for i, interp in enumerate(pca_result.interpretations):
            assert f"PC{i + 1}" in interp, f"Missing PC number in: {interp}"

        # Interpretations should contain meaningful terms
        all_interps = " ".join(pca_result.interpretations).lower()
        meaningful_terms = ["level", "slope", "curvature", "shift", "tilt", "butterfly", "pattern"]
        has_meaningful_term = any(term in all_interps for term in meaningful_terms)
        assert has_meaningful_term, f"No meaningful terms found in: {pca_result.interpretations}"

    def test_visualization_content(
        self,
        sample_rate_matrix: np.ndarray,
        sample_maturities: np.ndarray,
        sample_observation_dates: list[date],
    ) -> None:
        """Test that visualizations contain expected elements."""
        dataset = InterestRateDataset(
            file_path=Path("test.csv"),
            source_name="test",
            observation_dates=sample_observation_dates,
            maturities=sample_maturities,
            rates=sample_rate_matrix,
        )

        pca_result = compute_pca(dataset, n_components=5, use_changes=True)

        # Test explained variance plot
        fig1 = plot_explained_variance(pca_result)
        axes = fig1.get_axes()
        assert len(axes) >= 1, "Explained variance plot has no axes"

        # Should have bars on primary axis
        ax = axes[0]
        bars = [
            child for child in ax.get_children() if isinstance(child, matplotlib.patches.Rectangle)
        ]
        assert len(bars) > 0, "No bars in explained variance plot"

        # Cumulative line is on the secondary axis (twinx)
        # Check all axes for lines
        all_lines = []
        for axis in axes:
            all_lines.extend(axis.get_lines())
        assert len(all_lines) > 0, "No cumulative line in explained variance plot"
        plt.close(fig1)

        # Test loadings plot
        fig2 = plot_component_loadings(pca_result, n_components=3)
        axes2 = fig2.get_axes()
        assert len(axes2) >= 1, "Loadings plot has no axes"

        # Should have lines for each component
        ax2 = axes2[0]
        lines2 = ax2.get_lines()
        # We expect 3 component lines + 1 zero reference line = 4 lines minimum
        assert len(lines2) >= 3, f"Expected at least 3 loading lines, got {len(lines2)}"
        plt.close(fig2)
