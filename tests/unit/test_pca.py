"""Unit tests for PCA computation on yield curve data.

This module tests the Principal Component Analysis functionality including:
- Eigenvalue ordering and orthogonality
- Explained variance calculations
- Component interpretation (level, slope, curvature)
- Curve reconstruction from reduced components
"""

from datetime import date

import numpy as np
from hypothesis import given, settings
from hypothesis import strategies as st

from src.analysis.pca import (
    PCAResult,
    compute_pca,
    compute_yield_curve_changes,
    interpret_components,
    reconstruct_curve,
)
from src.data.loader import InterestRateDataset

# =============================================================================
# T058: Unit tests for PCA computation
# =============================================================================


class TestPCAComputation:
    """Tests for PCA computation on yield curve data."""

    def test_eigenvalue_ordering(
        self,
        sample_rate_matrix: np.ndarray,
        sample_maturities: np.ndarray,
        sample_observation_dates: list[date],
    ) -> None:
        """Test that eigenvalues are returned in descending order."""
        from pathlib import Path

        dataset = InterestRateDataset(
            file_path=Path("test.csv"),
            source_name="test",
            observation_dates=sample_observation_dates,
            maturities=sample_maturities,
            rates=sample_rate_matrix,
        )

        result = compute_pca(dataset, n_components=None, use_changes=True)

        # Eigenvalues should be in descending order
        eigenvalues = result.eigenvalues
        for i in range(len(eigenvalues) - 1):
            assert eigenvalues[i] >= eigenvalues[i + 1], (
                f"Eigenvalues not in descending order: {eigenvalues}"
            )

    def test_loadings_orthogonality(
        self,
        sample_rate_matrix: np.ndarray,
        sample_maturities: np.ndarray,
        sample_observation_dates: list[date],
    ) -> None:
        """Test that PCA loadings are orthogonal."""
        from pathlib import Path

        dataset = InterestRateDataset(
            file_path=Path("test.csv"),
            source_name="test",
            observation_dates=sample_observation_dates,
            maturities=sample_maturities,
            rates=sample_rate_matrix,
        )

        result = compute_pca(dataset, n_components=None, use_changes=True)

        # Loadings should be orthonormal: L.T @ L ≈ I
        loadings = result.loadings
        product = loadings.T @ loadings
        identity = np.eye(loadings.shape[1])

        np.testing.assert_array_almost_equal(
            product, identity, decimal=10, err_msg="Loadings are not orthonormal"
        )

    def test_explained_variance_sum(
        self,
        sample_rate_matrix: np.ndarray,
        sample_maturities: np.ndarray,
        sample_observation_dates: list[date],
    ) -> None:
        """Test that explained variance ratios sum to approximately 1."""
        from pathlib import Path

        dataset = InterestRateDataset(
            file_path=Path("test.csv"),
            source_name="test",
            observation_dates=sample_observation_dates,
            maturities=sample_maturities,
            rates=sample_rate_matrix,
        )

        result = compute_pca(dataset, n_components=None, use_changes=True)

        # All variance ratios should sum to 1.0 (when all components retained)
        total_variance = np.sum(result.explained_variance_ratio)
        assert abs(total_variance - 1.0) < 1e-10, (
            f"Explained variance does not sum to 1: {total_variance}"
        )

    def test_cumulative_variance_monotonic(
        self,
        sample_rate_matrix: np.ndarray,
        sample_maturities: np.ndarray,
        sample_observation_dates: list[date],
    ) -> None:
        """Test that cumulative variance is monotonically increasing."""
        from pathlib import Path

        dataset = InterestRateDataset(
            file_path=Path("test.csv"),
            source_name="test",
            observation_dates=sample_observation_dates,
            maturities=sample_maturities,
            rates=sample_rate_matrix,
        )

        result = compute_pca(dataset, n_components=None, use_changes=True)

        # Cumulative variance should be monotonically increasing
        cum_var = result.cumulative_variance
        for i in range(len(cum_var) - 1):
            assert cum_var[i] <= cum_var[i + 1], f"Cumulative variance not monotonic: {cum_var}"

        # Should end at 1.0
        assert abs(cum_var[-1] - 1.0) < 1e-10

    def test_n_components_limits_output(
        self,
        sample_rate_matrix: np.ndarray,
        sample_maturities: np.ndarray,
        sample_observation_dates: list[date],
    ) -> None:
        """Test that n_components parameter correctly limits output."""
        from pathlib import Path

        dataset = InterestRateDataset(
            file_path=Path("test.csv"),
            source_name="test",
            observation_dates=sample_observation_dates,
            maturities=sample_maturities,
            rates=sample_rate_matrix,
        )

        n_requested = 3
        result = compute_pca(dataset, n_components=n_requested, use_changes=True)

        assert result.n_components == n_requested
        assert result.loadings.shape[1] == n_requested
        assert len(result.explained_variance_ratio) == n_requested
        assert len(result.eigenvalues) == n_requested

    def test_pca_on_levels_vs_changes(
        self,
        sample_rate_matrix: np.ndarray,
        sample_maturities: np.ndarray,
        sample_observation_dates: list[date],
    ) -> None:
        """Test that PCA can be computed on both levels and changes."""
        from pathlib import Path

        dataset = InterestRateDataset(
            file_path=Path("test.csv"),
            source_name="test",
            observation_dates=sample_observation_dates,
            maturities=sample_maturities,
            rates=sample_rate_matrix,
        )

        result_changes = compute_pca(dataset, use_changes=True)
        result_levels = compute_pca(dataset, use_changes=False)

        # Both should produce valid results
        assert result_changes.n_components > 0
        assert result_levels.n_components > 0

        # Results should differ (changes vs levels have different properties)
        assert not np.allclose(result_changes.loadings, result_levels.loadings)

    def test_eigenvalues_non_negative(
        self,
        sample_rate_matrix: np.ndarray,
        sample_maturities: np.ndarray,
        sample_observation_dates: list[date],
    ) -> None:
        """Test that all eigenvalues are non-negative."""
        from pathlib import Path

        dataset = InterestRateDataset(
            file_path=Path("test.csv"),
            source_name="test",
            observation_dates=sample_observation_dates,
            maturities=sample_maturities,
            rates=sample_rate_matrix,
        )

        result = compute_pca(dataset, n_components=None, use_changes=True)

        assert np.all(result.eigenvalues >= 0), f"Found negative eigenvalues: {result.eigenvalues}"


class TestYieldCurveChanges:
    """Tests for yield curve change computation."""

    def test_compute_changes_shape(
        self,
        sample_rate_matrix: np.ndarray,
        sample_maturities: np.ndarray,
        sample_observation_dates: list[date],
    ) -> None:
        """Test that computed changes have correct shape."""
        from pathlib import Path

        dataset = InterestRateDataset(
            file_path=Path("test.csv"),
            source_name="test",
            observation_dates=sample_observation_dates,
            maturities=sample_maturities,
            rates=sample_rate_matrix,
        )

        changes = compute_yield_curve_changes(dataset)

        # Should have n-1 changes for n observations
        expected_n_changes = len(sample_observation_dates) - 1
        assert len(changes) == expected_n_changes

    def test_change_values_are_differences(
        self,
        sample_rate_matrix: np.ndarray,
        sample_maturities: np.ndarray,
        sample_observation_dates: list[date],
    ) -> None:
        """Test that change values equal consecutive rate differences."""
        from pathlib import Path

        dataset = InterestRateDataset(
            file_path=Path("test.csv"),
            source_name="test",
            observation_dates=sample_observation_dates,
            maturities=sample_maturities,
            rates=sample_rate_matrix,
        )

        changes = compute_yield_curve_changes(dataset)

        # Verify first change
        expected_first_change = sample_rate_matrix[1] - sample_rate_matrix[0]
        np.testing.assert_array_almost_equal(changes[0].changes, expected_first_change)

    def test_change_dates_are_correct(
        self,
        sample_rate_matrix: np.ndarray,
        sample_maturities: np.ndarray,
        sample_observation_dates: list[date],
    ) -> None:
        """Test that change dates are correctly assigned."""
        from pathlib import Path

        dataset = InterestRateDataset(
            file_path=Path("test.csv"),
            source_name="test",
            observation_dates=sample_observation_dates,
            maturities=sample_maturities,
            rates=sample_rate_matrix,
        )

        changes = compute_yield_curve_changes(dataset)

        # First change should be from date 0 to date 1
        assert changes[0].from_date == sample_observation_dates[0]
        assert changes[0].to_date == sample_observation_dates[1]


class TestComponentInterpretation:
    """Tests for PCA component interpretation."""

    def test_interpret_first_component_as_level(self) -> None:
        """Test that flat loadings are interpreted as level."""
        from datetime import datetime

        # Create a PCA result with flat first component (level)
        n_maturities = 7
        loadings = np.zeros((n_maturities, 3))
        loadings[:, 0] = 0.38  # Flat loadings - level
        loadings[:, 1] = np.linspace(0.5, -0.5, n_maturities)  # Slope
        loadings[:, 2] = np.array([-0.3, -0.1, 0.2, 0.4, 0.2, -0.1, -0.3])  # Curvature

        # Normalize
        for i in range(3):
            loadings[:, i] /= np.linalg.norm(loadings[:, i])

        pca_result = PCAResult(
            n_components=3,
            explained_variance_ratio=np.array([0.85, 0.10, 0.03]),
            cumulative_variance=np.array([0.85, 0.95, 0.98]),
            loadings=loadings,
            maturities=np.array([0.25, 0.5, 1.0, 2.0, 5.0, 10.0, 30.0]),
            interpretations=[],
            eigenvalues=np.array([0.85, 0.10, 0.03]),
            analysis_date=datetime.now(),
        )

        interpretations = interpret_components(pca_result, n_interpret=3)

        assert len(interpretations) == 3
        assert "level" in interpretations[0].lower() or "parallel" in interpretations[0].lower()

    def test_interpret_second_component_as_slope(self) -> None:
        """Test that tilted loadings are interpreted as slope."""
        from datetime import datetime

        n_maturities = 7
        loadings = np.zeros((n_maturities, 3))
        loadings[:, 0] = np.linspace(0.5, -0.5, n_maturities)  # Slope as PC1 for test
        loadings[:, 1] = 0.38  # Level as PC2
        loadings[:, 2] = np.array([-0.3, -0.1, 0.2, 0.4, 0.2, -0.1, -0.3])

        for i in range(3):
            loadings[:, i] /= np.linalg.norm(loadings[:, i])

        pca_result = PCAResult(
            n_components=3,
            explained_variance_ratio=np.array([0.85, 0.10, 0.03]),
            cumulative_variance=np.array([0.85, 0.95, 0.98]),
            loadings=loadings,
            maturities=np.array([0.25, 0.5, 1.0, 2.0, 5.0, 10.0, 30.0]),
            interpretations=[],
            eigenvalues=np.array([0.85, 0.10, 0.03]),
            analysis_date=datetime.now(),
        )

        interpretations = interpret_components(pca_result, n_interpret=1)

        assert "slope" in interpretations[0].lower() or "tilt" in interpretations[0].lower()

    def test_interpret_third_component_as_curvature(self) -> None:
        """Test that bow-shaped loadings are interpreted as curvature."""
        from datetime import datetime

        n_maturities = 7
        loadings = np.zeros((n_maturities, 3))
        # Curvature pattern: short and long same sign, middle opposite
        loadings[:, 0] = np.array([0.4, 0.3, -0.1, -0.4, -0.1, 0.3, 0.4])
        loadings[:, 1] = 0.38
        loadings[:, 2] = np.linspace(0.5, -0.5, n_maturities)

        for i in range(3):
            loadings[:, i] /= np.linalg.norm(loadings[:, i])

        pca_result = PCAResult(
            n_components=3,
            explained_variance_ratio=np.array([0.85, 0.10, 0.03]),
            cumulative_variance=np.array([0.85, 0.95, 0.98]),
            loadings=loadings,
            maturities=np.array([0.25, 0.5, 1.0, 2.0, 5.0, 10.0, 30.0]),
            interpretations=[],
            eigenvalues=np.array([0.85, 0.10, 0.03]),
            analysis_date=datetime.now(),
        )

        interpretations = interpret_components(pca_result, n_interpret=1)

        assert (
            "curvature" in interpretations[0].lower() or "butterfly" in interpretations[0].lower()
        )


class TestCurveReconstruction:
    """Tests for yield curve reconstruction from PCA components."""

    def test_full_reconstruction_recovers_original(
        self,
        sample_rate_matrix: np.ndarray,
        sample_maturities: np.ndarray,
        sample_observation_dates: list[date],
    ) -> None:
        """Test that using all components recovers original changes."""
        from pathlib import Path

        dataset = InterestRateDataset(
            file_path=Path("test.csv"),
            source_name="test",
            observation_dates=sample_observation_dates,
            maturities=sample_maturities,
            rates=sample_rate_matrix,
        )

        result = compute_pca(dataset, n_components=None, use_changes=True)
        changes = compute_yield_curve_changes(dataset)

        # Take first change and reconstruct with all components
        original_change = changes[0].changes
        reconstructed = reconstruct_curve(
            result, n_components=result.n_components, curve_changes=original_change
        )

        np.testing.assert_array_almost_equal(
            reconstructed,
            original_change,
            decimal=10,
            err_msg="Full reconstruction does not recover original",
        )

    def test_partial_reconstruction_loses_variance(
        self,
        sample_rate_matrix: np.ndarray,
        sample_maturities: np.ndarray,
        sample_observation_dates: list[date],
    ) -> None:
        """Test that partial reconstruction explains less variance."""
        from pathlib import Path

        dataset = InterestRateDataset(
            file_path=Path("test.csv"),
            source_name="test",
            observation_dates=sample_observation_dates,
            maturities=sample_maturities,
            rates=sample_rate_matrix,
        )

        result = compute_pca(dataset, n_components=None, use_changes=True)
        changes = compute_yield_curve_changes(dataset)
        original_change = changes[0].changes

        # Reconstruct with 1 component vs all
        reconstructed_1 = reconstruct_curve(result, n_components=1, curve_changes=original_change)
        reconstructed_all = reconstruct_curve(
            result, n_components=result.n_components, curve_changes=original_change
        )

        # Error should be larger with fewer components
        error_1 = np.sum((original_change - reconstructed_1) ** 2)
        error_all = np.sum((original_change - reconstructed_all) ** 2)

        assert error_1 >= error_all


# =============================================================================
# T059: Property-based tests for PCA
# =============================================================================


class TestPCAProperties:
    """Property-based tests using Hypothesis."""

    @given(
        n_obs=st.integers(min_value=50, max_value=200),
        n_mat=st.integers(min_value=3, max_value=10),
    )
    @settings(max_examples=10, deadline=None)
    def test_loadings_orthonormal(self, n_obs: int, n_mat: int) -> None:
        """Property: PCA loadings should always be orthonormal."""
        from datetime import datetime, timedelta
        from pathlib import Path

        rng = np.random.default_rng(42)

        # Generate random rate data
        rates = rng.standard_normal((n_obs, n_mat)) * 0.01 + 0.03
        maturities = np.linspace(0.25, 30.0, n_mat)
        dates = [datetime(2024, 1, 1).date() + timedelta(days=i) for i in range(n_obs)]

        dataset = InterestRateDataset(
            file_path=Path("test.csv"),
            source_name="test",
            observation_dates=dates,
            maturities=maturities,
            rates=rates,
        )

        result = compute_pca(dataset, n_components=None, use_changes=True)

        # Check orthonormality
        loadings = result.loadings
        product = loadings.T @ loadings
        identity = np.eye(loadings.shape[1])

        np.testing.assert_array_almost_equal(product, identity, decimal=8)

    @given(
        n_obs=st.integers(min_value=50, max_value=200),
        n_mat=st.integers(min_value=3, max_value=10),
    )
    @settings(max_examples=10, deadline=None)
    def test_variance_ratios_sum_lte_one(self, n_obs: int, n_mat: int) -> None:
        """Property: Explained variance ratios should sum to <= 1."""
        from datetime import datetime, timedelta
        from pathlib import Path

        rng = np.random.default_rng(42)

        rates = rng.standard_normal((n_obs, n_mat)) * 0.01 + 0.03
        maturities = np.linspace(0.25, 30.0, n_mat)
        dates = [datetime(2024, 1, 1).date() + timedelta(days=i) for i in range(n_obs)]

        dataset = InterestRateDataset(
            file_path=Path("test.csv"),
            source_name="test",
            observation_dates=dates,
            maturities=maturities,
            rates=rates,
        )

        result = compute_pca(dataset, n_components=3, use_changes=True)

        # Variance ratios should sum to <= 1 when not all components used
        total = np.sum(result.explained_variance_ratio)
        assert total <= 1.0 + 1e-10, f"Variance ratio sum exceeds 1: {total}"

    @given(
        n_obs=st.integers(min_value=50, max_value=200),
        n_mat=st.integers(min_value=3, max_value=10),
    )
    @settings(max_examples=10, deadline=None)
    def test_eigenvalues_non_negative(self, n_obs: int, n_mat: int) -> None:
        """Property: Eigenvalues should always be non-negative."""
        from datetime import datetime, timedelta
        from pathlib import Path

        rng = np.random.default_rng(42)

        rates = rng.standard_normal((n_obs, n_mat)) * 0.01 + 0.03
        maturities = np.linspace(0.25, 30.0, n_mat)
        dates = [datetime(2024, 1, 1).date() + timedelta(days=i) for i in range(n_obs)]

        dataset = InterestRateDataset(
            file_path=Path("test.csv"),
            source_name="test",
            observation_dates=dates,
            maturities=maturities,
            rates=rates,
        )

        result = compute_pca(dataset, n_components=None, use_changes=True)

        assert np.all(result.eigenvalues >= -1e-10), (
            f"Negative eigenvalues found: {result.eigenvalues}"
        )

    @given(
        n_obs=st.integers(min_value=50, max_value=100),
        n_mat=st.integers(min_value=4, max_value=7),
        n_comp=st.integers(min_value=1, max_value=3),
    )
    @settings(max_examples=10, deadline=None)
    def test_reconstruction_error_decreases_with_components(
        self, n_obs: int, n_mat: int, n_comp: int
    ) -> None:
        """Property: Reconstruction error should decrease with more components."""
        from datetime import datetime, timedelta
        from pathlib import Path

        rng = np.random.default_rng(42)

        rates = rng.standard_normal((n_obs, n_mat)) * 0.01 + 0.03
        maturities = np.linspace(0.25, 30.0, n_mat)
        dates = [datetime(2024, 1, 1).date() + timedelta(days=i) for i in range(n_obs)]

        dataset = InterestRateDataset(
            file_path=Path("test.csv"),
            source_name="test",
            observation_dates=dates,
            maturities=maturities,
            rates=rates,
        )

        result = compute_pca(dataset, n_components=None, use_changes=True)
        changes = compute_yield_curve_changes(dataset)

        if len(changes) == 0 or n_comp >= n_mat:
            return  # Skip edge cases

        original_change = changes[0].changes

        # Compare reconstruction with n_comp vs n_comp + 1 components
        if n_comp + 1 <= result.n_components:
            reconstructed_n = reconstruct_curve(
                result, n_components=n_comp, curve_changes=original_change
            )
            reconstructed_n1 = reconstruct_curve(
                result, n_components=n_comp + 1, curve_changes=original_change
            )

            error_n = np.sum((original_change - reconstructed_n) ** 2)
            error_n1 = np.sum((original_change - reconstructed_n1) ** 2)

            assert error_n >= error_n1 - 1e-10, f"Error did not decrease: {error_n} -> {error_n1}"
