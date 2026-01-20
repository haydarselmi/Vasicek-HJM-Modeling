"""Unit tests for the HJM (Heath-Jarrow-Morton) multifactor interest rate model.

This module tests HJM model initialization from PCA, volatility calibration,
no-arbitrage drift computation, and forward curve simulation.
"""

import numpy as np
import pytest
from hypothesis import given, settings, strategies as st, HealthCheck

from src.analysis.pca import PCAResult
from src.models.hjm import FactorContribution, HJMModel
from src.simulation.paths import SimulationResult
from src.utils.exceptions import CalibrationError, ModelNotCalibratedError


# =============================================================================
# Fixtures for HJM Tests
# =============================================================================


@pytest.fixture
def sample_pca_result(sample_maturities: np.ndarray) -> PCAResult:
    """Create a sample PCAResult for HJM initialization.

    This PCA result simulates typical yield curve PCA with:
    - PC1: Level (parallel shift) ~85% variance
    - PC2: Slope (tilt) ~10% variance
    - PC3: Curvature (bow) ~3% variance
    """
    n_maturities = len(sample_maturities)
    n_components = 3

    # Create loadings with typical patterns
    loadings = np.zeros((n_maturities, n_components))

    # PC1: Level - all similar positive loadings
    loadings[:, 0] = np.ones(n_maturities) / np.sqrt(n_maturities)

    # PC2: Slope - negative at short end, positive at long end
    slope_raw = np.linspace(-1, 1, n_maturities)
    loadings[:, 1] = slope_raw / np.linalg.norm(slope_raw)

    # PC3: Curvature - positive at ends, negative in middle
    x = np.linspace(-1, 1, n_maturities)
    curv_raw = x**2 - 0.5
    loadings[:, 2] = curv_raw / np.linalg.norm(curv_raw)

    # Variance explained (typical values)
    explained_variance_ratio = np.array([0.85, 0.10, 0.03])
    cumulative_variance = np.cumsum(explained_variance_ratio)

    # Eigenvalues (scaled by typical daily variance ~1e-6)
    eigenvalues = np.array([1e-6, 1.2e-7, 3.5e-8])

    return PCAResult(
        n_components=n_components,
        explained_variance_ratio=explained_variance_ratio,
        cumulative_variance=cumulative_variance,
        loadings=loadings,
        maturities=sample_maturities,
        interpretations=["PC1: Level", "PC2: Slope", "PC3: Curvature"],
        eigenvalues=eigenvalues,
    )


@pytest.fixture
def sample_initial_curve(sample_maturities: np.ndarray) -> np.ndarray:
    """Create a sample initial forward rate curve."""
    # Typical upward sloping curve starting at ~2%
    return 0.02 + 0.015 * np.log(1 + sample_maturities)


# =============================================================================
# T071: Unit tests for HJM model initialization from PCA
# =============================================================================


class TestHJMModelInitialization:
    """Test HJM model initialization from PCA results."""

    def test_init_from_pca_result(
        self,
        sample_pca_result: PCAResult,
    ) -> None:
        """HJM model should initialize from PCAResult."""
        model = HJMModel(pca_result=sample_pca_result)

        assert model.pca_result is sample_pca_result
        assert model.n_factors == sample_pca_result.n_components
        assert not model.is_calibrated

    def test_init_with_fewer_factors(
        self,
        sample_pca_result: PCAResult,
    ) -> None:
        """HJM model should accept fewer factors than PCA components."""
        model = HJMModel(pca_result=sample_pca_result, n_factors=2)

        assert model.n_factors == 2

    def test_init_with_too_many_factors_raises(
        self,
        sample_pca_result: PCAResult,
    ) -> None:
        """HJM model should reject n_factors > n_components."""
        with pytest.raises(ValueError, match="factors"):
            HJMModel(pca_result=sample_pca_result, n_factors=5)

    def test_maturities_from_pca(
        self,
        sample_pca_result: PCAResult,
        sample_maturities: np.ndarray,
    ) -> None:
        """HJM model should inherit maturities from PCA result."""
        model = HJMModel(pca_result=sample_pca_result)

        np.testing.assert_array_equal(model.maturities, sample_maturities)


# =============================================================================
# T071: Unit tests for HJM volatility calibration
# =============================================================================


class TestHJMCalibration:
    """Test HJM model calibration of volatility functions."""

    def test_calibrate_sets_volatility_functions(
        self,
        sample_pca_result: PCAResult,
    ) -> None:
        """Calibration should compute volatility functions."""
        model = HJMModel(pca_result=sample_pca_result)
        model.calibrate()

        assert model.is_calibrated
        assert model.volatility_functions is not None
        assert model.volatility_functions.shape == (
            sample_pca_result.n_components,
            len(sample_pca_result.maturities),
        )

    def test_calibrate_volatility_is_non_negative(
        self,
        sample_pca_result: PCAResult,
    ) -> None:
        """Volatility function magnitudes should be non-negative."""
        model = HJMModel(pca_result=sample_pca_result)
        model.calibrate()

        # Volatility functions can be negative (sign indicates direction)
        # but the volatility of each factor should be a reasonable value
        vol_magnitudes = np.sqrt(np.sum(model.volatility_functions**2, axis=1))
        assert np.all(vol_magnitudes >= 0)
        assert np.all(np.isfinite(vol_magnitudes))

    def test_calibrate_with_subset_of_factors(
        self,
        sample_pca_result: PCAResult,
    ) -> None:
        """Calibration should work with fewer factors."""
        model = HJMModel(pca_result=sample_pca_result, n_factors=2)
        model.calibrate()

        assert model.is_calibrated
        assert model.volatility_functions.shape[0] == 2

    def test_factor_variances_computed(
        self,
        sample_pca_result: PCAResult,
    ) -> None:
        """Calibration should compute factor variances."""
        model = HJMModel(pca_result=sample_pca_result)
        model.calibrate()

        assert model.factor_variances is not None
        assert len(model.factor_variances) == model.n_factors
        assert np.all(model.factor_variances >= 0)


# =============================================================================
# T071: Unit tests for HJM drift computation
# =============================================================================


class TestHJMDriftComputation:
    """Test HJM no-arbitrage drift computation."""

    def test_compute_drift_returns_array(
        self,
        sample_pca_result: PCAResult,
    ) -> None:
        """compute_drift should return drift for all maturities."""
        model = HJMModel(pca_result=sample_pca_result)
        model.calibrate()

        drift = model.compute_drift()

        assert isinstance(drift, np.ndarray)
        assert len(drift) == len(sample_pca_result.maturities)

    def test_compute_drift_is_finite(
        self,
        sample_pca_result: PCAResult,
    ) -> None:
        """Drift values should be finite."""
        model = HJMModel(pca_result=sample_pca_result)
        model.calibrate()

        drift = model.compute_drift()

        assert np.all(np.isfinite(drift))

    def test_compute_drift_raises_when_not_calibrated(
        self,
        sample_pca_result: PCAResult,
    ) -> None:
        """compute_drift should raise when model not calibrated."""
        model = HJMModel(pca_result=sample_pca_result)

        with pytest.raises(ModelNotCalibratedError):
            model.compute_drift()

    def test_drift_at_short_maturity(
        self,
        sample_pca_result: PCAResult,
    ) -> None:
        """Drift at very short maturity should be close to zero."""
        model = HJMModel(pca_result=sample_pca_result)
        model.calibrate()

        drift = model.compute_drift()

        # Drift at the shortest maturity should be relatively small
        # (integral from t to T is small when T-t is small)
        assert abs(drift[0]) < 0.1  # Reasonable bound for daily drift


# =============================================================================
# T071: Unit tests for HJM simulation
# =============================================================================


class TestHJMSimulation:
    """Test HJM forward curve simulation."""

    def test_simulate_returns_simulation_result(
        self,
        sample_pca_result: PCAResult,
        sample_initial_curve: np.ndarray,
    ) -> None:
        """simulate should return SimulationResult."""
        model = HJMModel(pca_result=sample_pca_result)
        model.calibrate()

        result = model.simulate(
            n_paths=10,
            time_horizon=0.25,
            time_step=1 / 252,
            initial_curve=sample_initial_curve,
            random_seed=42,
        )

        assert isinstance(result, SimulationResult)
        assert result.model_type == "hjm"
        assert result.path_count == 10

    def test_simulate_path_shape(
        self,
        sample_pca_result: PCAResult,
        sample_initial_curve: np.ndarray,
    ) -> None:
        """Simulated paths should have correct shape."""
        model = HJMModel(pca_result=sample_pca_result)
        model.calibrate()

        n_paths = 10
        time_horizon = 0.25
        time_step = 1 / 252
        n_steps = int(time_horizon / time_step)
        n_maturities = len(sample_initial_curve)

        result = model.simulate(
            n_paths=n_paths,
            time_horizon=time_horizon,
            time_step=time_step,
            initial_curve=sample_initial_curve,
            random_seed=42,
        )

        # Each path should have shape (n_steps + 1, n_maturities)
        for path in result.paths:
            assert path.rates.shape == (n_steps + 1, n_maturities)

    def test_simulate_initial_curve_preserved(
        self,
        sample_pca_result: PCAResult,
        sample_initial_curve: np.ndarray,
    ) -> None:
        """First curve in each path should match initial curve."""
        model = HJMModel(pca_result=sample_pca_result)
        model.calibrate()

        result = model.simulate(
            n_paths=5,
            time_horizon=0.1,
            time_step=1 / 252,
            initial_curve=sample_initial_curve,
            random_seed=42,
        )

        for path in result.paths:
            np.testing.assert_array_almost_equal(
                path.rates[0],
                sample_initial_curve,
                decimal=10,
            )

    def test_simulate_raises_when_not_calibrated(
        self,
        sample_pca_result: PCAResult,
        sample_initial_curve: np.ndarray,
    ) -> None:
        """simulate should raise when model not calibrated."""
        model = HJMModel(pca_result=sample_pca_result)

        with pytest.raises(ModelNotCalibratedError):
            model.simulate(
                n_paths=10,
                time_horizon=1.0,
                time_step=1 / 252,
                initial_curve=sample_initial_curve,
            )

    def test_simulate_reproducible_with_seed(
        self,
        sample_pca_result: PCAResult,
        sample_initial_curve: np.ndarray,
    ) -> None:
        """Simulation with same seed should produce identical results."""
        model = HJMModel(pca_result=sample_pca_result)
        model.calibrate()

        result1 = model.simulate(
            n_paths=5,
            time_horizon=0.1,
            time_step=1 / 252,
            initial_curve=sample_initial_curve,
            random_seed=42,
        )

        result2 = model.simulate(
            n_paths=5,
            time_horizon=0.1,
            time_step=1 / 252,
            initial_curve=sample_initial_curve,
            random_seed=42,
        )

        for p1, p2 in zip(result1.paths, result2.paths, strict=False):
            np.testing.assert_array_equal(p1.rates, p2.rates)

    def test_simulate_rates_finite(
        self,
        sample_pca_result: PCAResult,
        sample_initial_curve: np.ndarray,
    ) -> None:
        """All simulated rates should be finite."""
        model = HJMModel(pca_result=sample_pca_result)
        model.calibrate()

        result = model.simulate(
            n_paths=100,
            time_horizon=1.0,
            time_step=1 / 252,
            initial_curve=sample_initial_curve,
            random_seed=42,
        )

        for path in result.paths:
            assert np.all(np.isfinite(path.rates))


# =============================================================================
# T071: Unit tests for get_parameters
# =============================================================================


class TestHJMParameters:
    """Test HJM get_parameters and related methods."""

    def test_get_parameters_returns_dict(
        self,
        sample_pca_result: PCAResult,
    ) -> None:
        """get_parameters should return dictionary with model info."""
        model = HJMModel(pca_result=sample_pca_result)
        model.calibrate()

        params = model.get_parameters()

        assert isinstance(params, dict)
        assert "n_factors" in params
        assert params["n_factors"] == model.n_factors

    def test_get_parameters_raises_when_not_calibrated(
        self,
        sample_pca_result: PCAResult,
    ) -> None:
        """get_parameters should raise when not calibrated."""
        model = HJMModel(pca_result=sample_pca_result)

        with pytest.raises(ModelNotCalibratedError):
            model.get_parameters()


# =============================================================================
# T079: Tests for FactorContribution dataclass
# =============================================================================


class TestFactorContribution:
    """Test FactorContribution dataclass."""

    def test_factor_contribution_creation(self) -> None:
        """FactorContribution should store all fields correctly."""
        contribution = FactorContribution(
            factor_index=0,
            factor_label="Level",
            contribution=np.array([0.001, 0.001, 0.001]),
            percentage=85.0,
        )

        assert contribution.factor_index == 0
        assert contribution.factor_label == "Level"
        assert contribution.percentage == 85.0
        np.testing.assert_array_equal(
            contribution.contribution,
            np.array([0.001, 0.001, 0.001]),
        )


class TestGetFactorContributions:
    """Test get_factor_contributions method."""

    def test_get_factor_contributions_returns_list(
        self,
        sample_pca_result: PCAResult,
        sample_initial_curve: np.ndarray,
    ) -> None:
        """get_factor_contributions should return list of FactorContribution."""
        model = HJMModel(pca_result=sample_pca_result)
        model.calibrate()

        result = model.simulate(
            n_paths=10,
            time_horizon=0.1,
            time_step=1 / 252,
            initial_curve=sample_initial_curve,
            random_seed=42,
        )

        contributions = model.get_factor_contributions(result)

        assert isinstance(contributions, list)
        assert len(contributions) == model.n_factors
        assert all(isinstance(c, FactorContribution) for c in contributions)

    def test_factor_contributions_sum_approximately(
        self,
        sample_pca_result: PCAResult,
        sample_initial_curve: np.ndarray,
    ) -> None:
        """Factor contribution percentages should sum to approximately 100%."""
        model = HJMModel(pca_result=sample_pca_result)
        model.calibrate()

        result = model.simulate(
            n_paths=100,
            time_horizon=0.25,
            time_step=1 / 252,
            initial_curve=sample_initial_curve,
            random_seed=42,
        )

        contributions = model.get_factor_contributions(result)
        total_percentage = sum(c.percentage for c in contributions)

        # Should be close to 100% (allow for numerical precision)
        assert 95.0 <= total_percentage <= 105.0


# =============================================================================
# T072: Property-based tests for HJM no-arbitrage condition
# =============================================================================


class TestHJMNoArbitrageProperties:
    """Property-based tests verifying HJM no-arbitrage condition."""

    @given(
        scale=st.floats(min_value=0.1, max_value=10.0),
    )
    @settings(
        max_examples=20,
        suppress_health_check=[HealthCheck.function_scoped_fixture],
    )
    def test_drift_scales_with_volatility(
        self,
        scale: float,
        sample_pca_result: PCAResult,
    ) -> None:
        """Drift should scale quadratically with volatility."""
        # Create two models with different volatility scales
        model1 = HJMModel(pca_result=sample_pca_result)
        model1.calibrate()
        drift1 = model1.compute_drift()

        # Scale the eigenvalues (which determine volatility)
        scaled_pca = PCAResult(
            n_components=sample_pca_result.n_components,
            explained_variance_ratio=sample_pca_result.explained_variance_ratio,
            cumulative_variance=sample_pca_result.cumulative_variance,
            loadings=sample_pca_result.loadings,
            maturities=sample_pca_result.maturities,
            interpretations=sample_pca_result.interpretations,
            eigenvalues=sample_pca_result.eigenvalues * (scale**2),
        )

        model2 = HJMModel(pca_result=scaled_pca)
        model2.calibrate()
        drift2 = model2.compute_drift()

        # Drift should scale approximately as σ² (quadratically with volatility)
        # Since eigenvalues are σ², drift should scale linearly with eigenvalue scale
        # Handle zero drift values by checking the ratio where drift1 is non-zero
        non_zero_mask = np.abs(drift1) > 1e-15
        if np.any(non_zero_mask):
            ratio = drift2[non_zero_mask] / drift1[non_zero_mask]
            expected_ratio = scale**2
            np.testing.assert_array_almost_equal(
                ratio,
                np.full_like(ratio, expected_ratio),
                decimal=4,
            )
        # Also check that where drift1 is zero, drift2 is also zero
        zero_mask = ~non_zero_mask
        if np.any(zero_mask):
            np.testing.assert_array_almost_equal(
                drift2[zero_mask],
                np.zeros_like(drift2[zero_mask]),
                decimal=10,
            )

    @given(
        n_factors=st.integers(min_value=1, max_value=3),
    )
    @settings(
        max_examples=10,
        suppress_health_check=[HealthCheck.function_scoped_fixture],
    )
    def test_drift_finite_for_any_factor_count(
        self,
        n_factors: int,
        sample_pca_result: PCAResult,
    ) -> None:
        """Drift should be finite for any valid number of factors."""
        model = HJMModel(pca_result=sample_pca_result, n_factors=n_factors)
        model.calibrate()

        drift = model.compute_drift()

        assert np.all(np.isfinite(drift))

    def test_simulation_preserves_no_arbitrage(
        self,
        sample_pca_result: PCAResult,
        sample_initial_curve: np.ndarray,
    ) -> None:
        """Forward curves should evolve consistently with no-arbitrage drift.

        The HJM no-arbitrage condition states that the drift μ(t,T) is determined
        by the volatility functions. This test verifies that the simulated
        forward curves satisfy this condition on average.
        """
        model = HJMModel(pca_result=sample_pca_result)
        model.calibrate()

        # Run many simulations
        result = model.simulate(
            n_paths=1000,
            time_horizon=0.1,
            time_step=1 / 252,
            initial_curve=sample_initial_curve,
            random_seed=42,
        )

        # Get the rates matrix
        rates_matrix = result.get_rates_matrix()  # (n_paths, n_steps+1, n_maturities)

        # Compute empirical drift (mean change per time step)
        dt = 1 / 252
        n_steps = rates_matrix.shape[1] - 1

        # Average change across all paths and time steps
        changes = np.diff(rates_matrix, axis=1)  # (n_paths, n_steps, n_maturities)
        empirical_drift_per_step = np.mean(changes, axis=(0, 1))  # (n_maturities,)

        # Theoretical drift per time step
        theoretical_drift = model.compute_drift() * dt

        # The empirical drift should be close to theoretical drift
        # (within Monte Carlo error, which decreases with sqrt(n_paths * n_steps))
        mc_error = 0.001 / np.sqrt(1000 * n_steps)  # Approximate MC standard error

        np.testing.assert_array_almost_equal(
            empirical_drift_per_step,
            theoretical_drift,
            decimal=4,
            err_msg="Empirical drift deviates from no-arbitrage drift",
        )


class TestHJMLoadingsOrthogonality:
    """Test that HJM correctly uses orthogonal PCA loadings."""

    def test_volatility_functions_from_loadings(
        self,
        sample_pca_result: PCAResult,
    ) -> None:
        """Volatility functions should be derived from PCA loadings × sqrt(eigenvalue)."""
        model = HJMModel(pca_result=sample_pca_result)
        model.calibrate()

        # Each volatility function should be loading * sqrt(eigenvalue)
        for i in range(model.n_factors):
            expected_vol = sample_pca_result.loadings[:, i] * np.sqrt(
                sample_pca_result.eigenvalues[i]
            )
            np.testing.assert_array_almost_equal(
                model.volatility_functions[i],
                expected_vol,
                decimal=10,
            )

    def test_factors_are_orthogonal_in_simulation(
        self,
        sample_pca_result: PCAResult,
        sample_initial_curve: np.ndarray,
    ) -> None:
        """Factor contributions to curve changes should be orthogonal."""
        model = HJMModel(pca_result=sample_pca_result)
        model.calibrate()

        result = model.simulate(
            n_paths=500,
            time_horizon=0.25,
            time_step=1 / 252,
            initial_curve=sample_initial_curve,
            random_seed=42,
        )

        contributions = model.get_factor_contributions(result)

        # Check that factor contributions are approximately orthogonal
        # by verifying cross-correlations are low
        for i, ci in enumerate(contributions):
            for j, cj in enumerate(contributions):
                if i != j:
                    # Correlation should be close to zero for orthogonal factors
                    ci_flat = ci.contribution.flatten()
                    cj_flat = cj.contribution.flatten()
                    # Handle case where one contribution is constant (zero std)
                    if np.std(ci_flat) < 1e-10 or np.std(cj_flat) < 1e-10:
                        continue  # Skip if insufficient variance
                    corr = np.corrcoef(ci_flat, cj_flat)[0, 1]
                    # Allow for NaN if variance is very small
                    if not np.isnan(corr):
                        assert abs(corr) < 0.5, f"Factors {i} and {j} corr={corr}"
