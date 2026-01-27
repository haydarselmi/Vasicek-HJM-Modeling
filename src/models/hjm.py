"""Heath-Jarrow-Morton (HJM) multifactor forward rate model implementation.

This module implements the HJM multifactor model using PCA-derived volatility
functions. The forward rate follows:

    df(t,T) = μ(t,T)dt + Σᵢ σᵢ(t,T) dWᵢ(t)

where:
    - f(t,T) is the instantaneous forward rate at time t for maturity T
    - μ(t,T) is the no-arbitrage drift (determined by volatility)
    - σᵢ(t,T) are the volatility functions for each factor
    - Wᵢ(t) are independent Wiener processes

The no-arbitrage drift condition ensures:
    μ(t,T) = Σᵢ σᵢ(t,T) ∫ᵗᵀ σᵢ(t,s) ds
"""

from dataclasses import dataclass
from datetime import datetime
from typing import TYPE_CHECKING

import numpy as np
import numpy.typing as npt

from src.analysis.pca import PCAResult
from src.data.types import MaturityVector
from src.models.base import InterestRateModel
from src.simulation.monte_carlo import generate_normal_shocks
from src.simulation.paths import SimulationPath, SimulationResult
from src.utils.exceptions import ModelNotCalibratedError
from src.utils.logging import get_logger

# Import apply_hjm_dynamics lazily to avoid circular imports
# It's defined in monte_carlo.py which we update in this phase

if TYPE_CHECKING:
    from src.data.loader import InterestRateDataset

logger = get_logger("models.hjm")


# =============================================================================
# Data Classes
# =============================================================================


@dataclass
class FactorContribution:
    """Decomposition of rate movements by HJM factor.

    This dataclass stores the contribution of a single factor to the
    overall yield curve movement in a simulation.

    Attributes:
        factor_index: Which factor (0=level, 1=slope, 2=curvature, etc.).
        factor_label: Human-readable label from PCA interpretation.
        contribution: Array of contribution values to rate changes.
        percentage: Percentage of total variance explained by this factor.
    """

    factor_index: int
    factor_label: str
    contribution: npt.NDArray[np.float64]
    percentage: float


# =============================================================================
# HJM Model Implementation
# =============================================================================


class HJMModel(InterestRateModel):
    """Heath-Jarrow-Morton multifactor forward rate model with PCA factors.

    The HJM framework models the evolution of the entire forward rate curve,
    using principal components as volatility factors. This provides:
    - Dimensionality reduction from N maturities to K factors (typically 3)
    - Interpretable factors (level, slope, curvature)
    - No-arbitrage pricing framework

    The model is initialized from PCA results, which provide the volatility
    structure. Calibration computes the volatility functions from the PCA
    loadings and eigenvalues.

    Attributes:
        pca_result: Underlying PCA decomposition of yield curve changes.
        n_factors: Number of PCA factors used in the model.
        maturities: Maturity tenors from PCA result.
        volatility_functions: Volatility for each factor across maturities.
            Shape: (n_factors, n_maturities).
        factor_variances: Variance of each factor (eigenvalues).
        calibration_date: Timestamp when model was calibrated.

    Example:
        >>> pca_result = compute_pca(dataset, n_components=3)
        >>> model = HJMModel(pca_result=pca_result, n_factors=3)
        >>> model.calibrate()
        >>> result = model.simulate(n_paths=500, time_horizon=1.0, time_step=1/252)
        >>> contributions = model.get_factor_contributions(result)
    """

    def __init__(
        self,
        pca_result: PCAResult,
        n_factors: int | None = None,
    ) -> None:
        """Initialize the HJM model from PCA results.

        Args:
            pca_result: PCA decomposition of yield curve changes.
            n_factors: Number of factors to use. If None, use all PCA components.

        Raises:
            ValueError: If n_factors exceeds number of PCA components.
        """
        self._pca_result = pca_result

        # Determine number of factors
        if n_factors is None:
            n_factors = pca_result.n_components
        elif n_factors > pca_result.n_components:
            msg = (
                f"n_factors ({n_factors}) cannot exceed "
                f"PCA components ({pca_result.n_components})"
            )
            raise ValueError(msg)

        self._n_factors = n_factors
        self._maturities = pca_result.maturities.copy()
        self._calibrated = False

        # Calibration results (set during calibrate())
        self._volatility_functions: npt.NDArray[np.float64] | None = None
        self._factor_variances: npt.NDArray[np.float64] | None = None
        self.calibration_date: datetime | None = None

        logger.info(
            "Initialized HJM model with %d factors from PCA (%d components)",
            n_factors,
            pca_result.n_components,
        )

    # =========================================================================
    # Properties
    # =========================================================================

    @property
    def pca_result(self) -> PCAResult:
        """Return the underlying PCA decomposition."""
        return self._pca_result

    @property
    def n_factors(self) -> int:
        """Return the number of factors used in the model."""
        return self._n_factors

    @property
    def maturities(self) -> MaturityVector:
        """Return the maturity tenors."""
        return self._maturities

    @property
    def volatility_functions(self) -> npt.NDArray[np.float64]:
        """Return the volatility functions for each factor.

        Returns:
            Array of shape (n_factors, n_maturities).

        Raises:
            ModelNotCalibratedError: If model is not calibrated.
        """
        if self._volatility_functions is None:
            raise ModelNotCalibratedError(model_type="HJM")
        return self._volatility_functions

    @property
    def factor_variances(self) -> npt.NDArray[np.float64]:
        """Return the variance of each factor.

        Returns:
            Array of shape (n_factors,).

        Raises:
            ModelNotCalibratedError: If model is not calibrated.
        """
        if self._factor_variances is None:
            raise ModelNotCalibratedError(model_type="HJM")
        return self._factor_variances

    @property
    def is_calibrated(self) -> bool:
        """Return True if model has been calibrated."""
        return self._calibrated

    # =========================================================================
    # Calibration
    # =========================================================================

    def calibrate(
        self,
        data: InterestRateDataset | npt.NDArray[np.float64] | None = None,
        **kwargs: object,
    ) -> None:
        """Calibrate volatility functions from PCA loadings and eigenvalues.

        The HJM volatility functions are derived directly from the PCA results:
            σᵢ(T) = loadings[:, i] × √(eigenvalue[i])

        This gives each factor a volatility function that matches the PCA
        loading pattern, scaled by the standard deviation of that component.

        Args:
            data: Optional dataset (not used, kept for API compatibility).
                HJM calibration uses PCA results directly.
            **kwargs: Additional arguments (ignored).
        """
        logger.info("Calibrating HJM model with %d factors", self._n_factors)

        # Extract loadings for the factors we're using
        # loadings shape: (n_maturities, n_components)
        loadings = self._pca_result.loadings[:, : self._n_factors]

        # Extract eigenvalues for our factors
        eigenvalues = self._pca_result.eigenvalues[: self._n_factors]

        # Compute volatility functions
        # Each volatility function is: loading × sqrt(eigenvalue)
        # Result shape: (n_factors, n_maturities)
        self._volatility_functions = np.zeros((self._n_factors, len(self._maturities)))

        for i in range(self._n_factors):
            self._volatility_functions[i] = loadings[:, i] * np.sqrt(eigenvalues[i])

        # Store factor variances
        self._factor_variances = eigenvalues.copy()

        self._calibrated = True
        self.calibration_date = datetime.now()

        logger.info(
            "HJM calibration complete: %d factors, maturities=%s",
            self._n_factors,
            self._maturities.tolist(),
        )

        # Log volatility summary for each factor
        for i in range(self._n_factors):
            vol_norm = np.linalg.norm(self._volatility_functions[i])
            logger.debug(
                "Factor %d (%s): volatility norm=%.6f, variance=%.2e",
                i,
                (
                    self._pca_result.interpretations[i]
                    if i < len(self._pca_result.interpretations)
                    else f"PC{i + 1}"
                ),
                vol_norm,
                eigenvalues[i],
            )

    # =========================================================================
    # No-Arbitrage Drift
    # =========================================================================

    def compute_drift(self) -> npt.NDArray[np.float64]:
        """Compute the no-arbitrage drift for all maturities.

        The HJM no-arbitrage condition requires:
            μ(t,T) = Σᵢ σᵢ(t,T) × ∫ᵗᵀ σᵢ(t,s) ds

        For discrete maturities, we approximate the integral using the
        trapezoidal rule over the maturity grid.

        Returns:
            Array of drift values for each maturity point.

        Raises:
            ModelNotCalibratedError: If model is not calibrated.
        """
        if not self._calibrated:
            raise ModelNotCalibratedError(model_type="HJM")

        n_maturities = len(self._maturities)
        drift = np.zeros(n_maturities)

        # For each maturity T_j
        for j in range(n_maturities):
            # Compute drift at maturity T_j
            drift_j = 0.0

            # Get volatility functions (guaranteed non-None after calibration check)
            vol_funcs = self.volatility_functions
            for i in range(self._n_factors):
                # σᵢ(T_j)
                sigma_at_j = vol_funcs[i, j]

                # ∫₀^{T_j} σᵢ(s) ds (using trapezoidal rule)
                # Integrate from 0 to maturity j
                integral = self._integrate_volatility(i, 0, j)

                # Add to drift: σᵢ(T_j) × integral
                drift_j += sigma_at_j * integral

            drift[j] = drift_j

        return drift

    def _integrate_volatility(
        self,
        factor_idx: int,
        start_idx: int,
        end_idx: int,
    ) -> float:
        """Integrate volatility function using trapezoidal rule.

        Args:
            factor_idx: Which factor's volatility to integrate.
            start_idx: Starting maturity index.
            end_idx: Ending maturity index.

        Returns:
            Approximate integral value.
        """
        if start_idx >= end_idx:
            return 0.0

        # Get volatility functions (guaranteed non-None when called from compute_drift)
        vol_funcs = self.volatility_functions
        # Extract volatility values and maturities for integration range
        vol = vol_funcs[factor_idx, start_idx:end_idx + 1]
        mat = self._maturities[start_idx:end_idx + 1]

        # Trapezoidal integration
        return float(np.trapezoid(vol, mat))

    # =========================================================================
    # Simulation
    # =========================================================================

    def simulate(
        self,
        n_paths: int,
        time_horizon: float,
        time_step: float,
        random_seed: int | None = None,
        **kwargs: object,
    ) -> SimulationResult:
        """Simulate forward rate curve evolution using Monte Carlo.

        Uses Euler-Maruyama discretization to simulate the HJM dynamics:
            f(t+dt, T) = f(t, T) + μ(t,T)dt + Σᵢ σᵢ(t,T)√dt·εᵢ

        Args:
            n_paths: Number of simulation paths to generate.
            time_horizon: Simulation horizon in years.
            time_step: Time step size in years (e.g., 1/252 for daily).
            random_seed: Optional seed for reproducibility.
            **kwargs: Additional arguments.
                initial_curve: Initial forward rate curve. If None, use last
                    observed curve from PCA analysis (if available).

        Returns:
            SimulationResult containing all simulated forward curve paths.

        Raises:
            ModelNotCalibratedError: If model is not calibrated.
            ValueError: If initial_curve has wrong shape.
        """
        if not self._calibrated:
            raise ModelNotCalibratedError(model_type="HJM")

        n_maturities = len(self._maturities)

        # Extract initial_curve from kwargs
        initial_curve: npt.NDArray[np.float64] | None = kwargs.get("initial_curve")

        # Validate or create initial curve
        if initial_curve is None:
            # Use a flat curve at 3% as default
            initial_curve = np.full(n_maturities, 0.03)
            logger.warning(
                "No initial curve provided, using flat curve at 3%%"
            )
        elif len(initial_curve) != n_maturities:
            msg = (
                f"initial_curve length ({len(initial_curve)}) "
                f"must match maturities ({n_maturities})"
            )
            raise ValueError(msg)

        logger.info(
            "Simulating HJM: %d paths, horizon=%.2f years, dt=%.6f",
            n_paths,
            time_horizon,
            time_step,
        )

        # Calculate number of steps
        n_steps = int(np.ceil(time_horizon / time_step))
        actual_time_horizon = n_steps * time_step

        # Generate random shocks: shape (n_paths, n_factors, n_steps)
        shocks = generate_normal_shocks(
            n_paths=n_paths,
            n_steps=n_steps,
            n_factors=self._n_factors,
            random_seed=random_seed,
        )

        # Compute drift (per unit time)
        drift = self.compute_drift()

        # Import apply_hjm_dynamics here to avoid circular import
        from src.simulation.monte_carlo import apply_hjm_dynamics

        # Apply HJM dynamics
        # Result shape: (n_paths, n_steps + 1, n_maturities)
        forward_curves = apply_hjm_dynamics(
            initial_curve=initial_curve,
            volatility_functions=self.volatility_functions,
            drift=drift,
            shocks=shocks,
            dt=time_step,
        )

        # Build time grid
        time_grid = np.linspace(0, actual_time_horizon, n_steps + 1)

        # Create SimulationPath objects
        paths = [
            SimulationPath(
                path_id=i,
                model_type="hjm",
                time_grid=time_grid,
                rates=forward_curves[i],
                random_seed=random_seed,
            )
            for i in range(n_paths)
        ]

        # Log summary statistics
        terminal_curves = forward_curves[:, -1, :]
        logger.info(
            "Simulation complete: %d paths, terminal mean curve: [%.4f, ..., %.4f]",
            n_paths,
            np.mean(terminal_curves[:, 0]),
            np.mean(terminal_curves[:, -1]),
        )

        return SimulationResult(
            model_type="hjm",
            paths=paths,
            path_count=n_paths,
            time_horizon=actual_time_horizon,
            time_step=time_step,
            model_parameters=self.get_parameters(),
        )

    # =========================================================================
    # Factor Contributions
    # =========================================================================

    def get_factor_contributions(
        self,
        simulation_result: SimulationResult,
    ) -> list[FactorContribution]:
        """Decompose simulation results by factor contribution.

        Analyzes the simulated curve movements to determine how much each
        PCA factor contributed to the overall variance.

        Args:
            simulation_result: Simulation output to decompose.

        Returns:
            List of FactorContribution objects for each factor.

        Raises:
            ModelNotCalibratedError: If model is not calibrated.
        """
        if not self._calibrated:
            raise ModelNotCalibratedError(model_type="HJM")

        # Get all rate paths
        rates_matrix = simulation_result.get_rates_matrix()
        # Shape: (n_paths, n_steps + 1, n_maturities)

        # Compute total changes
        changes = np.diff(rates_matrix, axis=1)
        # Shape: (n_paths, n_steps, n_maturities)

        # Flatten for analysis
        all_changes = changes.reshape(-1, len(self._maturities))
        # Shape: (n_paths * n_steps, n_maturities)

        contributions = []

        for i in range(self._n_factors):
            # Get factor label
            if i < len(self._pca_result.interpretations):
                label = self._pca_result.interpretations[i]
            else:
                label = f"PC{i + 1}"

            # Project changes onto this factor's direction
            loading = self._pca_result.loadings[:, i]

            # Factor scores for each change
            factor_scores = all_changes @ loading  # Shape: (n_obs,)

            # Reconstruct contribution from this factor
            factor_contribution = np.outer(factor_scores, loading).mean(axis=0)

            # Variance explained by this factor (use property which checks for None)
            factor_variance = self.factor_variances[i]
            total_factor_variance = np.sum(self.factor_variances)

            percentage = 100.0 * factor_variance / total_factor_variance

            contributions.append(
                FactorContribution(
                    factor_index=i,
                    factor_label=label,
                    contribution=factor_contribution,
                    percentage=percentage,
                )
            )

        logger.debug(
            "Factor contributions: %s",
            [(c.factor_label, f"{c.percentage:.1f}%") for c in contributions],
        )

        return contributions

    # =========================================================================
    # Parameters
    # =========================================================================

    def get_parameters(self) -> dict[str, float]:
        """Return model parameters as a dictionary.

        Returns:
            Dictionary with n_factors and factor variance information.

        Raises:
            ModelNotCalibratedError: If model is not calibrated.
        """
        if not self._calibrated:
            raise ModelNotCalibratedError(model_type="HJM")

        params: dict[str, float] = {
            "n_factors": float(self._n_factors),
        }

        # Add factor variances (use property which checks for None)
        factor_vars = self.factor_variances
        for i in range(self._n_factors):
            params[f"factor_{i}_variance"] = float(factor_vars[i])

        # Add total variance explained
        total_var_explained = float(
            np.sum(self._pca_result.explained_variance_ratio[: self._n_factors])
        )
        params["total_variance_explained"] = total_var_explained

        return params

    def get_parameter_interpretations(self) -> dict[str, str]:
        """Get plain-language interpretations of model parameters.

        Returns:
            Dictionary mapping parameter names to interpretation strings.

        Raises:
            ModelNotCalibratedError: If model is not calibrated.
        """
        if not self._calibrated:
            raise ModelNotCalibratedError(model_type="HJM")

        interpretations: dict[str, str] = {
            "n_factors": (
                f"Number of factors: {self._n_factors}. "
                f"The HJM model uses {self._n_factors} PCA components "
                f"to model yield curve movements."
            ),
        }

        # Add interpretation for each factor
        for i in range(self._n_factors):
            label = (
                self._pca_result.interpretations[i]
                if i < len(self._pca_result.interpretations)
                else f"PC{i + 1}"
            )
            var_ratio = self._pca_result.explained_variance_ratio[i] * 100
            factor_vars = self.factor_variances
            interpretations[f"factor_{i}"] = (
                f"{label}: Explains {var_ratio:.1f}% of yield curve variance. "
                f"Daily volatility magnitude: {np.sqrt(factor_vars[i]) * 100:.4f}%."
            )

        total_explained = (
            np.sum(self._pca_result.explained_variance_ratio[: self._n_factors]) * 100
        )
        interpretations["coverage"] = (
            f"Total variance explained: {total_explained:.1f}%. "
            f"The {self._n_factors}-factor model captures this fraction of yield curve movements."
        )

        return interpretations
