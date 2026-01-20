"""Principal Component Analysis for yield curve data.

This module provides PCA functionality for analyzing yield curve movements,
including computation of principal components, interpretation of factors
(level, slope, curvature), and curve reconstruction.
"""

from dataclasses import dataclass, field
from datetime import date, datetime

import numpy as np
import numpy.typing as npt

from src.data.loader import InterestRateDataset
from src.data.types import LoadingMatrix, MaturityVector, RateVector
from src.utils.logging import get_logger

logger = get_logger("analysis.pca")


# =============================================================================
# Data Classes
# =============================================================================


@dataclass
class YieldCurveChange:
    """Represents the change in yield curve between two consecutive observations.

    Attributes:
        from_date: Start observation date.
        to_date: End observation date.
        maturities: Array of maturity tenors in years.
        changes: Array of rate changes (to - from) for each maturity.
    """

    from_date: date
    to_date: date
    maturities: MaturityVector
    changes: RateVector


@dataclass
class PCAResult:
    """Results from Principal Component Analysis on yield curve data.

    Attributes:
        n_components: Number of components computed/retained.
        explained_variance_ratio: Variance explained by each PC.
        cumulative_variance: Cumulative variance explained.
        loadings: Component loadings matrix (n_maturities, n_components).
        maturities: Maturity tenors used in analysis.
        interpretations: Human-readable labels for components.
        eigenvalues: Eigenvalues (variances) in descending order.
        analysis_date: When PCA was performed.
    """

    n_components: int
    explained_variance_ratio: npt.NDArray[np.float64]
    cumulative_variance: npt.NDArray[np.float64]
    loadings: LoadingMatrix
    maturities: MaturityVector
    interpretations: list[str]
    eigenvalues: npt.NDArray[np.float64]
    analysis_date: datetime = field(default_factory=datetime.now)


# =============================================================================
# Helper Functions
# =============================================================================


def compute_yield_curve_changes(
    dataset: InterestRateDataset,
) -> list[YieldCurveChange]:
    """Compute first differences of yield curve data.

    This function calculates the day-over-day changes in rates for each
    maturity point, which is the typical input for PCA analysis.

    Args:
        dataset: The interest rate dataset to analyze.

    Returns:
        List of YieldCurveChange objects, one for each consecutive date pair.
    """
    logger.info(
        "Computing yield curve changes for %d observations",
        dataset.n_observations,
    )

    changes: list[YieldCurveChange] = []

    for i in range(len(dataset.observation_dates) - 1):
        change = YieldCurveChange(
            from_date=dataset.observation_dates[i],
            to_date=dataset.observation_dates[i + 1],
            maturities=dataset.maturities.copy(),
            changes=dataset.rates[i + 1] - dataset.rates[i],
        )
        changes.append(change)

    logger.info("Computed %d yield curve changes", len(changes))
    return changes


def _get_changes_matrix(
    changes: list[YieldCurveChange],
) -> npt.NDArray[np.float64]:
    """Convert list of YieldCurveChange to a matrix.

    Args:
        changes: List of yield curve changes.

    Returns:
        Matrix of shape (n_changes, n_maturities).
    """
    return np.array([c.changes for c in changes])


# =============================================================================
# Core PCA Functions
# =============================================================================


def compute_pca(
    dataset: InterestRateDataset,
    n_components: int | None = None,
    use_changes: bool = True,
) -> PCAResult:
    """Perform Principal Component Analysis on yield curve data.

    This function computes PCA on either rate levels or rate changes,
    returning loadings, explained variance, and component interpretations.

    Args:
        dataset: The interest rate dataset to analyze.
        n_components: Number of components to retain. If None, retain all.
        use_changes: If True, compute PCA on rate changes; if False, on levels.

    Returns:
        PCAResult containing loadings, explained variance, and interpretations.
    """
    logger.info(
        "Computing PCA on %s with n_components=%s",
        "changes" if use_changes else "levels",
        n_components,
    )

    # Get data matrix
    if use_changes:
        changes = compute_yield_curve_changes(dataset)
        data = _get_changes_matrix(changes)
    else:
        data = dataset.rates

    # Center the data
    mean = np.mean(data, axis=0)
    centered_data = data - mean

    # Compute covariance matrix
    n_obs = centered_data.shape[0]
    cov_matrix = (centered_data.T @ centered_data) / (n_obs - 1)

    # Eigenvalue decomposition (using eigh for symmetric matrices)
    eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)

    # Sort in descending order (eigh returns ascending)
    sort_idx = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[sort_idx]
    eigenvectors = eigenvectors[:, sort_idx]

    # Handle numerical issues: clamp small negative eigenvalues to zero
    eigenvalues = np.maximum(eigenvalues, 0)

    # Compute explained variance
    total_variance = np.sum(eigenvalues)
    explained_variance_ratio = eigenvalues / total_variance
    cumulative_variance = np.cumsum(explained_variance_ratio)

    # Limit to n_components if specified
    n_maturities = len(dataset.maturities)
    n_components = (
        n_maturities if n_components is None else min(n_components, n_maturities)
    )

    # Extract relevant components
    loadings = eigenvectors[:, :n_components]
    eigenvalues = eigenvalues[:n_components]
    explained_variance_ratio = explained_variance_ratio[:n_components]
    cumulative_variance = cumulative_variance[:n_components]

    logger.info(
        "PCA complete: %d components explain %.2f%% of variance",
        n_components,
        cumulative_variance[-1] * 100,
    )

    # Create result (interpretations will be filled by interpret_components)
    result = PCAResult(
        n_components=n_components,
        explained_variance_ratio=explained_variance_ratio,
        cumulative_variance=cumulative_variance,
        loadings=loadings,
        maturities=dataset.maturities.copy(),
        interpretations=[],
        eigenvalues=eigenvalues,
    )

    # Compute interpretations
    result.interpretations = interpret_components(result, n_interpret=min(3, n_components))

    return result


def interpret_components(
    pca_result: PCAResult,
    n_interpret: int = 3,
) -> list[str]:
    """Generate human-readable interpretations for principal components.

    This function analyzes the loading patterns to identify:
    - Level: Flat loadings across maturities (parallel shift)
    - Slope: Monotonic pattern (short vs long maturity)
    - Curvature: Bow pattern (short and long same sign, middle opposite)

    Args:
        pca_result: The PCA results to interpret.
        n_interpret: Number of components to interpret (max 3 recommended).

    Returns:
        List of interpretation strings for each component.
    """
    n_interpret = min(n_interpret, pca_result.n_components)
    interpretations: list[str] = []

    for i in range(n_interpret):
        loadings = pca_result.loadings[:, i]
        interpretation = _interpret_single_component(loadings, i + 1)
        interpretations.append(interpretation)

    return interpretations


def _interpret_single_component(
    loadings: npt.NDArray[np.float64],
    component_number: int,
) -> str:
    """Interpret a single PCA component based on its loading pattern.

    Args:
        loadings: Loading vector for this component.
        component_number: 1-based component index.

    Returns:
        Interpretation string.
    """
    n = len(loadings)

    # Normalize loadings for comparison
    loadings_norm = loadings / (np.linalg.norm(loadings) + 1e-10)

    # Check for level pattern (all same sign, relatively flat)
    same_sign = np.all(loadings_norm > 0) or np.all(loadings_norm < 0)
    std_loadings = np.std(np.abs(loadings_norm))
    is_flat = std_loadings < 0.2

    if same_sign and is_flat:
        return (
            f"PC{component_number}: Level (parallel shift) "
            f"- uniform impact across maturities"
        )

    # Check for slope pattern (monotonic increase or decrease)
    diffs = np.diff(loadings_norm)
    is_monotonic_increasing = np.all(diffs >= -0.05)
    is_monotonic_decreasing = np.all(diffs <= 0.05)

    if is_monotonic_increasing or is_monotonic_decreasing:
        direction = "steepening" if is_monotonic_increasing else "flattening"
        return f"PC{component_number}: Slope (tilt) - {direction} of yield curve"

    # Check for curvature pattern (short and long same sign, middle opposite)
    third = n // 3
    short_end = loadings_norm[:third] if third > 0 else loadings_norm[:1]
    middle = loadings_norm[third : 2 * third] if 2 * third > third else loadings_norm[1:2]
    long_end = loadings_norm[2 * third :] if 2 * third < n else loadings_norm[-1:]

    short_sign = np.mean(short_end) > 0
    middle_sign = np.mean(middle) > 0
    long_sign = np.mean(long_end) > 0

    if short_sign == long_sign and short_sign != middle_sign:
        shape = "butterfly" if short_sign else "inverse butterfly"
        return f"PC{component_number}: Curvature ({shape}) - belly vs wings movement"

    # Check for twist pattern (short opposite to long, middle neutral)
    if short_sign != long_sign:
        return f"PC{component_number}: Slope (tilt) - short vs long maturity divergence"

    # Default interpretation
    return f"PC{component_number}: Mixed pattern - complex yield curve movement"


def reconstruct_curve(
    pca_result: PCAResult,
    n_components: int,
    curve_changes: npt.NDArray[np.float64],
) -> npt.NDArray[np.float64]:
    """Reconstruct yield curve changes using a reduced number of components.

    This function projects the original changes onto the principal component
    space and reconstructs using only the specified number of components.

    Args:
        pca_result: The PCA decomposition.
        n_components: Number of components to use for reconstruction.
        curve_changes: Original curve changes to reconstruct.

    Returns:
        Reconstructed curve changes.
    """
    n_components = min(n_components, pca_result.n_components)

    # Get the subset of loadings to use
    loadings = pca_result.loadings[:, :n_components]

    # Project onto PC space
    scores = curve_changes @ loadings

    # Reconstruct
    reconstructed = scores @ loadings.T

    logger.debug(
        "Reconstructed curve using %d components",
        n_components,
    )

    return reconstructed
