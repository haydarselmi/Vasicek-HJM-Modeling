"""Pytest fixtures for the HJM-Vasicek modeling system tests.

This module provides shared fixtures for sample interest rate data,
model instances, and test utilities used across the test suite.
"""

from datetime import date, timedelta

import numpy as np
import pytest

from src.data.types import (
    CorrelationMatrix,
    CovarianceMatrix,
    MaturityVector,
    RateMatrix,
    RateVector,
)

# =============================================================================
# Configuration Constants
# =============================================================================

# Standard treasury maturities in years
STANDARD_MATURITIES = [0.25, 0.5, 1.0, 2.0, 5.0, 10.0, 30.0]

# Number of observations for sample data (at least 100 as per task requirement)
NUM_OBSERVATIONS = 252  # ~1 year of daily data

# Random seed for reproducibility
RANDOM_SEED = 42


# =============================================================================
# Date Fixtures
# =============================================================================


@pytest.fixture
def sample_start_date() -> date:
    """Return a sample start date for test data.

    Returns:
        Start date for sample data (2024-01-02).
    """
    return date(2024, 1, 2)


@pytest.fixture
def sample_observation_dates(sample_start_date: date) -> list[date]:
    """Generate a list of business day observation dates.

    Args:
        sample_start_date: Starting date for observations.

    Returns:
        List of 252 observation dates (approximately 1 year).
    """
    dates = []
    current_date = sample_start_date
    while len(dates) < NUM_OBSERVATIONS:
        # Skip weekends (Saturday=5, Sunday=6)
        if current_date.weekday() < 5:
            dates.append(current_date)
        current_date += timedelta(days=1)
    return dates


# =============================================================================
# Maturity Fixtures
# =============================================================================


@pytest.fixture
def sample_maturities() -> MaturityVector:
    """Return standard treasury maturities.

    Returns:
        Array of 7 standard maturities: [0.25, 0.5, 1.0, 2.0, 5.0, 10.0, 30.0].
    """
    return np.array(STANDARD_MATURITIES, dtype=np.float64)


@pytest.fixture
def n_maturities() -> int:
    """Return the number of standard maturities.

    Returns:
        Number of maturities (7).
    """
    return len(STANDARD_MATURITIES)


# =============================================================================
# Rate Data Fixtures
# =============================================================================


@pytest.fixture
def rng() -> np.random.Generator:
    """Return a seeded random number generator for reproducibility.

    Returns:
        NumPy random generator with fixed seed.
    """
    return np.random.default_rng(RANDOM_SEED)


@pytest.fixture
def sample_rate_matrix(
    rng: np.random.Generator,
    n_maturities: int,
) -> RateMatrix:
    """Generate sample interest rate data with realistic properties.

    The generated rates have:
    - Mean around 3% (0.03) with upward sloping term structure
    - Realistic daily changes with autocorrelation
    - Values bounded between -1% and 15%

    Args:
        rng: Random number generator for reproducibility.
        n_maturities: Number of maturity tenors.

    Returns:
        Rate matrix of shape (252, 7) with realistic rate dynamics.
    """
    # Base rates for each maturity (upward sloping term structure)
    base_rates = np.array([0.02, 0.022, 0.025, 0.028, 0.032, 0.035, 0.038])

    # Generate correlated rate changes using Cholesky decomposition
    # Create correlation matrix (nearby maturities more correlated)
    n = n_maturities
    correlation = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            correlation[i, j] = np.exp(-0.3 * abs(i - j))

    # Cholesky decomposition for correlated random numbers
    cholesky = np.linalg.cholesky(correlation)

    # Generate rate paths with mean reversion
    rates = np.zeros((NUM_OBSERVATIONS, n_maturities))
    rates[0] = base_rates

    # Daily volatility
    daily_vol = 0.001  # 10 bps daily volatility

    for t in range(1, NUM_OBSERVATIONS):
        # Mean reversion towards base rates
        mean_reversion = 0.02 * (base_rates - rates[t - 1])

        # Correlated shocks
        uncorrelated = rng.standard_normal(n_maturities)
        correlated_shocks = cholesky @ uncorrelated * daily_vol

        rates[t] = rates[t - 1] + mean_reversion + correlated_shocks

    return rates


@pytest.fixture
def sample_short_rates(sample_rate_matrix: RateMatrix) -> RateVector:
    """Extract short-term rate series (3-month rate).

    Args:
        sample_rate_matrix: Full rate matrix.

    Returns:
        1D array of short-term rates for Vasicek calibration.
    """
    return sample_rate_matrix[:, 0].copy()


@pytest.fixture
def sample_rate_changes(sample_rate_matrix: RateMatrix) -> RateMatrix:
    """Compute first differences of rate data for PCA.

    Args:
        sample_rate_matrix: Full rate matrix.

    Returns:
        Rate changes matrix of shape (251, 7).
    """
    return np.diff(sample_rate_matrix, axis=0)


# =============================================================================
# Statistical Fixtures
# =============================================================================


@pytest.fixture
def sample_correlation_matrix(
    sample_rate_changes: RateMatrix,
) -> CorrelationMatrix:
    """Compute correlation matrix of rate changes.

    Args:
        sample_rate_changes: Rate changes matrix.

    Returns:
        Correlation matrix of shape (7, 7).
    """
    return np.corrcoef(sample_rate_changes.T)


@pytest.fixture
def sample_covariance_matrix(
    sample_rate_changes: RateMatrix,
) -> CovarianceMatrix:
    """Compute covariance matrix of rate changes.

    Args:
        sample_rate_changes: Rate changes matrix.

    Returns:
        Covariance matrix of shape (7, 7).
    """
    return np.cov(sample_rate_changes.T)


# =============================================================================
# Model Parameter Fixtures
# =============================================================================


@pytest.fixture
def sample_vasicek_params() -> tuple[float, float, float]:
    """Return sample Vasicek model parameters.

    Returns:
        Tuple of (kappa, theta, sigma) = (0.5, 0.03, 0.01).
    """
    return (0.5, 0.03, 0.01)


@pytest.fixture
def sample_pca_loadings(rng: np.random.Generator) -> np.ndarray:
    """Generate sample PCA loadings for 3 components.

    The loadings approximate typical yield curve PCA patterns:
    - PC1: Parallel shift (all similar loadings)
    - PC2: Slope (short vs long maturity)
    - PC3: Curvature (bow shape)

    Args:
        rng: Random number generator.

    Returns:
        Loading matrix of shape (7, 3).
    """
    n_maturities = 7
    loadings = np.zeros((n_maturities, 3))

    # PC1: Parallel shift (level)
    loadings[:, 0] = 0.35 + 0.05 * rng.standard_normal(n_maturities)

    # PC2: Slope (short vs long)
    loadings[:, 1] = np.linspace(0.5, -0.5, n_maturities)

    # PC3: Curvature (bow)
    x = np.linspace(-1, 1, n_maturities)
    loadings[:, 2] = -x**2 + 0.5

    # Normalize each column
    for i in range(3):
        loadings[:, i] /= np.linalg.norm(loadings[:, i])

    return loadings


# =============================================================================
# Simulation Fixtures
# =============================================================================


@pytest.fixture
def sample_time_grid() -> np.ndarray:
    """Generate a time grid for 1-year simulation with daily steps.

    Returns:
        Time grid array from 0 to 1 with 253 points (252 days + t=0).
    """
    return np.linspace(0, 1.0, 253)


@pytest.fixture
def sample_simulation_paths(
    rng: np.random.Generator,
    sample_vasicek_params: tuple[float, float, float],
) -> np.ndarray:
    """Generate sample Vasicek simulation paths.

    Args:
        rng: Random number generator.
        sample_vasicek_params: Vasicek parameters (kappa, theta, sigma).

    Returns:
        Simulated paths of shape (100, 253).
    """
    kappa, theta, sigma = sample_vasicek_params
    n_paths = 100
    n_steps = 252
    dt = 1.0 / 252

    paths = np.zeros((n_paths, n_steps + 1))
    paths[:, 0] = theta  # Start at long-run mean

    for t in range(n_steps):
        dW = rng.standard_normal(n_paths) * np.sqrt(dt)
        paths[:, t + 1] = (
            paths[:, t]
            + kappa * (theta - paths[:, t]) * dt
            + sigma * dW
        )

    return paths


# =============================================================================
# File Path Fixtures
# =============================================================================


@pytest.fixture
def tmp_csv_path(tmp_path: pytest.TempPathFactory) -> str:
    """Return a temporary path for CSV file tests.

    Args:
        tmp_path: Pytest temporary directory fixture.

    Returns:
        Path to temporary CSV file.
    """
    return str(tmp_path / "test_rates.csv")


@pytest.fixture
def tmp_output_dir(tmp_path: pytest.TempPathFactory) -> str:
    """Return a temporary directory for output files.

    Args:
        tmp_path: Pytest temporary directory fixture.

    Returns:
        Path to temporary output directory.
    """
    output_dir = tmp_path / "output"
    output_dir.mkdir()
    return str(output_dir)

