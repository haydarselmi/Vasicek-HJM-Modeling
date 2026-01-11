"""Type aliases for the HJM-Vasicek modeling system.

This module defines type aliases used throughout the codebase for
improved type safety and code readability. All types are designed
to work with numpy arrays and support strict type checking.
"""

from datetime import date

import numpy as np
import numpy.typing as npt

# =============================================================================
# Scalar Type Aliases
# =============================================================================

type RateValue = float
"""A single interest rate value (can be negative)."""

type MaturityTenor = float
"""Time to maturity in years (e.g., 0.25 for 3 months, 1.0 for 1 year)."""

type PathIndex = int
"""Index of a simulation path (0-based)."""

type ObservationDate = date
"""A single observation date for interest rate data."""

# =============================================================================
# Vector Type Aliases (1D Arrays)
# =============================================================================

type RateVector = npt.NDArray[np.float64]
"""1D array of interest rate values across time or maturities.

Shape: (n_observations,) or (n_maturities,)
"""

type MaturityVector = npt.NDArray[np.float64]
"""1D array of maturity tenors in years.

Shape: (n_maturities,)
Example: [0.25, 0.5, 1.0, 2.0, 5.0, 10.0, 30.0] for standard treasury maturities.
"""

type TimeGrid = npt.NDArray[np.float64]
"""1D array of time points for simulation.

Shape: (n_steps + 1,) including t=0
"""

# =============================================================================
# Matrix Type Aliases (2D Arrays)
# =============================================================================

type RateMatrix = npt.NDArray[np.float64]
"""2D array of interest rates with observations as rows and maturities as columns.

Shape: (n_observations, n_maturities)
Each row represents rates for all maturities on a single observation date.
"""

type LoadingMatrix = npt.NDArray[np.float64]
"""2D array of PCA factor loadings.

Shape: (n_maturities, n_components)
Each column represents loadings for a single principal component across maturities.
"""

type CovarianceMatrix = npt.NDArray[np.float64]
"""2D symmetric positive semi-definite covariance matrix.

Shape: (n_maturities, n_maturities)
"""

type CorrelationMatrix = npt.NDArray[np.float64]
"""2D symmetric correlation matrix with diagonal of ones.

Shape: (n_maturities, n_maturities)
"""

type SimulationPathMatrix = npt.NDArray[np.float64]
"""2D array of simulated rate paths.

Shape: (n_paths, n_steps + 1)
Each row represents a single simulated path across time.
"""

# =============================================================================
# Date Range Type
# =============================================================================

type DateRange = tuple[date, date]
"""A tuple of (start_date, end_date) defining a date range (inclusive)."""

# =============================================================================
# Parameter Type Aliases
# =============================================================================

type VasicekParams = tuple[float, float, float]
"""Vasicek model parameters: (kappa, theta, sigma).

- kappa: Mean reversion speed
- theta: Long-run mean rate
- sigma: Volatility
"""

# =============================================================================
# Validation Bounds
# =============================================================================

# Reasonable bounds for interest rates (allowing for negative rates)
RATE_MIN_BOUND: float = -0.10  # -10%
RATE_MAX_BOUND: float = 0.50  # 50%

# Reasonable bounds for maturity tenors in years
MATURITY_MIN_BOUND: float = 0.0  # Overnight
MATURITY_MAX_BOUND: float = 100.0  # 100 years (extreme long-term bonds)

# Minimum number of observations for calibration
MIN_OBSERVATIONS_FOR_CALIBRATION: int = 30
