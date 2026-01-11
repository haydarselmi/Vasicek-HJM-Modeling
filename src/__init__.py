"""HJM-Vasicek Modeling System.

A comprehensive interest rate modeling system implementing both Vasicek
(single-factor) and Heath-Jarrow-Morton (multifactor with PCA) models.

This package provides:
- Data loading and validation for interest rate time series
- Vasicek model calibration and simulation
- Principal Component Analysis (PCA) for yield curve dimensionality reduction
- HJM multifactor model with PCA-derived volatility functions
- Model comparison and backtesting utilities
- Visualization tools for yield curves and simulations
"""

__version__ = "0.1.0"
__author__ = "Haydar"
__package_name__ = "hjm-vasicek-modeling"

# Public API exports
__all__ = [
    "__version__",
    "__author__",
    "__package_name__",
]

