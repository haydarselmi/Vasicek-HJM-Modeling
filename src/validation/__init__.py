"""Model validation and backtesting utilities.

This module provides functionality for:
- Backtesting models against held-out historical data
- Computing error metrics (MAE, RMSE, MAPE)
- Comparing model performance
"""

from src.validation.backtesting import (
    ValidationResult,
    backtest_model,
    compute_hit_ratio,
    compute_mae,
    compute_mape,
    compute_rmse,
)
from src.validation.comparison import (
    ComparisonResult,
    compare_models,
    get_model_recommendation,
)

__all__ = [
    # Backtesting
    "ValidationResult",
    "backtest_model",
    "compute_mae",
    "compute_rmse",
    "compute_mape",
    "compute_hit_ratio",
    # Comparison
    "ComparisonResult",
    "compare_models",
    "get_model_recommendation",
]

