"""Statistical analysis and PCA for yield curve data.

This module provides functionality for:
- Computing summary statistics (mean, std, correlations)
- Principal Component Analysis (PCA) on yield curve changes
- Interpretation of PCA components (level, slope, curvature)
"""

from src.analysis.pca import (
    PCAResult,
    YieldCurveChange,
    compute_pca,
    compute_yield_curve_changes,
    interpret_components,
    reconstruct_curve,
)
from src.analysis.statistics import (
    SummaryStatistics,
    compute_correlation_matrix,
    compute_summary_statistics,
)

__all__ = [
    # Statistics
    "SummaryStatistics",
    "compute_summary_statistics",
    "compute_correlation_matrix",
    # PCA
    "PCAResult",
    "YieldCurveChange",
    "compute_pca",
    "compute_yield_curve_changes",
    "interpret_components",
    "reconstruct_curve",
]

