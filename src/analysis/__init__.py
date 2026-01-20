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
    # Autoencoder (optional - requires torch)
    "AutoencoderResult",
    "ComparisonResult",
    "YieldCurveAutoencoder",
    "train_autoencoder",
    "compare_dimensionality_methods",
    "get_latent_encodings",
]


# Conditional imports for autoencoder module (requires torch)
def __getattr__(name: str) -> object:
    """Lazy import for autoencoder module components."""
    if name in (
        "AutoencoderResult",
        "ComparisonResult",
        "YieldCurveAutoencoder",
        "train_autoencoder",
        "compare_dimensionality_methods",
        "get_latent_encodings",
    ):
        from src.analysis import autoencoder

        return getattr(autoencoder, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

