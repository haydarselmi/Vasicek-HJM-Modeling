"""Statistical analysis functionality for interest rate datasets.

This module provides functions for computing summary statistics and
correlation matrices for interest rate time series data.
"""

from dataclasses import dataclass
from datetime import datetime

import numpy as np
import numpy.typing as npt

from src.data.loader import InterestRateDataset
from src.data.types import CorrelationMatrix, MaturityVector, RateVector
from src.utils.logging import get_logger

logger = get_logger("analysis.statistics")


@dataclass
class SummaryStatistics:
    """Summary statistics for an interest rate dataset.

    Contains descriptive statistics computed across the time dimension
    for each maturity in the dataset.

    Attributes:
        means: Mean rate for each maturity.
        stds: Standard deviation for each maturity.
        mins: Minimum rate for each maturity.
        maxs: Maximum rate for each maturity.
        correlations: Correlation matrix across maturities.
        maturities: Array of maturity tenors.
        computed_at: Timestamp when statistics were computed.
    """

    means: RateVector
    stds: RateVector
    mins: RateVector
    maxs: RateVector
    correlations: CorrelationMatrix
    maturities: MaturityVector
    computed_at: datetime | None = None

    def __post_init__(self) -> None:
        """Set computed_at timestamp if not provided."""
        if self.computed_at is None:
            self.computed_at = datetime.now()

    @property
    def n_maturities(self) -> int:
        """Return the number of maturities."""
        return len(self.maturities)

    def get_stats_for_maturity(self, maturity: float) -> dict[str, float]:
        """Get statistics for a specific maturity.

        Args:
            maturity: Maturity tenor in years.

        Returns:
            Dictionary with mean, std, min, max for the maturity.

        Raises:
            ValueError: If maturity is not found.
        """
        idx = np.where(np.isclose(self.maturities, maturity))[0]
        if len(idx) == 0:
            msg = f"Maturity {maturity} not found"
            raise ValueError(msg)

        i = idx[0]
        return {
            "mean": float(self.means[i]),
            "std": float(self.stds[i]),
            "min": float(self.mins[i]),
            "max": float(self.maxs[i]),
        }

    def to_dataframe(self) -> pd.DataFrame:  # noqa: F821
        """Convert statistics to a pandas DataFrame.

        Returns:
            DataFrame with maturities as index and stats as columns.
        """
        import pandas as pd

        return pd.DataFrame(
            {
                "maturity": self.maturities,
                "mean": self.means,
                "std": self.stds,
                "min": self.mins,
                "max": self.maxs,
            }
        ).set_index("maturity")


def compute_summary_statistics(dataset: InterestRateDataset) -> SummaryStatistics:
    """Compute summary statistics for each maturity in the dataset.

    Computes mean, standard deviation, minimum, maximum, and correlation
    matrix across the time dimension for each maturity tenor.

    Args:
        dataset: The interest rate dataset to analyze.

    Returns:
        SummaryStatistics containing descriptive statistics.
    """
    logger.info(
        "Computing summary statistics for %d observations across %d maturities",
        dataset.n_observations,
        dataset.n_maturities,
    )

    rates = dataset.rates

    # Compute statistics along time axis (axis=0)
    means = np.nanmean(rates, axis=0)
    stds = np.nanstd(rates, axis=0, ddof=1)
    mins = np.nanmin(rates, axis=0)
    maxs = np.nanmax(rates, axis=0)

    # Compute correlation matrix
    correlations = compute_correlation_matrix(dataset)

    return SummaryStatistics(
        means=means,
        stds=stds,
        mins=mins,
        maxs=maxs,
        correlations=correlations,
        maturities=dataset.maturities.copy(),
    )


def compute_correlation_matrix(
    dataset: InterestRateDataset,
) -> npt.NDArray[np.float64]:
    """Compute the correlation matrix across maturities.

    Computes the Pearson correlation coefficient between rate time series
    for each pair of maturities.

    Args:
        dataset: The interest rate dataset.

    Returns:
        Correlation matrix of shape (n_maturities, n_maturities).
    """
    rates = dataset.rates

    # Handle NaN values by using pairwise complete observations
    n_maturities = dataset.n_maturities
    corr_matrix = np.zeros((n_maturities, n_maturities))

    for i in range(n_maturities):
        for j in range(n_maturities):
            if i == j:
                corr_matrix[i, j] = 1.0
            elif i < j:
                # Get valid pairs
                x = rates[:, i]
                y = rates[:, j]
                valid = ~(np.isnan(x) | np.isnan(y))

                if np.sum(valid) > 1:
                    corr = np.corrcoef(x[valid], y[valid])[0, 1]
                    corr_matrix[i, j] = corr
                    corr_matrix[j, i] = corr
                else:
                    corr_matrix[i, j] = np.nan
                    corr_matrix[j, i] = np.nan

    return corr_matrix

