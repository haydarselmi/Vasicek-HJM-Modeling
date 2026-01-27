"""Data loading functionality for interest rate datasets.

This module provides functions and classes for loading historical interest
rate data from CSV files and representing it in a structured format.
"""

import re
from dataclasses import dataclass, field
from datetime import date, datetime
from pathlib import Path

import numpy as np
import numpy.typing as npt
import pandas as pd

from src.data.types import DateRange, MaturityVector, RateMatrix
from src.utils.exceptions import DataValidationError
from src.utils.logging import get_logger

logger = get_logger("data.loader")


@dataclass
class YieldCurve:
    """A snapshot of interest rates across different maturities at a single point in time.

    Attributes:
        observation_date: Date of observation.
        maturities: Array of maturity tenors in years.
        rates: Array of rate values for each maturity.
    """

    observation_date: date
    maturities: npt.NDArray[np.float64]
    rates: npt.NDArray[np.float64]

    @property
    def n_points(self) -> int:
        """Return the number of maturity points in the curve."""
        return len(self.maturities)


@dataclass
class InterestRateDataset:
    """A collection of historical interest rate observations.

    This class represents a time series of yield curves loaded from an
    external source (typically a CSV file).

    Attributes:
        file_path: Path to the source file.
        source_name: Human-readable data source name.
        observation_dates: List of observation dates.
        maturities: Array of maturity tenors in years.
        rates: Rate matrix with shape (n_observations, n_maturities).
        loaded_at: Timestamp when data was loaded.
    """

    file_path: Path
    source_name: str
    observation_dates: list[date]
    maturities: MaturityVector
    rates: RateMatrix
    loaded_at: datetime = field(default_factory=datetime.now)

    @property
    def n_observations(self) -> int:
        """Return the number of observation dates."""
        return len(self.observation_dates)

    @property
    def n_maturities(self) -> int:
        """Return the number of maturity tenors."""
        return len(self.maturities)

    @property
    def date_range(self) -> DateRange:
        """Return the start and end dates of the dataset."""
        return (self.observation_dates[0], self.observation_dates[-1])

    def get_yield_curve(self, observation_date: date) -> YieldCurve:
        """Get the yield curve for a specific date.

        Args:
            observation_date: The date to retrieve.

        Returns:
            YieldCurve for the specified date.

        Raises:
            ValueError: If date is not in the dataset.
        """
        try:
            idx = self.observation_dates.index(observation_date)
        except ValueError as e:
            msg = f"Date {observation_date} not found in dataset"
            raise ValueError(msg) from e

        return YieldCurve(
            observation_date=observation_date,
            maturities=self.maturities.copy(),
            rates=self.rates[idx].copy(),
        )

    def get_yield_curves(self, dates: list[date] | None = None) -> list[YieldCurve]:
        """Get yield curves for multiple dates.

        Args:
            dates: List of dates to retrieve. If None, return all curves.

        Returns:
            List of YieldCurve objects.
        """
        if dates is None:
            dates = self.observation_dates

        return [self.get_yield_curve(d) for d in dates]

    def filter_dates(
        self,
        start_date: date | None = None,
        end_date: date | None = None,
    ) -> InterestRateDataset:
        """Filter the dataset to a date range for train/test splitting.

        Creates a new dataset containing only observations within the
        specified date range. Useful for creating training and validation
        splits for model backtesting.

        Args:
            start_date: Start date (inclusive). If None, uses first date.
            end_date: End date (inclusive). If None, uses last date.

        Returns:
            New InterestRateDataset with filtered data.

        Raises:
            ValueError: If date range results in empty dataset.

        Example:
            >>> # Split into train (first 80%) and test (last 20%)
            >>> n_obs = len(dataset.observation_dates)
            >>> split_idx = int(n_obs * 0.8)
            >>> split_date = dataset.observation_dates[split_idx]
            >>> train_data = dataset.filter_dates(end_date=split_date)
            >>> test_data = dataset.filter_dates(start_date=split_date)
        """
        if start_date is None:
            start_date = self.observation_dates[0]
        if end_date is None:
            end_date = self.observation_dates[-1]

        # Find indices for the date range
        indices = [
            i for i, d in enumerate(self.observation_dates)
            if start_date <= d <= end_date
        ]

        if len(indices) == 0:
            msg = f"No observations found in date range {start_date} to {end_date}"
            raise ValueError(msg)

        # Filter dates and rates
        filtered_dates = [self.observation_dates[i] for i in indices]
        filtered_rates = self.rates[indices, :]

        logger.info(
            "Filtered dataset from %s to %s: %d observations",
            start_date,
            end_date,
            len(filtered_dates),
        )

        return InterestRateDataset(
            file_path=self.file_path,
            source_name=f"{self.source_name}_filtered",
            observation_dates=filtered_dates,
            maturities=self.maturities.copy(),
            rates=filtered_rates,
        )


def _parse_maturity(maturity_str: str) -> float:
    """Parse a maturity string into years.

    Supported formats:
    - "3M", "6M" -> months converted to years
    - "1Y", "2Y", "10Y" -> years
    - "0.25", "1.0" -> direct numeric values

    Args:
        maturity_str: Maturity string to parse.

    Returns:
        Maturity in years as a float.

    Raises:
        ValueError: If the format is not recognized.
    """
    maturity_str = maturity_str.strip().upper()

    # Try months format (e.g., "3M", "6M")
    month_match = re.match(r"^(\d+)M$", maturity_str)
    if month_match:
        months = int(month_match.group(1))
        return months / 12.0

    # Try years format (e.g., "1Y", "10Y")
    year_match = re.match(r"^(\d+)Y$", maturity_str)
    if year_match:
        return float(year_match.group(1))

    # Try direct numeric
    try:
        return float(maturity_str)
    except ValueError as e:
        msg = f"Cannot parse maturity format: '{maturity_str}'"
        raise ValueError(msg) from e


def load_interest_rate_data(
    file_path: Path | str,
    date_column: str = "date",
    date_format: str = "%Y-%m-%d",
    source_name: str | None = None,
) -> InterestRateDataset:
    """Load historical interest rate data from a CSV file.

    The CSV file should have a date column and one or more columns for
    different maturities. Maturity columns can be named using standard
    conventions like "3M", "6M", "1Y", "10Y", etc.

    Args:
        file_path: Path to the CSV file containing interest rate data.
        date_column: Name of the column containing observation dates.
        date_format: Format string for parsing dates.
        source_name: Optional human-readable name for the data source.

    Returns:
        InterestRateDataset containing the loaded and validated data.

    Raises:
        FileNotFoundError: If the file does not exist.
        DataValidationError: If the data fails validation checks.
    """
    file_path = Path(file_path)

    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    logger.info("Loading interest rate data from %s", file_path)

    # Read CSV
    try:
        df = pd.read_csv(file_path)
    except pd.errors.EmptyDataError as e:
        raise DataValidationError("CSV file is empty", field="file") from e

    # Validate date column exists
    if date_column not in df.columns:
        raise DataValidationError(
            f"Date column '{date_column}' not found in CSV",
            field=date_column,
        )

    # Parse dates
    try:
        df[date_column] = pd.to_datetime(df[date_column], format=date_format)
    except ValueError:
        # Try auto-detection
        try:
            df[date_column] = pd.to_datetime(df[date_column])
        except ValueError as e:
            raise DataValidationError(
                f"Cannot parse dates in column '{date_column}'",
                field=date_column,
            ) from e

    # Get observation dates
    observation_dates = [d.date() for d in df[date_column]]

    # Validate we have data
    if len(observation_dates) == 0:
        raise DataValidationError("No observations found in CSV", field=date_column)

    # Identify maturity columns (all non-date columns)
    maturity_columns = [c for c in df.columns if c != date_column]

    if len(maturity_columns) == 0:
        raise DataValidationError("No maturity columns found in CSV")

    # Parse maturities
    maturities: list[float] = []
    for col in maturity_columns:
        try:
            mat = _parse_maturity(col)
            maturities.append(mat)
        except ValueError:
            # Skip non-maturity columns
            maturity_columns.remove(col)

    if len(maturities) == 0:
        raise DataValidationError("No valid maturity columns found")

    # Sort by maturity
    sorted_indices = np.argsort(maturities)
    maturities_array = np.array(maturities)[sorted_indices]
    sorted_columns = [maturity_columns[i] for i in sorted_indices]

    # Extract rate matrix
    rates = df[sorted_columns].values.astype(np.float64)

    # Determine source name
    if source_name is None:
        source_name = file_path.stem

    logger.info(
        "Loaded %d observations with %d maturities from %s",
        len(observation_dates),
        len(maturities_array),
        source_name,
    )

    return InterestRateDataset(
        file_path=file_path,
        source_name=source_name,
        observation_dates=observation_dates,
        maturities=maturities_array,
        rates=rates,
    )

