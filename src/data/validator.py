"""Data validation functionality for interest rate datasets.

This module provides functions for validating data quality, detecting
issues, and handling missing values in interest rate datasets.
"""

from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Literal

import numpy as np

from src.data.loader import InterestRateDataset
from src.data.types import RATE_MAX_BOUND, RATE_MIN_BOUND
from src.utils.exceptions import DataValidationError
from src.utils.logging import get_logger

logger = get_logger("data.validator")

# Maximum allowed percentage of missing values
MAX_MISSING_PERCENT = 5.0


@dataclass
class ValidationReport:
    """Report from validating an interest rate dataset.

    Attributes:
        warnings: List of warning messages (non-critical issues).
        errors: List of error messages (critical issues).
        is_valid: Whether the dataset passes validation.
        validated_at: Timestamp of validation.
    """

    warnings: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    is_valid: bool = True
    validated_at: datetime = field(default_factory=datetime.now)

    def add_warning(self, message: str) -> None:
        """Add a warning message.

        Args:
            message: Warning message to add.
        """
        self.warnings.append(message)

    def add_error(self, message: str) -> None:
        """Add an error message and mark as invalid.

        Args:
            message: Error message to add.
        """
        self.errors.append(message)
        self.is_valid = False


def validate_dataset(
    dataset: InterestRateDataset,
    strict: bool = False,
) -> ValidationReport:
    """Validate an interest rate dataset for completeness and consistency.

    Performs the following checks:
    - Missing/NaN values detection
    - Rate values within reasonable bounds
    - Chronological order of dates
    - No duplicate dates

    Args:
        dataset: The dataset to validate.
        strict: If True, raise on warnings; if False, only raise on errors.

    Returns:
        ValidationReport containing any warnings or issues found.

    Raises:
        DataValidationError: If validation fails critically or strict mode is on.
    """
    logger.info("Validating dataset with %d observations", dataset.n_observations)

    report = ValidationReport()

    # Check for missing/NaN values
    _check_missing_values(dataset, report)

    # Check rate bounds
    _check_rate_bounds(dataset, report)

    # Check chronological order
    _check_chronological_order(dataset, report)

    # Check for duplicate dates
    _check_duplicate_dates(dataset, report)

    # Check minimum observations
    _check_minimum_observations(dataset, report)

    # Log results
    if report.is_valid:
        logger.info("Validation passed with %d warnings", len(report.warnings))
    else:
        logger.warning("Validation failed with %d errors", len(report.errors))

    # Handle strict mode
    if strict and report.warnings:
        raise DataValidationError(
            f"Strict validation failed: {report.warnings[0]}",
            details={"warnings": report.warnings},
        )

    if not report.is_valid:
        raise DataValidationError(
            f"Validation failed: {report.errors[0]}",
            details={"errors": report.errors},
        )

    return report


def _check_missing_values(dataset: InterestRateDataset, report: ValidationReport) -> None:
    """Check for missing/NaN values in the rate matrix."""
    missing_mask = np.isnan(dataset.rates)
    n_missing = np.sum(missing_mask)

    if n_missing > 0:
        pct_missing = 100.0 * n_missing / dataset.rates.size
        report.add_warning(
            f"Found {n_missing} missing values ({pct_missing:.2f}% of data)"
        )


def _check_rate_bounds(dataset: InterestRateDataset, report: ValidationReport) -> None:
    """Check that rate values are within reasonable bounds."""
    valid_rates = dataset.rates[~np.isnan(dataset.rates)]

    if len(valid_rates) == 0:
        report.add_error("No valid rate values found")
        return

    min_rate = np.min(valid_rates)
    max_rate = np.max(valid_rates)

    # Check extreme bounds (error level)
    if min_rate < RATE_MIN_BOUND:
        report.add_warning(
            f"Rate value {min_rate:.4f} is below minimum bound {RATE_MIN_BOUND}"
        )

    if max_rate > RATE_MAX_BOUND:
        report.add_warning(
            f"Rate value {max_rate:.4f} is above maximum bound {RATE_MAX_BOUND}"
        )

    # Check typical bounds (warning level)
    if min_rate < -0.02 or max_rate > 0.20:
        report.add_warning(
            f"Rate range [{min_rate:.4f}, {max_rate:.4f}] is outside typical range [-0.02, 0.20]"
        )


def _check_chronological_order(dataset: InterestRateDataset, report: ValidationReport) -> None:
    """Check that dates are in chronological order."""
    dates = dataset.observation_dates

    for i in range(1, len(dates)):
        if dates[i] <= dates[i - 1]:
            report.add_error(
                f"Dates not in chronological order: {dates[i-1]} >= {dates[i]} at index {i}"
            )
            return  # Only report first instance


def _check_duplicate_dates(dataset: InterestRateDataset, report: ValidationReport) -> None:
    """Check for duplicate dates."""
    dates = dataset.observation_dates
    unique_dates = set(dates)

    if len(unique_dates) != len(dates):
        n_duplicates = len(dates) - len(unique_dates)
        report.add_warning(f"Found {n_duplicates} duplicate dates")


def _check_minimum_observations(dataset: InterestRateDataset, report: ValidationReport) -> None:
    """Check that we have minimum required observations."""
    if dataset.n_observations < 30:
        report.add_warning(
            f"Dataset has only {dataset.n_observations} observations; "
            "at least 30 recommended for meaningful calibration"
        )


def handle_missing_values(
    dataset: InterestRateDataset,
    method: Literal["interpolate", "forward_fill", "drop"] = "interpolate",
) -> InterestRateDataset:
    """Handle missing values in the dataset.

    Args:
        dataset: Dataset with potential missing values.
        method: Strategy for handling missing values:
            - "interpolate": Linear interpolation along time axis
            - "forward_fill": Fill with previous valid value
            - "drop": Remove rows with any missing values

    Returns:
        New dataset with missing values handled.

    Raises:
        DataValidationError: If missing values exceed 5% of total.
    """
    rates = dataset.rates.copy()
    missing_mask = np.isnan(rates)
    n_missing = np.sum(missing_mask)

    if n_missing == 0:
        logger.info("No missing values to handle")
        return dataset

    # Check threshold
    pct_missing = 100.0 * n_missing / rates.size
    if pct_missing > MAX_MISSING_PERCENT:
        raise DataValidationError(
            f"Missing values ({pct_missing:.1f}%) exceed maximum allowed ({MAX_MISSING_PERCENT}%)",
            details={"missing_count": int(n_missing), "missing_percent": pct_missing},
        )

    logger.info("Handling %d missing values using method '%s'", n_missing, method)

    if method == "interpolate":
        return _interpolate_missing(dataset, rates)
    elif method == "forward_fill":
        return _forward_fill_missing(dataset, rates)
    elif method == "drop":
        return _drop_missing_rows(dataset, rates)
    else:
        raise ValueError(f"Unknown method: {method}")


def _interpolate_missing(
    dataset: InterestRateDataset,
    rates: np.ndarray,
) -> InterestRateDataset:
    """Interpolate missing values along the time axis."""
    for col in range(rates.shape[1]):
        column = rates[:, col]
        valid_mask = ~np.isnan(column)

        if np.any(~valid_mask):
            valid_indices = np.where(valid_mask)[0]
            invalid_indices = np.where(~valid_mask)[0]

            if len(valid_indices) >= 2:
                column[invalid_indices] = np.interp(
                    invalid_indices,
                    valid_indices,
                    column[valid_indices],
                )
            elif len(valid_indices) == 1:
                # Only one valid value - use it for all
                column[invalid_indices] = column[valid_indices[0]]

    return InterestRateDataset(
        file_path=dataset.file_path,
        source_name=dataset.source_name,
        observation_dates=dataset.observation_dates,
        maturities=dataset.maturities.copy(),
        rates=rates,
        loaded_at=dataset.loaded_at,
    )


def _forward_fill_missing(
    dataset: InterestRateDataset,
    rates: np.ndarray,
) -> InterestRateDataset:
    """Forward fill missing values."""
    for col in range(rates.shape[1]):
        for row in range(1, rates.shape[0]):
            if np.isnan(rates[row, col]) and not np.isnan(rates[row - 1, col]):
                rates[row, col] = rates[row - 1, col]

    # Handle any remaining NaNs at the start with backward fill
    for col in range(rates.shape[1]):
        for row in range(rates.shape[0] - 2, -1, -1):
            if np.isnan(rates[row, col]) and not np.isnan(rates[row + 1, col]):
                rates[row, col] = rates[row + 1, col]

    return InterestRateDataset(
        file_path=dataset.file_path,
        source_name=dataset.source_name,
        observation_dates=dataset.observation_dates,
        maturities=dataset.maturities.copy(),
        rates=rates,
        loaded_at=dataset.loaded_at,
    )


def _drop_missing_rows(
    dataset: InterestRateDataset,
    rates: np.ndarray,
) -> InterestRateDataset:
    """Drop rows containing any missing values."""
    valid_rows = ~np.any(np.isnan(rates), axis=1)
    valid_indices = np.where(valid_rows)[0]

    new_dates = [dataset.observation_dates[i] for i in valid_indices]
    new_rates = rates[valid_rows]

    return InterestRateDataset(
        file_path=dataset.file_path,
        source_name=dataset.source_name,
        observation_dates=new_dates,
        maturities=dataset.maturities.copy(),
        rates=new_rates,
        loaded_at=dataset.loaded_at,
    )

