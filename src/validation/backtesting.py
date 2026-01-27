"""Model backtesting and validation utilities.

This module provides functionality for backtesting interest rate models
against held-out historical data and computing error metrics.

The validation framework supports:
- Rolling prediction validation
- Error metrics: MAE, RMSE, MAPE, hit ratio
- Quality assessment based on prediction accuracy
"""

from dataclasses import dataclass, field
from datetime import date, datetime
from typing import TYPE_CHECKING, Literal

import numpy as np
import numpy.typing as npt

from src.utils.logging import get_logger

if TYPE_CHECKING:
    from src.data.loader import InterestRateDataset
    from src.models.base import InterestRateModel
    from src.models.hjm import HJMModel
    from src.models.vasicek import VasicekModel

logger = get_logger("validation.backtesting")


# =============================================================================
# Error Metric Functions
# =============================================================================


def compute_mae(
    predicted: npt.NDArray[np.float64],
    actual: npt.NDArray[np.float64],
) -> float:
    """Compute Mean Absolute Error (MAE).

    MAE = (1/n) * Σ|predicted_i - actual_i|

    Args:
        predicted: Array of predicted values.
        actual: Array of actual values.

    Returns:
        Mean Absolute Error (non-negative).

    Raises:
        ValueError: If arrays have different lengths.
    """
    if len(predicted) != len(actual):
        msg = (
            f"Arrays must have same length: "
            f"{len(predicted)} != {len(actual)}"
        )
        raise ValueError(msg)

    return float(np.mean(np.abs(predicted - actual)))


def compute_rmse(
    predicted: npt.NDArray[np.float64],
    actual: npt.NDArray[np.float64],
) -> float:
    """Compute Root Mean Squared Error (RMSE).

    RMSE = sqrt((1/n) * Σ(predicted_i - actual_i)²)

    RMSE penalizes larger errors more heavily than MAE due to
    the squaring of errors.

    Args:
        predicted: Array of predicted values.
        actual: Array of actual values.

    Returns:
        Root Mean Squared Error (non-negative).

    Raises:
        ValueError: If arrays have different lengths.
    """
    if len(predicted) != len(actual):
        msg = (
            f"Arrays must have same length: "
            f"{len(predicted)} != {len(actual)}"
        )
        raise ValueError(msg)

    return float(np.sqrt(np.mean((predicted - actual) ** 2)))


def compute_mape(
    predicted: npt.NDArray[np.float64],
    actual: npt.NDArray[np.float64],
) -> float | None:
    """Compute Mean Absolute Percentage Error (MAPE).

    MAPE = (100/n) * Σ|predicted_i - actual_i| / |actual_i|

    MAPE is undefined when actual values are zero or cross zero.
    Returns None in such cases.

    Args:
        predicted: Array of predicted values.
        actual: Array of actual values.

    Returns:
        Mean Absolute Percentage Error as a percentage (e.g., 5.0 = 5%),
        or None if actual contains zero values.

    Raises:
        ValueError: If arrays have different lengths.
    """
    if len(predicted) != len(actual):
        msg = (
            f"Arrays must have same length: "
            f"{len(predicted)} != {len(actual)}"
        )
        raise ValueError(msg)

    # Check for zero values
    if np.any(actual == 0):
        logger.warning("MAPE undefined: actual values contain zeros")
        return None

    # Check for sign changes (crossing zero)
    if np.any(actual < 0) and np.any(actual > 0):
        logger.warning("MAPE may be misleading: actual values cross zero")
        # We can still compute it, but log warning

    absolute_percentage_errors = np.abs(predicted - actual) / np.abs(actual)
    return float(np.mean(absolute_percentage_errors) * 100)


def compute_hit_ratio(
    predicted: npt.NDArray[np.float64],
    actual: npt.NDArray[np.float64],
    std: npt.NDArray[np.float64],
    n_std: float = 1.0,
) -> float:
    """Compute hit ratio (proportion of predictions within tolerance).

    Hit ratio = (# predictions within n_std * std of actual) / n

    This measures what fraction of predictions fall within an acceptable
    error band around the actual values.

    Args:
        predicted: Array of predicted values.
        actual: Array of actual values.
        std: Array of standard deviations for tolerance.
        n_std: Number of standard deviations for tolerance (default: 1.0).

    Returns:
        Hit ratio between 0 and 1.

    Raises:
        ValueError: If arrays have different lengths.
    """
    if len(predicted) != len(actual) or len(predicted) != len(std):
        msg = "All arrays must have the same length"
        raise ValueError(msg)

    errors = np.abs(predicted - actual)
    tolerance = n_std * std

    hits = errors <= tolerance
    return float(np.mean(hits))


# =============================================================================
# ValidationResult Dataclass
# =============================================================================


@dataclass
class ValidationResult:
    """Results from model backtesting and validation.

    This dataclass stores the validation metrics and data from
    backtesting a model against held-out historical data.

    Attributes:
        model_type: Type of model being validated ("vasicek" or "hjm").
        validation_period: Tuple of (start_date, end_date) for validation data.
        mae: Mean Absolute Error.
        rmse: Root Mean Squared Error.
        mape: Mean Absolute Percentage Error (None if undefined).
        hit_ratio: Proportion of predictions within 1 std of actual.
        predicted_values: Array of model predictions.
        actual_values: Array of actual observed values.
        validation_date: Timestamp when validation was performed.

    Interpretation Guidelines:
        - MAE < 0.002 (20bp): Excellent fit
        - MAE 0.002-0.005 (20-50bp): Good fit
        - MAE > 0.005 (50bp): Poor fit, consider model revision
    """

    model_type: Literal["vasicek", "hjm"]
    validation_period: tuple[date, date]
    mae: float
    rmse: float
    mape: float | None
    hit_ratio: float
    predicted_values: npt.NDArray[np.float64]
    actual_values: npt.NDArray[np.float64]
    validation_date: datetime = field(default_factory=datetime.now)

    @property
    def quality_rating(self) -> Literal["excellent", "good", "poor"]:
        """Return a quality rating based on MAE.

        Returns:
            "excellent" if MAE < 20bp (0.002)
            "good" if MAE between 20-50bp (0.002-0.005)
            "poor" if MAE > 50bp (0.005)
        """
        if self.mae < 0.002:
            return "excellent"
        elif self.mae < 0.005:
            return "good"
        else:
            return "poor"

    @property
    def n_observations(self) -> int:
        """Return the number of validation observations."""
        return len(self.actual_values)

    def summary(self) -> str:
        """Return a human-readable summary of validation results.

        Returns:
            Formatted summary string with all metrics.
        """
        lines = [
            f"Validation Results for {self.model_type.upper()} Model",
            "=" * 50,
            f"Validation Period: {self.validation_period[0]} to {self.validation_period[1]}",
            f"Number of Observations: {self.n_observations}",
            "",
            "Error Metrics:",
            f"  MAE:  {self.mae:.6f} ({self.mae * 10000:.2f} basis points)",
            f"  RMSE: {self.rmse:.6f} ({self.rmse * 10000:.2f} basis points)",
        ]

        if self.mape is not None:
            lines.append(f"  MAPE: {self.mape:.2f}%")
        else:
            lines.append("  MAPE: N/A (undefined)")

        lines.extend([
            f"  Hit Ratio: {self.hit_ratio:.1%}",
            "",
            f"Quality Rating: {self.quality_rating.upper()}",
        ])

        return "\n".join(lines)


# =============================================================================
# Backtest Function
# =============================================================================


def backtest_model(
    model: InterestRateModel,
    validation_data: InterestRateDataset,
    prediction_horizon: int = 1,
) -> ValidationResult:
    """Backtest a calibrated model against held-out data.

    Performs rolling prediction validation where the model makes
    predictions for each time step and compares against actuals.

    For Vasicek models, the expected rate formula is used:
        E[r(t+h)] = r(t) * exp(-κh) + θ * (1 - exp(-κh))

    For HJM models, the forward rate simulation is used.

    Args:
        model: Calibrated interest rate model.
        validation_data: Held-out dataset for validation.
        prediction_horizon: Number of steps ahead to predict.
            Default is 1 (one-step-ahead prediction).

    Returns:
        ValidationResult with error metrics and predictions.

    Raises:
        ModelNotCalibratedError: If model is not calibrated.
        ValueError: If validation data is too short.
    """
    from src.models.hjm import HJMModel as HJMModelClass
    from src.models.vasicek import VasicekModel as VasicekModelClass
    from src.utils.exceptions import ModelNotCalibratedError

    # Check calibration status
    if not model.is_calibrated:
        msg = "Model must be calibrated before backtesting"
        raise ModelNotCalibratedError(
            msg, model_type=type(model).__name__
        )

    n_obs = validation_data.n_observations
    if n_obs < prediction_horizon + 1:
        msg = f"Validation data too short: {n_obs} < {prediction_horizon + 1}"
        raise ValueError(msg)

    logger.info(
        "Backtesting %s model with %d observations, horizon=%d",
        type(model).__name__,
        n_obs,
        prediction_horizon,
    )

    # Determine model type and compute predictions
    if isinstance(model, VasicekModelClass):
        model_type: Literal["vasicek", "hjm"] = "vasicek"
        predicted, actual, std = _backtest_vasicek(
            model, validation_data, prediction_horizon
        )
    elif isinstance(model, HJMModelClass):
        model_type = "hjm"
        predicted, actual, std = _backtest_hjm(
            model, validation_data, prediction_horizon
        )
    else:
        msg = f"Unknown model type: {type(model).__name__}"
        raise ValueError(msg)

    # Compute error metrics
    mae = compute_mae(predicted, actual)
    rmse = compute_rmse(predicted, actual)
    mape = compute_mape(predicted, actual)
    hit_ratio = compute_hit_ratio(predicted, actual, std)

    logger.info(
        "Backtest complete: MAE=%.6f, RMSE=%.6f, Hit Ratio=%.2f%%",
        mae,
        rmse,
        hit_ratio * 100,
    )

    return ValidationResult(
        model_type=model_type,
        validation_period=(
            validation_data.observation_dates[0],
            validation_data.observation_dates[-1],
        ),
        mae=mae,
        rmse=rmse,
        mape=mape,
        hit_ratio=hit_ratio,
        predicted_values=predicted,
        actual_values=actual,
    )


def _backtest_vasicek(
    model: VasicekModel,
    validation_data: InterestRateDataset,
    prediction_horizon: int,
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """Perform backtesting for Vasicek model.

    Uses the analytical expected rate formula for predictions.

    Args:
        model: Calibrated Vasicek model.
        validation_data: Validation dataset.
        prediction_horizon: Steps ahead to predict.

    Returns:
        Tuple of (predicted, actual, std) arrays.
    """
    # Extract short rates (first maturity)
    short_rates = validation_data.rates[:, 0]
    n_obs = len(short_rates)
    n_predictions = n_obs - prediction_horizon

    predicted = np.zeros(n_predictions)
    actual = np.zeros(n_predictions)
    std = np.zeros(n_predictions)

    # Time step (assuming daily data)
    dt = 1.0 / 252.0
    t_horizon = prediction_horizon * dt

    for i in range(n_predictions):
        r0 = short_rates[i]

        # Expected rate at t + horizon
        predicted[i] = model.expected_rate(t_horizon, r0)

        # Actual rate at t + horizon
        actual[i] = short_rates[i + prediction_horizon]

        # Standard deviation at t + horizon
        std[i] = np.sqrt(model.rate_variance(t_horizon))

    return predicted, actual, std


def _backtest_hjm(
    model: HJMModel,
    validation_data: InterestRateDataset,
    prediction_horizon: int,
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """Perform backtesting for HJM model.

    Uses the mean forward curve evolution for predictions.

    Args:
        model: Calibrated HJM model.
        validation_data: Validation dataset.
        prediction_horizon: Steps ahead to predict.

    Returns:
        Tuple of (predicted, actual, std) arrays for short rate.
    """
    n_obs = validation_data.n_observations
    n_predictions = n_obs - prediction_horizon

    predicted = np.zeros(n_predictions)
    actual = np.zeros(n_predictions)
    std = np.zeros(n_predictions)

    # Time step
    dt = 1.0 / 252.0
    t_horizon = prediction_horizon * dt

    # For HJM, we use short Monte Carlo simulations for each prediction
    n_sim_paths = 100  # Small for speed

    for i in range(n_predictions):
        initial_curve = validation_data.rates[i]

        # Simulate forward curves
        result = model.simulate(
            n_paths=n_sim_paths,
            time_horizon=t_horizon,
            time_step=dt,
            initial_curve=initial_curve,
            random_seed=42 + i,  # Different seed for each prediction
        )

        # Get terminal short rate (first maturity of terminal curve)
        terminal_curves = result.terminal_distribution()
        terminal_short_rates = terminal_curves[:, 0]

        predicted[i] = np.mean(terminal_short_rates)
        actual[i] = validation_data.rates[i + prediction_horizon, 0]
        std[i] = np.std(terminal_short_rates)

    return predicted, actual, std


# =============================================================================
# Export
# =============================================================================

__all__ = [
    "ValidationResult",
    "backtest_model",
    "compute_mae",
    "compute_rmse",
    "compute_mape",
    "compute_hit_ratio",
]
