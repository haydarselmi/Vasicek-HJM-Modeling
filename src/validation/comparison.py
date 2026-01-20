"""Model comparison utilities.

This module provides functionality for comparing multiple interest rate
models side-by-side and generating model selection recommendations.

The comparison framework supports:
- Side-by-side metrics comparison
- Use case based model recommendations
- Visual comparison outputs
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import TYPE_CHECKING, Literal

from src.utils.logging import get_logger
from src.validation.backtesting import ValidationResult, backtest_model

if TYPE_CHECKING:
    from src.data.loader import InterestRateDataset
    from src.models.base import InterestRateModel

logger = get_logger("validation.comparison")


# =============================================================================
# ComparisonResult Dataclass
# =============================================================================


@dataclass
class ComparisonResult:
    """Results from comparing multiple models.

    This dataclass stores the validation results and recommendations
    from comparing multiple interest rate models on the same data.

    Attributes:
        models: List of model names/types being compared.
        validation_results: Dictionary mapping model names to ValidationResult.
        recommendations: Dictionary of recommendations by use case.
        comparison_date: Timestamp when comparison was performed.
        best_model_by_metric: Dictionary mapping metrics to best model.
    """

    models: list[str]
    validation_results: dict[str, ValidationResult]
    recommendations: dict[str, str]
    comparison_date: datetime = field(default_factory=datetime.now)
    best_model_by_metric: dict[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Compute best model by each metric after initialization."""
        if not self.best_model_by_metric and self.validation_results:
            self._compute_best_models()

    def _compute_best_models(self) -> None:
        """Determine which model is best for each metric."""
        if len(self.validation_results) < 2:
            return

        metrics = ["mae", "rmse", "hit_ratio"]

        for metric in metrics:
            best_model = None
            best_value = None

            for model_name, result in self.validation_results.items():
                value = getattr(result, metric)

                if best_value is None:
                    best_model = model_name
                    best_value = value
                else:
                    # Lower is better for MAE/RMSE, higher is better for hit_ratio
                    if metric == "hit_ratio":
                        if value > best_value:
                            best_model = model_name
                            best_value = value
                    else:
                        if value < best_value:
                            best_model = model_name
                            best_value = value

            if best_model:
                self.best_model_by_metric[metric] = best_model

    def get_metrics_table(self) -> str:
        """Return a formatted table of comparison metrics.

        Returns:
            Formatted string table comparing all models.
        """
        header = (
            f"{'Model':<15} {'MAE (bp)':<12} {'RMSE (bp)':<12} "
            f"{'Hit Ratio':<12} {'Quality':<10}"
        )
        lines = [
            "Model Comparison Metrics",
            "=" * 70,
            header,
            "-" * 70,
        ]

        for model_name in self.models:
            result = self.validation_results[model_name]
            mae_bp = result.mae * 10000
            rmse_bp = result.rmse * 10000
            hit_pct = result.hit_ratio * 100
            quality = result.quality_rating

            line = (
                f"{model_name:<15} {mae_bp:<12.2f} {rmse_bp:<12.2f} "
                f"{hit_pct:<12.1f}% {quality:<10}"
            )
            lines.append(line)

        lines.append("-" * 70)

        # Add best model summary
        if self.best_model_by_metric:
            lines.append("")
            lines.append("Best Model by Metric:")
            for metric, model in self.best_model_by_metric.items():
                lines.append(f"  {metric.upper()}: {model}")

        return "\n".join(lines)

    def summary(self) -> str:
        """Return a complete summary of the comparison.

        Returns:
            Formatted summary string with all comparison data.
        """
        lines = [
            self.get_metrics_table(),
            "",
            "Recommendations by Use Case:",
            "-" * 40,
        ]

        for use_case, recommendation in self.recommendations.items():
            lines.append(f"  {use_case}: {recommendation}")

        return "\n".join(lines)


# =============================================================================
# Comparison Functions
# =============================================================================


def compare_models(
    models: list["InterestRateModel"],
    validation_data: "InterestRateDataset",
    model_names: list[str] | None = None,
    prediction_horizon: int = 1,
) -> ComparisonResult:
    """Compare multiple models on the same validation data.

    Runs backtesting for each model and aggregates results for
    side-by-side comparison.

    Args:
        models: List of calibrated models to compare.
        validation_data: Held-out data for validation.
        model_names: Optional custom names for models. If None, uses
            class names with indices.
        prediction_horizon: Steps ahead to predict (default: 1).

    Returns:
        ComparisonResult with side-by-side metrics and recommendations.

    Raises:
        ValueError: If fewer than 2 models provided.
        ModelNotCalibratedError: If any model is not calibrated.
    """
    if len(models) < 2:
        msg = "Need at least 2 models to compare"
        raise ValueError(msg)

    # Generate model names if not provided
    if model_names is None:
        model_names = []
        type_counts: dict[str, int] = {}
        for model in models:
            model_type = type(model).__name__
            count = type_counts.get(model_type, 0) + 1
            type_counts[model_type] = count
            if count > 1:
                model_names.append(f"{model_type}_{count}")
            else:
                model_names.append(model_type)

    if len(model_names) != len(models):
        msg = (
            f"model_names length ({len(model_names)}) "
            f"must match models ({len(models)})"
        )
        raise ValueError(msg)

    logger.info("Comparing %d models: %s", len(models), model_names)

    # Run backtesting for each model
    validation_results: dict[str, ValidationResult] = {}

    for name, model in zip(model_names, models, strict=True):
        logger.info("Backtesting model: %s", name)
        result = backtest_model(
            model=model,
            validation_data=validation_data,
            prediction_horizon=prediction_horizon,
        )
        validation_results[name] = result

    # Generate recommendations
    recommendations = _generate_recommendations(model_names, validation_results)

    logger.info("Comparison complete for %d models", len(models))

    return ComparisonResult(
        models=model_names,
        validation_results=validation_results,
        recommendations=recommendations,
    )


def _generate_recommendations(
    model_names: list[str],
    validation_results: dict[str, ValidationResult],
) -> dict[str, str]:
    """Generate model recommendations for different use cases.

    Args:
        model_names: Names of models being compared.
        validation_results: Validation results for each model.

    Returns:
        Dictionary mapping use cases to recommended models.
    """
    recommendations: dict[str, str] = {}

    # Find best by different criteria
    def get_mae(m: str) -> float:
        return validation_results[m].mae

    def get_rmse(m: str) -> float:
        return validation_results[m].rmse

    def get_hit_ratio(m: str) -> float:
        return validation_results[m].hit_ratio

    best_mae_model = min(model_names, key=get_mae)
    best_rmse_model = min(model_names, key=get_rmse)
    best_hit_model = max(model_names, key=get_hit_ratio)

    # Short rate prediction: prioritize MAE
    recommendations["short_rate"] = best_mae_model

    # Yield curve modeling: HJM preferred if available and performing well
    hjm_models = [m for m in model_names if "hjm" in m.lower()]
    if hjm_models:
        best_hjm = min(hjm_models, key=lambda m: validation_results[m].rmse)
        recommendations["yield_curve"] = best_hjm
    else:
        recommendations["yield_curve"] = best_rmse_model

    # Derivatives pricing: need consistent coverage, prioritize hit ratio
    recommendations["derivatives"] = best_hit_model

    # General purpose: balanced approach - use RMSE
    recommendations["general"] = best_rmse_model

    return recommendations


def get_model_recommendation(
    comparison: ComparisonResult,
    use_case: Literal["short_rate", "yield_curve", "derivatives", "general"],
) -> str:
    """Generate a recommendation for model selection based on use case.

    Provides detailed guidance on which model to use for specific
    applications based on the comparison results.

    Args:
        comparison: Model comparison results.
        use_case: Intended use case for the model:
            - "short_rate": Short-term rate prediction/forecasting
            - "yield_curve": Full yield curve modeling
            - "derivatives": Derivatives pricing and hedging
            - "general": General purpose rate modeling

    Returns:
        Human-readable recommendation string with justification.
    """
    if use_case not in comparison.recommendations:
        msg = f"Unknown use case: {use_case}"
        raise ValueError(msg)

    recommended_model = comparison.recommendations[use_case]
    result = comparison.validation_results[recommended_model]

    # Build recommendation text
    lines = [
        f"Recommendation for {use_case.replace('_', ' ').title()}:",
        f"  Model: {recommended_model}",
        "",
    ]

    # Add justification based on use case
    mae_bp = f"{result.mae * 10000:.2f}"
    rmse_bp = f"{result.rmse * 10000:.2f}"

    if use_case == "short_rate":
        lines.extend([
            "Justification:",
            f"  - Best Mean Absolute Error: {mae_bp} basis points",
            "  - Short rate predictions prioritize point accuracy",
            f"  - Quality rating: {result.quality_rating}",
        ])

    elif use_case == "yield_curve":
        lines.extend([
            "Justification:",
            "  - Full yield curve modeling requires curve dynamics",
        ])
        if "hjm" in recommended_model.lower():
            lines.append(
                "  - HJM captures multiple factors (level, slope, curvature)"
            )
        lines.extend([
            f"  - RMSE: {rmse_bp} basis points",
            f"  - Quality rating: {result.quality_rating}",
        ])

    elif use_case == "derivatives":
        lines.extend([
            "Justification:",
            "  - Derivatives pricing needs consistent rate distribution coverage",
            f"  - Best Hit Ratio: {result.hit_ratio:.1%}",
            "  - Higher hit ratio = more reliable probability estimates",
            f"  - Quality rating: {result.quality_rating}",
        ])

    else:  # general
        lines.extend([
            "Justification:",
            "  - General purpose modeling balances accuracy and robustness",
            f"  - RMSE: {result.rmse * 10000:.2f} basis points",
            f"  - Hit Ratio: {result.hit_ratio:.1%}",
            f"  - Quality rating: {result.quality_rating}",
        ])

    # Add warnings if quality is poor
    if result.quality_rating == "poor":
        lines.extend([
            "",
            "WARNING: Model fit quality is POOR.",
            "Consider: extending calibration period, using different model,",
            "         or investigating data quality issues.",
        ])

    return "\n".join(lines)


# =============================================================================
# Export
# =============================================================================

__all__ = [
    "ComparisonResult",
    "compare_models",
    "get_model_recommendation",
]
