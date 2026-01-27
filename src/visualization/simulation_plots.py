"""Visualization functions for simulation results.

This module provides functions for plotting Monte Carlo simulation
paths, summary statistics, and terminal rate distributions.
"""

from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt

from src.simulation.paths import SimulationResult

if TYPE_CHECKING:
    from src.models.hjm import HJMModel
    from src.validation.backtesting import ValidationResult
    from src.validation.comparison import ComparisonResult


def plot_simulation_paths(
    result: SimulationResult,
    n_paths: int = 100,
    show_mean: bool = True,
    show_percentiles: bool = True,
    percentiles: list[float] | None = None,
    ax: plt.Axes | None = None,
    title: str | None = None,
    alpha: float = 0.3,
) -> plt.Figure:
    """Plot simulated rate paths with summary statistics.

    Creates a visualization of Monte Carlo simulation paths showing
    individual paths, the mean path, and percentile bands.

    Args:
        result: SimulationResult containing paths to plot.
        n_paths: Maximum number of individual paths to display.
            Defaults to 100. If result has more paths, a random
            sample is shown.
        show_mean: Whether to show the mean path as a bold line.
        show_percentiles: Whether to show percentile bands.
        percentiles: List of percentiles for bands. Defaults to
            [5, 25, 75, 95] which creates a 90% and 50% band.
        ax: Optional matplotlib axes to plot on. If None, creates
            a new figure and axes.
        title: Optional plot title. If None, a default title is used.
        alpha: Transparency for individual path lines (0-1).

    Returns:
        Matplotlib Figure containing the plot.

    Example:
        >>> result = model.simulate(n_paths=1000, time_horizon=1.0, time_step=1/252)
        >>> fig = plot_simulation_paths(result, n_paths=50, show_percentiles=True)
        >>> fig.savefig("simulation.png")
    """
    if percentiles is None:
        percentiles = [5.0, 25.0, 75.0, 95.0]

    if ax is None:
        fig, ax = plt.subplots(figsize=(12, 6))
    else:
        fig = ax.figure

    time_grid = result.time_grid
    rates_matrix = result.get_rates_matrix()

    # Sample paths if there are too many
    n_total = result.path_count
    if n_total > n_paths:
        rng = np.random.default_rng(42)
        sample_indices = rng.choice(n_total, size=n_paths, replace=False)
        sampled_rates = rates_matrix[sample_indices, :]
    else:
        sampled_rates = rates_matrix
        n_paths = n_total

    # Plot individual paths
    for i in range(len(sampled_rates)):
        ax.plot(
            time_grid,
            sampled_rates[i, :] * 100,  # Convert to percentage
            color="steelblue",
            alpha=alpha,
            linewidth=0.5,
        )

    # Plot percentile bands
    if show_percentiles and len(percentiles) >= 2:
        percentile_paths = result.percentile_paths(percentiles)

        # Outer band (e.g., 5-95%)
        if 5.0 in percentile_paths and 95.0 in percentile_paths:
            ax.fill_between(
                time_grid,
                percentile_paths[5.0] * 100,
                percentile_paths[95.0] * 100,
                alpha=0.2,
                color="orange",
                label="5th-95th percentile",
            )

        # Inner band (e.g., 25-75%)
        if 25.0 in percentile_paths and 75.0 in percentile_paths:
            ax.fill_between(
                time_grid,
                percentile_paths[25.0] * 100,
                percentile_paths[75.0] * 100,
                alpha=0.3,
                color="orange",
                label="25th-75th percentile",
            )

    # Plot mean path
    if show_mean:
        mean_path = result.mean_path()
        ax.plot(
            time_grid,
            mean_path * 100,
            color="darkred",
            linewidth=2,
            label="Mean path",
        )

    # Formatting
    ax.set_xlabel("Time (years)")
    ax.set_ylabel("Interest Rate (%)")

    if title is None:
        model_type = result.model_type.capitalize()
        title = f"{model_type} Model Simulation ({result.path_count:,} paths)"
    ax.set_title(title)

    ax.legend(loc="upper right")
    ax.grid(True, alpha=0.3)

    # Set reasonable y-axis limits
    all_rates = rates_matrix.flatten() * 100
    y_min = np.percentile(all_rates, 1)
    y_max = np.percentile(all_rates, 99)
    margin = (y_max - y_min) * 0.1
    ax.set_ylim(y_min - margin, y_max + margin)

    fig.tight_layout()
    return fig


def plot_terminal_distribution(
    result: SimulationResult,
    bins: int = 50,
    show_stats: bool = True,
    show_density: bool = True,
    ax: plt.Axes | None = None,
    title: str | None = None,
) -> plt.Figure:
    """Plot the distribution of terminal (final) rates.

    Creates a histogram or density plot showing the distribution
    of rates at the end of the simulation horizon.

    Args:
        result: SimulationResult containing paths.
        bins: Number of histogram bins.
        show_stats: Whether to display summary statistics on plot.
        show_density: If True, plot normalized density. If False,
            plot frequency counts.
        ax: Optional matplotlib axes to plot on.
        title: Optional plot title.

    Returns:
        Matplotlib Figure containing the plot.

    Example:
        >>> result = model.simulate(n_paths=10000, time_horizon=1.0, time_step=1/252)
        >>> fig = plot_terminal_distribution(result, bins=100)
        >>> fig.savefig("terminal_distribution.png")
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 6))
    else:
        fig = ax.figure

    terminal_rates = result.terminal_distribution()

    # Convert to percentage for display
    terminal_rates_pct = terminal_rates * 100

    # Plot histogram
    ax.hist(
        terminal_rates_pct,
        bins=bins,
        density=show_density,
        color="steelblue",
        edgecolor="white",
        alpha=0.7,
    )

    # Add vertical lines for statistics
    mean_rate = np.mean(terminal_rates_pct)
    median_rate = np.median(terminal_rates_pct)

    ax.axvline(
        mean_rate,
        color="darkred",
        linestyle="-",
        linewidth=2,
        label=f"Mean: {mean_rate:.2f}%",
    )
    ax.axvline(
        median_rate,
        color="green",
        linestyle="--",
        linewidth=2,
        label=f"Median: {median_rate:.2f}%",
    )

    # Add percentile lines
    p5 = np.percentile(terminal_rates_pct, 5)
    p95 = np.percentile(terminal_rates_pct, 95)
    ax.axvline(
        p5,
        color="orange",
        linestyle=":",
        linewidth=1.5,
        label=f"5th percentile: {p5:.2f}%",
    )
    ax.axvline(
        p95,
        color="orange",
        linestyle=":",
        linewidth=1.5,
        label=f"95th percentile: {p95:.2f}%",
    )

    # Add statistics text box
    if show_stats:
        std_rate = np.std(terminal_rates_pct)
        skew = _compute_skewness(terminal_rates)
        kurt = _compute_kurtosis(terminal_rates)

        stats_text = (
            f"N = {len(terminal_rates):,}\n"
            f"Mean = {mean_rate:.4f}%\n"
            f"Std = {std_rate:.4f}%\n"
            f"Skew = {skew:.3f}\n"
            f"Kurt = {kurt:.3f}"
        )

        ax.text(
            0.02,
            0.98,
            stats_text,
            transform=ax.transAxes,
            verticalalignment="top",
            fontsize=10,
            fontfamily="monospace",
            bbox={"boxstyle": "round", "facecolor": "wheat", "alpha": 0.5},
        )

    # Formatting
    ax.set_xlabel("Terminal Interest Rate (%)")
    ylabel = "Density" if show_density else "Frequency"
    ax.set_ylabel(ylabel)

    if title is None:
        model_type = result.model_type.capitalize()
        horizon = result.time_horizon
        title = f"{model_type} Terminal Rate Distribution (T = {horizon:.1f} years)"
    ax.set_title(title)

    ax.legend(loc="upper right")
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    return fig


def _compute_skewness(data: npt.NDArray[np.float64]) -> float:
    """Compute Fisher skewness of data."""
    n = len(data)
    mean = np.mean(data)
    std = np.std(data, ddof=1)
    if std < 1e-10:
        return 0.0
    return np.sum(((data - mean) / std) ** 3) * n / ((n - 1) * (n - 2))


def _compute_kurtosis(data: npt.NDArray[np.float64]) -> float:
    """Compute excess kurtosis of data."""
    mean = np.mean(data)
    std = np.std(data, ddof=1)
    if std < 1e-10:
        return 0.0
    m4 = np.mean((data - mean) ** 4)
    return (m4 / (std**4)) - 3


# =============================================================================
# HJM-Specific Visualization Functions
# =============================================================================


def plot_volatility_functions(
    model: HJMModel,
    ax: plt.Axes | None = None,
    title: str | None = None,
) -> plt.Figure:
    """Plot HJM volatility functions by factor and maturity.

    Creates a visualization showing how each factor's volatility varies
    across maturities, revealing the level/slope/curvature patterns.

    Args:
        model: Calibrated HJMModel instance.
        ax: Optional matplotlib axes to plot on.
        title: Optional plot title.

    Returns:
        Matplotlib Figure containing the volatility plot.

    Raises:
        ModelNotCalibratedError: If model is not calibrated.

    Example:
        >>> model = HJMModel(pca_result=pca_result)
        >>> model.calibrate()
        >>> fig = plot_volatility_functions(model)
        >>> fig.savefig("hjm_volatility.png")
    """
    if not model.is_calibrated:
        from src.utils.exceptions import ModelNotCalibratedError

        raise ModelNotCalibratedError(model_type="HJM")

    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 6))
    else:
        fig = ax.figure

    maturities = model.maturities
    volatility_functions = model.volatility_functions
    n_factors = model.n_factors

    # Color palette for factors
    cmap = plt.colormaps["viridis"]
    colors = cmap(np.linspace(0.2, 0.8, n_factors))

    for i in range(n_factors):
        # Get factor label
        if i < len(model.pca_result.interpretations):
            label = model.pca_result.interpretations[i]
        else:
            label = f"Factor {i + 1}"

        # Add variance explained to label
        var_pct = model.pca_result.explained_variance_ratio[i] * 100
        label = f"{label} ({var_pct:.1f}%)"

        ax.plot(
            maturities,
            volatility_functions[i] * 100,  # Convert to percentage
            marker="o",
            markersize=6,
            linewidth=2,
            color=colors[i],
            label=label,
        )

    # Add zero line for reference
    ax.axhline(y=0, color="gray", linestyle="--", alpha=0.5)

    # Formatting
    ax.set_xlabel("Maturity (years)")
    ax.set_ylabel("Volatility (% per √day)")

    if title is None:
        title = f"HJM Volatility Functions ({n_factors} Factors)"
    ax.set_title(title)

    ax.legend(loc="best")
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    return fig


def plot_forward_curve_evolution(
    result: SimulationResult,
    n_curves: int = 10,
    time_indices: list[int] | None = None,
    path_index: int = 0,
    ax: plt.Axes | None = None,
    title: str | None = None,
    colormap: str = "viridis",
) -> plt.Figure:
    """Plot the evolution of forward curves over time.

    Shows how the forward rate curve changes through the simulation,
    with curves at different time points shown in different colors.

    Args:
        result: SimulationResult from HJM simulation.
        n_curves: Number of curves to display (evenly spaced in time).
        time_indices: Specific time indices to plot. If None, evenly space n_curves.
        path_index: Which path to visualize (default: first path).
        ax: Optional matplotlib axes to plot on.
        title: Optional plot title.
        colormap: Matplotlib colormap name for time coloring.

    Returns:
        Matplotlib Figure containing the forward curve evolution plot.

    Example:
        >>> result = hjm_model.simulate(n_paths=100, time_horizon=1.0, time_step=1/252)
        >>> fig = plot_forward_curve_evolution(result, n_curves=10)
        >>> fig.savefig("forward_curve_evolution.png")
    """
    if result.model_type != "hjm":
        raise ValueError(f"Expected HJM simulation result, got {result.model_type}")

    if ax is None:
        fig, ax = plt.subplots(figsize=(12, 6))
    else:
        fig = ax.figure

    # Get the path to visualize
    path = result.paths[path_index]
    rates = path.rates  # Shape: (n_time_points, n_maturities)
    time_grid = path.time_grid

    n_time_points = len(time_grid)

    # Determine which time indices to plot
    if time_indices is None:
        time_indices = np.linspace(0, n_time_points - 1, n_curves, dtype=int).tolist()

    # We need maturities - extract from first curve or use indices
    # For HJM, we typically have maturities stored somewhere
    # Use array indices as proxy if not available
    n_maturities = rates.shape[1]
    maturities: npt.NDArray[np.float64] = np.arange(
        n_maturities, dtype=np.float64
    )

    # Try to get actual maturities from model parameters
    if "n_factors" in result.model_parameters:
        # This is an HJM result, but we don't have maturities directly
        # Use standard treasury maturities as approximation
        if n_maturities == 7:
            maturities = np.array(
                [0.25, 0.5, 1.0, 2.0, 5.0, 10.0, 30.0], dtype=np.float64
            )
        else:
            maturities = np.linspace(0.25, 30, n_maturities)

    # Color mapping based on time
    cmap = plt.colormaps[colormap]
    colors = [cmap(t / (n_time_points - 1)) for t in time_indices]

    # Plot curves at selected times
    for idx, t_idx in enumerate(time_indices):
        curve = rates[t_idx]
        time_label = time_grid[t_idx]

        ax.plot(
            maturities,
            curve * 100,  # Convert to percentage
            color=colors[idx],
            linewidth=2,
            alpha=0.8,
            label=f"t = {time_label:.2f}y",
        )

    # Highlight initial and terminal curves
    ax.plot(
        maturities,
        rates[0] * 100,
        color="blue",
        linewidth=3,
        linestyle="--",
        label="Initial (t=0)",
    )
    ax.plot(
        maturities,
        rates[-1] * 100,
        color="red",
        linewidth=3,
        linestyle="--",
        label=f"Terminal (t={time_grid[-1]:.1f}y)",
    )

    # Formatting
    ax.set_xlabel("Maturity (years)")
    ax.set_ylabel("Forward Rate (%)")

    if title is None:
        title = f"Forward Curve Evolution (Path {path_index + 1})"
    ax.set_title(title)

    ax.legend(loc="best", fontsize=8)
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    return fig


def plot_model_comparison(
    comparison_result: ComparisonResult,
    ax: plt.Axes | None = None,
    title: str | None = None,
    metric: str = "all",
) -> plt.Figure:
    """Plot side-by-side model comparison metrics.

    Creates a bar chart comparing error metrics across multiple models,
    allowing visual assessment of model performance.

    Args:
        comparison_result: ComparisonResult from compare_models().
        ax: Optional matplotlib axes to plot on.
        title: Optional plot title.
        metric: Which metric(s) to plot:
            - "all": Plot MAE, RMSE, and hit ratio (default)
            - "mae": Plot only MAE
            - "rmse": Plot only RMSE
            - "hit_ratio": Plot only hit ratio

    Returns:
        Matplotlib Figure containing the comparison plot.

    Example:
        >>> comparison = compare_models([vasicek, hjm], validation_data)
        >>> fig = plot_model_comparison(comparison)
        >>> fig.savefig("model_comparison.png")
    """
    from src.validation.comparison import ComparisonResult

    if not isinstance(comparison_result, ComparisonResult):
        raise TypeError("Expected ComparisonResult instance")

    models = comparison_result.models

    if metric == "all":
        # Create subplot with 3 metrics
        fig, axes = plt.subplots(1, 3, figsize=(14, 5))

        # MAE comparison
        mae_values = [
            comparison_result.validation_results[m].mae * 10000
            for m in models
        ]
        axes[0].bar(models, mae_values, color="steelblue", edgecolor="black")
        axes[0].set_ylabel("MAE (basis points)")
        axes[0].set_title("Mean Absolute Error")
        axes[0].tick_params(axis="x", rotation=45)

        # RMSE comparison
        rmse_values = [
            comparison_result.validation_results[m].rmse * 10000
            for m in models
        ]
        axes[1].bar(models, rmse_values, color="coral", edgecolor="black")
        axes[1].set_ylabel("RMSE (basis points)")
        axes[1].set_title("Root Mean Squared Error")
        axes[1].tick_params(axis="x", rotation=45)

        # Hit ratio comparison
        hit_values = [
            comparison_result.validation_results[m].hit_ratio * 100
            for m in models
        ]
        axes[2].bar(models, hit_values, color="mediumseagreen", edgecolor="black")
        axes[2].set_ylabel("Hit Ratio (%)")
        axes[2].set_title("Hit Ratio (within 1σ)")
        axes[2].set_ylim(0, 100)
        axes[2].tick_params(axis="x", rotation=45)

        for ax_item in axes:
            ax_item.grid(True, alpha=0.3, axis="y")

        if title is None:
            title = "Model Comparison"
        fig.suptitle(title, fontsize=14, fontweight="bold")
        fig.tight_layout()

        return fig

    else:
        # Single metric plot
        if ax is None:
            fig, ax = plt.subplots(figsize=(8, 5))
        else:
            fig = ax.figure

        if metric == "mae":
            values = [
                comparison_result.validation_results[m].mae * 10000
                for m in models
            ]
            ylabel = "MAE (basis points)"
            color = "steelblue"
        elif metric == "rmse":
            values = [
                comparison_result.validation_results[m].rmse * 10000
                for m in models
            ]
            ylabel = "RMSE (basis points)"
            color = "coral"
        elif metric == "hit_ratio":
            values = [
                comparison_result.validation_results[m].hit_ratio * 100
                for m in models
            ]
            ylabel = "Hit Ratio (%)"
            color = "mediumseagreen"
        else:
            raise ValueError(f"Unknown metric: {metric}")

        bars = ax.bar(models, values, color=color, edgecolor="black")

        # Highlight best model
        best_idx = np.argmax(values) if metric == "hit_ratio" else np.argmin(values)

        bars[best_idx].set_color("gold")
        bars[best_idx].set_edgecolor("black")

        ax.set_ylabel(ylabel)
        if title is None:
            title = f"Model Comparison - {metric.upper()}"
        ax.set_title(title)
        ax.tick_params(axis="x", rotation=45)
        ax.grid(True, alpha=0.3, axis="y")

        fig.tight_layout()
        return fig


def plot_backtest_results(
    validation_result: ValidationResult,
    show_error_bands: bool = True,
    n_std: float = 1.0,
    ax: plt.Axes | None = None,
    title: str | None = None,
) -> plt.Figure:
    """Plot backtest results with predictions vs actuals.

    Creates a visualization showing the model's predictions against
    actual observed values, with optional error bands.

    Args:
        validation_result: ValidationResult from backtest_model().
        show_error_bands: Whether to show ±n_std error bands.
        n_std: Number of standard deviations for error bands.
        ax: Optional matplotlib axes to plot on.
        title: Optional plot title.

    Returns:
        Matplotlib Figure containing the backtest plot.

    Example:
        >>> result = backtest_model(model, validation_data)
        >>> fig = plot_backtest_results(result)
        >>> fig.savefig("backtest_results.png")
    """
    from src.validation.backtesting import ValidationResult

    if not isinstance(validation_result, ValidationResult):
        raise TypeError("Expected ValidationResult instance")

    if ax is None:
        fig, ax = plt.subplots(figsize=(12, 6))
    else:
        fig = ax.figure

    predicted = validation_result.predicted_values * 100  # Convert to percentage
    actual = validation_result.actual_values * 100
    n_obs = len(predicted)
    x = np.arange(n_obs)

    # Plot actual values
    ax.plot(
        x,
        actual,
        color="darkblue",
        linewidth=2,
        label="Actual",
        zorder=3,
    )

    # Plot predictions
    ax.plot(
        x,
        predicted,
        color="red",
        linewidth=1.5,
        linestyle="--",
        label="Predicted",
        alpha=0.8,
        zorder=2,
    )

    # Add error bands if requested
    if show_error_bands:
        # Estimate std from errors (simple approach)
        errors = np.abs(predicted - actual)
        rolling_std = np.convolve(errors, np.ones(10) / 10, mode="same")

        upper = predicted + n_std * rolling_std
        lower = predicted - n_std * rolling_std

        ax.fill_between(
            x,
            lower,
            upper,
            alpha=0.2,
            color="red",
            label=f"±{n_std}σ band",
            zorder=1,
        )

    # Add metrics annotation
    mae_bp = validation_result.mae * 10000
    rmse_bp = validation_result.rmse * 10000
    hit_pct = validation_result.hit_ratio * 100

    metrics_text = (
        f"MAE: {mae_bp:.2f} bp\n"
        f"RMSE: {rmse_bp:.2f} bp\n"
        f"Hit Ratio: {hit_pct:.1f}%\n"
        f"Quality: {validation_result.quality_rating}"
    )

    ax.text(
        0.02,
        0.98,
        metrics_text,
        transform=ax.transAxes,
        verticalalignment="top",
        fontsize=10,
        fontfamily="monospace",
        bbox={"boxstyle": "round", "facecolor": "wheat", "alpha": 0.8},
    )

    # Formatting
    ax.set_xlabel("Observation Index")
    ax.set_ylabel("Interest Rate (%)")

    if title is None:
        model_type = validation_result.model_type.capitalize()
        period = validation_result.validation_period
        title = f"{model_type} Backtest: {period[0]} to {period[1]}"
    ax.set_title(title)

    ax.legend(loc="upper right")
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    return fig


def plot_hjm_terminal_surfaces(
    result: SimulationResult,
    percentiles: list[float] | None = None,
    ax: plt.Axes | None = None,
    title: str | None = None,
) -> plt.Figure:
    """Plot percentile surfaces of terminal forward curves.

    Shows the distribution of terminal forward curves across all
    simulation paths, visualized as percentile bands.

    Args:
        result: SimulationResult from HJM simulation.
        percentiles: Percentiles to plot. Defaults to [5, 25, 50, 75, 95].
        ax: Optional matplotlib axes to plot on.
        title: Optional plot title.

    Returns:
        Matplotlib Figure containing the terminal surface plot.
    """
    if result.model_type != "hjm":
        raise ValueError(f"Expected HJM simulation result, got {result.model_type}")

    if percentiles is None:
        percentiles = [5.0, 25.0, 50.0, 75.0, 95.0]

    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 6))
    else:
        fig = ax.figure

    # Get terminal curves from all paths
    terminal_curves = result.terminal_distribution()  # (n_paths, n_maturities)
    n_maturities = terminal_curves.shape[1]

    # Use standard maturities or indices
    if n_maturities == 7:
        maturities = np.array([0.25, 0.5, 1.0, 2.0, 5.0, 10.0, 30.0])
    else:
        maturities = np.linspace(0.25, 30, n_maturities)

    # Compute percentiles
    pct_curves = {
        p: np.percentile(terminal_curves, p, axis=0) * 100
        for p in percentiles
    }

    # Plot median
    if 50.0 in pct_curves:
        ax.plot(
            maturities,
            pct_curves[50.0],
            color="darkblue",
            linewidth=2,
            label="Median",
        )

    # Plot bands
    if 5.0 in pct_curves and 95.0 in pct_curves:
        ax.fill_between(
            maturities,
            pct_curves[5.0],
            pct_curves[95.0],
            alpha=0.2,
            color="blue",
            label="5th-95th percentile",
        )

    if 25.0 in pct_curves and 75.0 in pct_curves:
        ax.fill_between(
            maturities,
            pct_curves[25.0],
            pct_curves[75.0],
            alpha=0.3,
            color="blue",
            label="25th-75th percentile",
        )

    # Plot initial curve for comparison
    initial_curve = result.paths[0].rates[0] * 100
    ax.plot(
        maturities,
        initial_curve,
        color="red",
        linewidth=2,
        linestyle="--",
        label="Initial Curve",
    )

    # Formatting
    ax.set_xlabel("Maturity (years)")
    ax.set_ylabel("Forward Rate (%)")

    if title is None:
        horizon = result.time_horizon
        title = f"Terminal Forward Curve Distribution (T = {horizon:.1f}y)"
    ax.set_title(title)

    ax.legend(loc="best")
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    return fig
