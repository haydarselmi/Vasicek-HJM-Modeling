"""Visualization functions for simulation results.

This module provides functions for plotting Monte Carlo simulation
paths, summary statistics, and terminal rate distributions.
"""

import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt

from src.simulation.paths import SimulationResult


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
    n = len(data)
    mean = np.mean(data)
    std = np.std(data, ddof=1)
    if std < 1e-10:
        return 0.0
    m4 = np.mean((data - mean) ** 4)
    return (m4 / (std**4)) - 3
