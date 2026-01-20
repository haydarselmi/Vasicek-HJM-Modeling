"""Visualization functions for yield curves.

This module provides functions for plotting yield curves and yield surfaces
from interest rate datasets.
"""

from datetime import date

import matplotlib.pyplot as plt
import numpy as np

from src.data.loader import InterestRateDataset
from src.utils.logging import get_logger

logger = get_logger("visualization.yield_curves")

# Color palette for yield curves
CURVE_COLORS = [
    "#1f77b4",  # Blue
    "#ff7f0e",  # Orange
    "#2ca02c",  # Green
    "#d62728",  # Red
    "#9467bd",  # Purple
    "#8c564b",  # Brown
    "#e377c2",  # Pink
    "#7f7f7f",  # Gray
    "#bcbd22",  # Olive
    "#17becf",  # Cyan
]


def plot_yield_curves(
    dataset: InterestRateDataset,
    dates: list[date] | None = None,
    n_curves: int = 5,
    ax: plt.Axes | None = None,
) -> plt.Figure:
    """Plot yield curves for selected dates.

    Creates a line plot showing the term structure of interest rates
    for one or more observation dates.

    Args:
        dataset: Interest rate dataset.
        dates: Specific dates to plot. If None, select evenly spaced dates.
        n_curves: Number of curves to plot if dates is None.
        ax: Optional matplotlib axes to plot on.

    Returns:
        Matplotlib figure with yield curve plot.
    """
    # Select dates to plot
    if dates is None:
        indices = np.linspace(0, dataset.n_observations - 1, n_curves, dtype=int)
        dates = [dataset.observation_dates[i] for i in indices]

    logger.info("Plotting %d yield curves", len(dates))

    # Create figure if needed
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 6))
    else:
        fig = ax.get_figure()

    # Plot each curve
    for i, obs_date in enumerate(dates):
        try:
            curve = dataset.get_yield_curve(obs_date)
            color = CURVE_COLORS[i % len(CURVE_COLORS)]

            ax.plot(
                curve.maturities,
                curve.rates * 100,  # Convert to percentage
                marker="o",
                markersize=4,
                color=color,
                label=obs_date.strftime("%Y-%m-%d"),
                linewidth=2,
            )
        except ValueError:
            logger.warning("Date %s not found in dataset, skipping", obs_date)
            continue

    # Formatting
    ax.set_xlabel("Maturity (years)", fontsize=12)
    ax.set_ylabel("Rate (%)", fontsize=12)
    ax.set_title(f"Yield Curves - {dataset.source_name}", fontsize=14)
    ax.legend(loc="best", fontsize=10)
    ax.grid(True, alpha=0.3)

    # Set x-axis to show nice maturity labels
    ax.set_xlim(left=0)

    plt.tight_layout()

    return fig


def plot_yield_surface(
    dataset: InterestRateDataset,
    ax: plt.Axes | None = None,
) -> plt.Figure:
    """Plot 3D yield surface (date × maturity × rate).

    Creates a 3D surface plot showing how the yield curve evolves
    over time.

    Args:
        dataset: Interest rate dataset.
        ax: Optional 3D matplotlib axes.

    Returns:
        Matplotlib figure with 3D surface plot.
    """
    logger.info("Plotting yield surface for %d observations", dataset.n_observations)

    # Create figure if needed
    if ax is None:
        fig = plt.figure(figsize=(12, 8))
        ax = fig.add_subplot(111, projection="3d")
    else:
        fig = ax.get_figure()

    # Create mesh grid
    n_dates = dataset.n_observations

    # Use numeric indices for dates (0 to n_dates-1)
    date_indices = np.arange(n_dates)
    maturities = dataset.maturities

    X, Y = np.meshgrid(maturities, date_indices)
    Z = dataset.rates * 100  # Convert to percentage

    # Plot surface
    surf = ax.plot_surface(
        X, Y, Z,
        cmap="viridis",
        edgecolor="none",
        alpha=0.8,
    )

    # Formatting
    ax.set_xlabel("Maturity (years)", fontsize=11, labelpad=10)
    ax.set_ylabel("Time (days)", fontsize=11, labelpad=10)
    ax.set_zlabel("Rate (%)", fontsize=11, labelpad=10)
    ax.set_title(f"Yield Surface - {dataset.source_name}", fontsize=14)

    # Add colorbar
    fig.colorbar(surf, ax=ax, shrink=0.5, aspect=10, label="Rate (%)")

    return fig


def plot_rate_time_series(
    dataset: InterestRateDataset,
    maturities: list[float] | None = None,
    ax: plt.Axes | None = None,
) -> plt.Figure:
    """Plot rate time series for selected maturities.

    Creates a line plot showing how rates evolve over time for
    one or more maturity tenors.

    Args:
        dataset: Interest rate dataset.
        maturities: Maturities to plot. If None, plot all.
        ax: Optional matplotlib axes.

    Returns:
        Matplotlib figure with time series plot.
    """
    if maturities is None:
        maturities = list(dataset.maturities)

    logger.info("Plotting time series for %d maturities", len(maturities))

    # Create figure if needed
    if ax is None:
        fig, ax = plt.subplots(figsize=(12, 6))
    else:
        fig = ax.get_figure()

    # Plot each maturity
    for i, mat in enumerate(maturities):
        mat_idx = np.where(np.isclose(dataset.maturities, mat))[0]
        if len(mat_idx) == 0:
            logger.warning("Maturity %.2f not found, skipping", mat)
            continue

        idx = mat_idx[0]
        color = CURVE_COLORS[i % len(CURVE_COLORS)]

        # Create date labels
        dates = dataset.observation_dates
        rates = dataset.rates[:, idx] * 100

        ax.plot(
            dates,
            rates,
            color=color,
            label=f"{mat:.2f}Y" if mat >= 1 else f"{int(mat*12)}M",
            linewidth=1.5,
        )

    # Formatting
    ax.set_xlabel("Date", fontsize=12)
    ax.set_ylabel("Rate (%)", fontsize=12)
    ax.set_title(f"Rate Time Series - {dataset.source_name}", fontsize=14)
    ax.legend(loc="best", fontsize=10)
    ax.grid(True, alpha=0.3)

    # Rotate date labels
    plt.xticks(rotation=45)
    plt.tight_layout()

    return fig

