"""Visualization functions for PCA analysis results.

This module provides plotting functions for visualizing Principal Component
Analysis results on yield curve data, including explained variance charts
and component loading plots.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import numpy as np

if TYPE_CHECKING:
    from src.analysis.pca import PCAResult


def plot_explained_variance(
    pca_result: PCAResult,
    ax: plt.Axes | None = None,
) -> plt.Figure:
    """Plot explained variance ratio and cumulative variance.

    Creates a combined bar chart showing individual component variance
    with a line overlay for cumulative explained variance.

    Args:
        pca_result: PCA analysis results.
        ax: Optional matplotlib axes. If None, creates new figure.

    Returns:
        Matplotlib figure with variance plot.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 6))
    else:
        fig = ax.get_figure()

    n_components = pca_result.n_components
    x = np.arange(1, n_components + 1)

    # Bar chart for individual variance
    bars = ax.bar(
        x,
        pca_result.explained_variance_ratio * 100,
        alpha=0.7,
        color="steelblue",
        label="Individual",
        edgecolor="navy",
    )

    # Add value labels on bars
    for bar, var in zip(bars, pca_result.explained_variance_ratio, strict=True):
        height = bar.get_height()
        if height > 5:  # Only label bars with significant height
            ax.annotate(
                f"{var * 100:.1f}%",
                xy=(bar.get_x() + bar.get_width() / 2, height),
                xytext=(0, 3),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=9,
            )

    # Line chart for cumulative variance
    ax2 = ax.twinx()
    ax2.plot(
        x,
        pca_result.cumulative_variance * 100,
        color="crimson",
        marker="o",
        linewidth=2,
        markersize=8,
        label="Cumulative",
    )

    # Add 90% and 95% reference lines
    ax2.axhline(y=90, color="gray", linestyle="--", alpha=0.5, label="90% threshold")
    ax2.axhline(y=95, color="gray", linestyle=":", alpha=0.5, label="95% threshold")

    # Styling
    ax.set_xlabel("Principal Component", fontsize=12)
    ax.set_ylabel("Individual Explained Variance (%)", fontsize=12, color="steelblue")
    ax2.set_ylabel("Cumulative Explained Variance (%)", fontsize=12, color="crimson")

    ax.set_xticks(x)
    ax.set_xticklabels([f"PC{i}" for i in x])
    ax.set_ylim(0, 100)
    ax2.set_ylim(0, 105)

    ax.tick_params(axis="y", labelcolor="steelblue")
    ax2.tick_params(axis="y", labelcolor="crimson")

    # Title
    ax.set_title(
        "PCA Explained Variance Analysis",
        fontsize=14,
        fontweight="bold",
    )

    # Combined legend
    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax2.legend(
        lines1 + lines2,
        labels1 + labels2,
        loc="center right",
        fontsize=10,
    )

    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    return fig


def plot_component_loadings(
    pca_result: PCAResult,
    n_components: int = 3,
    ax: plt.Axes | None = None,
) -> plt.Figure:
    """Plot component loadings across maturities.

    Creates a line plot showing how each principal component loads
    onto different maturity points, with interpretations in the legend.

    Args:
        pca_result: PCA analysis results.
        n_components: Number of components to plot (default: 3).
        ax: Optional matplotlib axes. If None, creates new figure.

    Returns:
        Matplotlib figure with loadings plot.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(12, 7))
    else:
        fig = ax.get_figure()

    n_components = min(n_components, pca_result.n_components)
    maturities = pca_result.maturities

    # Color palette for components
    colors = ["#2E86AB", "#A23B72", "#F18F01", "#C73E1D", "#3B1F2B"]

    for i in range(n_components):
        loadings = pca_result.loadings[:, i]
        var_pct = pca_result.explained_variance_ratio[i] * 100

        # Get interpretation if available
        if i < len(pca_result.interpretations):
            interp = pca_result.interpretations[i]
            # Extract just the type (Level, Slope, Curvature)
            interp_short = interp.split(":")[1].split("-")[0].strip() if ":" in interp else ""
            label = f"PC{i + 1} ({var_pct:.1f}%) - {interp_short}"
        else:
            label = f"PC{i + 1} ({var_pct:.1f}%)"

        ax.plot(
            maturities,
            loadings,
            marker="o",
            linewidth=2.5,
            markersize=8,
            color=colors[i % len(colors)],
            label=label,
        )

    # Add zero reference line
    ax.axhline(y=0, color="gray", linestyle="-", alpha=0.5, linewidth=1)

    # X-axis formatting for maturities
    ax.set_xlabel("Maturity (Years)", fontsize=12)
    ax.set_ylabel("Loading", fontsize=12)

    # Custom x-tick labels for common maturities
    maturity_labels = []
    for m in maturities:
        if m < 1:
            months = int(m * 12)
            maturity_labels.append(f"{months}M")
        else:
            years = int(m) if m == int(m) else m
            maturity_labels.append(f"{years}Y")

    ax.set_xticks(maturities)
    ax.set_xticklabels(maturity_labels)

    # Title and legend
    ax.set_title(
        "Principal Component Loadings Across Maturities",
        fontsize=14,
        fontweight="bold",
    )

    ax.legend(
        loc="best",
        fontsize=10,
        framealpha=0.9,
    )

    ax.grid(True, alpha=0.3)

    # Add interpretation text box
    interpretation_text = _format_interpretation_summary(pca_result, n_components)
    if interpretation_text:
        props = {"boxstyle": "round", "facecolor": "wheat", "alpha": 0.8}
        ax.text(
            0.02,
            0.02,
            interpretation_text,
            transform=ax.transAxes,
            fontsize=9,
            verticalalignment="bottom",
            bbox=props,
        )

    plt.tight_layout()
    return fig


def _format_interpretation_summary(
    pca_result: PCAResult,
    n_components: int,
) -> str:
    """Format a summary of component interpretations for display.

    Args:
        pca_result: PCA analysis results.
        n_components: Number of components to summarize.

    Returns:
        Formatted interpretation summary string.
    """
    lines = []
    n_components = min(n_components, len(pca_result.interpretations))

    for i in range(n_components):
        var_pct = pca_result.explained_variance_ratio[i] * 100
        interp = pca_result.interpretations[i] if i < len(pca_result.interpretations) else ""

        # Extract pattern type from interpretation
        if "Level" in interp or "parallel" in interp.lower():
            pattern = "Parallel shift"
        elif "Slope" in interp or "tilt" in interp.lower():
            pattern = "Yield curve tilt"
        elif "Curvature" in interp or "butterfly" in interp.lower():
            pattern = "Curvature change"
        else:
            pattern = "Complex pattern"

        lines.append(f"PC{i + 1}: {pattern} ({var_pct:.1f}%)")

    return "\n".join(lines)


def plot_pca_summary(
    pca_result: PCAResult,
    figsize: tuple[float, float] = (14, 6),
) -> plt.Figure:
    """Create a combined summary plot with variance and loadings.

    This function creates a side-by-side visualization with explained
    variance on the left and component loadings on the right.

    Args:
        pca_result: PCA analysis results.
        figsize: Figure size as (width, height) in inches.

    Returns:
        Matplotlib figure with combined plots.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)

    # Plot explained variance
    plot_explained_variance(pca_result, ax=ax1)

    # Plot loadings
    plot_component_loadings(pca_result, n_components=3, ax=ax2)

    plt.suptitle(
        "Principal Component Analysis Summary",
        fontsize=16,
        fontweight="bold",
        y=1.02,
    )

    plt.tight_layout()
    return fig
