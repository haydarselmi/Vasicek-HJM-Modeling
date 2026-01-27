"""Visualization functions for PCA analysis results.

This module provides plotting functions for visualizing Principal Component
Analysis results on yield curve data, including explained variance charts
and component loading plots.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

import matplotlib.pyplot as plt
import numpy as np

if TYPE_CHECKING:
    from matplotlib.figure import Figure

    from src.analysis.pca import PCAResult


def _get_figure_from_axes(ax: plt.Axes) -> Figure:
    """Get figure from axes, with type safety.

    Args:
        ax: Matplotlib axes.

    Returns:
        The figure containing the axes.

    Raises:
        ValueError: If axes has no associated figure.
    """
    fig = ax.get_figure()
    if fig is None:
        msg = "Cannot get figure from provided axes"
        raise ValueError(msg)
    return cast("Figure", fig)


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
        fig = _get_figure_from_axes(ax)

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
        fig = _get_figure_from_axes(ax)

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


def plot_training_loss(
    history: dict[str, list[float]],
    title: str = "Autoencoder Training Loss",
    ax: plt.Axes | None = None,
) -> plt.Figure:
    """Plot training and validation loss curves for autoencoder.

    Creates a line plot showing loss evolution over training epochs,
    with optional validation loss overlay.

    Args:
        history: Training history dictionary with 'train_loss' and
                 optionally 'val_loss' keys.
        title: Plot title.
        ax: Optional matplotlib axes. If None, creates new figure.

    Returns:
        Matplotlib figure with loss curves.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 6))
    else:
        fig = _get_figure_from_axes(ax)

    epochs = range(1, len(history["train_loss"]) + 1)

    # Plot training loss
    ax.plot(
        epochs,
        history["train_loss"],
        color="steelblue",
        linewidth=2,
        marker="o",
        markersize=4,
        markevery=max(1, len(epochs) // 20),
        label="Training Loss",
    )

    # Plot validation loss if available
    if "val_loss" in history and len(history["val_loss"]) > 0:
        ax.plot(
            epochs,
            history["val_loss"],
            color="crimson",
            linewidth=2,
            marker="s",
            markersize=4,
            markevery=max(1, len(epochs) // 20),
            label="Validation Loss",
        )

    # Find minimum loss point
    min_train_loss = min(history["train_loss"])
    min_epoch = history["train_loss"].index(min_train_loss) + 1
    ax.axvline(
        x=min_epoch,
        color="gray",
        linestyle="--",
        alpha=0.5,
        label=f"Min train loss (epoch {min_epoch})",
    )

    # Styling
    ax.set_xlabel("Epoch", fontsize=12)
    ax.set_ylabel("Loss (MSE)", fontsize=12)
    ax.set_title(title, fontsize=14, fontweight="bold")

    ax.legend(loc="upper right", fontsize=10)
    ax.grid(True, alpha=0.3)

    # Log scale for y-axis if range is large
    loss_range = max(history["train_loss"]) / (min(history["train_loss"]) + 1e-10)
    if loss_range > 100:
        ax.set_yscale("log")

    plt.tight_layout()
    return fig


def plot_latent_space(
    encodings: np.ndarray,
    labels: np.ndarray | list[str] | None = None,
    dims: tuple[int, int] | tuple[int, int, int] = (0, 1),
    title: str = "Latent Space Representation",
    ax: plt.Axes | None = None,
    colorbar_label: str = "Value",
) -> plt.Figure:
    """Plot latent space encodings in 2D or 3D.

    Creates a scatter plot of autoencoder latent representations,
    optionally colored by labels or continuous values.

    Args:
        encodings: Latent encodings of shape (n_samples, latent_dim).
        labels: Optional labels for coloring points. Can be categorical
                (list of strings) or numerical (array of floats).
        dims: Tuple of dimension indices to plot. Use 2 for 2D, 3 for 3D.
        title: Plot title.
        ax: Optional matplotlib axes. If None, creates new figure.
        colorbar_label: Label for colorbar if using numerical labels.

    Returns:
        Matplotlib figure with latent space visualization.
    """
    is_3d = len(dims) == 3
    latent_dim = encodings.shape[1]

    # Validate dimensions
    for d in dims:
        if d >= latent_dim:
            raise ValueError(
                f"Dimension {d} exceeds latent space size {latent_dim}"
            )

    # Create figure
    if ax is None:
        if is_3d:
            from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

            fig = plt.figure(figsize=(10, 8))
            ax = fig.add_subplot(111, projection="3d")
        else:
            fig, ax = plt.subplots(figsize=(10, 8))
    else:
        fig = _get_figure_from_axes(ax)

    # Prepare colors
    if labels is None:
        colors = "steelblue"
        scatter_kwargs = {"c": colors, "alpha": 0.6, "s": 50}
    elif isinstance(labels, list) and all(isinstance(lbl, str) for lbl in labels):
        # Categorical labels
        unique_labels = list(set(labels))
        color_map = {label: i for i, label in enumerate(unique_labels)}
        colors = [color_map[lbl] for lbl in labels]
        scatter_kwargs = {"c": colors, "cmap": "tab10", "alpha": 0.6, "s": 50}
    else:
        # Numerical labels
        colors = np.asarray(labels)
        scatter_kwargs = {"c": colors, "cmap": "viridis", "alpha": 0.6, "s": 50}

    # Create scatter plot
    if is_3d and len(dims) == 3:
        dim_z = dims[2]
        scatter = ax.scatter(
            encodings[:, dims[0]],
            encodings[:, dims[1]],
            encodings[:, dim_z],
            **scatter_kwargs,  # type: ignore[arg-type]
        )
        ax.set_xlabel(f"Latent Dim {dims[0] + 1}", fontsize=11)
        ax.set_ylabel(f"Latent Dim {dims[1] + 1}", fontsize=11)
        ax.set_zlabel(f"Latent Dim {dim_z + 1}", fontsize=11)  # type: ignore[union-attr]
    else:
        scatter = ax.scatter(
            encodings[:, dims[0]],
            encodings[:, dims[1]],
            **scatter_kwargs,
        )
        ax.set_xlabel(f"Latent Dim {dims[0] + 1}", fontsize=12)
        ax.set_ylabel(f"Latent Dim {dims[1] + 1}", fontsize=12)

    # Add colorbar for numerical labels
    if labels is not None and not (
        isinstance(labels, list) and all(isinstance(lbl, str) for lbl in labels)
    ):
        cbar = plt.colorbar(scatter, ax=ax, shrink=0.8)
        cbar.set_label(colorbar_label, fontsize=11)

    # Title and grid
    ax.set_title(title, fontsize=14, fontweight="bold")
    if not is_3d:
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    return fig


def plot_reconstruction_comparison(
    original: np.ndarray,
    pca_reconstructed: np.ndarray,
    autoencoder_reconstructed: np.ndarray,
    maturities: np.ndarray,
    sample_indices: list[int] | None = None,
    n_samples: int = 5,
) -> plt.Figure:
    """Plot original vs reconstructed curves for PCA and autoencoder.

    Creates a multi-panel figure comparing reconstruction quality
    of PCA and autoencoder on selected samples.

    Args:
        original: Original data of shape (n_samples, n_maturities).
        pca_reconstructed: PCA reconstructed data.
        autoencoder_reconstructed: Autoencoder reconstructed data.
        maturities: Maturity values for x-axis.
        sample_indices: Specific sample indices to plot. If None,
                       selects evenly spaced samples.
        n_samples: Number of samples to plot (if sample_indices is None).

    Returns:
        Matplotlib figure with reconstruction comparison.
    """
    n_total = original.shape[0]

    if sample_indices is None:
        # Select evenly spaced samples
        sample_indices = list(np.linspace(0, n_total - 1, n_samples, dtype=int))

    n_plots = len(sample_indices)
    fig, axes = plt.subplots(
        1,
        n_plots,
        figsize=(4 * n_plots, 4),
        sharey=True,
    )

    if n_plots == 1:
        axes = [axes]

    for ax, idx in zip(axes, sample_indices, strict=True):
        # Plot original
        ax.plot(
            maturities,
            original[idx],
            "k-",
            linewidth=2,
            marker="o",
            markersize=6,
            label="Original",
        )

        # Plot PCA reconstruction
        ax.plot(
            maturities,
            pca_reconstructed[idx],
            "b--",
            linewidth=1.5,
            marker="s",
            markersize=5,
            alpha=0.8,
            label="PCA",
        )

        # Plot autoencoder reconstruction
        ax.plot(
            maturities,
            autoencoder_reconstructed[idx],
            "r:",
            linewidth=1.5,
            marker="^",
            markersize=5,
            alpha=0.8,
            label="Autoencoder",
        )

        # Calculate errors
        pca_mse = np.mean((original[idx] - pca_reconstructed[idx]) ** 2)
        ae_mse = np.mean((original[idx] - autoencoder_reconstructed[idx]) ** 2)

        ax.set_title(f"Sample {idx}\nPCA MSE: {pca_mse:.2e}, AE MSE: {ae_mse:.2e}", fontsize=10)
        ax.set_xlabel("Maturity (Years)", fontsize=10)
        ax.grid(True, alpha=0.3)

    axes[0].set_ylabel("Rate Change", fontsize=11)
    axes[0].legend(loc="best", fontsize=9)

    fig.suptitle(
        "Reconstruction Comparison: Original vs PCA vs Autoencoder",
        fontsize=13,
        fontweight="bold",
    )

    plt.tight_layout()
    return fig
