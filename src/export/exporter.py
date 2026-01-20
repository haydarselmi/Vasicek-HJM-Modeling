"""Export functionality for models, simulations, and figures.

This module provides functions for exporting:
- Calibrated interest rate models (Vasicek, HJM) and PCA results to JSON/pickle
- Simulation results to JSON/CSV
- Matplotlib figures to PNG/PDF/SVG with configurable DPI

All export functions handle file path validation and provide meaningful
error messages through the ExportError exception.
"""

import csv
import json
import pickle
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

import numpy as np

from src.utils.exceptions import ExportError
from src.utils.logging import get_logger

if TYPE_CHECKING:
    import matplotlib.pyplot as plt

    from src.analysis.pca import PCAResult
    from src.models.base import InterestRateModel
    from src.simulation.paths import SimulationResult

logger = get_logger("export.exporter")


# =============================================================================
# Type Definitions
# =============================================================================

ModelExportFormat = Literal["json", "pickle"]
SimulationExportFormat = Literal["json", "csv"]
FigureExportFormat = Literal["png", "pdf", "svg"]


# =============================================================================
# Helper Functions
# =============================================================================


def _ensure_parent_directory(path: Path) -> None:
    """Create parent directories if they don't exist.

    Args:
        path: File path whose parent directories should be created.

    Raises:
        ExportError: If directory creation fails.
    """
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
    except OSError as e:
        msg = f"Failed to create directory {path.parent}"
        raise ExportError(msg, file_path=str(path)) from e


def _ndarray_to_list(obj: Any) -> Any:
    """Recursively convert numpy arrays to lists for JSON serialization.

    Args:
        obj: Object to convert (can be dict, list, ndarray, or primitive).

    Returns:
        Converted object with arrays as lists.
    """
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, dict):
        return {k: _ndarray_to_list(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_ndarray_to_list(item) for item in obj]
    if isinstance(obj, (np.float64, np.float32)):
        return float(obj)
    if isinstance(obj, (np.int64, np.int32)):
        return int(obj)
    if isinstance(obj, datetime):
        return obj.isoformat()
    return obj


def _get_model_type(model: "InterestRateModel") -> str:
    """Get the model type string from a model instance.

    Args:
        model: Interest rate model instance.

    Returns:
        Model type string ("vasicek", "hjm", or class name).
    """
    from src.models.hjm import HJMModel
    from src.models.vasicek import VasicekModel

    if isinstance(model, VasicekModel):
        return "vasicek"
    if isinstance(model, HJMModel):
        return "hjm"
    return model.__class__.__name__.lower()


def _serialize_model_to_dict(model: "InterestRateModel") -> dict[str, Any]:
    """Serialize a model to a dictionary for JSON export.

    Args:
        model: The calibrated model to serialize.

    Returns:
        Dictionary containing model parameters and metadata.

    Raises:
        ExportError: If model is not calibrated.
    """
    from src.models.hjm import HJMModel
    from src.models.vasicek import VasicekModel

    if not model.is_calibrated:
        msg = "Cannot export uncalibrated model"
        raise ExportError(msg, export_format="json")

    model_type = _get_model_type(model)
    params = model.get_parameters()

    result: dict[str, Any] = {
        "model_type": model_type,
        "parameters": _ndarray_to_list(params),
        "is_calibrated": True,
        "export_date": datetime.now().isoformat(),
    }

    # Add model-specific metadata
    if isinstance(model, VasicekModel):
        interpretations = model.get_parameter_interpretations()
        result["parameter_interpretations"] = interpretations
    elif isinstance(model, HJMModel):
        result["n_factors"] = model.n_factors
        result["maturities"] = _ndarray_to_list(model.maturities)
        if model.volatility_functions is not None:
            vol_funcs = model.volatility_functions
            result["volatility_functions"] = _ndarray_to_list(vol_funcs)
        if model.factor_variances is not None:
            factor_vars = model.factor_variances
            result["factor_variances"] = _ndarray_to_list(factor_vars)
        if model.calibration_date:
            result["calibration_date"] = model.calibration_date.isoformat()

    return result


def _serialize_pca_to_dict(pca_result: "PCAResult") -> dict[str, Any]:
    """Serialize a PCAResult to a dictionary for JSON export.

    Args:
        pca_result: The PCA result to serialize.

    Returns:
        Dictionary containing PCA results and metadata.
    """
    return {
        "type": "pca_result",
        "n_components": pca_result.n_components,
        "explained_variance_ratio": _ndarray_to_list(
            pca_result.explained_variance_ratio
        ),
        "cumulative_variance": _ndarray_to_list(pca_result.cumulative_variance),
        "loadings": _ndarray_to_list(pca_result.loadings),
        "maturities": _ndarray_to_list(pca_result.maturities),
        "interpretations": pca_result.interpretations,
        "eigenvalues": _ndarray_to_list(pca_result.eigenvalues),
        "analysis_date": pca_result.analysis_date.isoformat(),
        "export_date": datetime.now().isoformat(),
    }


def _serialize_simulation_to_dict(result: "SimulationResult") -> dict[str, Any]:
    """Serialize a SimulationResult to a dictionary for JSON export.

    Args:
        result: The simulation result to serialize.

    Returns:
        Dictionary containing simulation results and metadata.
    """
    terminal_mean = result.terminal_mean()
    terminal_std = result.terminal_std()

    return {
        "model_type": result.model_type,
        "path_count": result.path_count,
        "time_horizon": result.time_horizon,
        "time_step": result.time_step,
        "simulation_date": result.simulation_date.isoformat(),
        "model_parameters": _ndarray_to_list(result.model_parameters),
        "time_grid": _ndarray_to_list(result.time_grid),
        "n_time_points": result.n_time_points,
        "summary_statistics": {
            "mean_path": _ndarray_to_list(result.mean_path()),
            "std_path": _ndarray_to_list(result.std_path()),
            "terminal_mean": (
                _ndarray_to_list(terminal_mean)
                if isinstance(terminal_mean, np.ndarray)
                else terminal_mean
            ),
            "terminal_std": (
                _ndarray_to_list(terminal_std)
                if isinstance(terminal_std, np.ndarray)
                else terminal_std
            ),
            "percentile_paths": {
                str(p): _ndarray_to_list(v)
                for p, v in result.percentile_paths().items()
            },
        },
        "paths": [
            {
                "path_id": path.path_id,
                "rates": _ndarray_to_list(path.rates),
                "random_seed": path.random_seed,
            }
            for path in result.paths
        ],
        "export_date": datetime.now().isoformat(),
    }


# =============================================================================
# Export Functions
# =============================================================================


def export_model(
    model: "InterestRateModel | PCAResult",
    output_path: Path,
    format: ModelExportFormat = "json",
) -> None:
    """Export a calibrated model or PCA result to file.

    This function exports interest rate models (Vasicek, HJM) or PCA results
    to either JSON (human-readable) or pickle (for model reloading) format.

    Args:
        model: Calibrated model or PCA result to export.
        output_path: Path for output file.
        format: Export format ("json" or "pickle").

    Raises:
        ExportError: If export fails or model is not calibrated.

    Example:
        >>> export_model(vasicek_model, Path("output/models/vasicek.json"))
        >>> export_model(pca_result, Path("output/models/pca.pkl"), format="pickle")
    """
    from src.analysis.pca import PCAResult

    output_path = Path(output_path)
    _ensure_parent_directory(output_path)

    is_pca = isinstance(model, PCAResult)
    model_name = "PCA result" if is_pca else "model"

    logger.info(
        "Exporting %s to %s (format: %s)", model_name, output_path, format
    )

    try:
        if format == "json":
            # Serialize to JSON
            if is_pca:
                data = _serialize_pca_to_dict(model)
            else:
                data = _serialize_model_to_dict(model)

            with output_path.open("w", encoding="utf-8") as f:
                json.dump(data, f, indent=2)

        elif format == "pickle":
            # Serialize with pickle (preserves full object)
            with output_path.open("wb") as f:
                pickle.dump(model, f, protocol=pickle.HIGHEST_PROTOCOL)

        else:
            msg = f"Unsupported export format: {format}"
            raise ExportError(
                msg, export_format=format, file_path=str(output_path)
            )

        logger.info("Successfully exported %s to %s", model_name, output_path)

    except ExportError:
        raise
    except json.JSONDecodeError as e:
        msg = f"JSON serialization failed: {e}"
        raise ExportError(
            msg, export_format=format, file_path=str(output_path)
        ) from e
    except pickle.PicklingError as e:
        msg = f"Pickle serialization failed: {e}"
        raise ExportError(
            msg, export_format=format, file_path=str(output_path)
        ) from e
    except OSError as e:
        msg = f"Failed to write file: {e}"
        raise ExportError(
            msg, export_format=format, file_path=str(output_path)
        ) from e


def export_simulation(
    result: "SimulationResult",
    output_path: Path,
    format: SimulationExportFormat = "csv",
) -> None:
    """Export simulation results to file.

    This function exports Monte Carlo simulation results to either:
    - CSV: Tabular format with time grid and path data
    - JSON: Full structured export including metadata and statistics

    Args:
        result: Simulation results to export.
        output_path: Path for output file.
        format: Export format ("json" or "csv").

    Raises:
        ExportError: If export fails.

    Example:
        >>> export_simulation(result, Path("output/simulations/vasicek.csv"))
        >>> export_simulation(result, Path("output/simulations/hjm.json"), "json")
    """
    output_path = Path(output_path)
    _ensure_parent_directory(output_path)

    logger.info(
        "Exporting simulation (%d paths) to %s (format: %s)",
        result.path_count,
        output_path,
        format,
    )

    try:
        if format == "json":
            # Full structured export
            data = _serialize_simulation_to_dict(result)
            with output_path.open("w", encoding="utf-8") as f:
                json.dump(data, f, indent=2)

        elif format == "csv":
            # Tabular export - rows are time points, columns are paths
            _export_simulation_csv(result, output_path)

        else:
            msg = f"Unsupported export format: {format}"
            raise ExportError(
                msg, export_format=format, file_path=str(output_path)
            )

        logger.info("Successfully exported simulation to %s", output_path)

    except ExportError:
        raise
    except OSError as e:
        msg = f"Failed to write file: {e}"
        raise ExportError(
            msg, export_format=format, file_path=str(output_path)
        ) from e


def _export_simulation_csv(result: "SimulationResult", output_path: Path) -> None:
    """Export simulation result to CSV format.

    For Vasicek: Creates table with [time, path_0, path_1, ..., mean, std].
    For HJM: Creates table with [time, maturity_index, path_0, ..., mean, std].

    Args:
        result: Simulation result to export.
        output_path: Path to write CSV file.
    """
    rates_matrix = result.get_rates_matrix()
    time_grid = result.time_grid
    mean_path = result.mean_path()
    std_path = result.std_path()
    path_count = result.path_count

    with output_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        path_cols = [f"path_{i}" for i in range(path_count)]

        if result.model_type == "vasicek":
            _write_vasicek_csv(
                writer, time_grid, rates_matrix, mean_path, std_path, path_cols
            )
        else:
            _write_hjm_csv(
                writer, time_grid, rates_matrix, mean_path, std_path, path_cols
            )


def _write_vasicek_csv(
    writer: Any,
    time_grid: Any,
    rates_matrix: Any,
    mean_path: Any,
    std_path: Any,
    path_cols: list[str],
) -> None:
    """Write Vasicek simulation to CSV.

    Args:
        writer: CSV writer object.
        time_grid: Array of time points.
        rates_matrix: Matrix of rate paths.
        mean_path: Mean path values.
        std_path: Standard deviation path values.
        path_cols: Column names for paths.
    """
    header = ["time"] + path_cols + ["mean", "std"]
    writer.writerow(header)

    for t_idx, t in enumerate(time_grid):
        row = [f"{t:.6f}"]
        for p_idx in range(len(path_cols)):
            row.append(f"{rates_matrix[p_idx, t_idx]:.8f}")
        row.append(f"{mean_path[t_idx]:.8f}")
        row.append(f"{std_path[t_idx]:.8f}")
        writer.writerow(row)


def _write_hjm_csv(
    writer: Any,
    time_grid: Any,
    rates_matrix: Any,
    mean_path: Any,
    std_path: Any,
    path_cols: list[str],
) -> None:
    """Write HJM simulation to CSV.

    Args:
        writer: CSV writer object.
        time_grid: Array of time points.
        rates_matrix: Matrix of rate paths (3D for HJM).
        mean_path: Mean path values.
        std_path: Standard deviation path values.
        path_cols: Column names for paths.
    """
    n_maturities = rates_matrix.shape[2] if rates_matrix.ndim == 3 else 1

    header = ["time", "maturity_index"] + path_cols + ["mean", "std"]
    writer.writerow(header)

    for t_idx, t in enumerate(time_grid):
        for m_idx in range(n_maturities):
            row = [f"{t:.6f}", str(m_idx)]
            for p_idx in range(len(path_cols)):
                row.append(f"{rates_matrix[p_idx, t_idx, m_idx]:.8f}")
            row.append(f"{mean_path[t_idx, m_idx]:.8f}")
            row.append(f"{std_path[t_idx, m_idx]:.8f}")
            writer.writerow(row)


def export_figure(
    figure: "plt.Figure",
    output_path: Path,
    format: FigureExportFormat = "png",
    dpi: int = 300,
) -> None:
    """Export a matplotlib figure to file.

    This function exports matplotlib figures to various image formats
    with configurable resolution for raster formats.

    Args:
        figure: Matplotlib figure to export.
        output_path: Path for output file.
        format: Image format ("png", "pdf", or "svg").
        dpi: Resolution for raster formats (PNG). Default is 300.

    Raises:
        ExportError: If export fails.

    Example:
        >>> fig, ax = plt.subplots()
        >>> ax.plot([1, 2, 3], [1, 4, 9])
        >>> export_figure(fig, Path("output/figures/plot.png"), dpi=300)
        >>> export_figure(fig, Path("output/figures/plot.pdf"), format="pdf")
    """
    output_path = Path(output_path)
    _ensure_parent_directory(output_path)

    logger.info(
        "Exporting figure to %s (format: %s, dpi: %d)",
        output_path, format, dpi
    )

    try:
        # Validate format
        if format not in ("png", "pdf", "svg"):
            msg = f"Unsupported figure format: {format}"
            raise ExportError(
                msg, export_format=format, file_path=str(output_path)
            )

        # Save figure with appropriate settings
        save_kwargs: dict[str, Any] = {
            "format": format,
            "bbox_inches": "tight",
            "facecolor": figure.get_facecolor(),
            "edgecolor": "none",
        }

        # Add DPI for raster formats
        if format == "png":
            save_kwargs["dpi"] = dpi

        figure.savefig(output_path, **save_kwargs)

        logger.info("Successfully exported figure to %s", output_path)

    except ExportError:
        raise
    except ValueError as e:
        msg = f"Invalid figure or format: {e}"
        raise ExportError(
            msg, export_format=format, file_path=str(output_path)
        ) from e
    except OSError as e:
        msg = f"Failed to write file: {e}"
        raise ExportError(
            msg, export_format=format, file_path=str(output_path)
        ) from e


# =============================================================================
# Load Functions (for completeness)
# =============================================================================


def load_model_from_pickle(
    input_path: Path,
) -> "InterestRateModel | PCAResult":
    """Load a model or PCA result from a pickle file.

    Args:
        input_path: Path to the pickle file.

    Returns:
        Loaded model or PCA result.

    Raises:
        ExportError: If loading fails.
    """
    input_path = Path(input_path)
    logger.info("Loading model from %s", input_path)

    try:
        with input_path.open("rb") as f:
            model = pickle.load(f)  # noqa: S301
        logger.info("Successfully loaded model from %s", input_path)
        return model
    except FileNotFoundError as e:
        msg = f"File not found: {input_path}"
        raise ExportError(
            msg, export_format="pickle", file_path=str(input_path)
        ) from e
    except pickle.UnpicklingError as e:
        msg = f"Failed to unpickle: {e}"
        raise ExportError(
            msg, export_format="pickle", file_path=str(input_path)
        ) from e
