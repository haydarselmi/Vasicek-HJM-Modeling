# Module Interface Contracts

**Branch**: `001-hjm-vasicek-modeling` | **Date**: 2026-01-11

This document defines the public Python interfaces for each module in the interest rate modeling system. All functions follow Google-style docstrings and strict typing per the project constitution.

---

## 1. Data Module (`src/data/`)

### `loader.py`

```python
def load_interest_rate_data(
    file_path: Path,
    date_column: str = "date",
    date_format: str = "%Y-%m-%d",
    source_name: str | None = None,
) -> InterestRateDataset:
    """
    Load historical interest rate data from a CSV file.

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
```

### `validator.py`

```python
def validate_dataset(
    dataset: InterestRateDataset,
    strict: bool = False,
) -> ValidationReport:
    """
    Validate an interest rate dataset for completeness and consistency.

    Args:
        dataset: The dataset to validate.
        strict: If True, raise on warnings; if False, only raise on errors.

    Returns:
        ValidationReport containing any warnings or issues found.

    Raises:
        DataValidationError: If validation fails critically.
    """

def handle_missing_values(
    dataset: InterestRateDataset,
    method: Literal["interpolate", "forward_fill", "drop"] = "interpolate",
) -> InterestRateDataset:
    """
    Handle missing values in the dataset.

    Args:
        dataset: Dataset with potential missing values.
        method: Strategy for handling missing values.

    Returns:
        New dataset with missing values handled.

    Raises:
        DataValidationError: If missing values exceed 5% of total.
    """
```

---

## 2. Analysis Module (`src/analysis/`)

### `statistics.py`

```python
def compute_summary_statistics(
    dataset: InterestRateDataset,
) -> SummaryStatistics:
    """
    Compute summary statistics for each maturity in the dataset.

    Args:
        dataset: The interest rate dataset to analyze.

    Returns:
        SummaryStatistics containing mean, std, min, max, and correlations.
    """

def compute_correlation_matrix(
    dataset: InterestRateDataset,
) -> npt.NDArray[np.float64]:
    """
    Compute the correlation matrix across maturities.

    Args:
        dataset: The interest rate dataset.

    Returns:
        Correlation matrix of shape (n_maturities, n_maturities).
    """
```

### `pca.py`

```python
def compute_pca(
    dataset: InterestRateDataset,
    n_components: int | None = None,
    use_changes: bool = True,
) -> PCAResult:
    """
    Perform Principal Component Analysis on yield curve data.

    Args:
        dataset: The interest rate dataset to analyze.
        n_components: Number of components to retain. If None, retain all.
        use_changes: If True, compute PCA on rate changes; if False, on levels.

    Returns:
        PCAResult containing loadings, explained variance, and interpretations.
    """

def interpret_components(
    pca_result: PCAResult,
    n_interpret: int = 3,
) -> list[str]:
    """
    Generate human-readable interpretations for principal components.

    Args:
        pca_result: The PCA results to interpret.
        n_interpret: Number of components to interpret (max 3 recommended).

    Returns:
        List of interpretation strings (e.g., "Level", "Slope", "Curvature").
    """

def reconstruct_curve(
    pca_result: PCAResult,
    n_components: int,
    curve_changes: npt.NDArray[np.float64],
) -> npt.NDArray[np.float64]:
    """
    Reconstruct yield curve changes using a reduced number of components.

    Args:
        pca_result: The PCA decomposition.
        n_components: Number of components to use for reconstruction.
        curve_changes: Original curve changes to reconstruct.

    Returns:
        Reconstructed curve changes.
    """
```

---

## 3. Models Module (`src/models/`)

### `base.py`

```python
class InterestRateModel(ABC):
    """Abstract base class for interest rate models."""

    @abstractmethod
    def calibrate(
        self,
        data: InterestRateDataset | npt.NDArray[np.float64],
    ) -> None:
        """
        Calibrate the model to historical data.

        Args:
            data: Historical interest rate data.

        Raises:
            CalibrationError: If calibration fails to converge.
        """

    @abstractmethod
    def simulate(
        self,
        n_paths: int,
        time_horizon: float,
        time_step: float,
        random_seed: int | None = None,
    ) -> SimulationResult:
        """
        Generate simulated rate paths.

        Args:
            n_paths: Number of simulation paths to generate.
            time_horizon: Simulation horizon in years.
            time_step: Time step size in years.
            random_seed: Optional seed for reproducibility.

        Returns:
            SimulationResult containing all simulated paths.

        Raises:
            ModelNotCalibratedError: If simulate called before calibrate.
        """

    @abstractmethod
    def get_parameters(self) -> dict[str, float]:
        """Return model parameters as a dictionary."""

    @property
    @abstractmethod
    def is_calibrated(self) -> bool:
        """Return True if model has been successfully calibrated."""
```

### `vasicek.py`

```python
class VasicekModel(InterestRateModel):
    """
    Vasicek single-factor mean-reverting interest rate model.

    The short rate follows: dr = κ(θ - r)dt + σdW

    Attributes:
        kappa: Mean reversion speed (1/year).
        theta: Long-term mean rate.
        sigma: Volatility (rate/√year).
    """

    def __init__(
        self,
        kappa: float | None = None,
        theta: float | None = None,
        sigma: float | None = None,
    ) -> None:
        """
        Initialize the Vasicek model.

        Args:
            kappa: Optional initial mean reversion speed.
            theta: Optional initial long-term mean.
            sigma: Optional initial volatility.
        """

    def calibrate(
        self,
        short_rates: npt.NDArray[np.float64],
        dt: float = 1 / 252,
        method: Literal["mle", "ols"] = "mle",
    ) -> None:
        """
        Calibrate the Vasicek model to a short rate time series.

        Args:
            short_rates: Array of observed short rates.
            dt: Time step between observations in years (default: daily).
            method: Calibration method ("mle" for Maximum Likelihood, "ols" for OLS).

        Raises:
            CalibrationError: If optimization fails to converge.
        """

    def get_parameter_interpretations(self) -> dict[str, str]:
        """
        Get plain-language interpretations of model parameters.

        Returns:
            Dictionary mapping parameter names to interpretation strings.
        """

    def expected_rate(self, t: float, r0: float) -> float:
        """
        Compute expected rate at time t given initial rate r0.

        Args:
            t: Time horizon in years.
            r0: Initial rate.

        Returns:
            Expected rate at time t.
        """

    def rate_variance(self, t: float) -> float:
        """
        Compute variance of rate at time t.

        Args:
            t: Time horizon in years.

        Returns:
            Variance of rate at time t.
        """
```

### `hjm.py`

```python
class HJMModel(InterestRateModel):
    """
    Heath-Jarrow-Morton multifactor forward rate model with PCA factors.

    The forward rate follows: df(t,T) = μ(t,T)dt + Σᵢ σᵢ(t,T)dWᵢ

    Attributes:
        n_factors: Number of PCA factors used.
        volatility_functions: Volatility for each factor across maturities.
        pca_result: Underlying PCA decomposition.
    """

    def __init__(
        self,
        pca_result: PCAResult,
        n_factors: int | None = None,
    ) -> None:
        """
        Initialize the HJM model from PCA results.

        Args:
            pca_result: PCA decomposition of yield curve changes.
            n_factors: Number of factors to use (default: all from PCA).
        """

    def calibrate(
        self,
        dataset: InterestRateDataset,
    ) -> None:
        """
        Calibrate volatility functions from historical data.

        Args:
            dataset: Historical interest rate data.
        """

    def simulate(
        self,
        n_paths: int,
        time_horizon: float,
        time_step: float,
        initial_curve: npt.NDArray[np.float64] | None = None,
        random_seed: int | None = None,
    ) -> SimulationResult:
        """
        Simulate forward rate curve evolution.

        Args:
            n_paths: Number of simulation paths.
            time_horizon: Simulation horizon in years.
            time_step: Time step in years.
            initial_curve: Initial forward rate curve. If None, use last observed.
            random_seed: Optional seed for reproducibility.

        Returns:
            SimulationResult with simulated forward curve paths.
        """

    def get_factor_contributions(
        self,
        simulation_result: SimulationResult,
    ) -> list[FactorContribution]:
        """
        Decompose simulation results by factor contribution.

        Args:
            simulation_result: Simulation output to decompose.

        Returns:
            List of FactorContribution objects for each factor.
        """

    def compute_drift(
        self,
        maturity: float,
    ) -> float:
        """
        Compute the no-arbitrage drift at a given maturity.

        Args:
            maturity: Maturity tenor in years.

        Returns:
            Drift value satisfying the HJM drift condition.
        """
```

---

## 4. Simulation Module (`src/simulation/`)

### `monte_carlo.py`

```python
def generate_normal_shocks(
    n_paths: int,
    n_steps: int,
    n_factors: int = 1,
    random_seed: int | None = None,
) -> npt.NDArray[np.float64]:
    """
    Generate standard normal random shocks for Monte Carlo simulation.

    Args:
        n_paths: Number of simulation paths.
        n_steps: Number of time steps.
        n_factors: Number of independent factors (default: 1).
        random_seed: Optional random seed for reproducibility.

    Returns:
        Array of shape (n_paths, n_factors, n_steps) with N(0,1) values.
    """

def apply_vasicek_dynamics(
    initial_rate: float,
    kappa: float,
    theta: float,
    sigma: float,
    shocks: npt.NDArray[np.float64],
    dt: float,
) -> npt.NDArray[np.float64]:
    """
    Apply Vasicek dynamics to generate rate paths.

    Args:
        initial_rate: Starting short rate.
        kappa: Mean reversion speed.
        theta: Long-term mean.
        sigma: Volatility.
        shocks: Pre-generated normal shocks of shape (n_paths, n_steps).
        dt: Time step size.

    Returns:
        Rate paths of shape (n_paths, n_steps + 1).
    """

def apply_hjm_dynamics(
    initial_curve: npt.NDArray[np.float64],
    volatility_functions: npt.NDArray[np.float64],
    drift_function: Callable[[float], float],
    shocks: npt.NDArray[np.float64],
    dt: float,
) -> npt.NDArray[np.float64]:
    """
    Apply HJM dynamics to generate forward curve paths.

    Args:
        initial_curve: Initial forward rate curve.
        volatility_functions: Volatility for each factor × maturity.
        drift_function: Function computing drift at each maturity.
        shocks: Pre-generated normal shocks of shape (n_paths, n_factors, n_steps).
        dt: Time step size.

    Returns:
        Forward curve paths of shape (n_paths, n_steps + 1, n_maturities).
    """
```

---

## 5. Validation Module (`src/validation/`)

### `backtesting.py`

```python
def backtest_model(
    model: InterestRateModel,
    validation_data: InterestRateDataset,
    prediction_horizon: int = 1,
) -> ValidationResult:
    """
    Backtest a calibrated model against held-out data.

    Args:
        model: Calibrated interest rate model.
        validation_data: Held-out data for validation.
        prediction_horizon: Number of steps ahead to predict.

    Returns:
        ValidationResult with error metrics and predictions.
    """
```

### `comparison.py`

```python
def compare_models(
    models: list[InterestRateModel],
    validation_data: InterestRateDataset,
) -> ComparisonResult:
    """
    Compare multiple models on the same validation data.

    Args:
        models: List of calibrated models to compare.
        validation_data: Held-out data for comparison.

    Returns:
        ComparisonResult with side-by-side metrics and recommendations.
    """

def get_model_recommendation(
    comparison: ComparisonResult,
    use_case: Literal["short_rate", "yield_curve", "derivatives"],
) -> str:
    """
    Generate a recommendation for model selection based on use case.

    Args:
        comparison: Model comparison results.
        use_case: Intended use case for the model.

    Returns:
        Human-readable recommendation string.
    """
```

---

## 6. Visualization Module (`src/visualization/`)

### `yield_curves.py`

```python
def plot_yield_curves(
    dataset: InterestRateDataset,
    dates: list[date] | None = None,
    n_curves: int = 5,
    ax: plt.Axes | None = None,
) -> plt.Figure:
    """
    Plot yield curves for selected dates.

    Args:
        dataset: Interest rate dataset.
        dates: Specific dates to plot. If None, select evenly spaced.
        n_curves: Number of curves if dates is None.
        ax: Optional matplotlib axes to plot on.

    Returns:
        Matplotlib figure with yield curve plot.
    """

def plot_yield_surface(
    dataset: InterestRateDataset,
    ax: plt.Axes | None = None,
) -> plt.Figure:
    """
    Plot 3D yield surface (date × maturity × rate).

    Args:
        dataset: Interest rate dataset.
        ax: Optional 3D matplotlib axes.

    Returns:
        Matplotlib figure with 3D surface plot.
    """
```

### `pca_plots.py`

```python
def plot_explained_variance(
    pca_result: PCAResult,
    ax: plt.Axes | None = None,
) -> plt.Figure:
    """
    Plot explained variance ratio and cumulative variance.

    Args:
        pca_result: PCA analysis results.
        ax: Optional matplotlib axes.

    Returns:
        Matplotlib figure with variance plot.
    """

def plot_component_loadings(
    pca_result: PCAResult,
    n_components: int = 3,
    ax: plt.Axes | None = None,
) -> plt.Figure:
    """
    Plot component loadings across maturities.

    Args:
        pca_result: PCA analysis results.
        n_components: Number of components to plot.
        ax: Optional matplotlib axes.

    Returns:
        Matplotlib figure with loadings plot.
    """
```

### `simulation_plots.py`

```python
def plot_simulation_paths(
    result: SimulationResult,
    n_paths: int = 100,
    show_mean: bool = True,
    show_percentiles: bool = True,
    ax: plt.Axes | None = None,
) -> plt.Figure:
    """
    Plot simulated rate paths with summary statistics.

    Args:
        result: Simulation result to plot.
        n_paths: Number of individual paths to show.
        show_mean: Whether to show mean path.
        show_percentiles: Whether to show percentile bands.
        ax: Optional matplotlib axes.

    Returns:
        Matplotlib figure with simulation plot.
    """

def plot_terminal_distribution(
    result: SimulationResult,
    ax: plt.Axes | None = None,
) -> plt.Figure:
    """
    Plot distribution of terminal rates.

    Args:
        result: Simulation result.
        ax: Optional matplotlib axes.

    Returns:
        Matplotlib figure with histogram/density plot.
    """
```

---

## 7. Export Module (`src/export/`)

### `exporter.py`

```python
def export_model(
    model: InterestRateModel,
    output_path: Path,
    format: Literal["json", "pickle"] = "json",
) -> None:
    """
    Export a calibrated model to file.

    Args:
        model: Calibrated model to export.
        output_path: Path for output file.
        format: Export format.
    """

def export_simulation(
    result: SimulationResult,
    output_path: Path,
    format: Literal["json", "csv"] = "csv",
) -> None:
    """
    Export simulation results to file.

    Args:
        result: Simulation results to export.
        output_path: Path for output file.
        format: Export format.
    """

def export_figure(
    figure: plt.Figure,
    output_path: Path,
    format: Literal["png", "pdf", "svg"] = "png",
    dpi: int = 300,
) -> None:
    """
    Export a matplotlib figure to file.

    Args:
        figure: Figure to export.
        output_path: Path for output file.
        format: Image format.
        dpi: Resolution for raster formats.
    """
```

---

## Error Hierarchy

```python
class RateModelError(Exception):
    """Base exception for all rate modeling errors."""

class DataValidationError(RateModelError):
    """Raised when input data fails validation."""

class CalibrationError(RateModelError):
    """Raised when model calibration fails."""

class ModelNotCalibratedError(RateModelError):
    """Raised when simulation attempted on uncalibrated model."""

class SimulationError(RateModelError):
    """Raised when simulation encounters an error."""

class ExportError(RateModelError):
    """Raised when export operations fail."""
```

---

## Type Stubs Summary

All public interfaces are strictly typed with:
- `npt.NDArray[np.float64]` for numerical arrays
- `Path` from `pathlib` for file paths
- `Literal` for string enumerations
- Custom type aliases defined in `src/data/types.py`
- No use of `Any` except where interfacing with untyped libraries

