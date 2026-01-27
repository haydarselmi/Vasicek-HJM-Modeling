# Asset Rate Modeling System (Vasicek & HJM with PCA)

A comprehensive interest rate modeling system implementing both Vasicek (single-factor) and Heath-Jarrow-Morton (multifactor with PCA) models for quantitative finance applications.

## Overview

This project provides production-quality Python modules for:

- **Data Loading & Validation**: Load historical interest rate CSV data with automatic validation
- **Statistical Analysis**: Compute summary statistics and correlation matrices
- **PCA Dimensionality Reduction**: Decompose yield curve movements with interpretable components (level, slope, curvature)
- **Vasicek Model**: Single-factor short-rate model with MLE calibration
- **HJM Model**: Multifactor forward-rate model using PCA-derived volatility functions
- **Monte Carlo Simulation**: Vectorized path generation with configurable parameters
- **Model Validation**: Backtesting and model comparison tools
- **Visualization**: Publication-quality plots for yield curves, simulations, and PCA results

## Features

- **Modern Python**: Requires Python 3.14+ with full type annotations
- **Fast**: Vectorized NumPy operations - 10k Monte Carlo paths in <5 seconds
- **Well-tested**: Comprehensive test suite with >90% unit test coverage
- **Interpretable**: PCA components include automatic interpretation (level, slope, curvature)
- **Interactive**: Jupyter notebooks for exploration and presentation

## Installation

### Prerequisites

- Python 3.14+
- [uv](https://docs.astral.sh/uv/) package manager

### Quick Install

```bash
# Clone the repository
git clone <repository-url>
cd act

# Create virtual environment and install dependencies
uv venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
uv sync
```

### Optional Dependencies

```bash
# For interactive visualizations
uv sync --extra interactive

# For autoencoder-based dimensionality reduction
uv sync --extra autoencoder

# For LLM-powered interpretations
uv sync --extra llm

# For development (testing, linting)
uv sync --extra dev

# All optional dependencies
uv sync --extra all
```

## Quick Start

### 1. Load and Explore Data

```python
from pathlib import Path
from src.data.loader import load_interest_rate_data
from src.analysis.statistics import compute_summary_statistics

# Load interest rate data
dataset = load_interest_rate_data(
    file_path=Path("data/treasury_rates.csv"),
    date_column="date",
    source_name="US Treasury Rates"
)

# Compute summary statistics
stats = compute_summary_statistics(dataset)
print(f"Number of observations: {len(dataset.observation_dates)}")
print(f"Maturities: {dataset.maturities}")
```

### 2. Calibrate Vasicek Model

```python
from src.models.vasicek import VasicekModel

# Use short-rate (e.g., 3-month rate)
short_rates = dataset.rates[:, 0]

# Calibrate model
vasicek = VasicekModel()
vasicek.calibrate(short_rates, dt=1/252)

# Get parameters with interpretations
params = vasicek.get_parameters()
print(f"Mean reversion speed (kappa): {params['kappa']:.4f}")
print(f"Long-run mean (theta): {params['theta']:.4f}")
print(f"Volatility (sigma): {params['sigma']:.4f}")
print(vasicek.get_parameter_interpretations())
```

### 3. Perform PCA Analysis

```python
from src.analysis.pca import compute_pca, interpret_components

# Compute PCA on yield curve changes
pca_result = compute_pca(dataset, n_components=5)

# Get interpretations
interpretations = interpret_components(pca_result, n_interpret=3)
for i, interp in enumerate(interpretations):
    variance = pca_result.explained_variance_ratio[i]
    print(f"PC{i+1} ({variance:.1%}): {interp}")
```

### 4. Build HJM Model

```python
from src.models.hjm import HJMModel

# Create HJM model from PCA results
hjm = HJMModel(pca_result=pca_result, n_factors=3)
hjm.calibrate()

# Simulate forward curves
simulation = hjm.simulate(
    n_paths=1000,
    time_horizon=1.0,
    time_step=1/252,
    random_seed=42
)

print(f"Generated {len(simulation.paths)} simulation paths")
```

## Project Structure

```
act/
├── src/                    # Core Python modules
│   ├── data/              # Data loading and validation
│   ├── analysis/          # Statistics and PCA
│   ├── models/            # Vasicek and HJM models
│   ├── simulation/        # Monte Carlo engine
│   ├── validation/        # Model backtesting
│   ├── visualization/     # Plotting functions
│   └── export/            # Export utilities
├── notebooks/             # Interactive Jupyter notebooks
│   ├── 01_data_exploration.ipynb
│   ├── 02_vasicek_model.ipynb
│   ├── 03_pca_analysis.ipynb
│   ├── 04_hjm_model.ipynb
│   ├── 05_model_comparison.ipynb
│   └── 06_optional_autoencoder.ipynb
├── tests/                 # Test suite
│   ├── unit/             # Unit tests
│   └── integration/      # Integration tests
├── data/                  # Sample datasets
└── specs/                 # Design specifications
```

## Using the Notebooks

For interactive exploration:

```bash
jupyter lab notebooks/
```

**Recommended sequence:**
1. `01_data_exploration.ipynb` - Load and visualize data
2. `02_vasicek_model.ipynb` - Vasicek calibration and simulation
3. `03_pca_analysis.ipynb` - PCA with component interpretation
4. `04_hjm_model.ipynb` - HJM model with PCA factors
5. `05_model_comparison.ipynb` - Compare and validate models

## Testing

```bash
# Run all tests
uv run pytest

# Run with coverage report
uv run pytest --cov=src --cov-report=html

# Run only unit tests
uv run pytest tests/unit/

# Run benchmarks
uv run pytest tests/unit/test_performance.py -v
```

## Performance

The system is designed for performance:

- **Monte Carlo**: 10,000 simulation paths in <5 seconds
- **PCA**: 10 years of daily data processed in <2 seconds
- **Memory**: <2GB for standard simulation runs

All numerical operations use vectorized NumPy for optimal performance.

## Documentation

- [Quickstart Guide](specs/001-hjm-vasicek-modeling/quickstart.md)
- [Data Model](specs/001-hjm-vasicek-modeling/data-model.md)
- [API Contracts](specs/001-hjm-vasicek-modeling/contracts/module_interfaces.md)
- [Research Notes](specs/001-hjm-vasicek-modeling/research.md)

## Requirements

- Python >= 3.14
- NumPy >= 1.26.0
- SciPy >= 1.12.0
- Pandas >= 2.2.0
- Matplotlib >= 3.8.0

See `pyproject.toml` for the complete dependency list.

## License

MIT License

## Contributing

1. Ensure all tests pass: `uv run pytest`
2. Check linting: `uv run ruff check src/`
3. Check types: `uv run ty check src/`
4. Format code: `uv run ruff format src/`
