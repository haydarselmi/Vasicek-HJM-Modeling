# HJM-Vasicek Modeling System

A comprehensive interest rate modeling system implementing both Vasicek (single-factor) and Heath-Jarrow-Morton (multifactor with PCA) models.

## Features

- **Data Loading**: Load and validate real-world interest rate data from CSV files
- **Vasicek Model**: Calibrate and simulate single-factor mean-reverting short-rate model
- **PCA Analysis**: Principal Component Analysis on yield curves with interpretation (level, slope, curvature)
- **HJM Model**: Multifactor Heath-Jarrow-Morton model with PCA-derived volatility functions
- **Model Validation**: Backtesting and comparison between models
- **Visualization**: Yield curve plots, simulation paths, PCA component visualizations

## Installation

```bash
# Clone the repository
git clone <repository-url>
cd act

# Install with uv (recommended)
uv sync --extra dev

# Or install with pip
pip install -e ".[dev]"
```

## Quick Start

```python
from src.data.loader import load_interest_rate_data
from src.models.vasicek import VasicekModel
from src.analysis.pca import compute_pca

# Load interest rate data
dataset = load_interest_rate_data("data/sample_treasury_rates.csv")

# Calibrate Vasicek model
vasicek = VasicekModel()
vasicek.calibrate(dataset.rates[:, 0])  # Use short rate

# Run Monte Carlo simulation
result = vasicek.simulate(n_paths=1000, time_horizon=1.0, dt=1/252)
```

## Project Structure

```
src/
├── data/           # Data loading and validation
├── analysis/       # Statistics and PCA
├── models/         # Vasicek and HJM models
├── simulation/     # Monte Carlo engine
├── validation/     # Backtesting and comparison
├── visualization/  # Plotting utilities
├── export/         # Export functionality
└── utils/          # Logging and utilities

notebooks/          # Jupyter notebooks for exploration
tests/              # Unit and integration tests
```

## Development

```bash
# Run tests
pytest

# Run tests with coverage
pytest --cov=src --cov-report=html

# Run linter
ruff check src tests

# Format code
ruff format src tests

# Type checking
ty check src
```

## License

MIT

