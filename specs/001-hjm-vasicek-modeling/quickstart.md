# Quickstart Guide: Asset Rate Modeling System

**Branch**: `001-hjm-vasicek-modeling` | **Date**: 2026-01-11

This guide provides step-by-step instructions to set up and start using the interest rate modeling system.

---

## Prerequisites

- **Python 3.14+** installed on your system
- **uv** package manager installed ([installation guide](https://docs.astral.sh/uv/getting-started/installation/))
- Basic familiarity with yield curves and interest rate concepts

---

## Installation

### 1. Clone the Repository

```bash
git clone <repository-url>
cd act
git checkout 001-hjm-vasicek-modeling
```

### 2. Create Virtual Environment

```bash
uv venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

### 3. Install Dependencies

```bash
uv sync
```

This installs all required dependencies defined in `pyproject.toml`.

### 4. Verify Installation

```bash
python -c "from src.data.loader import load_interest_rate_data; print('Installation successful!')"
```

---

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
├── notebooks/             # Jupyter notebooks for exploration
├── tests/                 # Test suite
└── data/                  # Sample datasets (user-provided)
```

---

## Quick Start Workflow

### Step 1: Prepare Your Data

Create a CSV file with your interest rate data. Required format:

```csv
date,3M,6M,1Y,2Y,5Y,10Y,30Y
2020-01-02,0.0155,0.0160,0.0165,0.0162,0.0167,0.0182,0.0223
2020-01-03,0.0152,0.0157,0.0161,0.0158,0.0163,0.0178,0.0219
...
```

- **date**: Observation date in ISO format (YYYY-MM-DD)
- **Column headers**: Maturity tenors (3M, 6M, 1Y, etc.)
- **Values**: Interest rates as decimals (0.0155 = 1.55%)

### Step 2: Load and Explore Data

```python
from pathlib import Path
from src.data.loader import load_interest_rate_data
from src.analysis.statistics import compute_summary_statistics
from src.visualization.yield_curves import plot_yield_curves

# Load your data
dataset = load_interest_rate_data(
    file_path=Path("data/treasury_rates.csv"),
    date_column="date",
    source_name="US Treasury Rates"
)

# View summary statistics
stats = compute_summary_statistics(dataset)
print(stats)

# Visualize yield curves
fig = plot_yield_curves(dataset, n_curves=5)
fig.savefig("output/figures/yield_curves.png")
```

### Step 3: Calibrate Vasicek Model

```python
from src.models.vasicek import VasicekModel

# Use the shortest maturity as the short rate
short_rates = dataset.rates[:, 0]  # First column (e.g., 3M rate)

# Calibrate the model
vasicek = VasicekModel()
vasicek.calibrate(short_rates, dt=1/252)  # Daily data

# View parameters with interpretations
print(vasicek.get_parameters())
print(vasicek.get_parameter_interpretations())
```

### Step 4: Perform PCA Analysis

```python
from src.analysis.pca import compute_pca, interpret_components
from src.visualization.pca_plots import plot_component_loadings

# Run PCA on yield curve changes
pca_result = compute_pca(dataset, n_components=5, use_changes=True)

# Get interpretations for first 3 components
interpretations = interpret_components(pca_result, n_interpret=3)
for i, interp in enumerate(interpretations):
    print(f"PC{i+1}: {interp} ({pca_result.explained_variance_ratio[i]:.1%} variance)")

# Visualize loadings
fig = plot_component_loadings(pca_result, n_components=3)
fig.savefig("output/figures/pca_loadings.png")
```

### Step 5: Build and Simulate HJM Model

```python
from src.models.hjm import HJMModel
from src.visualization.simulation_plots import plot_simulation_paths

# Create HJM model with 3 PCA factors
hjm = HJMModel(pca_result=pca_result, n_factors=3)
hjm.calibrate(dataset)

# Simulate 1000 paths over 1 year
simulation = hjm.simulate(
    n_paths=1000,
    time_horizon=1.0,
    time_step=1/252,
    random_seed=42
)

# Visualize results
fig = plot_simulation_paths(simulation, n_paths=50)
fig.savefig("output/figures/hjm_simulation.png")
```

### Step 6: Compare Models

```python
from src.validation.comparison import compare_models
from datetime import date

# Split data for validation
train_data = dataset.filter_dates(end_date=date(2023, 6, 30))
test_data = dataset.filter_dates(start_date=date(2023, 7, 1))

# Calibrate both models on training data
vasicek_train = VasicekModel()
vasicek_train.calibrate(train_data.rates[:, 0])

hjm_train = HJMModel(pca_result=compute_pca(train_data), n_factors=3)
hjm_train.calibrate(train_data)

# Compare on test data
comparison = compare_models([vasicek_train, hjm_train], test_data)
print(comparison)
```

---

## Using the Notebooks

For interactive exploration, use the provided Jupyter notebooks:

```bash
jupyter lab notebooks/
```

**Notebook sequence:**
1. `01_data_exploration.ipynb` - Load and visualize your data
2. `02_vasicek_model.ipynb` - Vasicek calibration and simulation
3. `03_pca_analysis.ipynb` - PCA with component interpretation
4. `04_hjm_model.ipynb` - HJM model with PCA factors
5. `05_model_comparison.ipynb` - Compare and validate models
6. `06_optional_autoencoder.ipynb` - (Optional) Autoencoder exploration

Each notebook includes:
- Clear markdown explanations
- Step-by-step code cells
- Publication-quality visualizations
- Interpretation of results

---

## Running Tests

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=src --cov-report=html

# Run specific test file
pytest tests/unit/test_vasicek.py

# Run with verbose output
pytest -v
```

---

## Common Tasks

### Export Results

```python
from src.export.exporter import export_model, export_simulation, export_figure

# Save calibrated model
export_model(vasicek, Path("output/models/vasicek.json"))

# Save simulation results
export_simulation(simulation, Path("output/simulations/hjm_sim.csv"))

# Save figures in publication quality
export_figure(fig, Path("output/figures/plot.pdf"), format="pdf", dpi=300)
```

### Handle Missing Data

```python
from src.data.validator import handle_missing_values

# Interpolate missing values
clean_dataset = handle_missing_values(dataset, method="interpolate")
```

### Custom Visualization Theme

```python
import matplotlib.pyplot as plt

# Apply consistent styling
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['figure.figsize'] = (10, 6)
plt.rcParams['font.size'] = 12
```

---

## Troubleshooting

### "ModuleNotFoundError: No module named 'src'"

Ensure you're running from the project root and the virtual environment is activated:

```bash
cd act
source .venv/bin/activate
```

### "CalibrationError: Optimization failed to converge"

Try with different initial parameters or use OLS method:

```python
vasicek.calibrate(short_rates, method="ols")
```

### "DataValidationError: Too many missing values"

Check your data for gaps. Use the validator to diagnose:

```python
from src.data.validator import validate_dataset

report = validate_dataset(dataset)
print(report.warnings)
```

---

## Next Steps

After completing the quickstart:

1. **Explore the notebooks** for deeper analysis
2. **Read the data-model.md** to understand entity relationships
3. **Check module_interfaces.md** for full API documentation
4. **Run the test suite** to understand expected behaviors
5. **Experiment with your own datasets** from FRED, central banks, or Bloomberg

---

## Support

For questions or issues:
- Check the inline documentation (`help(function_name)`)
- Review test cases in `tests/` for usage examples
- Consult the research.md for implementation rationale

