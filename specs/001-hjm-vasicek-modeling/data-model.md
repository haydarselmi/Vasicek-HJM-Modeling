# Data Model: Asset Rate Modeling System (Vasicek & HJM with PCA)

**Branch**: `001-hjm-vasicek-modeling` | **Date**: 2026-01-11

This document defines the core entities, their fields, relationships, validation rules, and state transitions for the interest rate modeling system.

---

## Entity Diagram

```
┌─────────────────────┐       ┌──────────────────────┐
│  InterestRateData   │───────│      YieldCurve      │
│                     │ 1:N   │                      │
│ - file_path         │       │ - observation_date   │
│ - source_name       │       │ - rates[]            │
│ - date_range        │       │ - maturities[]       │
│ - maturities[]      │       └──────────┬───────────┘
└─────────────────────┘                  │
                                         │ N:1
                                         ▼
                        ┌────────────────────────────┐
                        │       YieldCurveChange     │
                        │                            │
                        │ - from_date                │
                        │ - to_date                  │
                        │ - changes[]                │
                        └────────────┬───────────────┘
                                     │
                    ┌────────────────┼────────────────┐
                    │                │                │
                    ▼                ▼                ▼
        ┌───────────────┐   ┌───────────────┐   ┌───────────────┐
        │ VasicekModel  │   │  PCAResult    │   │   HJMModel    │
        │               │   │               │   │               │
        │ - kappa       │   │ - components  │   │ - factors     │
        │ - theta       │   │ - loadings    │   │ - volatilities│
        │ - sigma       │   │ - variance    │   │ - drift       │
        └───────┬───────┘   └───────┬───────┘   └───────┬───────┘
                │                   │                   │
                └───────────────────┴───────────────────┘
                                    │
                                    ▼
                        ┌───────────────────────┐
                        │   SimulationResult    │
                        │                       │
                        │ - paths[]             │
                        │ - time_horizon        │
                        │ - path_count          │
                        │ - model_type          │
                        └───────────┬───────────┘
                                    │
                                    ▼
                        ┌───────────────────────┐
                        │  ValidationResult     │
                        │                       │
                        │ - error_metrics       │
                        │ - comparison_data     │
                        │ - recommendations     │
                        └───────────────────────┘
```

---

## Entity Definitions

### 1. InterestRateDataset

Represents a collection of historical interest rate observations loaded from an external source.

| Field | Type | Description | Validation |
|-------|------|-------------|------------|
| `file_path` | `Path` | Path to the source file | Must exist, must be readable |
| `source_name` | `str` | Human-readable data source name | Non-empty string |
| `observation_dates` | `list[date]` | List of observation dates | Chronologically ordered, no duplicates |
| `maturities` | `list[float]` | Maturity tenors in years | Positive values, ascending order |
| `rates` | `ndarray[float64]` | Rate matrix (dates × maturities) | Shape matches dates × maturities |
| `loaded_at` | `datetime` | Timestamp when data was loaded | Auto-generated |

**Validation Rules:**
- At least 30 observation dates required for meaningful calibration
- At least 3 maturity points required
- Rate values must be between -0.10 (-10%) and 0.50 (50%), warn outside -0.02 to 0.20
- Missing values allowed up to 5% of total cells

**Relationships:**
- Contains N `YieldCurve` instances (one per observation date)

---

### 2. YieldCurve

A snapshot of interest rates across different maturities at a single point in time.

| Field | Type | Description | Validation |
|-------|------|-------------|------------|
| `observation_date` | `date` | Date of observation | Valid date |
| `maturities` | `ndarray[float64]` | Maturity tenors in years | Same as parent dataset |
| `rates` | `ndarray[float64]` | Rate values for each maturity | Length matches maturities |

**Validation Rules:**
- All rate values must be finite (no NaN, no inf)
- Maturities must be positive

**Relationships:**
- Belongs to one `InterestRateDataset`
- Paired with adjacent curve to form `YieldCurveChange`

---

### 3. YieldCurveChange

Represents the change in yield curve between two consecutive observations.

| Field | Type | Description | Validation |
|-------|------|-------------|------------|
| `from_date` | `date` | Start observation date | Valid date |
| `to_date` | `date` | End observation date | Must be after from_date |
| `maturities` | `ndarray[float64]` | Maturity tenors | Same as parent dataset |
| `changes` | `ndarray[float64]` | Rate changes (to - from) | Length matches maturities |

**Validation Rules:**
- Change magnitude should typically be <0.01 (100bp) per day; warn if larger

**Relationships:**
- Derived from two consecutive `YieldCurve` instances
- Input to PCA analysis

---

### 4. VasicekModel

A calibrated single-factor mean-reverting interest rate model.

| Field | Type | Description | Validation |
|-------|------|-------------|------------|
| `kappa` | `float` | Mean reversion speed | Positive value |
| `theta` | `float` | Long-term mean rate | Finite value, typically -0.05 to 0.15 |
| `sigma` | `float` | Volatility | Positive value |
| `calibration_date` | `datetime` | When model was calibrated | Auto-generated |
| `data_start` | `date` | Start of calibration data | Valid date |
| `data_end` | `date` | End of calibration data | After data_start |
| `log_likelihood` | `float` | MLE objective value | Finite value |
| `convergence_status` | `str` | Optimization status | "converged" or "failed" |

**Parameter Interpretations:**
- `kappa`: How fast rates revert to mean. Higher = faster reversion. Unit: 1/year. Typical range: 0.01 to 2.0
- `theta`: Equilibrium rate level. Unit: rate (e.g., 0.03 = 3%). Typical range: -0.02 to 0.10
- `sigma`: Rate volatility. Higher = more volatile. Unit: rate/√year. Typical range: 0.001 to 0.05

**Validation Rules:**
- kappa must be positive (otherwise no mean reversion)
- sigma must be positive
- Model must have converged for simulation use

**State Transitions:**
```
UNCALIBRATED → CALIBRATING → CALIBRATED
                    ↓
                  FAILED
```

---

### 5. PCAResult

Results from Principal Component Analysis on yield curve changes.

| Field | Type | Description | Validation |
|-------|------|-------------|------------|
| `n_components` | `int` | Number of components computed | Positive, ≤ n_maturities |
| `explained_variance_ratio` | `ndarray[float64]` | Variance explained by each PC | Sum ≤ 1.0, all ≥ 0 |
| `cumulative_variance` | `ndarray[float64]` | Cumulative variance explained | Monotonically increasing to 1.0 |
| `loadings` | `ndarray[float64]` | Component loadings matrix | Shape: (n_components, n_maturities) |
| `maturities` | `ndarray[float64]` | Maturity tenors | Same as input data |
| `interpretations` | `list[str]` | Human-readable component labels | Length ≤ n_components |
| `eigenvalues` | `ndarray[float64]` | Eigenvalues (variance) | Descending order, all ≥ 0 |
| `analysis_date` | `datetime` | When PCA was performed | Auto-generated |

**Component Interpretation Rules:**
1. **Level (PC1)**: All loadings same sign, relatively flat across maturities
2. **Slope (PC2)**: Short maturities opposite sign to long maturities
3. **Curvature (PC3)**: Short and long same sign, middle opposite

**Validation Rules:**
- Loadings must be orthonormal (L @ L.T ≈ I)
- Explained variance ratios must sum to ≤ 1.0
- Eigenvalues must be non-negative

**Relationships:**
- Derived from `YieldCurveChange` data
- Provides factors for `HJMModel`

---

### 6. HJMModel

A calibrated multifactor Heath-Jarrow-Morton forward rate model using PCA factors.

| Field | Type | Description | Validation |
|-------|------|-------------|------------|
| `n_factors` | `int` | Number of PCA factors used | 1 to n_components |
| `volatility_functions` | `ndarray[float64]` | Volatility for each factor × maturity | Shape: (n_factors, n_maturities) |
| `factor_variances` | `ndarray[float64]` | Variance of each factor | Length = n_factors |
| `maturities` | `ndarray[float64]` | Maturity tenors | From PCA result |
| `pca_result` | `PCAResult` | Underlying PCA decomposition | Valid PCAResult |
| `calibration_date` | `datetime` | When model was calibrated | Auto-generated |

**No-Arbitrage Drift Condition:**
The drift μ(t,T) is determined by the volatility functions:
```
μ(t,T) = Σᵢ σᵢ(t,T) ∫ᵗᵀ σᵢ(t,s) ds
```

**Validation Rules:**
- n_factors ≤ number of components in pca_result
- Volatility functions must be finite
- Must satisfy no-arbitrage condition (verified during simulation)

**State Transitions:**
```
UNCALIBRATED → CALIBRATING → CALIBRATED
                    ↓
                  FAILED
```

---

### 7. SimulationPath

A single realization of future rate evolution from a model.

| Field | Type | Description | Validation |
|-------|------|-------------|------------|
| `path_id` | `int` | Unique path identifier | Non-negative integer |
| `model_type` | `str` | "vasicek" or "hjm" | Valid model type |
| `time_grid` | `ndarray[float64]` | Time points (years from start) | Positive, ascending |
| `rates` | `ndarray[float64]` | Rate values at each time point | Shape depends on model |
| `random_seed` | `int | None` | Random seed if reproducible | None or positive integer |

**Rate Shape:**
- Vasicek: (n_time_points,) - single rate per time
- HJM: (n_time_points, n_maturities) - forward curve per time

**Validation Rules:**
- Time grid must start at 0
- All rate values must be finite

---

### 8. SimulationResult

Collection of simulation paths with summary statistics.

| Field | Type | Description | Validation |
|-------|------|-------------|------------|
| `model_type` | `str` | "vasicek" or "hjm" | Valid model type |
| `paths` | `list[SimulationPath]` | Individual simulation paths | Non-empty list |
| `path_count` | `int` | Number of paths | Positive integer |
| `time_horizon` | `float` | Simulation horizon in years | Positive value |
| `time_step` | `float` | Time step size in years | Positive, ≤ time_horizon |
| `simulation_date` | `datetime` | When simulation was run | Auto-generated |
| `model_parameters` | `dict` | Copy of model parameters | Valid model snapshot |

**Derived Statistics (computed on demand):**
- `mean_path`: Average rate across all paths at each time
- `std_path`: Standard deviation across paths at each time
- `percentile_paths`: 5th, 25th, 75th, 95th percentiles
- `terminal_distribution`: Distribution of rates at final time

**Validation Rules:**
- All paths must have same time grid
- path_count must match len(paths)

---

### 9. ValidationResult

Results from model backtesting and comparison.

| Field | Type | Description | Validation |
|-------|------|-------------|------------|
| `model_type` | `str` | Model being validated | Valid model type |
| `validation_period` | `tuple[date, date]` | Start and end of validation data | Valid date range |
| `mae` | `float` | Mean Absolute Error | Non-negative |
| `rmse` | `float` | Root Mean Squared Error | Non-negative |
| `mape` | `float | None` | Mean Absolute Percentage Error | Non-negative or None if rates cross zero |
| `hit_ratio` | `float` | Proportion of predictions within 1 std | 0 to 1 |
| `predicted_values` | `ndarray[float64]` | Model predictions | Length matches validation period |
| `actual_values` | `ndarray[float64]` | Actual observed values | Length matches validation period |
| `validation_date` | `datetime` | When validation was performed | Auto-generated |

**Interpretation Guidelines:**
- MAE < 0.002 (20bp): Excellent fit
- MAE 0.002-0.005 (20-50bp): Good fit
- MAE > 0.005 (50bp): Poor fit, consider model revision

---

### 10. FactorContribution

Decomposition of rate movements by HJM factor.

| Field | Type | Description | Validation |
|-------|------|-------------|------------|
| `factor_index` | `int` | Which factor (0=level, 1=slope, etc.) | Valid factor index |
| `factor_label` | `str` | Human-readable label | Non-empty string |
| `contribution` | `ndarray[float64]` | Contribution to rate change | Same shape as rate change |
| `percentage` | `float` | Percentage of total variance | 0 to 100 |

**Validation Rules:**
- Sum of contributions should approximately equal total rate change
- Contributions should be orthogonal

---

## Type Aliases

```python
# Core types
RateValue = float  # Interest rate as decimal (0.03 = 3%)
MaturityTenor = float  # Years to maturity
PathIndex = int  # Simulation path identifier

# Array types
RateVector = npt.NDArray[np.float64]  # 1D array of rates
RateMatrix = npt.NDArray[np.float64]  # 2D array (time × maturity)
LoadingMatrix = npt.NDArray[np.float64]  # PCA loadings (component × maturity)

# Date types
ObservationDate = date  # Date of rate observation
DateRange = tuple[date, date]  # Start and end dates
```

---

## Data Flow

```
CSV File → InterestRateDataset → [N] YieldCurve
                                      │
                                      ▼
                              YieldCurveChange (differences)
                                      │
               ┌──────────────────────┴──────────────────────┐
               │                                             │
               ▼                                             ▼
        VasicekModel                                    PCAResult
        (short rate only)                                    │
               │                                             ▼
               │                                        HJMModel
               │                                    (full curve)
               ▼                                             │
        SimulationResult ◄───────────────────────────────────┘
               │
               ▼
        ValidationResult
```

---

## File Persistence

| Entity | Format | Location | Purpose |
|--------|--------|----------|---------|
| InterestRateDataset | CSV | User-provided | Input data |
| VasicekModel | JSON | `output/models/` | Model persistence |
| PCAResult | JSON | `output/models/` | PCA persistence |
| HJMModel | JSON | `output/models/` | Model persistence |
| SimulationResult | JSON/CSV | `output/simulations/` | Result export |
| ValidationResult | JSON | `output/validation/` | Validation export |
| Plots | PNG/PDF | `output/figures/` | Visualization export |

