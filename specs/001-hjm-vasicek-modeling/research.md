# Research: Asset Rate Modeling System (Vasicek & HJM with PCA)

**Branch**: `001-hjm-vasicek-modeling` | **Date**: 2026-01-11

This document consolidates research findings for all technology decisions, best practices, and implementation patterns required for the interest rate modeling system.

---

## 1. Vasicek Model Calibration

### Decision
Use Maximum Likelihood Estimation (MLE) with closed-form expressions for the Vasicek model parameters, implemented via `scipy.optimize.minimize`.

### Rationale
The Vasicek model has an analytical solution for the conditional distribution of the short rate, making MLE straightforward. The discretized Vasicek process follows:

```
r(t+Δt) = r(t) + κ(θ - r(t))Δt + σ√Δt · ε
```

where ε ~ N(0,1). This is an AR(1) process, so OLS regression can provide initial parameter estimates, with MLE refining them for better statistical properties.

### Alternatives Considered
- **Ordinary Least Squares (OLS)**: Simpler but provides biased estimates for small samples; used only for initial guesses.
- **Generalized Method of Moments (GMM)**: More robust to model misspecification but more complex; not needed for this academic application.
- **Kalman Filter**: Appropriate when rates are observed with measurement error; adds unnecessary complexity for direct rate observations.

---

## 2. HJM Model Implementation

### Decision
Implement the HJM model using discrete-time Euler-Maruyama discretization with PCA-derived volatility functions. Each principal component serves as a factor with its loading curve defining the volatility structure.

### Rationale
The HJM framework models the evolution of the entire forward rate curve:

```
df(t,T) = μ(t,T)dt + Σᵢ σᵢ(t,T) dWᵢ(t)
```

The no-arbitrage drift condition determines μ from σ. Using PCA components as volatility factors:
- Reduces dimensionality from N maturities to K factors (typically 3)
- Provides interpretable factors (level, slope, curvature)
- Captures 95%+ of variance with 3 components typically

### Alternatives Considered
- **Continuous-time simulation**: Analytically elegant but computationally expensive; Euler-Maruyama with daily steps provides sufficient accuracy.
- **LIBOR Market Model (LMM)**: Industry standard for derivatives pricing but out of scope; HJM is pedagogically clearer for demonstrating PCA integration.
- **Parametric volatility functions (Nelson-Siegel)**: Imposes structure that may not match data; PCA is data-driven and more flexible.

---

## 3. PCA Implementation for Yield Curves

### Decision
Apply PCA to yield curve changes (first differences) rather than levels, using `numpy.linalg.eigh` on the covariance matrix. Provide automatic interpretation of components based on loading patterns.

### Rationale
- **Changes vs. levels**: Yield curve levels are non-stationary; changes are approximately stationary, making PCA more meaningful.
- **Eigenvalue decomposition**: The symmetric positive semi-definite covariance matrix guarantees real, non-negative eigenvalues and orthogonal eigenvectors.
- **Interpretation**: First three components consistently represent:
  - PC1: Parallel shift (all loadings same sign, similar magnitude) - "level"
  - PC2: Tilt (short vs. long maturity loadings opposite sign) - "slope"
  - PC3: Bow (short and long same sign, medium opposite) - "curvature"

### Alternatives Considered
- **SVD on centered data**: Mathematically equivalent but eigendecomposition is more interpretable for PCA.
- **Factor Analysis**: Assumes specific error structure; PCA is simpler and sufficient for dimensionality reduction.
- **Kernel PCA**: Captures nonlinear relationships but harder to interpret; linear PCA preferred for this application.

---

## 4. Monte Carlo Simulation Engine

### Decision
Implement vectorized Monte Carlo simulation using NumPy broadcasting. Generate all random numbers upfront, then apply model dynamics in a single matrix operation per time step.

### Rationale
NumPy vectorization provides 100x speedup over Python loops. For N paths and T time steps:

```python
dW = np.random.standard_normal((N, T)) * np.sqrt(dt)  # All shocks at once
for t in range(T):
    rates[:, t+1] = f(rates[:, t], dW[:, t])  # Vectorized update
```

For HJM with K factors, dW becomes (N, K, T) and factor contributions are summed.

### Alternatives Considered
- **Python loops**: Simple but 100x slower; violates performance requirements.
- **Numba JIT compilation**: Additional complexity; NumPy vectorization sufficient for 10k paths.
- **GPU (CuPy/JAX)**: Overkill for this scale; adds deployment complexity.

---

## 5. Handling Negative Interest Rates

### Decision
Allow negative rates throughout the system. The Vasicek model naturally supports negative rates. Display and interpretation adjusted for negative rate environments.

### Rationale
Negative rates have been observed in Europe and Japan since 2014. The Vasicek model's Ornstein-Uhlenbeck process has no lower bound, making it suitable for negative rate environments. Unlike CIR (which requires positive rates), Vasicek does not need modification.

### Alternatives Considered
- **Shifted models (r + shift)**: Adds complexity without benefit since Vasicek handles negatives natively.
- **CIR model**: Maintains non-negative rates but would fail calibration in negative rate environments.
- **Log-rate models**: Prevent negatives but distort dynamics when rates are near zero.

---

## 6. Visualization Strategy

### Decision
Use matplotlib for static publication-quality plots in notebooks, with optional plotly for interactive exploration. Consistent styling via custom theme.

### Rationale
- **matplotlib**: Universal, well-documented, produces high-quality static figures suitable for academic papers and presentations.
- **plotly**: Optional for interactive exploration (hover, zoom, date selection) but not required for core functionality.
- **Consistent theme**: 
  - Color palette: Blues for rates, orange/red for volatility, green for fit quality
  - 6 decimal places for rates (e.g., 0.034521)
  - ISO date format on axes
  - Clear axis labels with units

### Alternatives Considered
- **seaborn**: Built on matplotlib, adds complexity without significant benefit for our plots.
- **bokeh**: Good for web apps but we're targeting notebooks.
- **altair**: Declarative but less control over financial chart specifics.

---

## 7. Testing Strategy for Financial Models

### Decision
Implement three-tier testing: unit tests for individual functions, integration tests for workflows, and property-based tests for mathematical invariants.

### Rationale
Financial models have known mathematical properties that serve as test oracles:
- **Vasicek**: Mean reversion (long-run mean approaches θ), variance formula
- **PCA**: Explained variance sums to 100%, components orthogonal
- **HJM**: No-arbitrage drift condition, positive variance

Property-based testing with hypothesis generates random inputs to verify these invariants hold universally.

### Alternatives Considered
- **Golden file testing**: Brittle for numerical outputs; property-based testing more robust.
- **Manual test cases only**: Insufficient coverage for edge cases.
- **Formal verification**: Overkill for this application.

---

## 8. Data Format and Validation

### Decision
Accept CSV files with columns: date, maturity tenor (or multiple columns for each tenor), rate value. Validate for completeness, chronological order, and reasonable rate ranges.

### Rationale
CSV is universal and human-readable. Standard interest rate data sources (FRED, central banks) provide CSV exports. Validation rules:
- Dates must be in ISO format (YYYY-MM-DD) or common variants
- Rates typically -2% to 20% (warn outside, error if >100%)
- Missing values: interpolate with warning, or exclude with warning
- At least 30 observations required for meaningful calibration

### Alternatives Considered
- **Parquet/HDF5**: More efficient for large datasets but adds dependency; CSV sufficient for our scale.
- **JSON**: More verbose, harder to edit manually.
- **Excel**: Requires openpyxl dependency; CSV covers main use case.

---

## 9. Notebook Organization

### Decision
Six notebooks following the analysis workflow, each self-contained with clear imports from `src/` modules.

### Rationale
The hybrid architecture separates concerns:
- **Notebooks**: Narrative, exploration, visualization, explanation
- **src/ modules**: Reusable, testable, documented business logic

Each notebook:
1. States its purpose and prerequisites
2. Imports only from `src/` (no business logic in notebooks)
3. Includes markdown explanations between code cells
4. Produces clearly labeled visualizations
5. Can run independently after data is loaded

### Alternatives Considered
- **Single monolithic notebook**: Difficult to navigate and maintain.
- **More granular notebooks**: Would require too much context switching.
- **No notebooks (CLI only)**: Loses interactive exploration value.

---

## 10. Logging and Observability

### Decision
Use Python's `logging` module with structured JSON output option. Log levels follow constitution guidelines. Calculation inputs logged for audit trail.

### Rationale
Financial calculations require audit trails for debugging and compliance. Structured logging enables:
- Filtering by log level
- Parsing for metrics extraction
- Correlation of related operations

Key logged events:
- Data load (file path, row count, date range)
- Calibration (model type, parameter estimates, convergence status)
- Simulation (path count, time horizon, random seed if set)

### Alternatives Considered
- **Print statements**: Not production quality, no filtering.
- **External logging service**: Overkill for local analysis tool.
- **Custom logging**: Reinventing the wheel.

---

## 11. Optional: Autoencoder Architecture

### Decision
If implemented, use a simple feedforward autoencoder with 3-5 latent dimensions, trained with MSE loss using PyTorch.

### Rationale
Autoencoders can capture nonlinear yield curve dynamics that PCA misses. Architecture:
- Input: N maturity points (standardized)
- Encoder: N → 64 → 32 → K (latent)
- Decoder: K → 32 → 64 → N
- Activation: ReLU (hidden), linear (output)
- Loss: Mean squared reconstruction error

Training: 100-500 epochs, Adam optimizer, batch size 32.

### Alternatives Considered
- **Variational Autoencoder (VAE)**: Provides probabilistic interpretation but adds complexity.
- **Convolutional autoencoder**: Overkill for 1D yield curve data.
- **Pre-trained foundation model**: Not applicable for this specific domain.

---

## 12. Optional: Foundation Model Integration

### Decision
If implemented, use OpenAI GPT-4.1-mini via LangChain for generating natural language interpretations of PCA components and model outputs.

### Rationale
Foundation models can translate quantitative results into accessible explanations. Use cases:
- Explain what PC1/PC2/PC3 loadings mean economically
- Summarize simulation results in plain language
- Provide context for model parameter values

LangChain provides orchestration for chaining prompts and managing context.

### Alternatives Considered
- **Local LLM**: Higher latency, more setup complexity.
- **Direct OpenAI API**: Less flexible than LangChain for prompt chaining.
- **Custom rule-based explanations**: Less flexible, more development effort.

---

## Dependency Summary

### Required Dependencies

| Package | Purpose | Version Constraint |
|---------|---------|-------------------|
| numpy | Array operations, linear algebra | >=1.26.0 |
| scipy | Optimization, statistics | >=1.12.0 |
| pandas | Time series data handling | >=2.2.0 |
| matplotlib | Static visualizations | >=3.8.0 |

### Development Dependencies

| Package | Purpose | Version Constraint |
|---------|---------|-------------------|
| pytest | Testing framework | >=8.0.0 |
| pytest-cov | Coverage reporting | >=4.1.0 |
| pytest-benchmark | Performance testing | >=4.0.0 |
| hypothesis | Property-based testing | >=6.100.0 |
| jupyter | Notebook support | >=1.0.0 |
| ipykernel | Jupyter Python kernel | >=6.0.0 |

### Optional Dependencies

| Package | Purpose | Version Constraint |
|---------|---------|-------------------|
| plotly | Interactive visualizations | >=5.18.0 |
| torch | Autoencoder implementation | >=2.2.0 |
| langchain | Foundation model integration | >=0.1.0 |
| openai | OpenAI API client | >=1.12.0 |

---

## Research Status

All technical decisions resolved. No outstanding NEEDS CLARIFICATION items. Ready to proceed to Phase 1: Design & Contracts.

