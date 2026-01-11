# Vasicek & HJM Interest Rate Modeling Project

## Complete Summary Document

---

# Table of Contents

1. [Project Overview](#1-project-overview)
2. [Why This Project?](#2-why-this-project)
3. [Key Concepts](#3-key-concepts)
4. [Mathematical Framework](#4-mathematical-framework)
5. [PCA for Yield Curves](#5-pca-for-yield-curves)
6. [Implementation Roadmap](#6-implementation-roadmap)
7. [Data Sources](#7-data-sources)
8. [Parameter Estimation](#8-parameter-estimation)
9. [Visualization Guide](#9-visualization-guide)
10. [Autoencoder Extension](#10-autoencoder-extension)
11. [Risk Metrics](#11-risk-metrics)
12. [Project Timeline](#12-project-timeline)

---

# 1. Project Overview

## 1.1 What Are We Building?

A complete interest rate modeling framework that:

- Models how interest rates evolve over time
- Captures the main patterns driving yield curve movements
- Simulates future rate scenarios for risk management
- Calculates key risk metrics (VaR, Duration, DV01)

## 1.2 The Two Models

| Model | Approach | Complexity | Use Case |
|-------|----------|------------|----------|
| **Vasicek** | Models short-term rate only | Simple | Quick calibration, bond pricing |
| **HJM** | Models entire yield curve | Advanced | Full curve dynamics, derivatives |

## 1.3 The Role of PCA

Principal Component Analysis bridges data and models by:

- Extracting the 3 main factors driving yield curves (Level, Slope, Curvature)
- Reducing dimensionality from 10+ rates to 3 factors
- Providing volatility functions for the HJM model

---

# 2. Why This Project?

## 2.1 Perfect for Finance + Data Science

| Finance Skills Developed | Data Science Skills Developed |
|--------------------------|-------------------------------|
| Interest rate dynamics | PCA for dimensionality reduction |
| Yield curve analysis | Time series modeling |
| Fixed income fundamentals | Stochastic simulation |
| Risk management concepts | Autoencoders (deep learning) |
| Regulatory frameworks | Working with real market data |

## 2.2 Career Relevance

- Quantitative finance roles (quant analyst, risk analyst)
- Asset-liability management (ALM)
- Insurance (Solvency II compliance)
- Banking (IRRBB, Basel requirements)

## 2.3 Project Differentiators

- Works with **real market data** (not synthetic)
- Combines **classical and modern** approaches
- Produces **interpretable** results
- Extensible to **deep learning** (autoencoders)

---

# 3. Key Concepts

## 3.1 The Yield Curve

The yield curve shows interest rates for different loan durations (maturities) at a single point in time.

**Key characteristics:**

- Shape can be upward sloping (normal), flat, or inverted
- Changes daily based on market conditions
- Affected by central bank policy, inflation expectations, economic outlook

**Why model it?**

- Bond pricing requires knowing discount rates
- Risk management needs to anticipate rate changes
- Regulatory capital depends on rate sensitivity

## 3.2 Short Rate vs. Forward Rate

| Concept | Definition | Model |
|---------|------------|-------|
| **Short Rate** | Instantaneous borrowing rate today | Vasicek |
| **Forward Rate** | Rate agreed today for future borrowing | HJM |

**Relationship:** The short rate is a special case of the forward rate where the borrowing starts immediately.

## 3.3 Mean Reversion

Interest rates exhibit **mean reversion**: when rates are high, they tend to fall; when low, they tend to rise. This is a key feature of the Vasicek model.

**Intuition:** Like a rubber band pulling rates back to a long-term average.

## 3.4 Stochastic Differential Equations (SDEs)

Both models use SDEs to describe rate dynamics:

- **Deterministic drift:** The expected direction of rate movement
- **Random diffusion:** Unpredictable market shocks

The combination creates realistic, uncertain future paths.

---

# 4. Mathematical Framework

## 4.1 Vasicek Model

### The Core Equation

```
dr(t) = κ(θ - r(t))dt + σdW(t)
```

### Parameters

| Symbol | Name | Meaning |
|--------|------|---------|
| r(t) | Short rate | Current instantaneous rate |
| κ (kappa) | Mean reversion speed | How fast rates return to average |
| θ (theta) | Long-term mean | Equilibrium rate level |
| σ (sigma) | Volatility | Randomness in rate changes |
| W(t) | Brownian motion | Random component |

### Key Properties

1. **Gaussian distribution:** Future rates are normally distributed
2. **Mean reversion:** Rates converge to θ over time
3. **Analytical solutions:** Bond prices have closed-form formulas
4. **Negative rates possible:** The normal distribution allows r < 0

### Bond Pricing Formula

Zero-coupon bond prices have an explicit formula:

```
P(t,T) = A(t,T) × exp(-B(t,T) × r(t))
```

Where A and B are functions of the parameters κ, θ, σ.

### Yield Curve Generation

Given the short rate r(t), the entire yield curve can be derived using the bond pricing formula for different maturities.

## 4.2 Heath-Jarrow-Morton (HJM) Framework

### The Core Equation

```
df(t,T) = α(t,T)dt + Σᵢ σᵢ(t,T)dWᵢ(t)
```

### Key Insight: No-Arbitrage Drift

The drift α is **not free**—it's determined by the volatility structure:

```
α(t,T) = Σᵢ σᵢ(t,T) × ∫ᵗᵀ σᵢ(t,u)du
```

This ensures the model is arbitrage-free by construction.

### Volatility Functions

The σᵢ(t,T) functions describe how different parts of the curve move. These come directly from PCA:

```
σᵢ(τ) = √λᵢ × vᵢ(τ)
```

Where λᵢ is the i-th eigenvalue and vᵢ(τ) is the i-th eigenvector loading at maturity τ.

### Multi-Factor Nature

HJM typically uses 3 factors:

| Factor | Interpretation | Variance Explained |
|--------|----------------|-------------------|
| Factor 1 | Level (parallel shift) | ~85% |
| Factor 2 | Slope (steepening/flattening) | ~10% |
| Factor 3 | Curvature (butterfly) | ~3% |

---

# 5. PCA for Yield Curves

## 5.1 Why PCA?

**The problem:** Yield curves have many points (10+ maturities), but they move in correlated ways.

**The solution:** PCA finds the underlying patterns that explain most of the variation.

## 5.2 The PCA Process

### Step 1: Compute Yield Changes

Analyze daily changes (Δy) rather than levels. Changes are stationary and suitable for PCA.

### Step 2: Center the Data

Subtract the mean from each column (maturity) to get zero-mean data.

### Step 3: Covariance Matrix

Compute the covariance between yield changes at different maturities. This captures how maturities co-move.

### Step 4: Eigenvalue Decomposition

Decompose the covariance matrix into eigenvectors (directions) and eigenvalues (importance).

### Step 5: Select Components

Keep the top 3 components, which explain 95%+ of the variance.

### Step 6: Interpret

- **PC1 (Level):** All loadings positive and similar — parallel shifts
- **PC2 (Slope):** Short positive, long negative — steepening/flattening
- **PC3 (Curvature):** Ends positive, middle negative — butterfly moves

## 5.3 Visualization of Factor Loadings

### PC1: Level Factor
All maturities have roughly equal, positive loadings. When PC1 increases, all rates increase together.

### PC2: Slope Factor
Short maturities have positive loadings; long maturities have negative loadings. When PC2 increases, the curve steepens.

### PC3: Curvature Factor
Ends (short and long) have positive loadings; middle has negative loadings. When PC3 increases, the curve becomes more "humped."

## 5.4 Variance Explained

Typical results for US Treasury yield curves:

| Component | Individual Variance | Cumulative Variance |
|-----------|--------------------|--------------------|
| PC1 | 85% | 85% |
| PC2 | 10% | 95% |
| PC3 | 3% | 98% |
| PC4+ | <2% | ~100% |

## 5.5 Connection to HJM

PCA results directly provide HJM volatility functions:

1. Eigenvalues → Scaling factors
2. Eigenvectors → Loading patterns across maturities
3. Combined → σᵢ(τ) volatility functions

---

# 6. Implementation Roadmap

## 6.1 Project Phases

### Phase 1: Data Pipeline (Week 1)

**Objective:** Fetch, clean, and prepare yield curve data.

**Key tasks:**
- Set up API connection to FRED (or alternative source)
- Fetch 10+ years of historical Treasury yields
- Handle missing values (forward fill, drop weekends)
- Compute daily yield changes
- Save processed data for analysis

**Output:** Clean yield curve dataset and yield changes dataset.

### Phase 2: PCA Analysis (Week 2)

**Objective:** Extract principal components from yield curve changes.

**Key tasks:**
- Center the yield change data
- Compute covariance matrix
- Perform eigenvalue decomposition
- Analyze variance explained (scree plot)
- Interpret factor loadings
- Validate with reconstruction quality

**Output:** Eigenvectors, eigenvalues, factor scores, volatility functions.

### Phase 3: Vasicek Model (Week 3)

**Objective:** Implement and calibrate the Vasicek short-rate model.

**Key tasks:**
- Extract short rate series (1-month or 3-month yield)
- Estimate parameters (κ, θ, σ) using MLE
- Implement exact simulation algorithm
- Derive bond pricing formulas
- Generate theoretical yield curves

**Output:** Calibrated Vasicek model, simulation engine, bond pricer.

### Phase 4: HJM Model (Week 4)

**Objective:** Build the HJM framework using PCA volatility functions.

**Key tasks:**
- Convert PCA results to volatility functions
- Implement HJM drift computation (no-arbitrage condition)
- Build forward rate simulation engine
- Convert forward curves to yield curves

**Output:** Full HJM simulator with PCA-based volatilities.

### Phase 5: Monte Carlo Simulation (Week 5)

**Objective:** Generate rate scenarios and compute statistics.

**Key tasks:**
- Configure simulation parameters (horizon, paths, steps)
- Run large-scale simulations (5,000-10,000 paths)
- Compute distributional statistics (mean, percentiles)
- Generate fan charts and confidence bands
- Store scenarios for risk calculations

**Output:** Simulated rate scenarios, distribution summaries.

### Phase 6: Risk Metrics (Week 6)

**Objective:** Calculate key risk measures from simulations.

**Key tasks:**
- Implement Duration and DV01 calculations
- Build VaR calculator (historical, parametric, Monte Carlo)
- Compute Expected Shortfall
- Create stress testing framework
- Generate risk dashboard

**Output:** Complete risk metrics suite.

## 6.2 Recommended Project Structure

```
interest-rate-models/
├── data/
│   ├── raw/                    # Downloaded data
│   ├── processed/              # Cleaned datasets
│   └── results/                # Model outputs
├── src/
│   ├── data_loader.py          # Data fetching
│   ├── pca_analysis.py         # PCA implementation
│   ├── vasicek_model.py        # Vasicek model
│   ├── hjm_model.py            # HJM framework
│   ├── simulation.py           # Monte Carlo engine
│   ├── risk_metrics.py         # VaR, Duration, etc.
│   └── visualization.py        # Plotting utilities
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_pca_analysis.ipynb
│   ├── 03_vasicek_calibration.ipynb
│   ├── 04_hjm_implementation.ipynb
│   └── 05_risk_metrics.ipynb
├── config.yaml                 # Configuration
└── README.md                   # Documentation
```

## 6.3 Key Dependencies

| Package | Purpose |
|---------|---------|
| numpy | Numerical computations |
| pandas | Data manipulation |
| scipy | Optimization, integration |
| scikit-learn | PCA implementation |
| matplotlib/plotly | Visualization |
| fredapi | FRED data access |
| yfinance | Alternative data source |

---

# 7. Data Sources

## 7.1 Recommended: FRED (Federal Reserve Economic Data)

**Website:** https://fred.stlouisfed.org/

**Why FRED?**
- Free API access
- High-quality official data
- Complete maturity spectrum
- Long history (decades)
- Python library available (fredapi)

**Available Treasury Series:**

| Maturity | Series ID |
|----------|-----------|
| 1 Month | DGS1MO |
| 3 Month | DGS3MO |
| 6 Month | DGS6MO |
| 1 Year | DGS1 |
| 2 Year | DGS2 |
| 3 Year | DGS3 |
| 5 Year | DGS5 |
| 7 Year | DGS7 |
| 10 Year | DGS10 |
| 20 Year | DGS20 |
| 30 Year | DGS30 |

**Setup:** Create free account → Request API key → Use in code.

## 7.2 Alternative: Yahoo Finance

**Pros:** Very easy setup, no API key needed
**Cons:** Limited maturities (5Y, 10Y, 30Y only)

**Best for:** Quick prototyping before switching to FRED.

## 7.3 Alternative: ECB (European Central Bank)

**Website:** ECB Statistical Data Warehouse

**Best for:** Euro area yield curves, European projects.

## 7.4 Data Quality Considerations

| Issue | Solution |
|-------|----------|
| Missing values | Forward fill (carry last observation) |
| Weekends/holidays | Filter to business days only |
| Data gaps | Linear interpolation for short gaps |
| Outliers | Review manually, winsorize if needed |

## 7.5 Recommended Data Scope

| Parameter | Recommendation |
|-----------|----------------|
| History | 10+ years (captures multiple cycles) |
| Maturities | 10 points spanning 1M to 30Y |
| Frequency | Daily |
| Region | US Treasury (most liquid) |

---

# 8. Parameter Estimation

## 8.1 Methods Overview

| Method | Complexity | Accuracy | Best For |
|--------|------------|----------|----------|
| OLS | Low | Moderate | Initial estimates |
| MLE | Medium | High | Standard calibration |
| GMM | Medium-High | High | Robust estimation |
| Kalman Filter | High | Highest | Multi-yield calibration |

## 8.2 Method 1: Ordinary Least Squares (OLS)

**Idea:** Discretize the Vasicek SDE into a linear regression.

**Process:**
1. Regress rate changes on rate levels
2. Extract parameters from regression coefficients
3. Estimate volatility from residual standard deviation

**When to use:** As a quick first pass to get starting values for MLE.

## 8.3 Method 2: Maximum Likelihood Estimation (MLE)

**Idea:** Find parameters that make observed data most probable.

**Process:**
1. Write down the likelihood function (Gaussian transitions)
2. Take the log for numerical stability
3. Maximize using numerical optimization (Nelder-Mead, BFGS)

**Key advantage:** Statistically efficient, provides standard errors.

**When to use:** Standard choice for Vasicek calibration.

## 8.4 Method 3: Generalized Method of Moments (GMM)

**Idea:** Match theoretical moments to sample moments.

**Moments used:**
- Mean of rates → θ
- Variance of rates → σ²/(2κ)
- Autocovariance → Var × exp(-κΔt)

**When to use:** When uncertain about normality assumption.

## 8.5 Method 4: Kalman Filter

**Idea:** Treat short rate as hidden state, observed through yields with noise.

**Process:**
1. Specify state-space model
2. Recursive predict-update cycle
3. Accumulate log-likelihood
4. Optimize over parameters

**When to use:** When calibrating to multiple yields simultaneously.

## 8.6 Practical Recommendations

1. **Start with OLS** for quick initial estimates
2. **Refine with MLE** for final calibration
3. **Use GMM** if you suspect non-normality
4. **Use Kalman Filter** for production systems with multiple yields

---

# 9. Visualization Guide

## 9.1 Data Exploration Visualizations

### 9.1.1 Yield Curve Time Series

**Purpose:** Show how individual rates evolve over time.

**Elements:**
- X-axis: Date
- Y-axis: Yield (%)
- Lines: One per maturity (color gradient)

**Insights:** Rate trends, relative movements, crisis periods.

### 9.1.2 Yield Curve 3D Surface

**Purpose:** Visualize entire curve evolution in one view.

**Elements:**
- X-axis: Date
- Y-axis: Maturity (years)
- Z-axis/Color: Yield level

**Insights:** Shape evolution, inversions, volatility patterns.

### 9.1.3 Correlation Heatmap

**Purpose:** Show how correlated different maturities are.

**Elements:**
- Matrix of correlations (yield changes)
- Color scale: -1 (blue) to +1 (red)

**Insights:** Adjacent maturities highly correlated, justifies PCA.

### 9.1.4 Yield Change Distribution

**Purpose:** Check normality assumption.

**Elements:**
- Histogram of daily changes
- Normal distribution overlay
- Q-Q plot

**Insights:** Fat tails, skewness, outliers.

## 9.2 PCA Visualizations

### 9.2.1 Scree Plot

**Purpose:** Determine how many components to keep.

**Elements:**
- Bars: Individual variance per PC
- Line: Cumulative variance
- Horizontal line at 95%

**Insights:** Elbow location, 3 PCs usually sufficient.

### 9.2.2 Factor Loadings Plot

**Purpose:** Interpret what each PC represents.

**Elements:**
- X-axis: Maturity (years)
- Y-axis: Loading value
- Lines: PC1, PC2, PC3

**Expected patterns:**
- PC1: Flat → Level
- PC2: Sloped → Slope
- PC3: Curved → Curvature

### 9.2.3 Factor Score Time Series

**Purpose:** See how factors evolve over time.

**Elements:**
- Three subplots (PC1, PC2, PC3)
- X-axis: Date
- Y-axis: Factor score

**Insights:** Correlation with Fed actions, economic cycles.

## 9.3 Model Output Visualizations

### 9.3.1 Rate Path Fan Chart

**Purpose:** Show range of possible future rate paths.

**Elements:**
- X-axis: Time (years)
- Y-axis: Rate (%)
- Shaded bands: Confidence intervals (5th-95th)
- Solid line: Median

**Insights:** Uncertainty growth, mean reversion.

### 9.3.2 Terminal Distribution

**Purpose:** Show rate distribution at horizon.

**Elements:**
- Histogram of terminal rates
- Vertical lines: Mean, percentiles

**Insights:** Distribution shape, tail risk.

### 9.3.3 Yield Curve Evolution (HJM)

**Purpose:** Show full curve dynamics.

**Elements:**
- Multiple curves at different times
- Color gradient: Start to end

**Insights:** Shape changes, realistic dynamics.

## 9.4 Risk Metric Visualizations

### 9.4.1 VaR Histogram

**Purpose:** Show loss distribution and VaR threshold.

**Elements:**
- P&L histogram
- Vertical lines: VaR levels
- Shaded tail

**Insights:** Tail thickness, extreme scenarios.

### 9.4.2 Key Rate Duration Profile

**Purpose:** Show sensitivity across the curve.

**Elements:**
- Bar chart by maturity
- Duration values

**Insights:** Risk concentration points.

## 9.5 Visualization Best Practices

| Aspect | Recommendation |
|--------|----------------|
| Color palette | Viridis (sequential), RdBu (diverging) |
| Labels | Always include title, axis labels, units |
| Grid | Light gridlines for readability |
| Legend | Clear, not overlapping data |

---

# 10. Autoencoder Extension

## 10.1 Why Autoencoders?

**PCA limitation:** Captures only linear relationships.

**Autoencoder advantage:** Captures non-linear patterns.

| Aspect | PCA | Autoencoder |
|--------|-----|-------------|
| Transformation | Linear | Non-linear |
| Basis functions | Fixed (eigenvectors) | Learned |
| Flexibility | Limited | High |
| Interpretability | Easy | Requires analysis |

## 10.2 Autoencoder Architecture

### Basic Structure

```
INPUT (10 yields)
    ↓
ENCODER (compress)
    ↓
LATENT SPACE (3 dimensions)
    ↓
DECODER (reconstruct)
    ↓
OUTPUT (10 reconstructed yields)
```

### Recommended Layer Sizes

| Layer | Neurons | Activation |
|-------|---------|------------|
| Input | 10 | - |
| Hidden 1 | 64 | ReLU |
| Hidden 2 | 32 | ReLU |
| Latent | 3 | Linear |
| Hidden 3 | 32 | ReLU |
| Hidden 4 | 64 | ReLU |
| Output | 10 | Linear |

## 10.3 Types of Autoencoders

### Vanilla Autoencoder
- Standard feedforward network
- MSE loss function
- Good for basic non-linear compression

### Variational Autoencoder (VAE)
- Latent space is probabilistic (not deterministic)
- Encodes to mean and variance
- Good for generating new scenarios

### Denoising Autoencoder
- Trained to remove noise from inputs
- More robust features
- Good for filtering noisy market data

### Temporal Autoencoder (LSTM-based)
- Uses recurrent layers
- Captures time dependencies
- Good for regime detection

## 10.4 Training Considerations

| Aspect | Recommendation |
|--------|----------------|
| Data split | 70% train, 15% validation, 15% test |
| Normalization | Z-score (subtract mean, divide by std) |
| Learning rate | 0.001 (Adam optimizer) |
| Batch size | 32-64 |
| Epochs | 100-500 (with early stopping) |
| Regularization | L2 (0.01) or Dropout (0.1) |

## 10.5 Interpreting Latent Space

Unlike PCA, autoencoder dimensions don't have automatic interpretations.

**Interpretation techniques:**
1. **Sensitivity analysis:** Vary one dimension, observe output changes
2. **Correlation with PCA:** Compare AE latent to PCA factors
3. **Visualization:** Plot latent space colored by curve features

**Expected finding:** If trained well, latent dimensions will correlate with Level, Slope, Curvature.

## 10.6 Connecting to HJM

**From autoencoder to volatility functions:**

1. Compute Jacobian of decoder: ∂ŷ/∂z
2. Each Jacobian column = response to latent dimension
3. Use as volatility loading in HJM

## 10.7 When to Use Each

| Situation | Recommendation |
|-----------|----------------|
| Standard project | PCA |
| Research with lots of data | Autoencoder |
| Production system | PCA (or validated AE) |
| Regime detection | Autoencoder |
| First analysis | PCA, then explore AE |

---

# 11. Risk Metrics

## 11.1 Sensitivity Metrics

### Duration

**Definition:** Measures price sensitivity to yield changes.

**Types:**
- **Macaulay Duration:** Weighted average time of cash flows (years)
- **Modified Duration:** Price sensitivity (% change per 1% yield change)
- **Effective Duration:** Numerical approximation for complex instruments

**Formula (Modified):**
```
D_mod = D_mac / (1 + y)
```

**Interpretation:**
- Duration 5 → 1% yield increase → ~5% price decrease
- Duration 10 → 1% yield increase → ~10% price decrease

### Convexity

**Definition:** Measures the curvature of price-yield relationship.

**Why it matters:** Duration assumes linear relationship; convexity captures the curve.

**Formula:**
```
Price change ≈ -Duration × Δy + 0.5 × Convexity × (Δy)²
```

**Key insight:** Positive convexity = bonds benefit from volatility.

### DV01 (Dollar Value of 01)

**Definition:** Dollar change in value for 1 basis point yield change.

**Formula:**
```
DV01 = Price × Duration × 0.0001
```

**Example:**
- $1M bond, Duration 5 → DV01 = $500
- 1bp yield increase → $500 loss

### Key Rate Duration (KRD)

**Definition:** Sensitivity to specific maturity point, others unchanged.

**Purpose:** Captures non-parallel curve moves.

**Standard key rates:** 3M, 6M, 1Y, 2Y, 3Y, 5Y, 7Y, 10Y, 30Y

## 11.2 Statistical Risk Metrics

### Value at Risk (VaR)

**Definition:** Maximum loss at a given confidence level over a given horizon.

**Example:** "1-day 95% VaR = $1M" means:
- With 95% confidence
- Daily losses won't exceed $1M
- But 5% of the time, losses could be worse

**Calculation methods:**

| Method | Approach | Pros | Cons |
|--------|----------|------|------|
| Historical | Use past returns | No assumptions | Past ≠ future |
| Parametric | Assume normal distribution | Fast, analytical | Underestimates tails |
| Monte Carlo | Simulate scenarios | Handles complexity | Computationally heavy |

**Formula (Parametric):**
```
VaR = |DV01| × σ_yield × z_α × √T
```

Where z_α = 1.645 (95%) or 2.33 (99%)

### Expected Shortfall (ES / CVaR)

**Definition:** Average loss in the worst cases beyond VaR.

**Why preferred over VaR:**
- Considers tail severity (VaR ignores what happens beyond threshold)
- Coherent risk measure (satisfies mathematical properties)
- Required by Basel III/IV

**Interpretation:**
- VaR tells you the threshold
- ES tells you how bad it gets when you cross it

## 11.3 Stress Testing

### Purpose

Evaluate performance under extreme but plausible scenarios.

### Standard Scenarios

| Scenario | Short Rate Change | Long Rate Change |
|----------|-------------------|------------------|
| Parallel Up | +200bp | +200bp |
| Parallel Down | -200bp | -200bp |
| Steepening | -100bp | +100bp |
| Flattening | +100bp | -100bp |
| Short Up | +200bp | 0bp |
| Short Down | -200bp | 0bp |

### Regulatory Scenarios

**Solvency II (European insurers):**
- Prescribed percentage shocks by maturity
- Asymmetric up/down scenarios
- SCR (Solvency Capital Requirement) calculation

**Basel (Banks):**
- IRRBB (Interest Rate Risk in Banking Book)
- EVE (Economic Value of Equity) impact
- NII (Net Interest Income) impact

## 11.4 Risk Dashboard Summary

A complete risk report should include:

| Category | Metrics |
|----------|---------|
| Position Sensitivity | Duration, DV01, Convexity |
| Key Rate Profile | KRD at each maturity |
| Statistical Risk | VaR (95%, 99%), ES |
| Stress Tests | P&L under each scenario |

---

# 12. Project Timeline

## 12.1 Suggested 8-Week Plan

| Week | Phase | Deliverables |
|------|-------|--------------|
| 1 | Data Pipeline | Clean yield curve dataset |
| 2 | PCA Analysis | Factor loadings, variance explained |
| 3 | Vasicek Model | Calibrated parameters, simulation |
| 4 | HJM Model | Full curve simulation |
| 5 | Monte Carlo | 10,000 scenario dataset |
| 6 | Risk Metrics | VaR, Duration, stress tests |
| 7 | Visualization | Complete dashboard |
| 8 | Documentation | Final report, code cleanup |

## 12.2 Quick Start Checklist

- [ ] Get FRED API key
- [ ] Set up Python environment
- [ ] Fetch 10 years of Treasury yields
- [ ] Clean and validate data
- [ ] Run PCA, verify 3 components explain 95%+
- [ ] Calibrate Vasicek parameters
- [ ] Build HJM with PCA volatilities
- [ ] Run simulations
- [ ] Calculate risk metrics
- [ ] Create visualizations
- [ ] Document findings

## 12.3 Extension Ideas

After completing the core project:

1. **Autoencoder:** Replace PCA with neural network
2. **Regime switching:** Add multiple Vasicek regimes
3. **International:** Compare US, EUR, UK curves
4. **Derivatives:** Price interest rate swaps and options
5. **Real portfolio:** Apply to actual bond portfolio

---

# Appendix: Key Formulas Reference

## Vasicek Model

| Formula | Expression |
|---------|------------|
| SDE | dr = κ(θ - r)dt + σdW |
| Mean (at time t) | μ(t) = θ + (r₀ - θ)e^(-κt) |
| Variance (at time t) | v(t) = σ²/(2κ) × (1 - e^(-2κt)) |
| Bond price | P(t,T) = A(t,T) × exp(-B(t,T) × r) |
| B function | B = (1 - e^(-κτ)) / κ |

## HJM Framework

| Formula | Expression |
|---------|------------|
| Forward rate SDE | df(t,T) = α(t,T)dt + Σᵢσᵢ(t,T)dWᵢ |
| Drift (no-arbitrage) | α(t,T) = Σᵢ σᵢ(t,T) ∫ᵗᵀ σᵢ(t,u)du |
| PCA volatility | σᵢ(τ) = √λᵢ × vᵢ(τ) |

## PCA

| Formula | Expression |
|---------|------------|
| Covariance matrix | Σ = X'X / (n-1) |
| Eigen decomposition | Σ = VΛV' |
| Variance explained | λᵢ / Σⱼλⱼ |
| Factor scores | F = XV |
| Reconstruction | X̂ = F × V' |

## Risk Metrics

| Metric | Formula |
|--------|---------|
| Modified Duration | D_mac / (1 + y) |
| DV01 | P × D_mod × 0.0001 |
| Parametric VaR | \|DV01\| × σ × z_α × √T |
| Price change (with convexity) | -D × Δy + 0.5 × C × (Δy)² |

---

*Document created: January 2026*
*Project: Vasicek-HJM Interest Rate Modeling*

