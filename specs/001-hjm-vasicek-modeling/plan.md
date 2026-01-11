# Implementation Plan: Asset Rate Modeling System (Vasicek & HJM with PCA)

**Branch**: `001-hjm-vasicek-modeling` | **Date**: 2026-01-11 | **Spec**: [spec.md](./spec.md)
**Input**: Feature specification from `/specs/001-hjm-vasicek-modeling/spec.md`

## Summary

Build a comprehensive interest rate modeling system implementing both Vasicek (single-factor) and Heath-Jarrow-Morton (multifactor with PCA) models. The system follows a hybrid architecture with production-quality Python modules in `src/` for core computational logic, and Jupyter notebooks in `notebooks/` for interactive exploration, visualization, and presentation. The core functionality includes: loading and validating real-world interest rate data, calibrating and simulating the Vasicek model, performing PCA dimensionality reduction on yield curves with meaningful interpretation of principal components (level, slope, curvature), constructing and simulating a multifactor HJM model, and validating models against historical data. Optional features include autoencoder-based dimensionality reduction and foundation model integration for natural language explanations.

## Technical Context

**Language/Version**: Python 3.14+ (required by constitution)  
**Package Manager**: uv (fast Rust-based by Astral)  
**Type Checker**: ty (fast Rust-based by Astral)  
**Linting**: ruff (max-complexity=10, Google import style)  
**Primary Dependencies**: numpy, scipy, pandas, matplotlib, plotly  
**Storage**: File-based (CSV for interest rate data input, JSON/pickle for model persistence)  
**Testing**: pytest + pytest-cov + pytest-benchmark + hypothesis  
**Target Platform**: Cross-platform (Linux, macOS, Windows)  
**Project Type**: Hybrid (Python modules + Jupyter notebooks)  
**Performance Goals**: 
- 10,000 Monte Carlo simulation paths in <5 seconds
- PCA decomposition for 10 years of daily data in <2 seconds
- Yield curve construction in <100ms
- Memory usage <2GB for standard simulation runs

**Constraints**: 
- Vectorized NumPy operations (no Python loops over arrays)
- All calculations must handle negative interest rates
- Maximum 20 maturity points for PCA (configurable)

**Scale/Scope**: 
- Up to 10 years of daily interest rate observations (~2,600 data points)
- Up to 20 maturity tenors per observation
- Up to 1,000 simulation paths for HJM model per run

## Constitution Check

*GATE: Must pass before Phase 0 research. Re-check after Phase 1 design.*

### Pre-Design Check (Phase 0)

| Principle | Requirement | Status |
|-----------|-------------|--------|
| I. Strict Typing | All code typed with `ty` validation | ☑ PLANNED |
| II. Code Quality | PEP8 + Google Style + ruff passing | ☑ PLANNED |
| III. Testing Standards | pytest with ≥90% unit / ≥80% integration | ☑ PLANNED |
| IV. Performance | Benchmarks defined and met | ☑ PLANNED |
| V. Dependency Management | uv + pyproject.toml | ☑ PLANNED |
| VI. UX Consistency | CLI/output standards followed | ☑ PLANNED |
| VII. Observability | Structured logging implemented | ☑ PLANNED |

**Gate Status**: ✅ PASS - All principles planned for compliance.

### Post-Design Check (Phase 1)

| Principle | Requirement | Design Compliance |
|-----------|-------------|-------------------|
| I. Strict Typing | All code typed with `ty` validation | ✅ All module interfaces in contracts/module_interfaces.md use strict type annotations with `npt.NDArray[np.float64]`, `Literal`, `Path`, and custom type aliases. No `Any` types used. |
| II. Code Quality | PEP8 + Google Style + ruff passing | ✅ All function signatures include Google-style docstrings with Args, Returns, Raises sections. snake_case for functions, PascalCase for classes. |
| III. Testing Standards | pytest with ≥90% unit / ≥80% integration | ✅ Test structure defined in plan.md with unit/, integration/, contract/ directories. Property-based testing with hypothesis planned for mathematical invariants. |
| IV. Performance | Benchmarks defined and met | ✅ Performance goals specified: 10k MC paths <5s, PCA <2s, <2GB memory. Vectorized NumPy operations mandated in research.md. |
| V. Dependency Management | uv + pyproject.toml | ✅ All dependencies documented in research.md with version constraints. uv workflow documented in quickstart.md. |
| VI. UX Consistency | CLI/output standards followed | ✅ Notebook-based UX with consistent visualization theme. 6 decimal places for rates. Export formats (JSON, CSV, PNG/PDF) standardized. |
| VII. Observability | Structured logging implemented | ✅ Logging module planned in src/utils/logging.py. Key events (data load, calibration, simulation) to be logged per research.md. |

**Post-Design Gate Status**: ✅ PASS - All principles addressed in design artifacts. No violations.

## Project Structure

### Documentation (this feature)

```text
specs/001-hjm-vasicek-modeling/
├── plan.md              # This file (/speckit.plan command output)
├── research.md          # Phase 0 output (/speckit.plan command)
├── data-model.md        # Phase 1 output (/speckit.plan command)
├── quickstart.md        # Phase 1 output (/speckit.plan command)
├── contracts/           # Phase 1 output (/speckit.plan command)
└── tasks.md             # Phase 2 output (/speckit.tasks command - NOT created by /speckit.plan)
```

### Source Code (repository root)

```text
src/
├── __init__.py
├── data/
│   ├── __init__.py
│   ├── loader.py            # Interest rate data loading and validation
│   ├── validator.py         # Data quality checks and missing value handling
│   └── types.py             # Type aliases (YieldCurve, RatePath, etc.)
├── analysis/
│   ├── __init__.py
│   ├── statistics.py        # Summary statistics and correlations
│   └── pca.py               # Principal Component Analysis with interpretation
├── models/
│   ├── __init__.py
│   ├── base.py              # Abstract base class for interest rate models
│   ├── vasicek.py           # Vasicek model calibration and simulation
│   └── hjm.py               # HJM multifactor model with PCA integration
├── simulation/
│   ├── __init__.py
│   ├── monte_carlo.py       # Monte Carlo simulation engine
│   └── paths.py             # Simulation path management and export
├── validation/
│   ├── __init__.py
│   ├── backtesting.py       # Model validation against held-out data
│   └── comparison.py        # Model comparison utilities
├── visualization/
│   ├── __init__.py
│   ├── yield_curves.py      # Yield curve plotting
│   ├── pca_plots.py         # PCA component visualizations
│   └── simulation_plots.py  # Simulation path visualizations
├── export/
│   ├── __init__.py
│   └── exporter.py          # Export model parameters and results
└── utils/
    ├── __init__.py
    └── logging.py           # Structured logging configuration

notebooks/
├── 01_data_exploration.ipynb        # Load data, visualize yield curves, summary stats
├── 02_vasicek_model.ipynb           # Vasicek calibration and simulation demo
├── 03_pca_analysis.ipynb            # PCA on yield curves with interpretation
├── 04_hjm_model.ipynb               # HJM model with PCA factors
├── 05_model_comparison.ipynb        # Compare Vasicek vs HJM, backtesting
└── 06_optional_autoencoder.ipynb    # (Optional) Autoencoder exploration

tests/
├── conftest.py                      # Pytest fixtures for test data
├── unit/
│   ├── test_loader.py
│   ├── test_validator.py
│   ├── test_statistics.py
│   ├── test_pca.py
│   ├── test_vasicek.py
│   ├── test_hjm.py
│   └── test_monte_carlo.py
├── integration/
│   ├── test_data_pipeline.py
│   ├── test_vasicek_workflow.py
│   └── test_hjm_workflow.py
└── contract/
    └── test_model_interface.py
```

**Structure Decision**: Hybrid architecture with `src/` containing modular, well-documented Python files for all core computational logic, and `notebooks/` containing Jupyter notebooks for interactive exploration and presentation. This follows the user's specified approach: production-quality code for software engineering demonstration, and visual notebooks for academic/professional presentations.

## Complexity Tracking

No constitution violations anticipated. The project follows standard patterns:
- Single `src/` module hierarchy (not multiple projects)
- Direct file-based storage (no database abstraction layer needed)
- Simple CLI via module entry points (no complex CLI framework)
