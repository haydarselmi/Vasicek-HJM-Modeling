# Tasks: Asset Rate Modeling System (Vasicek & HJM with PCA)

**Input**: Design documents from `/specs/001-hjm-vasicek-modeling/`
**Prerequisites**: plan.md ✓, spec.md ✓, research.md ✓, data-model.md ✓, contracts/ ✓

**Tests**: Tests included per constitution requirements (≥90% unit coverage, ≥80% integration coverage).

**Organization**: Tasks grouped by user story for independent implementation and testing.

## Format: `[ID] [P?] [Story?] Description`

- **[P]**: Can run in parallel (different files, no dependencies)
- **[Story]**: User story label (US1-US7) for traceability
- Exact file paths included in all task descriptions

---

## Phase 1: Setup (Project Initialization)

**Purpose**: Initialize Python project with constitution-compliant tooling

- [X] T001 Create project directory structure per plan.md in src/, tests/, notebooks/
- [X] T002 Initialize Python 3.14+ project with `uv init` and create pyproject.toml with all dependencies from research.md
- [X] T003 [P] Configure ruff linting in pyproject.toml (max-complexity=10, Google import style, select rules E/F/I/W/UP/ANN/B/C4/SIM)
- [X] T004 [P] Configure ty type checker settings in pyproject.toml
- [X] T005 [P] Configure pytest with pytest-cov, pytest-benchmark, hypothesis in pyproject.toml
- [X] T006 [P] Create .gitignore for Python project (venv, __pycache__, .pytest_cache, output/)
- [X] T007 Install dependencies with `uv sync` and verify installation

---

## Phase 2: Foundational (Blocking Prerequisites)

**Purpose**: Core infrastructure required by ALL user stories - MUST complete before any story implementation

**⚠️ CRITICAL**: No user story work can begin until this phase is complete

- [X] T008 Implement custom exceptions in src/utils/exceptions.py (RateModelError, DataValidationError, CalibrationError, ModelNotCalibratedError, SimulationError, ExportError)
- [X] T009 [P] Implement structured logging configuration in src/utils/logging.py (JSON output option, log levels per constitution)
- [X] T010 [P] Create type aliases in src/data/types.py (RateValue, MaturityTenor, PathIndex, RateVector, RateMatrix, LoadingMatrix, ObservationDate, DateRange)
- [X] T011 [P] Create src/__init__.py with package metadata
- [X] T012 [P] Create src/data/__init__.py
- [X] T013 [P] Create src/analysis/__init__.py
- [X] T014 [P] Create src/models/__init__.py
- [X] T015 [P] Create src/simulation/__init__.py
- [X] T016 [P] Create src/validation/__init__.py
- [X] T017 [P] Create src/visualization/__init__.py
- [X] T018 [P] Create src/export/__init__.py
- [X] T019 [P] Create src/utils/__init__.py
- [X] T020 Create tests/conftest.py with pytest fixtures for sample interest rate data (at least 100 observations, 7 maturities)
- [X] T021 [P] Setup pre-commit hooks script for ruff check, ruff format, ty check, pytest

**Checkpoint**: Foundation ready - user story implementation can begin

---

## Phase 3: User Story 1 - Load and Analyze Real-World Interest Rate Data (Priority: P1) 🎯 MVP

**Goal**: Enable analysts to load historical interest rate CSV data, validate it, compute summary statistics, and visualize yield curves.

**Independent Test**: Load sample treasury rates CSV, display summary stats (mean, std, min, max, correlations), and render yield curve plots for 5 dates.

### Tests for User Story 1

- [ ] T022 [P] [US1] Unit test for data loading in tests/unit/test_loader.py (test CSV parsing, date formats, maturity extraction)
- [ ] T023 [P] [US1] Unit test for data validation in tests/unit/test_validator.py (test missing values, rate bounds, chronological order)
- [ ] T024 [P] [US1] Unit test for statistics in tests/unit/test_statistics.py (test mean, std, min, max, correlation matrix)
- [ ] T025 [US1] Integration test for data pipeline in tests/integration/test_data_pipeline.py (load → validate → statistics → visualize)

### Implementation for User Story 1

- [ ] T026 [P] [US1] Implement InterestRateDataset dataclass in src/data/loader.py with fields: file_path, source_name, observation_dates, maturities, rates, loaded_at
- [ ] T027 [P] [US1] Implement YieldCurve dataclass in src/data/loader.py with fields: observation_date, maturities, rates
- [ ] T028 [US1] Implement load_interest_rate_data() function in src/data/loader.py per contract (CSV parsing, date handling, maturity extraction)
- [ ] T029 [US1] Implement ValidationReport dataclass in src/data/validator.py with fields: warnings, errors, is_valid
- [ ] T030 [US1] Implement validate_dataset() function in src/data/validator.py (check completeness, chronological order, rate bounds)
- [ ] T031 [US1] Implement handle_missing_values() function in src/data/validator.py (interpolate, forward_fill, drop strategies)
- [ ] T032 [P] [US1] Implement SummaryStatistics dataclass in src/analysis/statistics.py with fields: means, stds, mins, maxs, correlations
- [ ] T033 [US1] Implement compute_summary_statistics() function in src/analysis/statistics.py
- [ ] T034 [US1] Implement compute_correlation_matrix() function in src/analysis/statistics.py
- [ ] T035 [P] [US1] Implement plot_yield_curves() function in src/visualization/yield_curves.py (matplotlib, date selection, styling)
- [ ] T036 [P] [US1] Implement plot_yield_surface() function in src/visualization/yield_curves.py (3D surface plot)
- [ ] T037 [US1] Add logging to data loading and validation operations in src/data/loader.py and src/data/validator.py
- [ ] T038 [US1] Create sample data file data/sample_treasury_rates.csv with 1 year of daily data, 7 maturities (3M, 6M, 1Y, 2Y, 5Y, 10Y, 30Y)
- [ ] T039 [US1] Create notebook notebooks/01_data_exploration.ipynb demonstrating data loading, statistics, and yield curve visualization

**Checkpoint**: User Story 1 complete - analysts can load CSV data, view statistics, and visualize yield curves

---

## Phase 4: User Story 2 - Calibrate and Simulate Vasicek Model (Priority: P2)

**Goal**: Enable researchers to calibrate a Vasicek model to short-rate data, understand parameters with plain-language interpretations, and generate Monte Carlo simulations.

**Independent Test**: Calibrate Vasicek to sample short rates, display parameters (kappa, theta, sigma) with interpretations, simulate 1000 paths over 1 year.

### Tests for User Story 2

- [ ] T040 [P] [US2] Unit test for Vasicek calibration in tests/unit/test_vasicek.py (test MLE convergence, parameter bounds, OLS fallback)
- [ ] T041 [P] [US2] Unit test for Monte Carlo engine in tests/unit/test_monte_carlo.py (test shock generation, vectorization, reproducibility with seed)
- [ ] T042 [P] [US2] Property-based test for Vasicek mean reversion in tests/unit/test_vasicek.py (hypothesis: long-run mean approaches theta)
- [ ] T043 [US2] Integration test for Vasicek workflow in tests/integration/test_vasicek_workflow.py (load data → calibrate → simulate → visualize)

### Implementation for User Story 2

- [ ] T044 [US2] Implement InterestRateModel abstract base class in src/models/base.py with methods: calibrate(), simulate(), get_parameters(), is_calibrated property
- [ ] T045 [US2] Implement VasicekModel class in src/models/vasicek.py with calibration via scipy.optimize.minimize (MLE method)
- [ ] T046 [US2] Implement OLS initial parameter estimation in src/models/vasicek.py for calibration warm start
- [ ] T047 [US2] Implement get_parameter_interpretations() in src/models/vasicek.py (plain-language explanations for kappa, theta, sigma)
- [ ] T048 [US2] Implement expected_rate() and rate_variance() analytical formulas in src/models/vasicek.py
- [ ] T049 [P] [US2] Implement SimulationPath dataclass in src/simulation/paths.py with fields: path_id, model_type, time_grid, rates, random_seed
- [ ] T050 [P] [US2] Implement SimulationResult dataclass in src/simulation/paths.py with fields: model_type, paths, path_count, time_horizon, time_step, simulation_date, model_parameters
- [ ] T051 [US2] Implement generate_normal_shocks() function in src/simulation/monte_carlo.py (vectorized NumPy, optional seed)
- [ ] T052 [US2] Implement apply_vasicek_dynamics() function in src/simulation/monte_carlo.py (vectorized Euler-Maruyama)
- [ ] T053 [US2] Implement simulate() method in VasicekModel class using monte_carlo module
- [ ] T054 [P] [US2] Implement plot_simulation_paths() function in src/visualization/simulation_plots.py (paths, mean, percentile bands)
- [ ] T055 [P] [US2] Implement plot_terminal_distribution() function in src/visualization/simulation_plots.py (histogram/density)
- [ ] T056 [US2] Add logging to Vasicek calibration and simulation in src/models/vasicek.py
- [ ] T057 [US2] Create notebook notebooks/02_vasicek_model.ipynb demonstrating calibration, parameter interpretation, simulation, and visualization

**Checkpoint**: User Story 2 complete - researchers can calibrate Vasicek and run simulations

---

## Phase 5: User Story 3 - Perform PCA on Yield Curve Data with Interpretation (Priority: P3)

**Goal**: Enable analysts to run PCA on yield curve changes, visualize explained variance and loadings, and receive meaningful interpretations (level, slope, curvature).

**Independent Test**: Run PCA on sample yield curve changes, display explained variance (expect PC1 ~85%, PC2 ~10%, PC3 ~3%), show loadings plot, get interpretations.

### Tests for User Story 3

- [ ] T058 [P] [US3] Unit test for PCA computation in tests/unit/test_pca.py (test eigenvalue ordering, orthogonality, explained variance sum)
- [ ] T059 [P] [US3] Property-based test for PCA in tests/unit/test_pca.py (hypothesis: loadings orthonormal, variance ratios sum ≤1)
- [ ] T060 [US3] Integration test for PCA workflow in tests/integration/test_pca_workflow.py (load data → compute changes → PCA → interpret → visualize)

### Implementation for User Story 3

- [ ] T061 [US3] Implement YieldCurveChange dataclass in src/analysis/pca.py with fields: from_date, to_date, maturities, changes
- [ ] T062 [US3] Implement PCAResult dataclass in src/analysis/pca.py with fields: n_components, explained_variance_ratio, cumulative_variance, loadings, maturities, interpretations, eigenvalues, analysis_date
- [ ] T063 [US3] Implement compute_yield_curve_changes() helper function in src/analysis/pca.py (first differences of rate matrix)
- [ ] T064 [US3] Implement compute_pca() function in src/analysis/pca.py using numpy.linalg.eigh on covariance matrix
- [ ] T065 [US3] Implement interpret_components() function in src/analysis/pca.py with pattern-based rules (level: flat loadings, slope: tilt, curvature: bow)
- [ ] T066 [US3] Implement reconstruct_curve() function in src/analysis/pca.py for reduced-component reconstruction
- [ ] T067 [P] [US3] Implement plot_explained_variance() function in src/visualization/pca_plots.py (bar + cumulative line)
- [ ] T068 [P] [US3] Implement plot_component_loadings() function in src/visualization/pca_plots.py (loadings across maturities for first 3 PCs)
- [ ] T069 [US3] Add logging to PCA analysis in src/analysis/pca.py
- [ ] T070 [US3] Create notebook notebooks/03_pca_analysis.ipynb demonstrating PCA, interpretation, variance plots, loadings visualization

**Checkpoint**: User Story 3 complete - analysts can run PCA and interpret components

---

## Phase 6: User Story 4 - Calibrate and Simulate Multifactor HJM Model (Priority: P4)

**Goal**: Enable researchers to build an HJM model using PCA factors, calibrate volatility functions, simulate forward rate curve evolution, and analyze factor contributions.

**Independent Test**: Create 3-factor HJM from PCA results, calibrate volatilities, simulate 500 forward curve paths over 1 year, show factor contribution breakdown.

**Dependency**: Requires User Story 3 (PCA) to be complete

### Tests for User Story 4

- [ ] T071 [P] [US4] Unit test for HJM model in tests/unit/test_hjm.py (test initialization from PCA, volatility calibration, drift computation)
- [ ] T072 [P] [US4] Property-based test for HJM no-arbitrage in tests/unit/test_hjm.py (hypothesis: drift satisfies HJM condition)
- [ ] T073 [US4] Integration test for HJM workflow in tests/integration/test_hjm_workflow.py (PCA → HJM setup → calibrate → simulate)

### Implementation for User Story 4

- [ ] T074 [US4] Implement HJMModel class in src/models/hjm.py with initialization from PCAResult
- [ ] T075 [US4] Implement calibrate() method in HJMModel (estimate volatility functions from PCA loadings and eigenvalues)
- [ ] T076 [US4] Implement compute_drift() method in HJMModel (no-arbitrage drift condition)
- [ ] T077 [US4] Implement apply_hjm_dynamics() function in src/simulation/monte_carlo.py (vectorized Euler-Maruyama for forward curves)
- [ ] T078 [US4] Implement simulate() method in HJMModel class for forward rate curve simulation
- [ ] T079 [P] [US4] Implement FactorContribution dataclass in src/models/hjm.py with fields: factor_index, factor_label, contribution, percentage
- [ ] T080 [US4] Implement get_factor_contributions() method in HJMModel for decomposing simulated movements
- [ ] T081 [P] [US4] Implement plot_volatility_functions() function in src/visualization/simulation_plots.py (volatility by factor × maturity)
- [ ] T082 [P] [US4] Implement plot_forward_curve_evolution() function in src/visualization/simulation_plots.py (3D or animated)
- [ ] T083 [US4] Add logging to HJM calibration and simulation in src/models/hjm.py
- [ ] T084 [US4] Create notebook notebooks/04_hjm_model.ipynb demonstrating HJM from PCA, simulation, factor contributions

**Checkpoint**: User Story 4 complete - researchers can build and simulate HJM models

---

## Phase 7: User Story 5 - Compare Model Outputs and Validate Against Historical Data (Priority: P5)

**Goal**: Enable risk managers to compare Vasicek vs HJM outputs, run backtesting against held-out data, and receive model selection guidance.

**Independent Test**: Split data into train/test, calibrate both models on train, backtest on test, display MAE/RMSE, show side-by-side comparison.

**Dependency**: Requires User Stories 2 (Vasicek) and 4 (HJM) to be complete

### Tests for User Story 5

- [ ] T085 [P] [US5] Unit test for backtesting in tests/unit/test_backtesting.py (test error metric calculations: MAE, RMSE, MAPE, hit ratio)
- [ ] T086 [US5] Integration test for model comparison in tests/integration/test_comparison_workflow.py (train both → backtest → compare → recommend)

### Implementation for User Story 5

- [ ] T087 [US5] Implement ValidationResult dataclass in src/validation/backtesting.py with fields: model_type, validation_period, mae, rmse, mape, hit_ratio, predicted_values, actual_values, validation_date
- [ ] T088 [US5] Implement backtest_model() function in src/validation/backtesting.py (rolling predictions vs actuals)
- [ ] T089 [P] [US5] Implement ComparisonResult dataclass in src/validation/comparison.py with fields: models, validation_results, recommendations
- [ ] T090 [US5] Implement compare_models() function in src/validation/comparison.py (side-by-side metrics)
- [ ] T091 [US5] Implement get_model_recommendation() function in src/validation/comparison.py (use case based guidance)
- [ ] T092 [P] [US5] Implement plot_model_comparison() function in src/visualization/simulation_plots.py (side-by-side outputs)
- [ ] T093 [P] [US5] Implement plot_backtest_results() function in src/visualization/simulation_plots.py (predictions vs actuals with error bands)
- [ ] T094 [US5] Add filter_dates() method to InterestRateDataset in src/data/loader.py for train/test splitting
- [ ] T095 [US5] Add logging to validation and comparison operations in src/validation/
- [ ] T096 [US5] Create notebook notebooks/05_model_comparison.ipynb demonstrating model comparison and backtesting

**Checkpoint**: User Story 5 complete - risk managers can compare and validate models

---

## Phase 8: Export Functionality (Cross-Story)

**Purpose**: Export capabilities used across multiple user stories

- [ ] T097 [P] Implement export_model() function in src/export/exporter.py (JSON and pickle formats for Vasicek/HJM/PCA)
- [ ] T098 [P] Implement export_simulation() function in src/export/exporter.py (JSON and CSV formats)
- [ ] T099 [P] Implement export_figure() function in src/export/exporter.py (PNG, PDF, SVG formats with configurable DPI)
- [ ] T100 Create output/ directory structure (output/models/, output/simulations/, output/figures/, output/validation/)

---

## Phase 9: User Story 6 - Explore Autoencoder-Based Dimensionality Reduction (Priority: P6 - Optional)

**Goal**: Enable researchers to explore autoencoder as alternative to PCA for nonlinear dimensionality reduction.

**Independent Test**: Train autoencoder on yield curve data, display training loss curve, compare reconstruction error to PCA.

**Optional**: This phase can be skipped if only core functionality is needed.

### Tests for User Story 6 (Optional)

- [ ] T101 [P] [US6] Unit test for autoencoder in tests/unit/test_autoencoder.py (test encoder/decoder shapes, loss computation)

### Implementation for User Story 6 (Optional)

- [ ] T102 [P] [US6] Create src/analysis/autoencoder.py module
- [ ] T103 [US6] Implement YieldCurveAutoencoder class in src/analysis/autoencoder.py (PyTorch feedforward: N→64→32→K→32→64→N)
- [ ] T104 [US6] Implement train_autoencoder() function in src/analysis/autoencoder.py (MSE loss, Adam optimizer, batch training)
- [ ] T105 [US6] Implement compare_dimensionality_methods() function in src/analysis/autoencoder.py (PCA vs autoencoder reconstruction)
- [ ] T106 [P] [US6] Implement plot_training_loss() function in src/visualization/pca_plots.py
- [ ] T107 [P] [US6] Implement plot_latent_space() function in src/visualization/pca_plots.py (2D/3D scatter of encodings)
- [ ] T108 [US6] Create notebook notebooks/06_optional_autoencoder.ipynb demonstrating autoencoder training and comparison to PCA

**Checkpoint**: User Story 6 complete (optional) - autoencoder exploration available

---

## Phase 10: User Story 7 - Integrate Foundation Models for Enhanced Analysis (Priority: P7 - Optional)

**Goal**: Enable analysts to get AI-powered natural language explanations of model outputs and PCA components.

**Independent Test**: Submit PCA loadings to LLM, receive economic interpretation; submit simulation results, receive summary insights.

**Optional**: This phase can be skipped if only core functionality is needed.

### Implementation for User Story 7 (Optional)

- [ ] T109 [P] [US7] Create src/analysis/llm_integration.py module
- [ ] T110 [US7] Implement LLMInterpreter class in src/analysis/llm_integration.py (OpenAI GPT-4.1-mini via LangChain)
- [ ] T111 [US7] Implement interpret_pca_components() method for natural language PCA explanations
- [ ] T112 [US7] Implement summarize_simulation() method for simulation insights
- [ ] T113 [US7] Implement recommend_model() method for use-case based recommendations
- [ ] T114 [US7] Add optional LLM interpretation cells to notebooks 03, 04, 05

**Checkpoint**: User Story 7 complete (optional) - LLM-powered insights available

---

## Phase 11: Polish & Cross-Cutting Concerns

**Purpose**: Final quality improvements and constitution compliance verification

- [ ] T115 [P] Add Google-style docstrings to all public functions and classes (verify with ruff)
- [ ] T116 [P] Run ruff check and fix all linting issues
- [ ] T117 [P] Run ty check and fix all type errors
- [ ] T118 Run pytest --cov and verify ≥90% unit test coverage
- [ ] T119 Verify ≥80% integration test coverage
- [ ] T120 [P] Add benchmark tests in tests/unit/test_performance.py (10k MC paths <5s, PCA <2s)
- [ ] T121 Run benchmarks and verify performance meets constitution requirements
- [ ] T122 [P] Create README.md with project overview, installation, and quickstart
- [ ] T123 Validate quickstart.md by following all steps on fresh environment
- [ ] T124 Final constitution compliance check (all 7 principles verified in code)

---

## Dependencies & Execution Order

### Phase Dependencies

```
Phase 1: Setup ─────────────────────────────────────┐
                                                    ▼
Phase 2: Foundational ──────────────────────────────┤ (BLOCKS ALL USER STORIES)
                                                    │
         ┌──────────────────────────────────────────┤
         │                                          │
         ▼                                          │
Phase 3: US1 (Data Loading) ────────────────────────┤
         │                                          │
         ▼                                          │
Phase 4: US2 (Vasicek) ─────────────────────────────┤
         │                                          │
         ▼                                          │
Phase 5: US3 (PCA) ─────────────────────────────────┤
         │                                          │
         ▼                                          │
Phase 6: US4 (HJM) ─────► Requires US3              │
         │                                          │
         ▼                                          │
Phase 7: US5 (Comparison) ─► Requires US2 + US4     │
         │                                          │
         ▼                                          │
Phase 8: Export ────────────────────────────────────┤
         │                                          │
         ▼                                          │
Phase 9: US6 (Autoencoder - Optional)               │
         │                                          │
         ▼                                          │
Phase 10: US7 (LLM - Optional)                      │
         │                                          │
         ▼                                          │
Phase 11: Polish ◄──────────────────────────────────┘
```

### User Story Dependencies

| Story | Can Start After | Depends On |
|-------|-----------------|------------|
| US1 (P1) | Phase 2 complete | None |
| US2 (P2) | US1 complete | US1 for data loading |
| US3 (P3) | US1 complete | US1 for data loading |
| US4 (P4) | US3 complete | US3 for PCA results |
| US5 (P5) | US2 + US4 complete | US2 (Vasicek), US4 (HJM) |
| US6 (P6) | US3 complete | US3 for comparison |
| US7 (P7) | US3 complete | Any story for interpretation |

### Within Each User Story

1. Tests FIRST (write and verify they FAIL)
2. Dataclasses/entities before functions
3. Core logic before visualization
4. Notebook last (demonstrates complete story)

### Parallel Opportunities

**Phase 1 (Setup)**: T003, T004, T005, T006 can run in parallel

**Phase 2 (Foundational)**: T009-T019 can run in parallel (all __init__.py files)

**Phase 3 (US1)**: 
- Tests: T022, T023, T024 in parallel
- Implementation: T026, T027 in parallel; T032, T035, T036 in parallel

**Phase 4 (US2)**:
- Tests: T040, T041, T042 in parallel
- Implementation: T049, T050 in parallel; T054, T055 in parallel

**Phase 5 (US3)**:
- Tests: T058, T059 in parallel
- Implementation: T067, T068 in parallel

**Phase 6 (US4)**:
- Tests: T071, T072 in parallel
- Implementation: T079, T081, T082 in parallel

---

## Parallel Example: User Story 1

```bash
# Parallel test writing for US1:
Task T022: "Unit test for data loading in tests/unit/test_loader.py"
Task T023: "Unit test for data validation in tests/unit/test_validator.py"
Task T024: "Unit test for statistics in tests/unit/test_statistics.py"

# Parallel model creation for US1:
Task T026: "Implement InterestRateDataset dataclass in src/data/loader.py"
Task T027: "Implement YieldCurve dataclass in src/data/loader.py"
Task T032: "Implement SummaryStatistics dataclass in src/analysis/statistics.py"

# Parallel visualization for US1:
Task T035: "Implement plot_yield_curves() in src/visualization/yield_curves.py"
Task T036: "Implement plot_yield_surface() in src/visualization/yield_curves.py"
```

---

## Implementation Strategy

### MVP First (User Story 1 Only)

1. Complete Phase 1: Setup
2. Complete Phase 2: Foundational (CRITICAL)
3. Complete Phase 3: User Story 1
4. **STOP and VALIDATE**: Can load CSV, see stats, view yield curves
5. Deploy/demo if ready

### Core Features (User Stories 1-5)

1. Complete Setup + Foundational
2. US1: Data Loading → Test independently ✓
3. US2: Vasicek Model → Test independently ✓
4. US3: PCA Analysis → Test independently ✓
5. US4: HJM Model → Test independently ✓
6. US5: Comparison → Test independently ✓
7. Export functionality
8. Polish phase

### Full Feature Set (All Stories)

Add optional phases 9-10 for autoencoder and LLM integration.

---

## Summary

| Phase | Story | Task Count | Parallel Tasks |
|-------|-------|------------|----------------|
| 1 | Setup | 7 | 4 |
| 2 | Foundational | 14 | 12 |
| 3 | US1 (Data) | 18 | 10 |
| 4 | US2 (Vasicek) | 18 | 9 |
| 5 | US3 (PCA) | 13 | 5 |
| 6 | US4 (HJM) | 14 | 6 |
| 7 | US5 (Comparison) | 12 | 5 |
| 8 | Export | 4 | 3 |
| 9 | US6 (Optional) | 8 | 4 |
| 10 | US7 (Optional) | 6 | 1 |
| 11 | Polish | 10 | 5 |
| **Total** | | **124** | **64** |

**MVP Scope**: Phases 1-3 (39 tasks) - Load data, view stats, visualize yield curves

**Core Scope**: Phases 1-8 (100 tasks) - All core modeling functionality

**Full Scope**: All phases (124 tasks) - Including optional autoencoder and LLM features

---

## Notes

- [P] tasks = different files, no blocking dependencies
- [Story] label enables filtering tasks by user story
- Each notebook serves as integration validation for its story
- Stop at any checkpoint to validate story independently
- Constitution compliance verified in Phase 11 (all 7 principles)

