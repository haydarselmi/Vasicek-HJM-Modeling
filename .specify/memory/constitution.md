<!--
================================================================================
SYNC IMPACT REPORT
================================================================================
Version change: 0.0.0 → 1.0.0 (MAJOR - Initial constitution creation)

Modified Principles: N/A (Initial creation)

Added Sections:
  - Core Principles (7 principles: Strict Typing, Code Quality, Testing 
    Standards, Performance Requirements, Dependency Management, UX Consistency, 
    Observability)
  - Technology Stack Requirements
  - Development Workflow
  - Governance

Removed Sections: N/A (Initial creation)

Templates Requiring Updates:
  - .specify/templates/plan-template.md ✅ (Compatible - Technical Context 
    section supports Python/pytest)
  - .specify/templates/spec-template.md ✅ (Compatible - No conflicts)
  - .specify/templates/tasks-template.md ✅ (Compatible - Testing phases align)

Follow-up TODOs: None
================================================================================
-->

# Vasicek & HJM Interest Rate Modeling Project Constitution

## Core Principles

### I. Strict Typing (NON-NEGOTIABLE)

All Python code MUST be strictly typed using Python 3.14+ type hints and validated
with **ty** (the fast Rust-based type checker by Astral).

**Requirements:**
- Every function MUST have complete type annotations for all parameters and return types
- Every class attribute and instance variable MUST be typed
- Generic types (`list[T]`, `dict[K, V]`) MUST use concrete type parameters
- `Any` type MUST NOT be used except when interfacing with untyped external libraries
- Type aliases MUST be defined for complex domain types (e.g., `YieldCurve`, `RatePath`)
- All code MUST pass `ty` validation with zero errors before merge

**Rationale:** Financial modeling requires mathematical precision. Type errors in 
quantitative code can lead to silent calculation errors with significant impact.

### II. Code Quality & Style (NON-NEGOTIABLE)

All Python code MUST comply with PEP8 and the Google Python Style Guide 
(https://google.github.io/styleguide/pyguide.html).

**Requirements:**
- All code MUST pass **ruff** linting with zero errors
- Import order MUST follow Google style (stdlib → third-party → local)
- Maximum cyclomatic complexity: 10 (enforced by ruff)
- Line length MUST NOT exceed 100 characters
- All public modules, classes, and functions MUST have Google-style docstrings
- All functions MUST include docstrings with Args, Returns, and Raises sections
- Variable and function names MUST be descriptive and follow snake_case convention
- Class names MUST follow PascalCase convention
- Constants MUST be UPPER_SNAKE_CASE

**Rationale:** Consistent code style reduces cognitive load during code review and 
enables automated quality gates, ensuring maintainability of complex financial models.

### III. Testing Standards (NON-NEGOTIABLE)

All code MUST be tested using **pytest** with comprehensive unit and integration tests.

**Requirements:**
- Test-Driven Development (TDD) MUST be followed: tests written → tests fail → implement
- Unit test coverage MUST be ≥90% for core mathematical modules
- Integration test coverage MUST be ≥80% for service/API layers
- All public functions MUST have at least one unit test
- Critical calculation paths MUST have property-based tests using `hypothesis`
- Test files MUST be organized as:
  - `tests/unit/` for isolated unit tests
  - `tests/integration/` for component interaction tests
  - `tests/contract/` for API contract tests
- Test naming MUST follow pattern: `test_<function>_<scenario>_<expected_outcome>`
- Fixtures MUST be used for test data setup (no hardcoded values in tests)
- All tests MUST be deterministic (no random seeds without explicit control)

**Rationale:** Financial calculations require mathematical correctness. Comprehensive 
testing ensures model accuracy and prevents regression in critical pricing logic.

### IV. Performance Requirements

All computational components MUST meet defined performance benchmarks.

**Requirements:**
- Monte Carlo simulations MUST complete 10,000 paths in <5 seconds
- Single bond pricing MUST complete in <10ms
- Yield curve construction MUST complete in <100ms
- PCA decomposition for 10 years of daily data MUST complete in <2 seconds
- Memory usage MUST NOT exceed 2GB for standard simulation runs
- NumPy vectorization MUST be used for array operations (no Python loops over arrays)
- Performance-critical paths MUST have benchmark tests using `pytest-benchmark`
- All API endpoints MUST respond in <200ms (p95)

**Rationale:** Real-time risk management requires fast model execution. Performance
constraints ensure the system remains usable for production risk calculations.

### V. Dependency Management

All project dependencies MUST be managed via **uv** and declared in `pyproject.toml`.

**Requirements:**
- All dependencies MUST be declared with version constraints (minimum version or range)
- Virtual environment MUST be created and managed via `uv venv`
- Dependencies MUST be installed via `uv pip install` or `uv sync`
- Development dependencies MUST be separated from production dependencies
- Lock file MUST be committed and kept up-to-date
- Dependency updates MUST be reviewed for breaking changes before upgrade
- Only actively maintained libraries with security support MUST be used
- Core scientific stack: `numpy`, `scipy`, `pandas` for numerical computation

**Rationale:** Reproducible builds require explicit dependency management. Using uv 
ensures fast, deterministic installations across all development environments.

### VI. User Experience Consistency

All interfaces MUST provide a consistent, intuitive experience for users.

**Requirements:**
- CLI commands MUST follow consistent naming: `<verb>-<noun>` pattern
- All CLI commands MUST support `--help` with clear usage examples
- Output formats MUST support both JSON (for automation) and human-readable (for terminal)
- Error messages MUST be actionable with clear remediation steps
- Progress indicators MUST be shown for operations >2 seconds
- All numerical outputs MUST include appropriate precision (6 decimal places for rates)
- Plots and visualizations MUST use consistent color schemes and labeling
- All user-facing strings MUST be externalized for future internationalization

**Rationale:** Quantitative analysts need clear, consistent interfaces to efficiently
interpret model outputs and integrate results into their workflows.

### VII. Observability & Logging

All components MUST implement structured logging and observability.

**Requirements:**
- All modules MUST use Python `logging` with structured JSON output option
- Log levels MUST be used appropriately: DEBUG (trace), INFO (events), WARNING 
  (recoverable), ERROR (failures), CRITICAL (system down)
- All external calls (API, database) MUST log request/response timing
- All calculation entries MUST log input parameters (sanitized for sensitive data)
- Correlation IDs MUST be propagated through async call chains
- Performance metrics MUST be collected for critical paths

**Rationale:** Debugging financial calculations requires complete audit trails.
Observability enables root cause analysis when model outputs are questioned.

## Technology Stack Requirements

**Runtime Environment:**
- Python 3.14+ (required for modern type syntax and performance improvements)
- Virtual environment managed via `uv`

**Core Dependencies:**
- `numpy` - Numerical array operations
- `scipy` - Scientific computing (optimization, statistics)
- `pandas` - Time series data manipulation
- `matplotlib` / `plotly` - Visualization

**Development Tools:**
- **Package Management:** `uv` (fast Rust-based package manager by Astral)
- **Type Checking:** `ty` (fast Rust-based type checker by Astral)
- **Linting:** `ruff` with configuration:

```toml
[tool.ruff]
exclude = [
    ".git",
    "__pycache__",
    "docs/source/conf.py",
    "old",
    "build",
    "dist",
    ".venv",
    "wiki",
    "docs",
]

[tool.ruff.lint]
select = ["E", "F", "I", "W", "UP", "ANN", "B", "C4", "SIM"]

[tool.ruff.lint.isort]
force-single-line = false
known-first-party = ["src"]

[tool.ruff.lint.mccabe]
max-complexity = 10
```

- **Testing:** `pytest` with `pytest-cov`, `pytest-benchmark`, `hypothesis`
- **Formatting:** `ruff format` (Black-compatible)

**Project Configuration:**
All configuration MUST reside in `pyproject.toml` following PEP 621.

## Development Workflow

**Pre-Commit Checks (MANDATORY):**
1. `ruff check .` - Linting MUST pass with zero errors
2. `ruff format --check .` - Formatting MUST be consistent
3. `ty check .` - Type checking MUST pass with zero errors
4. `pytest` - All tests MUST pass

**Code Review Requirements:**
- All changes MUST be reviewed by at least one team member
- Reviewers MUST verify:
  - Type annotations are complete and accurate
  - Tests cover new/changed functionality
  - Docstrings follow Google style
  - Performance considerations for numerical code
  - No hardcoded magic numbers (use named constants)

**Branch Strategy:**
- Feature branches: `feature/<ticket-id>-<description>`
- Bugfix branches: `fix/<ticket-id>-<description>`
- All branches MUST be rebased on main before merge
- Squash merge for feature branches to maintain clean history

**Continuous Integration:**
- All PRs MUST pass CI pipeline before merge
- CI pipeline MUST include: lint, type-check, test, coverage report
- Coverage MUST NOT decrease on merge

## Governance

**Amendment Process:**
1. Propose amendment via pull request to this constitution
2. Document rationale for change
3. Obtain approval from project maintainers
4. Update version number per semantic versioning rules
5. Update all dependent templates if principle changes affect them

**Versioning Policy:**
- MAJOR: Backward incompatible principle removals or redefinitions
- MINOR: New principles added or existing guidance materially expanded
- PATCH: Clarifications, typo fixes, non-semantic refinements

**Compliance Review:**
- All pull requests MUST verify compliance with this constitution
- Constitution Check in implementation plans MUST reference current version
- Violations MUST be justified in Complexity Tracking section of plan

**Version**: 1.0.0 | **Ratified**: 2026-01-11 | **Last Amended**: 2026-01-11
