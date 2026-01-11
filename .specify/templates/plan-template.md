# Implementation Plan: [FEATURE]

**Branch**: `[###-feature-name]` | **Date**: [DATE] | **Spec**: [link]
**Input**: Feature specification from `/specs/[###-feature-name]/spec.md`

**Note**: This template is filled in by the `/speckit.plan` command. See `.specify/templates/commands/plan.md` for the execution workflow.

## Summary

[Extract from feature spec: primary requirement + technical approach from research]

## Technical Context

<!--
  ACTION REQUIRED: Replace the content in this section with the technical details
  for the project. The structure here is presented in advisory capacity to guide
  the iteration process.
-->

**Language/Version**: Python 3.14+ (required by constitution)  
**Package Manager**: uv (fast Rust-based by Astral)  
**Type Checker**: ty (fast Rust-based by Astral)  
**Linting**: ruff (max-complexity=10, Google import style)  
**Primary Dependencies**: numpy, scipy, pandas, [additional as needed]  
**Storage**: [if applicable, e.g., PostgreSQL, files, or N/A]  
**Testing**: pytest + pytest-cov + pytest-benchmark + hypothesis  
**Target Platform**: [e.g., Linux server, cross-platform or NEEDS CLARIFICATION]  
**Project Type**: [single/web/mobile - determines source structure]  
**Performance Goals**: [per constitution: 10k MC paths <5s, bond price <10ms, API <200ms p95]  
**Constraints**: [per constitution: <2GB memory, <200ms p95 or NEEDS CLARIFICATION]  
**Scale/Scope**: [domain-specific, e.g., 10 years daily data or NEEDS CLARIFICATION]

## Constitution Check

*GATE: Must pass before Phase 0 research. Re-check after Phase 1 design.*

| Principle | Requirement | Status |
|-----------|-------------|--------|
| I. Strict Typing | All code typed with `ty` validation | ☐ |
| II. Code Quality | PEP8 + Google Style + ruff passing | ☐ |
| III. Testing Standards | pytest with ≥90% unit / ≥80% integration | ☐ |
| IV. Performance | Benchmarks defined and met | ☐ |
| V. Dependency Management | uv + pyproject.toml | ☐ |
| VI. UX Consistency | CLI/output standards followed | ☐ |
| VII. Observability | Structured logging implemented | ☐ |

## Project Structure

### Documentation (this feature)

```text
specs/[###-feature]/
├── plan.md              # This file (/speckit.plan command output)
├── research.md          # Phase 0 output (/speckit.plan command)
├── data-model.md        # Phase 1 output (/speckit.plan command)
├── quickstart.md        # Phase 1 output (/speckit.plan command)
├── contracts/           # Phase 1 output (/speckit.plan command)
└── tasks.md             # Phase 2 output (/speckit.tasks command - NOT created by /speckit.plan)
```

### Source Code (repository root)
<!--
  ACTION REQUIRED: Replace the placeholder tree below with the concrete layout
  for this feature. Delete unused options and expand the chosen structure with
  real paths (e.g., apps/admin, packages/something). The delivered plan must
  not include Option labels.
-->

```text
# [REMOVE IF UNUSED] Option 1: Single project (DEFAULT)
src/
├── models/
├── services/
├── cli/
└── lib/

tests/
├── contract/
├── integration/
└── unit/

# [REMOVE IF UNUSED] Option 2: Web application (when "frontend" + "backend" detected)
backend/
├── src/
│   ├── models/
│   ├── services/
│   └── api/
└── tests/

frontend/
├── src/
│   ├── components/
│   ├── pages/
│   └── services/
└── tests/

# [REMOVE IF UNUSED] Option 3: Mobile + API (when "iOS/Android" detected)
api/
└── [same as backend above]

ios/ or android/
└── [platform-specific structure: feature modules, UI flows, platform tests]
```

**Structure Decision**: [Document the selected structure and reference the real
directories captured above]

## Complexity Tracking

> **Fill ONLY if Constitution Check has violations that must be justified**

| Violation | Why Needed | Simpler Alternative Rejected Because |
|-----------|------------|-------------------------------------|
| [e.g., 4th project] | [current need] | [why 3 projects insufficient] |
| [e.g., Repository pattern] | [specific problem] | [why direct DB access insufficient] |
