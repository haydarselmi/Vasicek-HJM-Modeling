#!/usr/bin/env bash
# =============================================================================
# Pre-commit hooks for HJM-Vasicek modeling project
#
# This script runs code quality checks before each commit:
# 1. ruff check - Linting
# 2. ruff format --check - Format verification
# 3. ty check - Type checking
# 4. pytest - Unit tests
#
# Usage:
#   ./scripts/pre-commit.sh           # Run all checks
#   ./scripts/pre-commit.sh --quick   # Skip tests (faster)
#
# To install as git pre-commit hook:
#   ln -sf ../../scripts/pre-commit.sh .git/hooks/pre-commit
# =============================================================================

set -e  # Exit on first error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Parse arguments
QUICK_MODE=false
if [[ "$1" == "--quick" ]]; then
    QUICK_MODE=true
fi

echo "=============================================="
echo "Running pre-commit checks..."
echo "=============================================="

# Function to print status
print_status() {
    if [ $1 -eq 0 ]; then
        echo -e "${GREEN}✓ $2 passed${NC}"
    else
        echo -e "${RED}✗ $2 failed${NC}"
        exit 1
    fi
}

# 1. Ruff linting
echo ""
echo -e "${YELLOW}[1/4] Running ruff check...${NC}"
uv run ruff check src tests
print_status $? "Ruff linting"

# 2. Ruff format check
echo ""
echo -e "${YELLOW}[2/4] Running ruff format check...${NC}"
uv run ruff format --check src tests
print_status $? "Ruff format"

# 3. Type checking with ty
echo ""
echo -e "${YELLOW}[3/4] Running ty type check...${NC}"
if command -v ty &> /dev/null; then
    uv run ty check src
    print_status $? "Type checking"
else
    echo -e "${YELLOW}⚠ ty not installed, skipping type check${NC}"
fi

# 4. Run tests (unless in quick mode)
if [ "$QUICK_MODE" = false ]; then
    echo ""
    echo -e "${YELLOW}[4/4] Running pytest...${NC}"
    uv run pytest tests/ -q --tb=short
    print_status $? "Tests"
else
    echo ""
    echo -e "${YELLOW}[4/4] Skipping tests (quick mode)${NC}"
fi

echo ""
echo "=============================================="
echo -e "${GREEN}All pre-commit checks passed!${NC}"
echo "=============================================="

