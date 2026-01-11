"""Utility modules for logging and exception handling.

This module provides functionality for:
- Structured logging with JSON output option
- Custom exceptions for the modeling system
"""

from src.utils.exceptions import (
    CalibrationError,
    DataValidationError,
    ExportError,
    ModelNotCalibratedError,
    RateModelError,
    SimulationError,
)
from src.utils.logging import (
    configure_logging,
    get_logger,
    log_operation,
)

__all__ = [
    # Exceptions
    "RateModelError",
    "DataValidationError",
    "CalibrationError",
    "ModelNotCalibratedError",
    "SimulationError",
    "ExportError",
    # Logging
    "get_logger",
    "configure_logging",
    "log_operation",
]

