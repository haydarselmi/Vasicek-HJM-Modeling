"""Structured logging configuration for the HJM-Vasicek modeling system.

This module provides a centralized logging configuration with support for
both human-readable console output and structured JSON output for
programmatic log analysis.
"""

import json
import logging
import sys
from datetime import UTC, datetime
from typing import Literal

# Log level type alias
LogLevel = Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]

# Default format for console output
DEFAULT_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
DEFAULT_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"

# Logger name for this package
PACKAGE_LOGGER_NAME = "hjm_vasicek"


class JSONFormatter(logging.Formatter):
    """Custom formatter that outputs log records as JSON objects.

    This formatter produces structured log output suitable for log
    aggregation systems and programmatic analysis.
    """

    def format(self, record: logging.LogRecord) -> str:
        """Format the log record as a JSON string.

        Args:
            record: The log record to format.

        Returns:
            JSON-formatted string representation of the log record.
        """
        log_entry = {
            "timestamp": datetime.now(UTC).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno,
        }

        # Add exception info if present
        if record.exc_info:
            log_entry["exception"] = self.formatException(record.exc_info)

        # Add any extra fields passed to the log call
        if hasattr(record, "extra_fields"):
            log_entry["extra"] = record.extra_fields

        return json.dumps(log_entry, default=str)


class ContextAdapter(logging.LoggerAdapter):
    """Logger adapter that adds contextual information to log records.

    This adapter allows adding structured context to log messages,
    useful for tracing operations across the modeling workflow.
    """

    def process(
        self, msg: str, kwargs: dict
    ) -> tuple[str, dict]:
        """Process the log message and add extra context.

        Args:
            msg: The log message.
            kwargs: Additional keyword arguments.

        Returns:
            Tuple of processed message and updated kwargs.
        """
        extra = kwargs.get("extra", {})
        if self.extra:
            extra.update(self.extra)
        kwargs["extra"] = extra
        return msg, kwargs


def get_logger(
    name: str | None = None,
    level: LogLevel = "INFO",
    json_output: bool = False,
) -> logging.Logger:
    """Get or create a configured logger.

    Args:
        name: Logger name. If None, returns the package root logger.
        level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL).
        json_output: If True, output logs as JSON objects.

    Returns:
        Configured logger instance.
    """
    # Build full logger name under package namespace
    logger_name = f"{PACKAGE_LOGGER_NAME}.{name}" if name else PACKAGE_LOGGER_NAME

    logger = logging.getLogger(logger_name)

    # Only configure if no handlers exist (avoid duplicate handlers)
    if not logger.handlers:
        logger.setLevel(getattr(logging, level))

        # Create console handler
        handler = logging.StreamHandler(sys.stdout)
        handler.setLevel(getattr(logging, level))

        # Set formatter based on output mode
        if json_output:
            formatter = JSONFormatter()
        else:
            formatter = logging.Formatter(DEFAULT_FORMAT, DEFAULT_DATE_FORMAT)

        handler.setFormatter(formatter)
        logger.addHandler(handler)

        # Prevent propagation to root logger to avoid duplicate logs
        logger.propagate = False

    return logger


def configure_logging(
    level: LogLevel = "INFO",
    json_output: bool = False,
    log_file: str | None = None,
) -> None:
    """Configure the root package logger with specified settings.

    This function should be called once at application startup to
    configure logging for the entire package.

    Args:
        level: Logging level for the package.
        json_output: If True, output logs as JSON objects.
        log_file: Optional file path to write logs to.
    """
    logger = logging.getLogger(PACKAGE_LOGGER_NAME)
    logger.setLevel(getattr(logging, level))

    # Clear existing handlers
    logger.handlers.clear()

    # Create formatter
    if json_output:
        formatter = JSONFormatter()
    else:
        formatter = logging.Formatter(DEFAULT_FORMAT, DEFAULT_DATE_FORMAT)

    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(getattr(logging, level))
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # File handler if specified
    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(getattr(logging, level))
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    # Prevent propagation to root logger
    logger.propagate = False


def log_operation(
    logger: logging.Logger,
    operation: str,
    **context: object,
) -> None:
    """Log an operation with structured context.

    This is a convenience function for logging operations with
    consistent formatting and context information.

    Args:
        logger: Logger instance to use.
        operation: Name of the operation being performed.
        **context: Additional context key-value pairs to include.
    """
    context_str = ", ".join(f"{k}={v}" for k, v in context.items())
    if context_str:
        logger.info("%s | %s", operation, context_str)
    else:
        logger.info("%s", operation)

