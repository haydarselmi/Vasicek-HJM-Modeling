"""Custom exceptions for the HJM-Vasicek modeling system.

This module defines a hierarchy of custom exceptions used throughout the
interest rate modeling system for consistent error handling and reporting.
"""


class RateModelError(Exception):
    """Base exception for all rate modeling errors.

    All custom exceptions in this module inherit from this base class,
    allowing callers to catch all rate modeling errors with a single
    except clause if desired.

    Args:
        message: Human-readable error description.
        details: Optional dictionary with additional error context.
    """

    def __init__(self, message: str, details: dict | None = None) -> None:
        """Initialize the exception with message and optional details.

        Args:
            message: Human-readable error description.
            details: Optional dictionary with additional error context.
        """
        super().__init__(message)
        self.message = message
        self.details = details or {}

    def __str__(self) -> str:
        """Return string representation of the exception."""
        if self.details:
            return f"{self.message} | Details: {self.details}"
        return self.message


class DataValidationError(RateModelError):
    """Raised when interest rate data fails validation checks.

    This exception is raised when loaded data does not meet the required
    quality standards, such as missing values, invalid date formats,
    or rate values outside acceptable bounds.

    Args:
        message: Description of the validation failure.
        field: Name of the field that failed validation.
        value: The invalid value that caused the failure.
        details: Additional context about the validation error.
    """

    def __init__(
        self,
        message: str,
        field: str | None = None,
        value: object = None,
        details: dict | None = None,
    ) -> None:
        """Initialize the data validation error.

        Args:
            message: Description of the validation failure.
            field: Name of the field that failed validation.
            value: The invalid value that caused the failure.
            details: Additional context about the validation error.
        """
        error_details = details or {}
        if field is not None:
            error_details["field"] = field
        if value is not None:
            error_details["value"] = value
        super().__init__(message, error_details)
        self.field = field
        self.value = value


class CalibrationError(RateModelError):
    """Raised when model calibration fails to converge or produces invalid parameters.

    This exception indicates that the optimization algorithm used for
    parameter estimation did not converge, or the resulting parameters
    are outside acceptable bounds.

    Args:
        message: Description of the calibration failure.
        model_type: Type of model being calibrated (e.g., "Vasicek", "HJM").
        iterations: Number of iterations before failure.
        details: Additional context about the calibration error.
    """

    def __init__(
        self,
        message: str,
        model_type: str | None = None,
        iterations: int | None = None,
        details: dict | None = None,
    ) -> None:
        """Initialize the calibration error.

        Args:
            message: Description of the calibration failure.
            model_type: Type of model being calibrated.
            iterations: Number of iterations before failure.
            details: Additional context about the calibration error.
        """
        error_details = details or {}
        if model_type is not None:
            error_details["model_type"] = model_type
        if iterations is not None:
            error_details["iterations"] = iterations
        super().__init__(message, error_details)
        self.model_type = model_type
        self.iterations = iterations


class ModelNotCalibratedError(RateModelError):
    """Raised when attempting to use a model that has not been calibrated.

    This exception is raised when methods that require calibrated parameters
    (such as simulate or predict) are called before the calibrate method.

    Args:
        message: Description of the error.
        model_type: Type of model that is not calibrated.
        details: Additional context about the error.
    """

    def __init__(
        self,
        message: str = "Model must be calibrated before use",
        model_type: str | None = None,
        details: dict | None = None,
    ) -> None:
        """Initialize the model not calibrated error.

        Args:
            message: Description of the error.
            model_type: Type of model that is not calibrated.
            details: Additional context about the error.
        """
        error_details = details or {}
        if model_type is not None:
            error_details["model_type"] = model_type
        super().__init__(message, error_details)
        self.model_type = model_type


class SimulationError(RateModelError):
    """Raised when Monte Carlo simulation encounters an error.

    This exception indicates problems during path generation, such as
    numerical instability, invalid parameters, or memory issues.

    Args:
        message: Description of the simulation failure.
        n_paths: Number of paths being simulated.
        time_horizon: Time horizon of the simulation.
        details: Additional context about the simulation error.
    """

    def __init__(
        self,
        message: str,
        n_paths: int | None = None,
        time_horizon: float | None = None,
        details: dict | None = None,
    ) -> None:
        """Initialize the simulation error.

        Args:
            message: Description of the simulation failure.
            n_paths: Number of paths being simulated.
            time_horizon: Time horizon of the simulation.
            details: Additional context about the simulation error.
        """
        error_details = details or {}
        if n_paths is not None:
            error_details["n_paths"] = n_paths
        if time_horizon is not None:
            error_details["time_horizon"] = time_horizon
        super().__init__(message, error_details)
        self.n_paths = n_paths
        self.time_horizon = time_horizon


class ExportError(RateModelError):
    """Raised when exporting model results or figures fails.

    This exception indicates problems during export operations, such as
    file permission issues, invalid formats, or serialization failures.

    Args:
        message: Description of the export failure.
        export_format: Format being exported (e.g., "JSON", "CSV", "PNG").
        file_path: Path where export was attempted.
        details: Additional context about the export error.
    """

    def __init__(
        self,
        message: str,
        export_format: str | None = None,
        file_path: str | None = None,
        details: dict | None = None,
    ) -> None:
        """Initialize the export error.

        Args:
            message: Description of the export failure.
            export_format: Format being exported.
            file_path: Path where export was attempted.
            details: Additional context about the export error.
        """
        error_details = details or {}
        if export_format is not None:
            error_details["format"] = export_format
        if file_path is not None:
            error_details["file_path"] = file_path
        super().__init__(message, error_details)
        self.export_format = export_format
        self.file_path = file_path

