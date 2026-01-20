"""Data loading, validation, and type definitions.

This module provides functionality for:
- Loading interest rate data from CSV files
- Validating data quality and completeness
- Type aliases for rate values, matrices, and date ranges
"""

from src.data.loader import (
    InterestRateDataset,
    YieldCurve,
    load_interest_rate_data,
)
from src.data.types import (
    CorrelationMatrix,
    CovarianceMatrix,
    DateRange,
    LoadingMatrix,
    MaturityTenor,
    MaturityVector,
    ObservationDate,
    PathIndex,
    RateMatrix,
    RateValue,
    RateVector,
    SimulationPathMatrix,
    TimeGrid,
    VasicekParams,
)
from src.data.validator import (
    ValidationReport,
    handle_missing_values,
    validate_dataset,
)

__all__ = [
    # Data classes
    "InterestRateDataset",
    "YieldCurve",
    "ValidationReport",
    # Functions
    "load_interest_rate_data",
    "validate_dataset",
    "handle_missing_values",
    # Scalar types
    "RateValue",
    "MaturityTenor",
    "PathIndex",
    "ObservationDate",
    # Vector types
    "RateVector",
    "MaturityVector",
    "TimeGrid",
    # Matrix types
    "RateMatrix",
    "LoadingMatrix",
    "CovarianceMatrix",
    "CorrelationMatrix",
    "SimulationPathMatrix",
    # Compound types
    "DateRange",
    "VasicekParams",
]

