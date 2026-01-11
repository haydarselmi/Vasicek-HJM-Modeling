"""Data loading, validation, and type definitions.

This module provides functionality for:
- Loading interest rate data from CSV files
- Validating data quality and completeness
- Type aliases for rate values, matrices, and date ranges
"""

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

__all__ = [
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

