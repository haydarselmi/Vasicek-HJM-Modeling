"""Abstract base class for interest rate models.

This module defines the common interface that all interest rate models
(Vasicek, HJM) must implement, ensuring consistent API across model types.
"""

from abc import ABC, abstractmethod

import numpy as np
import numpy.typing as npt

from src.data.loader import InterestRateDataset
from src.simulation.paths import SimulationResult


class InterestRateModel(ABC):
    """Abstract base class for interest rate models.

    All interest rate models (Vasicek, HJM) inherit from this class
    and must implement the abstract methods to ensure a consistent API.

    Subclasses must implement:
        - calibrate(): Fit model parameters to historical data
        - simulate(): Generate Monte Carlo simulation paths
        - get_parameters(): Return model parameters as dictionary
        - is_calibrated: Property indicating calibration status
    """

    @abstractmethod
    def calibrate(
        self,
        data: InterestRateDataset | npt.NDArray[np.float64],
        **kwargs: object,
    ) -> None:
        """Calibrate the model to historical interest rate data.

        This method estimates the model parameters from observed data
        using the appropriate estimation technique (e.g., MLE, OLS).

        Args:
            data: Historical interest rate data. Can be either:
                - InterestRateDataset: Full dataset with multiple maturities
                - NDArray: Simple array of short rates (for Vasicek)
            **kwargs: Model-specific calibration options.

        Raises:
            CalibrationError: If calibration fails to converge or
                produces invalid parameters.
        """

    @abstractmethod
    def simulate(
        self,
        n_paths: int,
        time_horizon: float,
        time_step: float,
        random_seed: int | None = None,
        **kwargs: object,
    ) -> SimulationResult:
        """Generate simulated rate paths using Monte Carlo simulation.

        Uses the calibrated model parameters to simulate future rate
        evolution via Euler-Maruyama discretization.

        Args:
            n_paths: Number of simulation paths to generate.
            time_horizon: Simulation horizon in years.
            time_step: Time step size in years (e.g., 1/252 for daily).
            random_seed: Optional seed for reproducibility.
            **kwargs: Model-specific simulation options.

        Returns:
            SimulationResult containing all simulated paths and metadata.

        Raises:
            ModelNotCalibratedError: If simulate is called before calibrate.
            SimulationError: If simulation encounters numerical issues.
        """

    @abstractmethod
    def get_parameters(self) -> dict[str, float]:
        """Return model parameters as a dictionary.

        Returns:
            Dictionary mapping parameter names to their values.
            For example: {"kappa": 0.5, "theta": 0.03, "sigma": 0.01}
        """

    @property
    @abstractmethod
    def is_calibrated(self) -> bool:
        """Return True if model has been successfully calibrated.

        Returns:
            True if the model has valid calibrated parameters,
            False otherwise.
        """
