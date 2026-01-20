"""Simulation path and result dataclasses.

This module defines the data structures for storing Monte Carlo
simulation paths and aggregated simulation results.
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Literal

import numpy as np
import numpy.typing as npt


@dataclass
class SimulationPath:
    """A single realization of future rate evolution from a model.

    This dataclass represents one simulated path from a Monte Carlo
    simulation, containing the time grid and corresponding rate values.

    Attributes:
        path_id: Unique identifier for this path within a simulation.
        model_type: Type of model used ("vasicek" or "hjm").
        time_grid: Array of time points in years from simulation start.
        rates: Array of rate values at each time point.
            For Vasicek: shape (n_time_points,) - single rate per time.
            For HJM: shape (n_time_points, n_maturities) - forward curve per time.
        random_seed: Random seed used for this path, if reproducible.
    """

    path_id: int
    model_type: Literal["vasicek", "hjm"]
    time_grid: npt.NDArray[np.float64]
    rates: npt.NDArray[np.float64]
    random_seed: int | None = None

    @property
    def n_time_points(self) -> int:
        """Return the number of time points in the path."""
        return len(self.time_grid)

    @property
    def initial_rate(self) -> float | npt.NDArray[np.float64]:
        """Return the initial rate (at t=0).

        Returns:
            For Vasicek: single rate value.
            For HJM: array of forward rates.
        """
        return self.rates[0]

    @property
    def terminal_rate(self) -> float | npt.NDArray[np.float64]:
        """Return the terminal rate (at final time point).

        Returns:
            For Vasicek: single rate value.
            For HJM: array of forward rates.
        """
        return self.rates[-1]

    @property
    def time_horizon(self) -> float:
        """Return the simulation time horizon in years."""
        return float(self.time_grid[-1])


@dataclass
class SimulationResult:
    """Collection of simulation paths with metadata and summary statistics.

    This dataclass aggregates multiple simulation paths from a Monte Carlo
    simulation, providing access to individual paths as well as computed
    summary statistics.

    Attributes:
        model_type: Type of model used ("vasicek" or "hjm").
        paths: List of individual simulation paths.
        path_count: Number of paths in the simulation.
        time_horizon: Simulation horizon in years.
        time_step: Time step size in years.
        simulation_date: Timestamp when simulation was run.
        model_parameters: Dictionary copy of model parameters at simulation time.
    """

    model_type: Literal["vasicek", "hjm"]
    paths: list[SimulationPath]
    path_count: int
    time_horizon: float
    time_step: float
    simulation_date: datetime = field(default_factory=datetime.now)
    model_parameters: dict[str, float] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Validate the simulation result after initialization."""
        if self.path_count != len(self.paths):
            msg = f"path_count ({self.path_count}) does not match len(paths) ({len(self.paths)})"
            raise ValueError(msg)

    @property
    def time_grid(self) -> npt.NDArray[np.float64]:
        """Return the common time grid for all paths.

        Returns:
            Time grid array from the first path.

        Raises:
            ValueError: If no paths are available.
        """
        if not self.paths:
            msg = "No paths available in simulation result"
            raise ValueError(msg)
        return self.paths[0].time_grid

    @property
    def n_time_points(self) -> int:
        """Return the number of time points per path."""
        return len(self.time_grid)

    def get_rates_matrix(self) -> npt.NDArray[np.float64]:
        """Return all rate paths as a matrix.

        Returns:
            For Vasicek: shape (n_paths, n_time_points).
            For HJM: shape (n_paths, n_time_points, n_maturities).
        """
        if self.model_type == "vasicek":
            return np.array([p.rates for p in self.paths])
        else:  # hjm
            return np.array([p.rates for p in self.paths])

    def mean_path(self) -> npt.NDArray[np.float64]:
        """Compute the mean rate across all paths at each time point.

        Returns:
            Mean rate array with shape (n_time_points,) for Vasicek
            or (n_time_points, n_maturities) for HJM.
        """
        rates_matrix = self.get_rates_matrix()
        return np.mean(rates_matrix, axis=0)

    def std_path(self) -> npt.NDArray[np.float64]:
        """Compute the standard deviation across all paths at each time point.

        Returns:
            Standard deviation array with same shape as mean_path.
        """
        rates_matrix = self.get_rates_matrix()
        return np.std(rates_matrix, axis=0)

    def percentile_paths(
        self,
        percentiles: list[float] | None = None,
    ) -> dict[float, npt.NDArray[np.float64]]:
        """Compute percentile paths across all simulation paths.

        Args:
            percentiles: List of percentiles to compute (e.g., [5, 25, 50, 75, 95]).
                Defaults to [5, 25, 50, 75, 95].

        Returns:
            Dictionary mapping percentile values to rate arrays.
        """
        if percentiles is None:
            percentiles = [5.0, 25.0, 50.0, 75.0, 95.0]

        rates_matrix = self.get_rates_matrix()
        return {
            p: np.percentile(rates_matrix, p, axis=0)
            for p in percentiles
        }

    def terminal_distribution(self) -> npt.NDArray[np.float64]:
        """Return the distribution of terminal rates across all paths.

        Returns:
            1D array of terminal rates with shape (n_paths,) for Vasicek.
            For HJM, returns (n_paths, n_maturities).
        """
        return np.array([p.terminal_rate for p in self.paths])

    def terminal_mean(self) -> float | npt.NDArray[np.float64]:
        """Return the mean of terminal rates.

        Returns:
            Mean terminal rate (scalar for Vasicek, array for HJM).
        """
        return np.mean(self.terminal_distribution(), axis=0)

    def terminal_std(self) -> float | npt.NDArray[np.float64]:
        """Return the standard deviation of terminal rates.

        Returns:
            Standard deviation of terminal rates.
        """
        return np.std(self.terminal_distribution(), axis=0)

    def terminal_percentiles(
        self,
        percentiles: list[float] | None = None,
    ) -> dict[float, float | npt.NDArray[np.float64]]:
        """Compute percentiles of the terminal rate distribution.

        Args:
            percentiles: List of percentiles to compute.
                Defaults to [5, 25, 50, 75, 95].

        Returns:
            Dictionary mapping percentile values to terminal rates.
        """
        if percentiles is None:
            percentiles = [5.0, 25.0, 50.0, 75.0, 95.0]

        terminal_dist = self.terminal_distribution()
        return {
            p: np.percentile(terminal_dist, p, axis=0)
            for p in percentiles
        }
