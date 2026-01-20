"""Monte Carlo simulation engine for interest rate models.

This module provides functionality for:
- Generating correlated normal shocks
- Simulating Vasicek and HJM model paths
- Managing and exporting simulation results
"""

from src.simulation.monte_carlo import (
    apply_vasicek_dynamics,
    generate_normal_shocks,
)
from src.simulation.paths import SimulationPath, SimulationResult

__all__ = [
    "generate_normal_shocks",
    "apply_vasicek_dynamics",
    "SimulationPath",
    "SimulationResult",
]

