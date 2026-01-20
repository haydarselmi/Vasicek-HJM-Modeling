"""Monte Carlo simulation engine for interest rate models.

This module provides functions for generating random shocks and
applying model dynamics to simulate rate paths.
"""

import numpy as np
import numpy.typing as npt

from src.utils.logging import get_logger

logger = get_logger("simulation.monte_carlo")


def generate_normal_shocks(
    n_paths: int,
    n_steps: int,
    n_factors: int = 1,
    random_seed: int | None = None,
) -> npt.NDArray[np.float64]:
    """Generate standard normal random shocks for Monte Carlo simulation.

    Creates an array of independent N(0,1) random variables for use in
    simulating stochastic processes via Euler-Maruyama discretization.

    Args:
        n_paths: Number of simulation paths to generate.
        n_steps: Number of time steps per path.
        n_factors: Number of independent factors (default: 1).
            For Vasicek, use 1. For HJM, use number of PCA factors.
        random_seed: Optional random seed for reproducibility.
            If None, uses non-deterministic random state.

    Returns:
        Array of shape (n_paths, n_factors, n_steps) containing N(0,1) values.
        All elements are independent standard normal random variables.

    Example:
        >>> shocks = generate_normal_shocks(1000, 252, n_factors=1, random_seed=42)
        >>> shocks.shape
        (1000, 1, 252)
        >>> np.abs(np.mean(shocks)) < 0.1  # Close to zero mean
        True
    """
    logger.debug(
        "Generating normal shocks: n_paths=%d, n_steps=%d, n_factors=%d, seed=%s",
        n_paths,
        n_steps,
        n_factors,
        random_seed,
    )

    rng = np.random.default_rng(random_seed)
    shocks = rng.standard_normal((n_paths, n_factors, n_steps))

    return shocks.astype(np.float64)


def apply_vasicek_dynamics(
    initial_rate: float,
    kappa: float,
    theta: float,
    sigma: float,
    shocks: npt.NDArray[np.float64],
    dt: float,
) -> npt.NDArray[np.float64]:
    """Apply Vasicek dynamics to generate rate paths using Euler-Maruyama discretization.

    Simulates the Vasicek short rate process:
        dr(t) = κ(θ - r(t))dt + σdW(t)

    Using the discretization:
        r(t+dt) = r(t) + κ(θ - r(t))dt + σ√dt·ε

    where ε ~ N(0,1).

    Args:
        initial_rate: Starting short rate value (r₀).
        kappa: Mean reversion speed (κ). Must be positive.
        theta: Long-term mean rate (θ).
        sigma: Volatility (σ). Must be positive.
        shocks: Pre-generated normal shocks of shape (n_paths, n_steps).
            Each element should be N(0,1) distributed.
        dt: Time step size in years (e.g., 1/252 for daily).

    Returns:
        Rate paths of shape (n_paths, n_steps + 1), where the first column
        contains the initial rate and subsequent columns contain simulated rates.

    Raises:
        ValueError: If shocks array has wrong dimensions.

    Example:
        >>> shocks = generate_normal_shocks(100, 252, n_factors=1, random_seed=42)
        >>> shocks = shocks.squeeze(axis=1)  # Shape: (100, 252)
        >>> rates = apply_vasicek_dynamics(
        ...     initial_rate=0.03,
        ...     kappa=0.5,
        ...     theta=0.03,
        ...     sigma=0.01,
        ...     shocks=shocks,
        ...     dt=1/252
        ... )
        >>> rates.shape
        (100, 253)
    """
    if shocks.ndim != 2:
        msg = f"shocks must be 2D array, got {shocks.ndim}D"
        raise ValueError(msg)

    n_paths, n_steps = shocks.shape

    logger.debug(
        "Applying Vasicek dynamics: n_paths=%d, n_steps=%d, dt=%.6f",
        n_paths,
        n_steps,
        dt,
    )

    # Pre-allocate output array (n_paths × (n_steps + 1))
    # +1 for initial rate at t=0
    rates = np.zeros((n_paths, n_steps + 1), dtype=np.float64)
    rates[:, 0] = initial_rate

    # Pre-compute constants
    sqrt_dt = np.sqrt(dt)

    # Euler-Maruyama simulation (vectorized over paths)
    for t in range(n_steps):
        # Drift term: κ(θ - r(t))dt
        drift = kappa * (theta - rates[:, t]) * dt

        # Diffusion term: σ√dt·ε
        diffusion = sigma * sqrt_dt * shocks[:, t]

        # Update: r(t+dt) = r(t) + drift + diffusion
        rates[:, t + 1] = rates[:, t] + drift + diffusion

    return rates
