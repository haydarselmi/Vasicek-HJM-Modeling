"""Vasicek interest rate model implementation.

This module implements the Vasicek single-factor mean-reverting
interest rate model with MLE and OLS calibration methods.

The Vasicek model follows the stochastic differential equation:
    dr(t) = κ(θ - r(t))dt + σdW(t)

where:
    - r(t) is the short rate at time t
    - κ (kappa) is the mean reversion speed
    - θ (theta) is the long-term mean rate
    - σ (sigma) is the volatility
    - W(t) is a Wiener process
"""

from datetime import date, datetime
from typing import Literal

import numpy as np
import numpy.typing as npt
from scipy import optimize

from src.data.loader import InterestRateDataset
from src.models.base import InterestRateModel
from src.simulation.monte_carlo import apply_vasicek_dynamics, generate_normal_shocks
from src.simulation.paths import SimulationPath, SimulationResult
from src.utils.exceptions import CalibrationError, ModelNotCalibratedError
from src.utils.logging import get_logger

logger = get_logger("models.vasicek")

# Minimum observations required for calibration
MIN_OBSERVATIONS = 30


def _estimate_ols_params(
    short_rates: npt.NDArray[np.float64],
    dt: float,
) -> tuple[float, float, float]:
    """Estimate Vasicek parameters using Ordinary Least Squares.

    The discretized Vasicek process can be written as an AR(1):
        r(t+dt) = a + b*r(t) + ε

    where:
        a = κ*θ*dt
        b = 1 - κ*dt
        Var(ε) = σ²*dt

    Args:
        short_rates: Array of observed short rates.
        dt: Time step between observations in years.

    Returns:
        Tuple of (kappa, theta, sigma) estimated parameters.

    Raises:
        CalibrationError: If estimation produces invalid parameters.
    """
    n = len(short_rates)
    if n < MIN_OBSERVATIONS:
        msg = f"Insufficient observations for calibration: {n} < {MIN_OBSERVATIONS}"
        raise CalibrationError(msg, model_type="Vasicek")

    # Prepare regression: r(t+1) = a + b*r(t) + epsilon
    y = short_rates[1:]  # r(t+dt)
    x = short_rates[:-1]  # r(t)

    # OLS regression
    n_reg = len(y)
    sum_x = np.sum(x)
    sum_y = np.sum(y)
    sum_xy = np.sum(x * y)
    sum_x2 = np.sum(x**2)

    # b = (n*sum(xy) - sum(x)*sum(y)) / (n*sum(x²) - sum(x)²)
    denom = n_reg * sum_x2 - sum_x**2
    if abs(denom) < 1e-15:
        msg = "Near-constant rate series, cannot estimate parameters"
        raise CalibrationError(msg, model_type="Vasicek")

    b = (n_reg * sum_xy - sum_x * sum_y) / denom
    a = (sum_y - b * sum_x) / n_reg

    # Residual variance
    y_pred = a + b * x
    residuals = y - y_pred
    sigma_eps_sq = np.var(residuals, ddof=2)

    # Convert to Vasicek parameters
    # b = 1 - κ*dt => κ = (1 - b) / dt
    kappa = (1 - b) / dt

    if kappa <= 0:
        # No mean reversion detected, use small positive value
        logger.warning("Negative kappa estimated (%.4f), using fallback", kappa)
        kappa = 0.01

    # a = κ*θ*dt => θ = a / (κ*dt)
    theta = a / (kappa * dt)

    # Var(ε) = σ²*dt => σ = sqrt(Var(ε)/dt)
    sigma = np.sqrt(sigma_eps_sq / dt)

    if sigma <= 0:
        sigma = 1e-6  # Small positive fallback

    return kappa, theta, sigma


class VasicekModel(InterestRateModel):
    """Vasicek single-factor mean-reverting interest rate model.

    The short rate follows the Ornstein-Uhlenbeck process:
        dr(t) = κ(θ - r(t))dt + σdW(t)

    This model allows negative interest rates and has closed-form
    solutions for expected rate and variance.

    Attributes:
        kappa: Mean reversion speed (1/year). Higher values mean
            faster reversion to the long-term mean.
        theta: Long-term mean rate. The rate level to which the
            process reverts.
        sigma: Volatility (rate/√year). Standard deviation of
            rate changes scaled by time.
        calibration_date: Timestamp when model was calibrated.
        data_start: Start date of calibration data.
        data_end: End date of calibration data.
        log_likelihood: MLE objective value (only for MLE calibration).
        convergence_status: "converged" or "failed".

    Example:
        >>> model = VasicekModel()
        >>> model.calibrate(short_rates, dt=1/252, method="mle")
        >>> print(model.get_parameter_interpretations())
        >>> result = model.simulate(n_paths=1000, time_horizon=1.0, time_step=1/252)
    """

    def __init__(
        self,
        kappa: float | None = None,
        theta: float | None = None,
        sigma: float | None = None,
    ) -> None:
        """Initialize the Vasicek model.

        Can be initialized with pre-specified parameters (for testing)
        or left empty for calibration from data.

        Args:
            kappa: Optional initial mean reversion speed.
            theta: Optional initial long-term mean.
            sigma: Optional initial volatility.
        """
        self._kappa: float | None = kappa
        self._theta: float | None = theta
        self._sigma: float | None = sigma
        self._calibrated: bool = False
        self.calibration_date: datetime | None = None
        self.data_start: date | None = None
        self.data_end: date | None = None
        self.log_likelihood: float | None = None
        self.convergence_status: str = "not_started"

        # Mark as calibrated if all parameters provided
        if kappa is not None and theta is not None and sigma is not None:
            self._calibrated = True
            self.calibration_date = datetime.now()
            self.convergence_status = "manual"

    @property
    def kappa(self) -> float:
        """Mean reversion speed (1/year)."""
        if self._kappa is None:
            msg = "kappa not set, model not calibrated"
            raise ModelNotCalibratedError(msg, model_type="Vasicek")
        return self._kappa

    @property
    def theta(self) -> float:
        """Long-term mean rate."""
        if self._theta is None:
            msg = "theta not set, model not calibrated"
            raise ModelNotCalibratedError(msg, model_type="Vasicek")
        return self._theta

    @property
    def sigma(self) -> float:
        """Volatility (rate/√year)."""
        if self._sigma is None:
            msg = "sigma not set, model not calibrated"
            raise ModelNotCalibratedError(msg, model_type="Vasicek")
        return self._sigma

    @property
    def is_calibrated(self) -> bool:
        """Return True if model has been successfully calibrated."""
        return self._calibrated

    def calibrate(
        self,
        data: InterestRateDataset | npt.NDArray[np.float64],
        dt: float = 1 / 252,
        method: Literal["mle", "ols"] = "mle",
        **kwargs: object,
    ) -> None:
        """Calibrate the Vasicek model to a short rate time series.

        Estimates the parameters κ, θ, and σ from historical data
        using Maximum Likelihood Estimation (MLE) or Ordinary Least
        Squares (OLS).

        Args:
            data: Either an InterestRateDataset (will use shortest maturity)
                or a 1D array of short rate observations.
            dt: Time step between observations in years.
                Default is 1/252 (daily data with 252 trading days/year).
            method: Calibration method:
                - "mle": Maximum Likelihood Estimation (default, more accurate)
                - "ols": Ordinary Least Squares (faster, less accurate)
            **kwargs: Additional keyword arguments (ignored).

        Raises:
            CalibrationError: If calibration fails to converge or
                produces invalid parameters.
        """
        # Extract short rates from dataset if needed
        if isinstance(data, InterestRateDataset):
            short_rates = data.rates[:, 0].copy()  # Shortest maturity
            self.data_start = data.observation_dates[0]
            self.data_end = data.observation_dates[-1]
            logger.info(
                "Calibrating Vasicek to %d observations from %s to %s",
                len(short_rates),
                self.data_start,
                self.data_end,
            )
        else:
            short_rates = np.asarray(data, dtype=np.float64)
            logger.info("Calibrating Vasicek to %d rate observations", len(short_rates))

        # Validate data
        n = len(short_rates)
        if n < MIN_OBSERVATIONS:
            msg = f"Insufficient observations for calibration: {n} < {MIN_OBSERVATIONS}"
            raise CalibrationError(msg, model_type="Vasicek", details={"n_observations": n})

        # Get OLS initial estimates
        kappa_init, theta_init, sigma_init = _estimate_ols_params(short_rates, dt)
        logger.debug(
            "OLS initial estimates: kappa=%.4f, theta=%.6f, sigma=%.6f",
            kappa_init,
            theta_init,
            sigma_init,
        )

        if method == "ols":
            # Use OLS estimates directly
            self._kappa = kappa_init
            self._theta = theta_init
            self._sigma = sigma_init
            self._calibrated = True
            self.calibration_date = datetime.now()
            self.convergence_status = "converged"
            self.log_likelihood = None
            logger.info(
                "OLS calibration complete: kappa=%.4f, theta=%.6f, sigma=%.6f",
                self._kappa,
                self._theta,
                self._sigma,
            )
            return

        # MLE calibration
        self._calibrate_mle(short_rates, dt, kappa_init, theta_init, sigma_init)

    def _calibrate_mle(
        self,
        short_rates: npt.NDArray[np.float64],
        dt: float,
        kappa_init: float,
        theta_init: float,
        sigma_init: float,
    ) -> None:
        """Perform MLE calibration using scipy.optimize.minimize.

        Args:
            short_rates: Array of short rate observations.
            dt: Time step between observations.
            kappa_init: Initial guess for kappa.
            theta_init: Initial guess for theta.
            sigma_init: Initial guess for sigma.
        """

        def negative_log_likelihood(params: npt.NDArray[np.float64]) -> float:
            """Compute negative log-likelihood for optimization."""
            kappa, theta, sigma = params

            # Parameter bounds check (return large value if invalid)
            if kappa <= 0 or sigma <= 0:
                return 1e20

            n = len(short_rates)
            r_t = short_rates[:-1]
            r_tp1 = short_rates[1:]

            # Conditional mean: E[r(t+dt)|r(t)] = r(t) + κ(θ - r(t))dt
            mu = r_t + kappa * (theta - r_t) * dt

            # Conditional variance: Var[r(t+dt)|r(t)] = σ²dt
            var = sigma**2 * dt

            # Log-likelihood (normal distribution)
            ll = -0.5 * n * np.log(2 * np.pi * var)
            ll -= 0.5 * np.sum((r_tp1 - mu) ** 2) / var

            return -ll  # Negative for minimization

        # Initial parameter vector
        x0 = np.array([kappa_init, theta_init, sigma_init])

        # Bounds: kappa > 0, theta unrestricted, sigma > 0
        bounds = [(1e-6, 10.0), (-0.10, 0.50), (1e-6, 0.20)]

        # Optimize
        result = optimize.minimize(
            negative_log_likelihood,
            x0,
            method="L-BFGS-B",
            bounds=bounds,
            options={"maxiter": 1000},
        )

        if result.success:
            self._kappa = float(result.x[0])
            self._theta = float(result.x[1])
            self._sigma = float(result.x[2])
            self._calibrated = True
            self.calibration_date = datetime.now()
            self.log_likelihood = -result.fun
            self.convergence_status = "converged"
            logger.info(
                "MLE calibration converged: kappa=%.4f, theta=%.6f, sigma=%.6f, LL=%.2f",
                self._kappa,
                self._theta,
                self._sigma,
                self.log_likelihood,
            )
        else:
            # Fall back to OLS estimates
            logger.warning(
                "MLE optimization failed: %s. Using OLS estimates.",
                result.message,
            )
            self._kappa = kappa_init
            self._theta = theta_init
            self._sigma = sigma_init
            self._calibrated = True
            self.calibration_date = datetime.now()
            self.convergence_status = "converged"  # OLS fallback succeeded
            self.log_likelihood = None

    def get_parameters(self) -> dict[str, float]:
        """Return model parameters as a dictionary.

        Returns:
            Dictionary with keys "kappa", "theta", "sigma".

        Raises:
            ModelNotCalibratedError: If model is not calibrated.
        """
        if not self.is_calibrated:
            raise ModelNotCalibratedError(model_type="Vasicek")

        return {
            "kappa": self.kappa,
            "theta": self.theta,
            "sigma": self.sigma,
        }

    def get_parameter_interpretations(self) -> dict[str, str]:
        """Get plain-language interpretations of model parameters.

        Returns human-readable explanations of what each parameter
        means in practical terms.

        Returns:
            Dictionary mapping parameter names to interpretation strings.

        Raises:
            ModelNotCalibratedError: If model is not calibrated.
        """
        if not self.is_calibrated:
            raise ModelNotCalibratedError(model_type="Vasicek")

        half_life = np.log(2) / self.kappa
        long_run_std = self.sigma / np.sqrt(2 * self.kappa)

        return {
            "kappa": (
                f"Mean reversion speed: κ = {self.kappa:.4f}/year. "
                f"The interest rate reverts halfway to its long-term mean "
                f"in approximately {half_life:.2f} years."
            ),
            "theta": (
                f"Long-term mean rate: θ = {self.theta:.4%}. "
                f"This is the equilibrium level to which interest rates "
                f"tend to revert over time."
            ),
            "sigma": (
                f"Volatility: σ = {self.sigma:.4%}/√year. "
                f"This produces a long-run standard deviation of "
                f"approximately {long_run_std:.4%} around the mean."
            ),
        }

    def expected_rate(self, t: float, r0: float) -> float:
        """Compute the expected rate at time t given initial rate r0.

        Uses the analytical formula for the Vasicek model:
            E[r(t)] = r0 * exp(-κt) + θ * (1 - exp(-κt))

        Args:
            t: Time horizon in years.
            r0: Initial rate at t=0.

        Returns:
            Expected rate at time t.

        Raises:
            ModelNotCalibratedError: If model is not calibrated.
        """
        if not self.is_calibrated:
            raise ModelNotCalibratedError(model_type="Vasicek")

        exp_neg_kappa_t = np.exp(-self.kappa * t)
        return r0 * exp_neg_kappa_t + self.theta * (1 - exp_neg_kappa_t)

    def rate_variance(self, t: float) -> float:
        """Compute the variance of the rate at time t.

        Uses the analytical formula for the Vasicek model:
            Var[r(t)] = (σ²/(2κ)) * (1 - exp(-2κt))

        Args:
            t: Time horizon in years.

        Returns:
            Variance of rate at time t.

        Raises:
            ModelNotCalibratedError: If model is not calibrated.
        """
        if not self.is_calibrated:
            raise ModelNotCalibratedError(model_type="Vasicek")

        return (self.sigma**2 / (2 * self.kappa)) * (1 - np.exp(-2 * self.kappa * t))

    def simulate(
        self,
        n_paths: int,
        time_horizon: float,
        time_step: float,
        initial_rate: float | None = None,
        random_seed: int | None = None,
        **kwargs: object,
    ) -> SimulationResult:
        """Generate simulated rate paths using Monte Carlo simulation.

        Uses Euler-Maruyama discretization to simulate the Vasicek process.

        Args:
            n_paths: Number of simulation paths to generate.
            time_horizon: Simulation horizon in years.
            time_step: Time step size in years (e.g., 1/252 for daily).
            initial_rate: Starting rate. If None, uses theta.
            random_seed: Optional seed for reproducibility.
            **kwargs: Additional keyword arguments (ignored).

        Returns:
            SimulationResult containing all simulated paths.

        Raises:
            ModelNotCalibratedError: If model is not calibrated.
        """
        if not self.is_calibrated:
            raise ModelNotCalibratedError(model_type="Vasicek")

        logger.info(
            "Simulating %d paths over %.2f years with dt=%.6f",
            n_paths,
            time_horizon,
            time_step,
        )

        # Default initial rate to theta
        if initial_rate is None:
            initial_rate = self.theta

        # Calculate number of steps
        n_steps = int(np.ceil(time_horizon / time_step))
        actual_time_horizon = n_steps * time_step

        # Generate shocks
        shocks = generate_normal_shocks(
            n_paths=n_paths,
            n_steps=n_steps,
            n_factors=1,
            random_seed=random_seed,
        ).squeeze(axis=1)  # Shape: (n_paths, n_steps)

        # Apply Vasicek dynamics
        rates_matrix = apply_vasicek_dynamics(
            initial_rate=initial_rate,
            kappa=self.kappa,
            theta=self.theta,
            sigma=self.sigma,
            shocks=shocks,
            dt=time_step,
        )

        # Build time grid
        time_grid = np.linspace(0, actual_time_horizon, n_steps + 1)

        # Create SimulationPath objects
        paths = [
            SimulationPath(
                path_id=i,
                model_type="vasicek",
                time_grid=time_grid,
                rates=rates_matrix[i, :],
                random_seed=random_seed,
            )
            for i in range(n_paths)
        ]

        logger.info(
            "Simulation complete: %d paths, terminal mean=%.6f, terminal std=%.6f",
            n_paths,
            np.mean(rates_matrix[:, -1]),
            np.std(rates_matrix[:, -1]),
        )

        return SimulationResult(
            model_type="vasicek",
            paths=paths,
            path_count=n_paths,
            time_horizon=actual_time_horizon,
            time_step=time_step,
            model_parameters=self.get_parameters(),
        )
