"""Mathematical utilities for GBWM RL"""

import numpy as np
import torch
from typing import Tuple, Optional


def geometric_brownian_motion(initial_value: float,
                              mu: float,
                              sigma: float,
                              dt: float = 1.0,
                              random_state: Optional[int] = None) -> float:
    """
    Generate single step of geometric Brownian motion

    Args:
        initial_value: Starting value
        mu: Drift (mean return)
        sigma: Volatility (standard deviation)
        dt: Time step
        random_state: Random seed

    Returns:
        New value after one time step
    """
    if random_state is not None:
        np.random.seed(random_state)

    # Generate random shock
    dW = np.random.normal(0, np.sqrt(dt))

    # GBM formula: S(t+dt) = S(t) * exp((mu - 0.5*sigma^2)*dt + sigma*dW)
    drift_term = (mu - 0.5 * sigma ** 2) * dt
    diffusion_term = sigma * dW

    return initial_value * np.exp(drift_term + diffusion_term)


def efficient_frontier_portfolio(risk_level: int,
                                 num_portfolios: int = 15) -> Tuple[float, float]:
    """
    Get mean return and standard deviation for portfolio on efficient frontier

    Args:
        risk_level: Portfolio risk level (0 = most conservative, 14 = most aggressive)
        num_portfolios: Total number of portfolios on frontier

    Returns:
        Tuple of (mean_return, std_deviation)
    """
    # From paper: equally spaced portfolios on efficient frontier
    min_return, max_return = 0.052632, 0.088636
    min_std, max_std = 0.037351, 0.195437

    # Linear interpolation
    weight = risk_level / (num_portfolios - 1)

    mean_return = min_return + weight * (max_return - min_return)
    std_deviation = min_std + weight * (max_std - min_std)

    return mean_return, std_deviation


def markowitz_efficient_frontier(asset_returns: np.ndarray,
                                 asset_cov: np.ndarray,
                                 num_portfolios: int = 15) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate portfolios on Markowitz efficient frontier

    Args:
        asset_returns: Expected returns for each asset
        asset_cov: Covariance matrix of asset returns
        num_portfolios: Number of portfolios to generate

    Returns:
        Tuple of (portfolio_returns, portfolio_stds)
    """
    n_assets = len(asset_returns)

    # Target returns
    min_return = np.min(asset_returns)
    max_return = np.max(asset_returns)
    target_returns = np.linspace(min_return, max_return, num_portfolios)

    portfolio_returns = []
    portfolio_stds = []

    for target in target_returns:
        # Solve quadratic programming problem for minimum variance portfolio
        # This is a simplified version - in practice, use cvxpy or scipy.optimize

        # Equal weight portfolio as approximation
        weights = np.ones(n_assets) / n_assets

        port_return = np.dot(weights, asset_returns)
        port_std = np.sqrt(np.dot(weights.T, np.dot(asset_cov, weights)))

        portfolio_returns.append(port_return)
        portfolio_stds.append(port_std)

    return np.array(portfolio_returns), np.array(portfolio_stds)


def sharpe_ratio(returns: np.ndarray, risk_free_rate: float = 0.02) -> float:
    """
    Calculate Sharpe ratio

    Args:
        returns: Array of returns
        risk_free_rate: Risk-free rate

    Returns:
        Sharpe ratio
    """
    excess_returns = returns - risk_free_rate
    return np.mean(excess_returns) / np.std(excess_returns) if np.std(excess_returns) > 0 else 0


def max_drawdown(wealth_series: np.ndarray) -> float:
    """
    Calculate maximum drawdown

    Args:
        wealth_series: Time series of wealth values

    Returns:
        Maximum drawdown (positive value)
    """
    peak = np.maximum.accumulate(wealth_series)
    drawdown = (peak - wealth_series) / peak
    return np.max(drawdown)


def normalize_state(time: int, wealth: float, max_time: int = 16, max_wealth: float = 1e7) -> np.ndarray:
    """
    Normalize state variables to [0, 1] range

    Args:
        time: Current time step
        wealth: Current wealth
        max_time: Maximum time horizon
        max_wealth: Maximum wealth for normalization

    Returns:
        Normalized state vector
    """
    norm_time = time / max_time
    norm_wealth = wealth / max_wealth
    return np.array([norm_time, norm_wealth], dtype=np.float32)


def denormalize_wealth(norm_wealth: float, max_wealth: float = 1e7) -> float:
    """Denormalize wealth from [0, 1] to actual value"""
    return norm_wealth * max_wealth