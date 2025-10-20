"""
Dynamic Programming Algorithm for Goals-Based Wealth Management

Implementation of the GBWM DP algorithm as described in:
"Dynamic Portfolio Allocation in Goals-Based Wealth Management" 
by Das, Ostrov, Radhakrishnan, and Srivastav (2019)
"""

import numpy as np
import time
from typing import Tuple, Dict, Optional, List
from scipy import stats
import logging
from dataclasses import dataclass

from config.environment_config import EnvironmentConfig, DEFAULT_ENV_CONFIG


@dataclass
class DPConfig:
    """Configuration for Dynamic Programming algorithm"""
    # Base case parameters from paper
    initial_wealth: float = 100000.0  # W(0) = $100k
    goal_wealth: float = 200000.0     # G = $200k  
    time_horizon: int = 10            # T = 10 years
    
    # Portfolio parameters (efficient frontier bounds)
    mu_min: float = 0.0526           # Minimum expected return
    mu_max: float = 0.0886           # Maximum expected return  
    sigma_min: float = 0.0374        # Minimum volatility
    sigma_max: float = 0.1954        # Maximum volatility
    
    # Efficient frontier parameters calculated from paper's market data
    # Based on Table 1: US Bonds, International Stocks, US Stocks (1998-2017)
    eff_frontier_a: float = None     # Will be calculated from market data
    eff_frontier_b: float = None     # Will be calculated from market data  
    eff_frontier_c: float = None     # Will be calculated from market data
    
    # Algorithm parameters
    num_portfolios: int = 15         # m = number of portfolio choices
    grid_density: float = 1.5        # ρ_grid = wealth grid density (reduced for speed)
    
    # Cash flows (empty for base case)
    cash_flows: Dict[int, float] = None
    
    # Random seed
    random_seed: int = 42
    
    def __post_init__(self):
        """Calculate efficient frontier parameters from paper's market data"""
        if self.eff_frontier_a is None:
            self._calculate_efficient_frontier_params()
    
    def _calculate_efficient_frontier_params(self):
        """
        Calculate efficient frontier coefficients from paper's market data
        Based on Table 1 from Das et al. (2019) - January 1998 to December 2017
        """
        # Mean returns from Table 1
        mu = np.array([0.0493, 0.0770, 0.0886])  # [US Bonds, Intl Stocks, US Stocks]
        
        # Covariance matrix from Table 1  
        Sigma = np.array([
            [ 0.0017, -0.0017, -0.0021],
            [-0.0017,  0.0396,  0.0309], 
            [-0.0021,  0.0309,  0.0392]
        ])
        
        # Vector of ones
        ones = np.ones(3)
        
        # Calculate scalars k, l, p from paper's formulas
        Sigma_inv = np.linalg.inv(Sigma)
        k = mu.T @ Sigma_inv @ ones
        l = mu.T @ Sigma_inv @ mu  
        p = ones.T @ Sigma_inv @ ones
        
        # Calculate g and h vectors
        denominator = l * p - k**2
        g = (l * Sigma_inv @ ones - k * Sigma_inv @ mu) / denominator
        h = (p * Sigma_inv @ mu - k * Sigma_inv @ ones) / denominator
        
        # Calculate efficient frontier coefficients: σ = √(aμ² + bμ + c)
        self.eff_frontier_a = h.T @ Sigma @ h
        self.eff_frontier_b = 2 * g.T @ Sigma @ h  
        self.eff_frontier_c = g.T @ Sigma @ g
        
        # Verify bounds match expected values
        mu_test_min = 0.0526  # Should give σ ≈ 0.0374
        mu_test_max = 0.0886  # Should give σ ≈ 0.1954
        
        sigma_min_calc = np.sqrt(self.eff_frontier_a * mu_test_min**2 + 
                                self.eff_frontier_b * mu_test_min + 
                                self.eff_frontier_c)
        sigma_max_calc = np.sqrt(self.eff_frontier_a * mu_test_max**2 + 
                                self.eff_frontier_b * mu_test_max + 
                                self.eff_frontier_c)
        
        # Update bounds based on calculated values
        self.sigma_min = sigma_min_calc
        self.sigma_max = sigma_max_calc


class GBWMDynamicProgramming:
    """
    Goals-Based Wealth Management Dynamic Programming Algorithm
    
    Solves the optimization problem:
    max P[W(T) ≥ G] over all portfolio allocation strategies
    
    Uses backward recursion with the Bellman equation:
    V(Wi, t) = max[μ] Σ[j] V(Wj, t+1) × P(Wj at t+1 | Wi at t, portfolio μ)
    """
    
    def __init__(self, config: DPConfig = None):
        """Initialize the DP algorithm with configuration"""
        self.config = config or DPConfig()
        self.logger = logging.getLogger(__name__)
        
        # Set random seed
        np.random.seed(self.config.random_seed)
        
        # Initialize results storage first
        self.value_function = None  # V(Wi, t)
        self.policy = None          # μ*(Wi, t)
        self.wealth_grid = None     # W0, W1, ..., Wimax
        self.solve_time = None
        
        # Initialize components in correct order
        self._setup_efficient_frontier()
        self._setup_wealth_grid()
        
        self.logger.info("GBWM Dynamic Programming algorithm initialized")
    
    def _setup_efficient_frontier(self):
        """Setup the efficient frontier portfolios"""
        # Create m equally spaced μ values
        self.mu_array = np.linspace(self.config.mu_min, self.config.mu_max, self.config.num_portfolios)
        
        # Calculate corresponding σ values using efficient frontier equation
        # σ = √(aμ² + bμ + c)
        mu_squared = self.mu_array ** 2
        variance = (self.config.eff_frontier_a * mu_squared + 
                   self.config.eff_frontier_b * self.mu_array + 
                   self.config.eff_frontier_c)
        self.sigma_array = np.sqrt(np.maximum(variance, 0.0001))  # Ensure positive variance
        
        self.logger.info(f"Efficient frontier: {self.config.num_portfolios} portfolios")
        self.logger.info(f"μ range: [{self.config.mu_min:.4f}, {self.config.mu_max:.4f}]")
        self.logger.info(f"σ range: [{self.sigma_array.min():.4f}, {self.sigma_array.max():.4f}]")
    
    def _compute_wealth_bounds(self) -> Tuple[float, float]:
        """Compute extreme wealth bounds using worst/best case scenarios"""
        W0 = self.config.initial_wealth
        T = self.config.time_horizon
        
        # Worst case: minimum return + bad luck (-3σ)
        mu_worst = self.config.mu_min
        sigma_worst = self.config.sigma_max
        
        # Best case: maximum return + good luck (+3σ)  
        mu_best = self.config.mu_max
        sigma_best = self.config.sigma_max
        
        # Account for cash flows if present
        cash_flow_impact = 0.0
        if self.config.cash_flows:
            # Simple approximation - sum all cash flows
            cash_flow_impact = sum(self.config.cash_flows.values())
        
        # Extreme scenarios using geometric Brownian motion
        W_min = W0 * np.exp((mu_worst - 0.5 * sigma_worst**2) * T - 3 * sigma_worst * np.sqrt(T))
        W_max = W0 * np.exp((mu_best - 0.5 * sigma_best**2) * T + 3 * sigma_best * np.sqrt(T))
        
        # Add cash flow impact
        W_min = max(1.0, W_min + cash_flow_impact)  # Minimum $1 to avoid log issues
        W_max = W_max + abs(cash_flow_impact)
        
        return W_min, W_max
    
    def _setup_wealth_grid(self):
        """Create logarithmically-spaced wealth grid"""
        W_min, W_max = self._compute_wealth_bounds()
        
        # Logarithmic spacing
        ln_W_min = np.log(W_min)
        ln_W_max = np.log(W_max)
        
        # Grid size based on density parameter
        # Number of grid points per σ_min unit
        grid_span = ln_W_max - ln_W_min
        self.i_max = int(np.ceil(grid_span * self.config.grid_density / self.config.sigma_min))
        
        # Create logarithmically spaced grid
        ln_W_grid = np.linspace(ln_W_min, ln_W_max, self.i_max + 1)
        self.wealth_grid = np.exp(ln_W_grid)
        
        # Ensure initial wealth is in grid (find closest and replace)
        closest_idx = np.argmin(np.abs(self.wealth_grid - self.config.initial_wealth))
        self.wealth_grid[closest_idx] = self.config.initial_wealth
        self.initial_wealth_idx = closest_idx
        
        self.logger.info(f"Wealth grid: {len(self.wealth_grid)} points")
        self.logger.info(f"Range: [${self.wealth_grid.min():,.0f}, ${self.wealth_grid.max():,.0f}]")
        self.logger.info(f"Initial wealth at index: {self.initial_wealth_idx}")
    
    def _get_cash_flow(self, t: int) -> float:
        """Get cash flow at time t"""
        if self.config.cash_flows and t in self.config.cash_flows:
            return self.config.cash_flows[t]
        return 0.0
    
    def _compute_transition_probabilities(self, Wi: float, mu: float, sigma: float) -> np.ndarray:
        """
        Compute transition probabilities P(Wj at t+1 | Wi at t, portfolio μ)
        
        Uses geometric Brownian motion:
        W(t+1) = W(t) * exp((μ - σ²/2) + σ*Z) where Z ~ N(0,1)
        
        Returns normalized probability vector for all wealth grid points
        """
        # Add cash flow to current wealth  
        Wi_after_cashflow = Wi + self._get_cash_flow(0)  # Simplified - no time-dependent flows in base case
        
        if Wi_after_cashflow <= 0:
            # If bankrupt, stay at zero wealth
            probs = np.zeros(len(self.wealth_grid))
            zero_idx = np.argmin(np.abs(self.wealth_grid))
            probs[zero_idx] = 1.0
            return probs
        
        # Use discrete approximation for more stable computation
        probs = np.zeros(len(self.wealth_grid))
        
        # For each grid point, compute the probability of transitioning to it
        # using the discretized normal distribution
        for j, Wj in enumerate(self.wealth_grid):
            if Wj > 0:
                # Required log return to reach Wj from Wi
                log_return = np.log(Wj / Wi_after_cashflow)
                
                # Calculate probability using normal distribution
                # Log return ~ N(mu - sigma^2/2, sigma^2)
                mean_log_return = mu - 0.5 * sigma**2
                
                # Probability density
                z_score = (log_return - mean_log_return) / sigma
                
                # Use normal CDF for interval probability (more stable)
                if j == 0:
                    # Leftmost interval: (-inf, midpoint]
                    if j + 1 < len(self.wealth_grid):
                        mid_log = 0.5 * (np.log(self.wealth_grid[j] / Wi_after_cashflow) + 
                                        np.log(self.wealth_grid[j+1] / Wi_after_cashflow))
                        z_mid = (mid_log - mean_log_return) / sigma
                        probs[j] = stats.norm.cdf(z_mid)
                    else:
                        probs[j] = stats.norm.pdf(z_score)
                elif j == len(self.wealth_grid) - 1:
                    # Rightmost interval: [midpoint, +inf)
                    mid_log = 0.5 * (np.log(self.wealth_grid[j-1] / Wi_after_cashflow) + 
                                    np.log(self.wealth_grid[j] / Wi_after_cashflow))
                    z_mid = (mid_log - mean_log_return) / sigma
                    probs[j] = 1.0 - stats.norm.cdf(z_mid)
                else:
                    # Middle intervals: [left_mid, right_mid]
                    left_mid_log = 0.5 * (np.log(self.wealth_grid[j-1] / Wi_after_cashflow) + 
                                         np.log(self.wealth_grid[j] / Wi_after_cashflow))
                    right_mid_log = 0.5 * (np.log(self.wealth_grid[j] / Wi_after_cashflow) + 
                                          np.log(self.wealth_grid[j+1] / Wi_after_cashflow))
                    
                    z_left = (left_mid_log - mean_log_return) / sigma
                    z_right = (right_mid_log - mean_log_return) / sigma
                    
                    probs[j] = stats.norm.cdf(z_right) - stats.norm.cdf(z_left)
        
        # Ensure probabilities are non-negative and finite
        probs = np.maximum(probs, 0.0)
        probs = np.where(np.isfinite(probs), probs, 0.0)
        
        # Normalize probabilities to sum to 1
        total_prob = np.sum(probs)
        if total_prob > 1e-10:  # Avoid division by very small numbers
            probs = probs / total_prob
        else:
            # Fallback: uniform distribution
            probs = np.ones(len(self.wealth_grid)) / len(self.wealth_grid)
        
        return probs
    
    def solve(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Solve the GBWM problem using backward dynamic programming
        
        Returns:
            Tuple of (value_function, optimal_policy)
        """
        start_time = time.time()
        
        # Initialize value function V(Wi, t) and policy μ*(Wi, t)
        T = self.config.time_horizon
        n_wealth = len(self.wealth_grid)
        
        V = np.zeros((n_wealth, T + 1))  # Value function
        policy = np.zeros((n_wealth, T), dtype=int)  # Optimal portfolio indices
        
        # Terminal condition: V(Wi, T) = 1 if Wi ≥ G, 0 otherwise
        V[:, T] = (self.wealth_grid >= self.config.goal_wealth).astype(float)
        
        self.logger.info("Starting backward recursion...")
        
        # Backward recursion
        for t in range(T - 1, -1, -1):
            if t % 5 == 0:  # Progress update every 5 time steps
                self.logger.info(f"Processing time period {t}")
            
            for i in range(n_wealth):
                Wi = self.wealth_grid[i]
                
                # Check for bankruptcy (simplified - skip for base case)
                if Wi <= 0:
                    V[i, t] = 0.0
                    policy[i, t] = 0  # Default to most conservative portfolio
                    continue
                
                # Optimize over portfolio choices
                best_value = 0.0
                best_portfolio = 0
                
                for mu_idx, (mu, sigma) in enumerate(zip(self.mu_array, self.sigma_array)):
                    # Compute transition probabilities
                    transition_probs = self._compute_transition_probabilities(Wi, mu, sigma)
                    
                    # Expected value = sum of (probability × next-period value)
                    expected_value = np.sum(transition_probs * V[:, t + 1])
                    
                    if expected_value > best_value:
                        best_value = expected_value
                        best_portfolio = mu_idx
                
                V[i, t] = max(0.0, min(1.0, best_value))  # Ensure bounds [0,1]
                policy[i, t] = best_portfolio
        
        self.solve_time = time.time() - start_time
        
        # Store results
        self.value_function = V
        self.policy = policy
        
        # Get optimal probability for initial wealth
        optimal_prob = V[self.initial_wealth_idx, 0]
        
        self.logger.info(f"✅ DP solved in {self.solve_time:.2f} seconds")
        self.logger.info(f"✅ Optimal success probability: {optimal_prob:.3f}")
        
        return V, policy
    
    def get_optimal_probability(self) -> float:
        """Get the optimal probability of reaching the goal"""
        if self.value_function is None:
            raise ValueError("Must solve the problem first using solve()")
        
        return self.value_function[self.initial_wealth_idx, 0]
    
    def get_optimal_strategy(self, wealth: float, time: int) -> Tuple[int, float, float]:
        """
        Get optimal portfolio choice for given wealth and time
        
        Returns:
            Tuple of (portfolio_index, expected_return, volatility)
        """
        if self.policy is None:
            raise ValueError("Must solve the problem first using solve()")
        
        if time >= self.config.time_horizon:
            return 0, 0.0, 0.0
        
        # Find closest wealth grid point
        wealth_idx = np.argmin(np.abs(self.wealth_grid - wealth))
        portfolio_idx = self.policy[wealth_idx, time]
        
        mu = self.mu_array[portfolio_idx]
        sigma = self.sigma_array[portfolio_idx]
        
        return portfolio_idx, mu, sigma
    
    def simulate_trajectory(self, num_simulations: int = 1000, seed: int = None) -> Dict:
        """
        Simulate trajectories using the optimal policy
        
        Returns:
            Dictionary with simulation results
        """
        if self.value_function is None:
            raise ValueError("Must solve the problem first using solve()")
        
        if seed is not None:
            np.random.seed(seed)
        
        successes = 0
        final_wealths = []
        
        for _ in range(num_simulations):
            wealth = self.config.initial_wealth
            
            for t in range(self.config.time_horizon):
                # Get optimal portfolio choice
                portfolio_idx, mu, sigma = self.get_optimal_strategy(wealth, t)
                
                # Apply cash flow
                wealth += self._get_cash_flow(t)
                
                # Evolve wealth with geometric Brownian motion
                if wealth > 0:
                    drift = mu - 0.5 * sigma**2
                    diffusion = sigma * np.random.normal(0, 1)
                    wealth = wealth * np.exp(drift + diffusion)
                    wealth = max(0.0, wealth)  # Cannot be negative
            
            final_wealths.append(wealth)
            if wealth >= self.config.goal_wealth:
                successes += 1
        
        success_rate = successes / num_simulations
        
        return {
            'success_rate': success_rate,
            'mean_final_wealth': np.mean(final_wealths),
            'std_final_wealth': np.std(final_wealths),
            'min_final_wealth': np.min(final_wealths),
            'max_final_wealth': np.max(final_wealths),
            'final_wealths': final_wealths,
            'num_simulations': num_simulations
        }
    
    def get_policy_summary(self) -> Dict:
        """Get a summary of the optimal policy"""
        if self.policy is None:
            raise ValueError("Must solve the problem first using solve()")
        
        policy_summary = {}
        T = self.config.time_horizon
        
        for t in range(T):
            policy_summary[f'time_{t}'] = {}
            
            for i in range(0, len(self.wealth_grid), max(1, len(self.wealth_grid) // 10)):
                wealth = self.wealth_grid[i]
                portfolio_idx = self.policy[i, t]
                mu = self.mu_array[portfolio_idx]
                sigma = self.sigma_array[portfolio_idx]
                
                risk_level = "Conservative" if sigma < 0.08 else "Moderate" if sigma < 0.15 else "Aggressive"
                
                policy_summary[f'time_{t}'][f'wealth_{wealth:.0f}'] = {
                    'portfolio_index': int(portfolio_idx),
                    'expected_return': float(mu),
                    'volatility': float(sigma),
                    'risk_level': risk_level
                }
        
        return policy_summary
    
    def get_results_summary(self) -> Dict:
        """Get comprehensive results summary"""
        if self.value_function is None:
            raise ValueError("Must solve the problem first using solve()")
        
        # Run simulation to validate
        sim_results = self.simulate_trajectory(num_simulations=10000, seed=42)
        
        return {
            'algorithm': 'Dynamic Programming',
            'config': {
                'initial_wealth': self.config.initial_wealth,
                'goal_wealth': self.config.goal_wealth,
                'time_horizon': self.config.time_horizon,
                'num_portfolios': self.config.num_portfolios,
                'grid_size': len(self.wealth_grid),
                'grid_density': self.config.grid_density
            },
            'theoretical_results': {
                'optimal_probability': float(self.get_optimal_probability()),
                'solve_time': float(self.solve_time)
            },
            'simulation_validation': sim_results,
            'grid_info': {
                'wealth_range': [float(self.wealth_grid.min()), float(self.wealth_grid.max())],
                'grid_size': len(self.wealth_grid),
                'initial_wealth_index': int(self.initial_wealth_idx)
            }
        }


def solve_gbwm_dp(initial_wealth: float = 100000,
                  goal_wealth: float = 200000,
                  time_horizon: int = 10,
                  num_portfolios: int = 15,
                  grid_density: float = 3.0) -> GBWMDynamicProgramming:
    """
    Convenience function to solve GBWM problem with specified parameters
    
    Args:
        initial_wealth: Starting wealth ($)
        goal_wealth: Target wealth ($)  
        time_horizon: Time to goal (years)
        num_portfolios: Number of portfolio choices
        grid_density: Wealth grid density parameter
        
    Returns:
        Solved GBWMDynamicProgramming instance
    """
    config = DPConfig(
        initial_wealth=initial_wealth,
        goal_wealth=goal_wealth,
        time_horizon=time_horizon,
        num_portfolios=num_portfolios,
        grid_density=grid_density
    )
    
    dp = GBWMDynamicProgramming(config)
    dp.solve()
    
    return dp