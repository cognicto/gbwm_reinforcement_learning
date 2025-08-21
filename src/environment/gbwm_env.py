"""
Goals-Based Wealth Management Environment

This module implements the core GBWM environment as described in the paper.
"""

import gymnasium as gym
from gymnasium import spaces
import numpy as np
from typing import Tuple, Dict, Any, Optional, List
import logging

from config.environment_config import EnvironmentConfig, DEFAULT_ENV_CONFIG


class GBWMEnvironment(gym.Env):
    """
    Goals-Based Wealth Management Gymnasium Environment

    State space: [time, wealth] (continuous, 2D)
    Action space: [goal_decision, portfolio_choice] (multi-discrete, 2D)

    The agent must decide:
    1. Whether to take available goals (binary)
    2. Which portfolio to invest in (categorical)
    
    Supports two data modes:
    - 'simulation': Synthetic data using Geometric Brownian Motion (default)
    - 'historical': Real market data from historical sequences
    """

    metadata = {"render_modes": ["human"]}

    def __init__(self, 
                 config: EnvironmentConfig = None,
                 data_mode: str = "simulation",
                 historical_loader=None):
        """
        Initialize GBWM Environment
        
        Args:
            config: Environment configuration
            data_mode: 'simulation' or 'historical'
            historical_loader: HistoricalDataLoader instance (required for historical mode)
        """
        super().__init__()

        self.config = config or DEFAULT_ENV_CONFIG
        
        # Data mode configuration
        self.data_mode = data_mode
        self.historical_loader = historical_loader
        
        # Historical data state
        self.historical_sequence = None  # Current historical return sequence
        self.historical_step = 0         # Current step in historical sequence

        # Set random seed
        np.random.seed(self.config.random_seed)
        
        # Setup logging first
        self.logger = logging.getLogger(__name__)
        
        # Validate configuration
        self._validate_configuration()

        # Initialize environment parameters
        self._setup_environment()

        # Initialize state
        self.current_time = 0
        self.current_wealth = self.config.initial_wealth
        self.goals_taken = []
        self.total_utility = 0.0

        # Episode tracking
        self.episode_count = 0

        self.logger.info(f"GBWM Environment initialized in {data_mode} mode")

    def _validate_configuration(self):
        """Validate environment configuration"""
        if self.data_mode not in ["simulation", "historical"]:
            raise ValueError(f"Invalid data_mode: {self.data_mode}. Must be 'simulation' or 'historical'")
        
        if self.data_mode == "historical":
            if self.historical_loader is None:
                raise ValueError("historical_loader must be provided when data_mode='historical'")
            
            # Check if historical loader has enough data for episode length
            available_sequences = self.historical_loader.get_available_sequences(self.config.time_horizon)
            if available_sequences < 1:
                raise ValueError(f"Not enough historical data for {self.config.time_horizon}-period episodes")
            
            self.logger.info(f"Historical mode: {available_sequences} sequences available")

    def _setup_environment(self):
        """Setup action and observation spaces"""

        # Action space: [goal_decision, portfolio_choice]
        # goal_decision: 0=skip, 1=take
        # portfolio_choice: 0 to n_portfolios-1
        self.action_space = spaces.MultiDiscrete([
            2,  # Goal decision: skip or take
            len(self.config.portfolio_config.mean_returns)  # Portfolio choice
        ])

        # Observation space: [time, wealth] (normalized)
        self.observation_space = spaces.Box(
            low=np.array([0.0, 0.0]),
            high=np.array([1.0, 1.0]),
            dtype=np.float32
        )

        # Goal schedule
        self.goal_schedule = self.config.goal_config.goal_years

        # Portfolio parameters
        self.portfolio_means = self.config.portfolio_config.mean_returns
        self.portfolio_stds = self.config.portfolio_config.return_stds

    def _normalize_state(self, time: int, wealth: float) -> np.ndarray:
        """Normalize state to [0, 1] range"""
        normalized_time = time / self.config.time_horizon
        normalized_wealth = wealth / self.config.max_wealth
        return np.array([normalized_time, normalized_wealth], dtype=np.float32)

    def _denormalize_wealth(self, normalized_wealth: float) -> float:
        """Convert normalized wealth back to actual value"""
        return normalized_wealth * self.config.max_wealth

    def _is_goal_available(self, time: int) -> bool:
        """Check if a goal is available at current time"""
        return time in self.goal_schedule

    def _get_goal_cost(self, time: int) -> float:
        """Get cost of goal at given time"""
        return self.config.goal_config.get_goal_cost(time)

    def _get_goal_utility(self, time: int) -> float:
        """Get utility of goal at given time"""
        return self.config.goal_config.get_goal_utility(time)

    def _can_afford_goal(self, time: int, wealth: float) -> bool:
        """Check if agent can afford goal at given time"""
        if not self._is_goal_available(time):
            return False
        return wealth >= self._get_goal_cost(time)

    def _execute_goal_action(self, goal_action: int) -> Tuple[float, float]:
        """
        Execute goal decision and return (reward, wealth_change)

        Args:
            goal_action: 0=skip, 1=take

        Returns:
            Tuple of (reward, wealth_after_goal)
        """
        reward = 0.0
        wealth_after_goal = self.current_wealth

        if self._is_goal_available(self.current_time):
            goal_cost = self._get_goal_cost(self.current_time)
            goal_utility = self._get_goal_utility(self.current_time)

            if goal_action == 1:  # Take goal
                if self._can_afford_goal(self.current_time, self.current_wealth):
                    # Take the goal
                    wealth_after_goal = self.current_wealth - goal_cost
                    reward = goal_utility
                    self.goals_taken.append(self.current_time)
                    self.total_utility += goal_utility

                    self.logger.debug(f"Goal taken at t={self.current_time}, "
                                      f"cost={goal_cost:.0f}, utility={goal_utility:.1f}")
                else:
                    # Cannot afford goal - no action taken
                    reward = 0.0
            # If goal_action == 0 (skip), no action taken

        return reward, wealth_after_goal

    def _evolve_portfolio(self, portfolio_choice: int, wealth: float) -> float:
        """
        Evolve wealth through portfolio investment
        
        Uses either geometric Brownian motion (simulation) or historical returns (historical)

        Args:
            portfolio_choice: Index of chosen portfolio
            wealth: Current wealth to invest

        Returns:
            New wealth after one time period
        """
        if wealth <= 0:
            return 0.0

        if self.data_mode == "simulation":
            # Original GBM simulation
            mu = self.portfolio_means[portfolio_choice]
            sigma = self.portfolio_stds[portfolio_choice]

            # Geometric Brownian motion: W(t+1) = W(t) * exp((mu - 0.5*sigma^2) + sigma*Z)
            drift = mu - 0.5 * sigma ** 2
            diffusion = sigma * np.random.normal(0, 1)
            portfolio_return = np.exp(drift + diffusion)
            
            self.logger.debug(f"Simulation - Portfolio {portfolio_choice}: "
                              f"wealth {wealth:.0f} -> {wealth * portfolio_return:.0f} "
                              f"(return: {portfolio_return - 1:.1%})")

        elif self.data_mode == "historical":
            # Use historical returns
            if self.historical_sequence is None:
                raise ValueError("No historical sequence loaded for current episode")
            
            if self.historical_step >= len(self.historical_sequence[portfolio_choice]):
                raise ValueError(f"Historical step {self.historical_step} exceeds sequence length")
            
            historical_return = self.historical_sequence[portfolio_choice][self.historical_step]
            
            # Handle NaN values (replace with 0 return)
            if np.isnan(historical_return):
                historical_return = 0.0
                self.logger.debug(f"NaN return replaced with 0 for portfolio {portfolio_choice} at step {self.historical_step}")
            
            portfolio_return = 1.0 + historical_return  # Convert to multiplier
            self.historical_step += 1
            
            self.logger.debug(f"Historical - Portfolio {portfolio_choice}: "
                              f"wealth {wealth:.0f} -> {wealth * portfolio_return:.0f} "
                              f"(return: {historical_return:.1%})")

        new_wealth = wealth * portfolio_return
        return max(0.0, new_wealth)  # Wealth cannot be negative

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """
        Execute one time step in the environment

        Args:
            action: [goal_decision, portfolio_choice]

        Returns:
            Tuple of (observation, reward, terminated, truncated, info)
        """
        goal_action, portfolio_action = action

        # Execute goal decision
        reward, wealth_after_goal = self._execute_goal_action(goal_action)

        # Evolve portfolio (if not at final time step)
        if self.current_time < self.config.time_horizon - 1:
            new_wealth = self._evolve_portfolio(portfolio_action, wealth_after_goal)
        else:
            new_wealth = wealth_after_goal

        # Update state
        self.current_time += 1
        self.current_wealth = new_wealth

        # Check if episode is done
        terminated = self.current_time >= self.config.time_horizon
        truncated = False  # We don't use truncation in this environment

        # Create info dictionary
        info = {
            'time': self.current_time,
            'wealth': self.current_wealth,
            'goal_available': self._is_goal_available(self.current_time - 1),
            'goal_taken': goal_action == 1 if self._is_goal_available(self.current_time - 1) else False,
            'goals_taken_so_far': len(self.goals_taken),
            'total_utility': self.total_utility,
            'portfolio_choice': portfolio_action
        }

        # Get normalized observation
        observation = self._normalize_state(self.current_time, self.current_wealth)

        return observation, reward, terminated, truncated, info

    def reset(self, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None) -> Tuple[
        np.ndarray, Dict[str, Any]]:
        """
        Reset the environment to initial state

        Args:
            seed: Random seed
            options: Additional options

        Returns:
            Tuple of (initial_observation, info)
        """
        super().reset(seed=seed)

        if seed is not None:
            np.random.seed(seed)

        # Load historical sequence if in historical mode
        if self.data_mode == "historical":
            self.historical_sequence = self.historical_loader.get_random_sequence(
                length=self.config.time_horizon
            )
            self.historical_step = 0
            self.logger.debug(f"Loaded new historical sequence for episode {self.episode_count + 1}")

        # Reset state
        self.current_time = 0
        self.current_wealth = self.config.initial_wealth
        self.goals_taken = []
        self.total_utility = 0.0
        self.episode_count += 1

        # Initial observation
        observation = self._normalize_state(self.current_time, self.current_wealth)

        # Info dictionary
        info = {
            'time': self.current_time,
            'wealth': self.current_wealth,
            'episode': self.episode_count,
            'goal_schedule': self.goal_schedule
        }

        return observation, info

    def render(self, mode: str = "human"):
        """Render the environment state"""
        if mode == "human":
            print(f"Time: {self.current_time}, Wealth: ${self.current_wealth:,.0f}, "
                  f"Goals taken: {len(self.goals_taken)}, Total utility: {self.total_utility:.1f}")

    def get_trajectory_summary(self) -> Dict[str, Any]:
        """Get summary of completed trajectory"""
        return {
            'total_utility': self.total_utility,
            'goals_taken': self.goals_taken,
            'final_wealth': self.current_wealth,
            'goal_success_rate': len(self.goals_taken) / len(self.goal_schedule),
            'trajectory_length': self.current_time
        }


# Convenience function for creating environment
def make_gbwm_env(num_goals: int = 4, 
                  initial_wealth: float = None,
                  data_mode: str = "simulation",
                  historical_loader = None,
                  **kwargs) -> GBWMEnvironment:
    """
    Create GBWM environment with specified parameters

    Args:
        num_goals: Number of goals (1, 2, 4, 8, or 16)
        initial_wealth: Initial wealth (if None, calculated from paper formula)
        data_mode: 'simulation' or 'historical'
        historical_loader: HistoricalDataLoader instance (required for historical mode)
        **kwargs: Additional environment parameters

    Returns:
        Configured GBWM environment
    """
    config = DEFAULT_ENV_CONFIG

    # Set goal schedule based on number of goals
    if num_goals == 1:
        config.goal_config.goal_years = [16]
    elif num_goals == 2:
        config.goal_config.goal_years = [8, 16]
    elif num_goals == 4:
        config.goal_config.goal_years = [4, 8, 12, 16]
    elif num_goals == 8:
        config.goal_config.goal_years = [2, 4, 6, 8, 10, 12, 14, 16]
    elif num_goals == 16:
        config.goal_config.goal_years = list(range(1, 17))
    else:
        raise ValueError(f"Unsupported number of goals: {num_goals}")

    # Set initial wealth
    if initial_wealth is not None:
        config.initial_wealth = initial_wealth
    else:
        # Use paper formula: W0 = 12 * (NG)^0.85 * 10000
        config.initial_wealth = 12 * (num_goals ** 0.85) * 10000

    # Apply any additional configuration
    for key, value in kwargs.items():
        if hasattr(config, key):
            setattr(config, key, value)

    return GBWMEnvironment(config, data_mode=data_mode, historical_loader=historical_loader)