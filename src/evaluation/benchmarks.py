"""
Benchmark strategies for comparison with trained RL agent
"""

import numpy as np
from typing import Tuple, Any
import logging


class BenchmarkStrategy:
    """Base class for benchmark strategies"""

    def __init__(self, name: str):
        self.name = name
        self.logger = logging.getLogger(__name__)

    def get_action(self, observation: np.ndarray, env) -> np.ndarray:
        """Get action for given observation"""
        raise NotImplementedError

    def reset(self):
        """Reset strategy state if needed"""
        pass


class GreedyStrategy(BenchmarkStrategy):
    """
    Greedy goal-taking strategy

    Always takes goals if sufficient funds available.
    Uses moderate portfolio allocation.
    """

    def __init__(self):
        super().__init__("Greedy")
        self.default_portfolio = 7  # Moderate portfolio (middle of 0-14 range)

    def get_action(self, observation: np.ndarray, env) -> np.ndarray:
        """
        Greedy action selection

        Args:
            observation: [normalized_time, normalized_wealth]
            env: Environment instance

        Returns:
            Action [goal_decision, portfolio_choice]
        """
        time_norm, wealth_norm = observation

        # Denormalize to get actual values
        current_time = int(time_norm * env.config.time_horizon)
        current_wealth = wealth_norm * env.config.max_wealth

        # Goal decision: take if available and affordable
        goal_action = 0  # Default: don't take goal

        if env._is_goal_available(current_time):
            goal_cost = env._get_goal_cost(current_time)
            if current_wealth >= goal_cost:
                goal_action = 1  # Take goal

        # Portfolio decision: use moderate portfolio
        portfolio_action = self.default_portfolio

        return np.array([goal_action, portfolio_action])


class BuyAndHoldStrategy(BenchmarkStrategy):
    """
    Buy and hold strategy

    Maintains single portfolio throughout.
    Random goal-taking with 50% probability.
    """

    def __init__(self, portfolio_choice: int = 8):
        super().__init__("Buy and Hold")
        self.portfolio_choice = portfolio_choice  # Moderate-aggressive portfolio
        np.random.seed(42)  # For reproducible randomness

    def get_action(self, observation: np.ndarray, env) -> np.ndarray:
        """
        Buy and hold action selection

        Args:
            observation: [normalized_time, normalized_wealth]
            env: Environment instance

        Returns:
            Action [goal_decision, portfolio_choice]
        """
        time_norm, wealth_norm = observation
        current_time = int(time_norm * env.config.time_horizon)
        current_wealth = wealth_norm * env.config.max_wealth

        # Goal decision: random 50/50 if affordable
        goal_action = 0

        if env._is_goal_available(current_time):
            goal_cost = env._get_goal_cost(current_time)
            if current_wealth >= goal_cost:
                goal_action = np.random.choice([0, 1])  # 50/50 chance

        # Portfolio decision: always same portfolio
        portfolio_action = self.portfolio_choice

        return np.array([goal_action, portfolio_action])


class RandomStrategy(BenchmarkStrategy):
    """
    Random strategy

    Random goal decisions and portfolio choices.
    """

    def __init__(self):
        super().__init__("Random")
        np.random.seed(42)

    def get_action(self, observation: np.ndarray, env) -> np.ndarray:
        """
        Random action selection

        Args:
            observation: [normalized_time, normalized_wealth]
            env: Environment instance

        Returns:
            Action [goal_decision, portfolio_choice]
        """
        time_norm, wealth_norm = observation
        current_time = int(time_norm * env.config.time_horizon)
        current_wealth = wealth_norm * env.config.max_wealth

        # Goal decision: random, but only if affordable
        goal_action = 0

        if env._is_goal_available(current_time):
            goal_cost = env._get_goal_cost(current_time)
            if current_wealth >= goal_cost:
                goal_action = np.random.choice([0, 1])

        # Portfolio decision: random
        portfolio_action = np.random.randint(0, 15)

        return np.array([goal_action, portfolio_action])


class ConservativeStrategy(BenchmarkStrategy):
    """
    Conservative strategy

    Only takes early, affordable goals.
    Uses conservative portfolios.
    """

    def __init__(self):
        super().__init__("Conservative")

    def get_action(self, observation: np.ndarray, env) -> np.ndarray:
        """Conservative action selection"""
        time_norm, wealth_norm = observation
        current_time = int(time_norm * env.config.time_horizon)
        current_wealth = wealth_norm * env.config.max_wealth

        # Goal decision: only take early goals if very affordable
        goal_action = 0

        if env._is_goal_available(current_time) and current_time <= 8:  # Only first half
            goal_cost = env._get_goal_cost(current_time)
            if current_wealth >= goal_cost * 2:  # Only if wealth is 2x goal cost
                goal_action = 1

        # Portfolio decision: conservative (first 5 portfolios)
        portfolio_action = min(4, int(time_norm * 5))  # More conservative over time

        return np.array([goal_action, portfolio_action])