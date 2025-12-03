"""
Sentiment-Augmented Goals-Based Wealth Management Environment

This module extends the base GBWM environment to include market sentiment
as an additional state variable, enabling regime-aware decision making.
"""

import gymnasium as gym
from gymnasium import spaces
import numpy as np
from typing import Tuple, Dict, Any, Optional, List
from datetime import datetime, timedelta
import pandas as pd
import logging

from .gbwm_env import GBWMEnvironment
from ..data.sentiment_provider import SentimentProvider
from config.environment_config import EnvironmentConfig, DEFAULT_ENV_CONFIG


class GBWMEnvironmentWithSentiment(GBWMEnvironment):
    """
    GBWM Environment with sentiment integration
    
    Extends base GBWM environment to include market sentiment in state space.
    
    State space: [time, wealth, vix_sentiment, vix_momentum] (continuous, 4D)
    Action space: [goal_decision, portfolio_choice] (multi-discrete, 2D)
    
    The agent now considers market sentiment when making:
    1. Goal-taking decisions (e.g., skip goals during market stress)
    2. Portfolio allocation decisions (e.g., more conservative during high VIX)
    """
    
    def __init__(
        self,
        config: EnvironmentConfig = None,
        data_mode: str = "simulation",
        historical_loader = None,
        sentiment_provider: Optional[SentimentProvider] = None,
        sentiment_start_date: str = "2015-01-01",
        **kwargs
    ):
        """
        Initialize sentiment-augmented GBWM environment
        
        Args:
            config: Environment configuration
            data_mode: 'simulation' or 'historical'
            historical_loader: HistoricalDataLoader instance
            sentiment_provider: SentimentProvider instance for market sentiment
            sentiment_start_date: Start date for sentiment data simulation
            **kwargs: Additional parameters
        """
        # Initialize base environment first
        super().__init__(config=config, data_mode=data_mode, historical_loader=historical_loader)
        
        # Sentiment configuration
        self.sentiment_provider = sentiment_provider
        self.sentiment_start_date = pd.to_datetime(sentiment_start_date)
        self.sentiment_enabled = sentiment_provider is not None
        
        # Sentiment state tracking
        self.current_date: Optional[pd.Timestamp] = None
        self.current_sentiment_features: Optional[np.ndarray] = None
        
        # Update observation space to include sentiment
        if self.sentiment_enabled:
            # 4D state: [time, wealth, vix_sentiment, vix_momentum]
            self.observation_space = spaces.Box(
                low=np.array([0.0, 0.0, -1.0, -1.0]),
                high=np.array([1.0, 1.0, 1.0, 1.0]),
                dtype=np.float32
            )
        else:
            # Keep original 2D state for compatibility
            self.observation_space = spaces.Box(
                low=np.array([0.0, 0.0]),
                high=np.array([1.0, 1.0]),
                dtype=np.float32
            )
        
        # Initialize sentiment provider if provided
        if self.sentiment_enabled:
            try:
                success = self.sentiment_provider.initialize()
                if not success:
                    self.logger.warning("Failed to initialize sentiment provider, disabling sentiment")
                    self.sentiment_enabled = False
            except Exception as e:
                self.logger.error(f"Sentiment provider initialization error: {e}")
                self.sentiment_enabled = False
        
        self.logger.info(f"Sentiment-augmented GBWM Environment initialized (sentiment_enabled={self.sentiment_enabled})")
    
    def _normalize_state_with_sentiment(
        self, 
        time: int, 
        wealth: float, 
        sentiment_features: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """
        Normalize state including sentiment features
        
        Args:
            time: Current time step
            wealth: Current wealth
            sentiment_features: Sentiment features [vix_sentiment, vix_momentum]
            
        Returns:
            Normalized state array
        """
        # Base state normalization
        normalized_time = time / self.config.time_horizon
        normalized_wealth = wealth / self.config.max_wealth
        
        if self.sentiment_enabled and sentiment_features is not None:
            # 4D state with sentiment
            state = np.array([
                normalized_time,
                normalized_wealth,
                float(sentiment_features[0]),  # vix_sentiment [-1, 1]
                float(sentiment_features[1])   # vix_momentum [-1, 1]
            ], dtype=np.float32)
        else:
            # 2D state without sentiment
            state = np.array([
                normalized_time,
                normalized_wealth
            ], dtype=np.float32)
        
        return state
    
    def _get_sentiment_features_for_date(self, date: pd.Timestamp) -> np.ndarray:
        """
        Get sentiment features for a specific date
        
        Args:
            date: Target date
            
        Returns:
            Sentiment features array [vix_sentiment, vix_momentum]
        """
        if not self.sentiment_enabled:
            return np.array([0.0, 0.0], dtype=np.float32)
        
        try:
            return self.sentiment_provider.get_sentiment_features(date)
        except Exception as e:
            self.logger.warning(f"Failed to get sentiment for {date}: {e}")
            return np.array([0.0, 0.0], dtype=np.float32)  # Neutral sentiment
    
    def _advance_date(self) -> pd.Timestamp:
        """
        Advance current date by one time period (annual steps)
        
        Returns:
            New current date
        """
        if self.current_date is None:
            # Initialize date
            if self.data_mode == "historical" and hasattr(self.historical_loader, 'get_start_date'):
                # Use actual start date from historical data if available
                self.current_date = self.historical_loader.get_start_date()
            else:
                # Use configured start date
                self.current_date = self.sentiment_start_date
        else:
            # Advance by one year
            self.current_date += pd.DateOffset(years=1)
        
        return self.current_date
    
    def _evolve_portfolio_with_sentiment(
        self, 
        portfolio_choice: int, 
        wealth: float,
        sentiment_features: np.ndarray
    ) -> float:
        """
        Evolve wealth considering sentiment-based market conditions
        
        Args:
            portfolio_choice: Index of chosen portfolio
            wealth: Current wealth to invest
            sentiment_features: Current sentiment [vix_sentiment, vix_momentum]
            
        Returns:
            New wealth after one time period
        """
        if wealth <= 0:
            return 0.0
        
        # Get base portfolio return using parent method
        base_wealth = self._evolve_portfolio(portfolio_choice, wealth)
        base_return = (base_wealth / wealth) - 1.0 if wealth > 0 else 0.0
        
        # Apply sentiment-based adjustments if enabled
        if self.sentiment_enabled and self.data_mode == "simulation":
            vix_sentiment, vix_momentum = sentiment_features
            
            # Sentiment-based return adjustment
            # High VIX (negative sentiment) -> potential for higher future returns (mean reversion)
            # This captures the empirical relationship that high VIX periods often precede market recoveries
            vix_adjustment = -vix_sentiment * 0.01  # 1% adjustment per unit of sentiment
            
            # Momentum adjustment
            # Negative momentum (VIX increasing) -> slightly lower returns in near term
            momentum_adjustment = -vix_momentum * 0.005  # 0.5% adjustment per unit of momentum
            
            # Total adjustment (bounded to prevent extreme values)
            total_adjustment = np.clip(vix_adjustment + momentum_adjustment, -0.05, 0.05)  # ±5% max
            
            # Apply adjustment
            adjusted_return = base_return + total_adjustment
            new_wealth = wealth * (1.0 + adjusted_return)
            
            self.logger.debug(f"Sentiment adjustment - Portfolio {portfolio_choice}: "
                             f"base_return={base_return:.3f}, adjustment={total_adjustment:.3f}, "
                             f"final_return={adjusted_return:.3f}")
        else:
            # No sentiment adjustment
            new_wealth = base_wealth
        
        return max(0.0, new_wealth)  # Wealth cannot be negative
    
    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """
        Execute one time step with sentiment awareness
        
        Args:
            action: [goal_decision, portfolio_choice]
            
        Returns:
            Tuple of (observation, reward, terminated, truncated, info)
        """
        goal_action, portfolio_action = action
        
        # Store sentiment info for logging
        sentiment_info = {}
        if self.sentiment_enabled and self.current_date is not None:
            try:
                sentiment_info = self.sentiment_provider.get_sentiment_info(self.current_date)
            except Exception as e:
                self.logger.debug(f"Failed to get sentiment info: {e}")
        
        # Execute goal decision (unchanged from base class)
        reward, wealth_after_goal = self._execute_goal_action(goal_action)
        
        # Advance time and date
        old_time = self.current_time
        self.current_time += 1
        
        # Get current sentiment features
        if self.current_time < self.config.time_horizon:
            # Advance date for portfolio evolution
            current_date = self._advance_date()
            current_sentiment = self._get_sentiment_features_for_date(current_date)
            
            # Evolve portfolio with sentiment consideration
            new_wealth = self._evolve_portfolio_with_sentiment(
                portfolio_action, 
                wealth_after_goal,
                current_sentiment
            )
        else:
            # Final time step - no portfolio evolution
            new_wealth = wealth_after_goal
            current_sentiment = self.current_sentiment_features if self.current_sentiment_features is not None else np.array([0.0, 0.0])
        
        # Update state
        self.current_wealth = new_wealth
        self.current_sentiment_features = current_sentiment
        
        # Check if episode is done
        terminated = self.current_time >= self.config.time_horizon
        truncated = False
        
        # Enhanced info dictionary with sentiment data
        info = {
            'time': self.current_time,
            'wealth': self.current_wealth,
            'goal_available': self._is_goal_available(old_time),
            'goal_taken': goal_action == 1 if self._is_goal_available(old_time) else False,
            'goals_taken_so_far': len(self.goals_taken),
            'total_utility': self.total_utility,
            'portfolio_choice': portfolio_action,
            'sentiment_enabled': self.sentiment_enabled
        }
        
        # Add sentiment information to info
        if self.sentiment_enabled:
            info['sentiment_features'] = current_sentiment.tolist()
            info['vix_sentiment'] = float(current_sentiment[0])
            info['vix_momentum'] = float(current_sentiment[1])
            info['current_date'] = str(self.current_date.date()) if self.current_date else None
            
            # Add detailed sentiment info if available
            if sentiment_info:
                info['sentiment_info'] = sentiment_info
        
        # Get observation with sentiment
        observation = self._normalize_state_with_sentiment(
            self.current_time, 
            self.current_wealth, 
            current_sentiment
        )
        
        return observation, reward, terminated, truncated, info
    
    def reset(
        self, 
        seed: Optional[int] = None, 
        options: Optional[Dict[str, Any]] = None
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Reset environment with sentiment initialization
        
        Args:
            seed: Random seed
            options: Additional options (can include 'start_date')
            
        Returns:
            Tuple of (initial_observation, info)
        """
        # Call parent reset
        base_observation, base_info = super().reset(seed=seed, options=options)
        
        # Initialize sentiment state
        if options and 'start_date' in options:
            self.current_date = pd.to_datetime(options['start_date'])
        else:
            self.current_date = None  # Will be initialized on first step
        
        # Get initial sentiment features
        if self.current_date is not None:
            initial_sentiment = self._get_sentiment_features_for_date(self.current_date)
        else:
            initial_sentiment = np.array([0.0, 0.0], dtype=np.float32)
        
        self.current_sentiment_features = initial_sentiment
        
        # Create sentiment-aware observation
        observation = self._normalize_state_with_sentiment(
            self.current_time,
            self.current_wealth,
            initial_sentiment
        )
        
        # Enhanced info with sentiment
        info = base_info.copy()
        if self.sentiment_enabled:
            info['sentiment_enabled'] = True
            info['sentiment_features'] = initial_sentiment.tolist()
            info['vix_sentiment'] = float(initial_sentiment[0])
            info['vix_momentum'] = float(initial_sentiment[1])
            info['sentiment_start_date'] = str(self.current_date.date()) if self.current_date else None
        else:
            info['sentiment_enabled'] = False
        
        return observation, info
    
    def render(self, mode: str = "human"):
        """Render environment state with sentiment information"""
        base_info = f"Time: {self.current_time}, Wealth: ${self.current_wealth:,.0f}, " \
                   f"Goals taken: {len(self.goals_taken)}, Total utility: {self.total_utility:.1f}"
        
        if self.sentiment_enabled and self.current_sentiment_features is not None:
            sentiment_info = f", VIX Sentiment: {self.current_sentiment_features[0]:.2f}, " \
                           f"VIX Momentum: {self.current_sentiment_features[1]:.2f}"
            full_info = base_info + sentiment_info
        else:
            full_info = base_info + " (no sentiment)"
        
        if mode == "human":
            print(full_info)
    
    def get_trajectory_summary(self) -> Dict[str, Any]:
        """Get enhanced trajectory summary with sentiment statistics"""
        base_summary = super().get_trajectory_summary()
        
        if self.sentiment_enabled:
            # Add sentiment-specific metrics
            base_summary['sentiment_enabled'] = True
            base_summary['final_sentiment_features'] = (
                self.current_sentiment_features.tolist() 
                if self.current_sentiment_features is not None 
                else [0.0, 0.0]
            )
        else:
            base_summary['sentiment_enabled'] = False
        
        return base_summary


def make_sentiment_gbwm_env(
    num_goals: int = 4,
    initial_wealth: float = None,
    data_mode: str = "simulation", 
    historical_loader = None,
    sentiment_provider: Optional[SentimentProvider] = None,
    sentiment_start_date: str = "2015-01-01",
    **kwargs
) -> GBWMEnvironmentWithSentiment:
    """
    Create sentiment-augmented GBWM environment
    
    Args:
        num_goals: Number of goals (1, 2, 4, 8, or 16)
        initial_wealth: Initial wealth (if None, calculated from formula)
        data_mode: 'simulation' or 'historical'
        historical_loader: HistoricalDataLoader instance
        sentiment_provider: SentimentProvider for market sentiment
        sentiment_start_date: Start date for sentiment simulation
        **kwargs: Additional environment parameters
        
    Returns:
        Configured sentiment-augmented GBWM environment
    """
    config = DEFAULT_ENV_CONFIG
    
    # Set goal schedule based on number of goals (same as base environment)
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
    
    # Apply additional configuration
    for key, value in kwargs.items():
        if hasattr(config, key):
            setattr(config, key, value)
    
    return GBWMEnvironmentWithSentiment(
        config=config,
        data_mode=data_mode,
        historical_loader=historical_loader,
        sentiment_provider=sentiment_provider,
        sentiment_start_date=sentiment_start_date
    )


def test_sentiment_environment():
    """Test function for sentiment-augmented GBWM environment"""
    try:
        # Create a simple sentiment provider for testing
        from ..data.sentiment_provider import SentimentProvider
        
        sentiment_provider = SentimentProvider(
            cache_dir='./test_sentiment_env_cache',
            lookback_days=30
        )
        
        # Initialize sentiment provider
        if not sentiment_provider.initialize():
            print("Warning: Could not initialize sentiment provider, testing without sentiment")
            sentiment_provider = None
        
        # Create sentiment-augmented environment
        env = make_sentiment_gbwm_env(
            num_goals=4,
            initial_wealth=500000,
            sentiment_provider=sentiment_provider,
            sentiment_start_date="2020-01-01"
        )
        
        # Test environment
        print(f"✓ Environment created: observation_space={env.observation_space}")
        print(f"✓ Action space: {env.action_space}")
        
        # Test reset
        obs, info = env.reset()
        expected_obs_shape = 4 if sentiment_provider else 2
        assert obs.shape == (expected_obs_shape,), f"Wrong observation shape: {obs.shape}"
        print(f"✓ Reset successful: obs={obs}")
        
        # Test a few steps
        for i in range(3):
            action = env.action_space.sample()
            obs, reward, done, truncated, info = env.step(action)
            
            assert obs.shape == (expected_obs_shape,), f"Wrong observation shape at step {i}: {obs.shape}"
            print(f"✓ Step {i}: reward={reward:.2f}, done={done}")
            
            if 'sentiment_features' in info:
                print(f"    Sentiment: {info['sentiment_features']}")
            
            if done:
                break
        
        # Test trajectory summary
        summary = env.get_trajectory_summary()
        assert 'sentiment_enabled' in summary, "Missing sentiment info in summary"
        print(f"✓ Trajectory summary: {summary}")
        
        # Cleanup
        import shutil
        import os
        if os.path.exists('./test_sentiment_env_cache'):
            shutil.rmtree('./test_sentiment_env_cache')
        
        return True
        
    except Exception as e:
        print(f"✗ Sentiment environment test failed: {e}")
        # Cleanup on failure
        import shutil
        import os
        if os.path.exists('./test_sentiment_env_cache'):
            shutil.rmtree('./test_sentiment_env_cache')
        return False


if __name__ == "__main__":
    test_sentiment_environment()