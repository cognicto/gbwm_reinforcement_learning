"""Environment configuration for GBWM"""

from dataclasses import dataclass
from typing import List, Tuple
import numpy as np


@dataclass
class GoalConfig:
    """Configuration for financial goals"""

    # Goal timing (which years goals are available)
    goal_years: List[int] = None

    # Cost function parameters: C(t) = base_cost * growth_rate^t
    base_goal_cost: float = 10.0
    goal_cost_growth_rate: float = 1.08  # 8% annual growth

    # Utility function parameters: U(t) = base_utility + t
    base_utility: float = 10.0
    utility_time_bonus: float = 1.0  # Linear increase with time

    def __post_init__(self):
        if self.goal_years is None:
            # Default: 4 goals at years 4, 8, 12, 16
            self.goal_years = [4, 8, 12, 16]

    def get_goal_cost(self, time: int) -> float:
        """Calculate goal cost at given time"""
        return self.base_goal_cost * (self.goal_cost_growth_rate ** time)

    def get_goal_utility(self, time: int) -> float:
        """Calculate goal utility at given time"""
        return self.base_utility + self.utility_time_bonus * time


@dataclass
class PortfolioConfig:
    """Configuration for investment portfolios"""

    # Portfolio returns (from paper - efficient frontier)
    mean_returns: np.ndarray = None
    return_stds: np.ndarray = None
    correlation_matrix: np.ndarray = None

    # Asset parameters (US bonds, US stocks, International stocks)
    asset_means: Tuple[float, float, float] = (0.0493, 0.0770, 0.0886)
    asset_stds: Tuple[float, float, float] = (0.0412, 0.1990, 0.1978)
    asset_correlations: Tuple[Tuple[float, float, float], ...] = (
        (1.0, -0.2077, -0.2685),
        (-0.2077, 1.0, 0.7866),
        (-0.2685, 0.7866, 1.0)
    )

    def __post_init__(self):
        if self.mean_returns is None:
            # Generate 15 portfolios on efficient frontier
            self.mean_returns = np.linspace(0.052632, 0.088636, 15)
            self.return_stds = np.linspace(0.037351, 0.195437, 15)

        if self.correlation_matrix is None:
            self.correlation_matrix = np.array(self.asset_correlations)


@dataclass
class EnvironmentConfig:
    """Complete environment configuration"""

    # Time settings
    time_horizon: int = 16
    dt: float = 1.0  # Time step (1 year)

    # Wealth settings
    initial_wealth: float = 120000.0
    max_wealth: float = 10000000.0  # For normalization

    # Goal and portfolio configurations
    goal_config: GoalConfig = None
    portfolio_config: PortfolioConfig = None

    # Environment parameters
    random_seed: int = 42

    # Historical data configuration
    data_mode: str = "simulation"  # "simulation" or "historical"
    historical_data_path: str = "data/raw/market_data/"
    processed_data_path: str = "data/processed/"
    
    # Historical data parameters
    min_sequence_length: int = 200  # Minimum time periods needed for training
    historical_validation_split: float = 0.2  # Reserve 20% for validation
    historical_start_date: str = "2010-01-01"  # Default start date for historical data
    historical_end_date: str = "2023-12-31"    # Default end date for historical data
    
    # Data augmentation options
    use_data_augmentation: bool = True  # Enable data augmentation techniques
    augmentation_noise_std: float = 0.01  # Standard deviation for noise injection
    augmentation_return_scaling: float = 0.05  # Scaling factor for return perturbation
    
    # Historical data validation
    allow_missing_data: bool = True  # Allow some missing values in historical data
    max_missing_ratio: float = 0.05  # Maximum allowed ratio of missing data
    interpolate_missing: bool = True  # Interpolate missing values
    
    # Logging and debugging
    log_historical_stats: bool = False  # Log detailed historical data statistics
    save_historical_sequences: bool = False  # Save used sequences for debugging

    def __post_init__(self):
        if self.goal_config is None:
            self.goal_config = GoalConfig()
        if self.portfolio_config is None:
            self.portfolio_config = PortfolioConfig()
        
        # Validate configuration
        self._validate_config()
    
    def _validate_config(self):
        """Validate environment configuration parameters"""
        if self.data_mode not in ["simulation", "historical"]:
            raise ValueError(f"Invalid data_mode: {self.data_mode}. Must be 'simulation' or 'historical'")
        
        if self.time_horizon <= 0:
            raise ValueError(f"time_horizon must be positive, got {self.time_horizon}")
        
        if self.min_sequence_length < self.time_horizon:
            raise ValueError(f"min_sequence_length ({self.min_sequence_length}) must be >= time_horizon ({self.time_horizon})")
        
        if not (0.0 <= self.historical_validation_split <= 1.0):
            raise ValueError(f"historical_validation_split must be in [0,1], got {self.historical_validation_split}")
        
        if not (0.0 <= self.max_missing_ratio <= 1.0):
            raise ValueError(f"max_missing_ratio must be in [0,1], got {self.max_missing_ratio}")
    
    def get_historical_config(self) -> dict:
        """Get historical data configuration as dictionary"""
        return {
            'data_path': self.historical_data_path,
            'processed_path': self.processed_data_path,
            'start_date': self.historical_start_date,
            'end_date': self.historical_end_date,
            'min_sequence_length': self.min_sequence_length,
            'validation_split': self.historical_validation_split,
            'use_augmentation': self.use_data_augmentation,
            'noise_std': self.augmentation_noise_std,
            'return_scaling': self.augmentation_return_scaling,
            'allow_missing': self.allow_missing_data,
            'max_missing_ratio': self.max_missing_ratio,
            'interpolate_missing': self.interpolate_missing
        }


# Helper function to create historical environment config
def create_historical_config(num_goals: int = 4,
                            initial_wealth: float = None,
                            historical_data_path: str = "data/raw/market_data/",
                            start_date: str = "2010-01-01",
                            end_date: str = "2023-12-31",
                            **kwargs) -> EnvironmentConfig:
    """
    Create environment configuration for historical data mode
    
    Args:
        num_goals: Number of financial goals
        initial_wealth: Initial wealth (calculated if None)
        historical_data_path: Path to historical market data
        start_date: Start date for historical data
        end_date: End date for historical data
        **kwargs: Additional configuration parameters
        
    Returns:
        Configured EnvironmentConfig for historical mode
    """
    config = EnvironmentConfig(
        data_mode="historical",
        historical_data_path=historical_data_path,
        historical_start_date=start_date,
        historical_end_date=end_date
    )
    
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
    
    # Apply additional parameters
    for key, value in kwargs.items():
        if hasattr(config, key):
            setattr(config, key, value)
    
    return config


def create_simulation_config(num_goals: int = 4,
                           initial_wealth: float = None,
                           **kwargs) -> EnvironmentConfig:
    """
    Create environment configuration for simulation mode (default behavior)
    
    Args:
        num_goals: Number of financial goals
        initial_wealth: Initial wealth (calculated if None)
        **kwargs: Additional configuration parameters
        
    Returns:
        Configured EnvironmentConfig for simulation mode
    """
    config = EnvironmentConfig(data_mode="simulation")
    
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
    
    # Apply additional parameters
    for key, value in kwargs.items():
        if hasattr(config, key):
            setattr(config, key, value)
    
    return config


# Default environment configuration (simulation mode)
DEFAULT_ENV_CONFIG = EnvironmentConfig()