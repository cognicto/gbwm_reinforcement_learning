"""
Sentiment Configuration for GBWM

This module defines configuration parameters for sentiment integration
in the Goals-Based Wealth Management system.
"""

from dataclasses import dataclass
from typing import Optional, Dict, Any


@dataclass
class SentimentConfig:
    """Configuration for sentiment data provider"""
    
    # Data source configuration
    cache_dir: str = './data/sentiment'
    vix_symbol: str = '^VIX'
    lookback_days: int = 365
    
    # Feature weights
    vix_weight: float = 1.0
    news_weight: float = 0.0  # Reserved for future news sentiment integration
    
    # VIX processing parameters
    long_term_vix_mean: float = 20.0
    vix_low_threshold: float = 15.0   # Below this: low fear/complacency
    vix_high_threshold: float = 25.0  # Above this: high fear/stress
    
    # Normalization parameters
    vix_min_bound: float = 10.0  # Historical minimum for normalization
    vix_max_bound: float = 80.0  # Historical maximum for normalization
    vix_momentum_scale: float = 0.5  # Scale for momentum normalization (50% change = full range)
    
    # Cache management
    cache_max_age_hours: int = 24
    cache_max_age_days: int = 7
    auto_cache_cleanup: bool = True
    
    # Data validation
    validate_data_quality: bool = True
    max_missing_data_gap_days: int = 7
    
    # Sentiment adjustment parameters for environment
    sentiment_return_adjustment_scale: float = 0.01  # Max ±1% return adjustment per unit sentiment
    momentum_return_adjustment_scale: float = 0.005  # Max ±0.5% return adjustment per unit momentum
    max_sentiment_return_adjustment: float = 0.05    # Max ±5% total adjustment


@dataclass
class ModelSentimentConfig:
    """Configuration for sentiment-aware model architecture"""
    
    # State representation
    state_dim: int = 4  # [time, wealth, vix_sentiment, vix_momentum]
    sentiment_feature_dim: int = 2  # [vix_sentiment, vix_momentum]
    
    # Feature encoding
    encoder_type: str = "feature"  # Options: 'feature', 'simple', 'adaptive', 'attention'
    time_encoder_dim: int = 16
    wealth_encoder_dim: int = 32
    sentiment_encoder_dim: int = 16
    
    # Network architecture  
    policy_type: str = "standard"  # Options: 'standard', 'hierarchical'
    value_type: str = "standard"   # Options: 'standard', 'dual_head', 'ensemble'
    hidden_dim: int = 64
    
    # Training enhancements
    use_batch_norm: bool = False
    dropout_rate: float = 0.0
    
    # Sentiment-specific training
    sentiment_loss_weight: float = 1.0
    sentiment_regularization: float = 0.0


@dataclass
class ExperimentSentimentConfig:
    """Configuration for sentiment-aware experiments"""
    
    # Experiment setup
    sentiment_enabled: bool = True
    sentiment_start_date: str = "2015-01-01"
    
    # Comparison settings
    baseline_comparison: bool = True
    comparison_metrics: list = None
    
    # Evaluation settings
    eval_sentiment_analysis: bool = True
    eval_regime_analysis: bool = True  # Analyze performance in different VIX regimes
    eval_correlation_analysis: bool = True  # Analyze sentiment-performance correlations
    
    # Logging and tracking
    log_sentiment_metrics: bool = True
    log_portfolio_sentiment_correlation: bool = True
    track_regime_decisions: bool = True
    
    def __post_init__(self):
        if self.comparison_metrics is None:
            self.comparison_metrics = [
                'episode_reward',
                'goal_success_rate', 
                'portfolio_entropy',
                'sentiment_reward_correlation'
            ]


@dataclass
class SentimentIntegrationConfig:
    """Complete configuration for sentiment integration"""
    
    # Component configurations
    sentiment: SentimentConfig = None
    model: ModelSentimentConfig = None
    experiment: ExperimentSentimentConfig = None
    
    def __post_init__(self):
        if self.sentiment is None:
            self.sentiment = SentimentConfig()
        if self.model is None:
            self.model = ModelSentimentConfig()
        if self.experiment is None:
            self.experiment = ExperimentSentimentConfig()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary"""
        return {
            'sentiment': self.sentiment.__dict__,
            'model': self.model.__dict__,
            'experiment': self.experiment.__dict__
        }
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'SentimentIntegrationConfig':
        """Create configuration from dictionary"""
        sentiment = SentimentConfig(**config_dict.get('sentiment', {}))
        model = ModelSentimentConfig(**config_dict.get('model', {}))
        experiment = ExperimentSentimentConfig(**config_dict.get('experiment', {}))
        
        return cls(sentiment=sentiment, model=model, experiment=experiment)


# Default configurations
DEFAULT_SENTIMENT_CONFIG = SentimentConfig()
DEFAULT_MODEL_SENTIMENT_CONFIG = ModelSentimentConfig()
DEFAULT_EXPERIMENT_SENTIMENT_CONFIG = ExperimentSentimentConfig()
DEFAULT_SENTIMENT_INTEGRATION_CONFIG = SentimentIntegrationConfig()

# Preset configurations for different use cases

# Conservative sentiment integration (minimal impact)
CONSERVATIVE_SENTIMENT_CONFIG = SentimentIntegrationConfig(
    sentiment=SentimentConfig(
        sentiment_return_adjustment_scale=0.005,
        momentum_return_adjustment_scale=0.0025,
        max_sentiment_return_adjustment=0.025
    ),
    model=ModelSentimentConfig(
        encoder_type="simple",
        sentiment_encoder_dim=8,
        dropout_rate=0.1
    )
)

# Aggressive sentiment integration (maximum impact)
AGGRESSIVE_SENTIMENT_CONFIG = SentimentIntegrationConfig(
    sentiment=SentimentConfig(
        sentiment_return_adjustment_scale=0.02,
        momentum_return_adjustment_scale=0.01,
        max_sentiment_return_adjustment=0.08
    ),
    model=ModelSentimentConfig(
        encoder_type="attention",
        sentiment_encoder_dim=32,
        policy_type="hierarchical",
        value_type="dual_head"
    )
)

# Research configuration (comprehensive analysis)
RESEARCH_SENTIMENT_CONFIG = SentimentIntegrationConfig(
    sentiment=SentimentConfig(
        validate_data_quality=True,
        lookback_days=730  # 2 years of data
    ),
    model=ModelSentimentConfig(
        encoder_type="feature",
        hidden_dim=128,
        use_batch_norm=True
    ),
    experiment=ExperimentSentimentConfig(
        eval_regime_analysis=True,
        eval_correlation_analysis=True,
        track_regime_decisions=True,
        log_portfolio_sentiment_correlation=True
    )
)


def get_sentiment_config(config_name: str = "default") -> SentimentIntegrationConfig:
    """
    Get predefined sentiment configuration
    
    Args:
        config_name: Name of configuration ('default', 'conservative', 'aggressive', 'research')
        
    Returns:
        SentimentIntegrationConfig instance
    """
    config_map = {
        'default': DEFAULT_SENTIMENT_INTEGRATION_CONFIG,
        'conservative': CONSERVATIVE_SENTIMENT_CONFIG,
        'aggressive': AGGRESSIVE_SENTIMENT_CONFIG,
        'research': RESEARCH_SENTIMENT_CONFIG
    }
    
    if config_name not in config_map:
        available_configs = list(config_map.keys())
        raise ValueError(f"Unknown config '{config_name}'. Available: {available_configs}")
    
    return config_map[config_name]


def create_custom_sentiment_config(
    sentiment_enabled: bool = True,
    encoder_type: str = "feature",
    policy_type: str = "standard",
    return_adjustment_scale: float = 0.01,
    **kwargs
) -> SentimentIntegrationConfig:
    """
    Create custom sentiment configuration
    
    Args:
        sentiment_enabled: Whether to enable sentiment features
        encoder_type: Type of state encoder
        policy_type: Type of policy network
        return_adjustment_scale: Scale for sentiment-based return adjustments
        **kwargs: Additional configuration parameters
        
    Returns:
        Custom SentimentIntegrationConfig
    """
    sentiment_config = SentimentConfig(
        sentiment_return_adjustment_scale=return_adjustment_scale,
        **{k: v for k, v in kwargs.items() if k in SentimentConfig.__annotations__}
    )
    
    model_config = ModelSentimentConfig(
        encoder_type=encoder_type,
        policy_type=policy_type,
        **{k: v for k, v in kwargs.items() if k in ModelSentimentConfig.__annotations__}
    )
    
    experiment_config = ExperimentSentimentConfig(
        sentiment_enabled=sentiment_enabled,
        **{k: v for k, v in kwargs.items() if k in ExperimentSentimentConfig.__annotations__}
    )
    
    return SentimentIntegrationConfig(
        sentiment=sentiment_config,
        model=model_config,
        experiment=experiment_config
    )


# Configuration validation
def validate_sentiment_config(config: SentimentIntegrationConfig) -> list:
    """
    Validate sentiment configuration
    
    Args:
        config: Configuration to validate
        
    Returns:
        List of validation warnings/errors
    """
    warnings = []
    
    # Validate sentiment configuration
    if config.sentiment.vix_weight + config.sentiment.news_weight <= 0:
        warnings.append("Total sentiment weights must be positive")
    
    if config.sentiment.vix_low_threshold >= config.sentiment.vix_high_threshold:
        warnings.append("VIX low threshold must be less than high threshold")
    
    if config.sentiment.sentiment_return_adjustment_scale > 0.05:
        warnings.append("Very high sentiment return adjustment scale (>5%) may cause instability")
    
    # Validate model configuration
    if config.model.state_dim < 4 and config.experiment.sentiment_enabled:
        warnings.append("State dimension should be 4 for sentiment-enabled models")
    
    if config.model.encoder_type not in ['feature', 'simple', 'adaptive', 'attention']:
        warnings.append(f"Unknown encoder type: {config.model.encoder_type}")
    
    # Validate experiment configuration
    if not config.experiment.sentiment_enabled and config.model.state_dim == 4:
        warnings.append("Sentiment disabled but model configured for sentiment state")
    
    return warnings


if __name__ == "__main__":
    # Test configuration creation and validation
    print("Testing sentiment configuration...")
    
    # Test default configuration
    default_config = get_sentiment_config("default")
    warnings = validate_sentiment_config(default_config)
    print(f"Default config warnings: {warnings}")
    
    # Test custom configuration
    custom_config = create_custom_sentiment_config(
        sentiment_enabled=True,
        encoder_type="attention",
        return_adjustment_scale=0.015
    )
    warnings = validate_sentiment_config(custom_config)
    print(f"Custom config warnings: {warnings}")
    
    # Test configuration serialization
    config_dict = default_config.to_dict()
    restored_config = SentimentIntegrationConfig.from_dict(config_dict)
    print("✓ Configuration serialization test passed")
    
    print("Sentiment configuration tests completed!")