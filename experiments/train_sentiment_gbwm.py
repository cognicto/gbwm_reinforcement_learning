"""
Training Script for Sentiment-Aware GBWM Models

This script trains PPO agents with market sentiment integration for 
Goals-Based Wealth Management.
"""

import os
import sys
import argparse
import logging
import json
from pathlib import Path
from datetime import datetime
import numpy as np
import torch

# Add src to path for imports
sys.path.append(str(Path(__file__).parent.parent / 'src'))

from src.data.sentiment_provider import SentimentProvider
from src.environment.gbwm_env_sentiment import make_sentiment_gbwm_env
from src.models.sentiment_ppo_agent import SentimentAwarePPOAgent
from config.training_config import TrainingConfig
from config.environment_config import EnvironmentConfig


def setup_logging(log_dir: str, experiment_name: str) -> logging.Logger:
    """Setup logging configuration"""
    log_dir = Path(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    
    log_file = log_dir / f"{experiment_name}_training.log"
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    
    return logging.getLogger(__name__)


def create_experiment_config(args) -> dict:
    """Create experiment configuration from arguments"""
    return {
        'experiment_name': args.experiment_name,
        'num_goals': args.num_goals,
        'timesteps': args.timesteps,
        'batch_size': args.batch_size,
        'learning_rate': args.learning_rate,
        'sentiment_enabled': args.sentiment_enabled,
        'policy_type': args.policy_type,
        'encoder_type': args.encoder_type,
        'data_mode': args.data_mode,
        'sentiment_start_date': args.sentiment_start_date,
        'cache_dir': args.cache_dir,
        'save_interval': args.save_interval,
        'device': args.device,
        'random_seed': args.random_seed
    }


def setup_sentiment_provider(config: dict, logger: logging.Logger) -> SentimentProvider:
    """Setup sentiment provider if sentiment is enabled"""
    if not config['sentiment_enabled']:
        return None
    
    logger.info("Initializing sentiment provider...")
    
    sentiment_provider = SentimentProvider(
        cache_dir=config['cache_dir'],
        vix_weight=1.0,
        news_weight=0.0,
        long_term_vix_mean=20.0,
        lookback_days=365
    )
    
    # Initialize with data
    success = sentiment_provider.initialize()
    if not success:
        logger.warning("Failed to initialize sentiment provider, proceeding without sentiment")
        return None
    
    # Get sentiment statistics
    stats = sentiment_provider.get_statistics()
    logger.info(f"Sentiment provider initialized: {stats.get('vix', {}).get('count', 0)} VIX records")
    
    return sentiment_provider


def setup_environment(config: dict, sentiment_provider: SentimentProvider, logger: logging.Logger):
    """Setup the GBWM environment"""
    logger.info(f"Creating environment: {config['num_goals']} goals, sentiment_enabled={config['sentiment_enabled']}")
    
    env = make_sentiment_gbwm_env(
        num_goals=config['num_goals'],
        data_mode=config['data_mode'],
        sentiment_provider=sentiment_provider,
        sentiment_start_date=config['sentiment_start_date']
    )
    
    logger.info(f"Environment created: state_dim={env.observation_space.shape[0]}, action_space={env.action_space}")
    
    return env


def setup_agent(env, config: dict, logger: logging.Logger) -> SentimentAwarePPOAgent:
    """Setup the PPO agent"""
    logger.info(f"Creating agent: policy_type={config['policy_type']}, encoder_type={config['encoder_type']}")
    
    # Create training configuration
    training_config = TrainingConfig(
        batch_size=config['batch_size'],
        learning_rate=config['learning_rate'],
        n_neurons=64,
        ppo_epochs=4,
        mini_batch_size=256,
        gamma=0.99,
        gae_lambda=0.95,
        clip_epsilon=0.5,
        entropy_coeff=0.01,
        max_grad_norm=0.5,
        time_horizon=16,
        device=config['device']
    )
    
    # Create agent
    agent = SentimentAwarePPOAgent(
        env=env,
        config=training_config,
        policy_type=config['policy_type'],
        encoder_type=config['encoder_type'],
        sentiment_enabled=config['sentiment_enabled']
    )
    
    logger.info(f"Agent created: {agent.agent_config}")
    
    return agent


def train_model(agent: SentimentAwarePPOAgent, config: dict, logger: logging.Logger) -> list:
    """Train the model and return training history"""
    logger.info(f"Starting training: {config['timesteps']} timesteps")
    
    # Calculate number of iterations
    steps_per_batch = config['batch_size'] * 16  # 16 years per episode
    total_iterations = config['timesteps'] // steps_per_batch
    
    logger.info(f"Training plan: {total_iterations} iterations, {steps_per_batch} steps per batch")
    
    training_history = []
    
    # Training loop
    for iteration in range(total_iterations):
        # Single training iteration
        metrics = agent.train_iteration()
        training_history.append(metrics)
        
        # Logging
        if iteration % 10 == 0:
            logger.info(
                f"Iteration {iteration}/{total_iterations}: "
                f"Reward={metrics['mean_episode_reward']:.2f}, "
                f"GoalSuccess={metrics['mean_goal_success_rate']:.2%}, "
                f"PolicyLoss={metrics['policy_loss']:.4f}"
            )
            
            # Sentiment-specific logging
            if config['sentiment_enabled'] and 'mean_vix_sentiment' in metrics:
                logger.info(
                    f"  Sentiment: VIX={metrics['mean_vix_sentiment']:.2f}, "
                    f"Momentum={metrics['mean_vix_momentum']:.2f}"
                )
        
        # Save checkpoint
        if config['save_interval'] > 0 and (iteration + 1) % config['save_interval'] == 0:
            save_path = Path(config['output_dir']) / f"checkpoint_iter_{iteration + 1}.pth"
            agent.save(str(save_path))
            logger.info(f"Checkpoint saved: {save_path}")
    
    logger.info("Training completed!")
    return training_history


def save_results(agent: SentimentAwarePPOAgent, training_history: list, config: dict, logger: logging.Logger):
    """Save training results and final model"""
    output_dir = Path(config['output_dir'])
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save final model
    final_model_path = output_dir / "final_model_safe.pth"
    agent.save(str(final_model_path))
    logger.info(f"Final model saved: {final_model_path}")
    
    # Save training history
    history_path = output_dir / "training_history.json"
    
    # Convert to JSON-serializable format
    def convert_to_serializable(obj):
        if isinstance(obj, (np.float32, np.float64)):
            return float(obj)
        elif isinstance(obj, (np.int32, np.int64)):
            return int(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {k: convert_to_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_to_serializable(v) for v in obj]
        else:
            return obj
    
    with open(history_path, 'w') as f:
        json.dump(convert_to_serializable(training_history), f, indent=2)
    logger.info(f"Training history saved: {history_path}")
    
    # Save experiment configuration
    config_path = output_dir / "experiment_config.json"
    config_copy = config.copy()
    # Convert non-serializable values
    if 'device' in config_copy and hasattr(config_copy['device'], 'type'):
        config_copy['device'] = str(config_copy['device'])
    
    with open(config_path, 'w') as f:
        json.dump(config_copy, f, indent=2)
    logger.info(f"Configuration saved: {config_path}")
    
    # Save sentiment analysis if available
    if config['sentiment_enabled']:
        sentiment_analysis = agent.get_sentiment_analysis()
        sentiment_path = output_dir / "sentiment_analysis.json"
        with open(sentiment_path, 'w') as f:
            json.dump(convert_to_serializable(sentiment_analysis), f, indent=2)
        logger.info(f"Sentiment analysis saved: {sentiment_path}")
    
    # Generate summary report
    if training_history:
        final_metrics = training_history[-1]
        summary = {
            'experiment_name': config['experiment_name'],
            'training_completed': True,
            'total_iterations': len(training_history),
            'final_metrics': {
                'mean_episode_reward': final_metrics.get('mean_episode_reward', 0),
                'mean_goal_success_rate': final_metrics.get('mean_goal_success_rate', 0),
                'policy_loss': final_metrics.get('policy_loss', 0),
                'value_loss': final_metrics.get('value_loss', 0)
            },
            'sentiment_enabled': config['sentiment_enabled'],
            'num_goals': config['num_goals'],
            'training_time': datetime.now().isoformat()
        }
        
        if config['sentiment_enabled'] and 'sentiment_analysis' in locals():
            summary['sentiment_summary'] = sentiment_analysis
        
        summary_path = output_dir / "summary.json"
        with open(summary_path, 'w') as f:
            json.dump(convert_to_serializable(summary), f, indent=2)
        logger.info(f"Summary report saved: {summary_path}")


def main():
    """Main training function"""
    parser = argparse.ArgumentParser(description="Train Sentiment-Aware GBWM Models")
    
    # Experiment configuration
    parser.add_argument('--experiment_name', type=str, 
                       default=f"sentiment_gbwm_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                       help='Name of the experiment')
    parser.add_argument('--num_goals', type=int, default=4,
                       choices=[1, 2, 4, 8, 16],
                       help='Number of goals')
    parser.add_argument('--timesteps', type=int, default=1000000,
                       help='Total training timesteps')
    
    # Training parameters
    parser.add_argument('--batch_size', type=int, default=4800,
                       help='Batch size for training')
    parser.add_argument('--learning_rate', type=float, default=0.01,
                       help='Learning rate')
    
    # Sentiment configuration
    parser.add_argument('--sentiment_enabled', action='store_true', default=True,
                       help='Enable sentiment features')
    parser.add_argument('--no_sentiment', action='store_true',
                       help='Disable sentiment features (for baseline)')
    parser.add_argument('--sentiment_start_date', type=str, default='2015-01-01',
                       help='Start date for sentiment data')
    parser.add_argument('--cache_dir', type=str, default='./data/sentiment',
                       help='Directory for sentiment data cache')
    
    # Model architecture
    parser.add_argument('--policy_type', type=str, default='standard',
                       choices=['standard', 'hierarchical'],
                       help='Type of policy network')
    parser.add_argument('--encoder_type', type=str, default='feature',
                       choices=['feature', 'simple', 'adaptive', 'attention'],
                       help='Type of state encoder')
    
    # Environment configuration
    parser.add_argument('--data_mode', type=str, default='simulation',
                       choices=['simulation', 'historical'],
                       help='Data mode for environment')
    
    # System configuration
    parser.add_argument('--device', type=str, default='auto',
                       help='Device to use (auto, cuda, cpu)')
    parser.add_argument('--random_seed', type=int, default=42,
                       help='Random seed for reproducibility')
    
    # Output configuration
    parser.add_argument('--output_dir', type=str, default='./data/results',
                       help='Output directory for results')
    parser.add_argument('--log_dir', type=str, default='./logs',
                       help='Directory for log files')
    parser.add_argument('--save_interval', type=int, default=50,
                       help='Save checkpoint every N iterations')
    
    args = parser.parse_args()
    
    # Handle sentiment flags
    if args.no_sentiment:
        args.sentiment_enabled = False
    
    # Setup device
    if args.device == 'auto':
        args.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Create experiment configuration
    config = create_experiment_config(args)
    
    # Create output directory
    config['output_dir'] = os.path.join(args.output_dir, args.experiment_name)
    
    # Setup logging
    logger = setup_logging(args.log_dir, args.experiment_name)
    logger.info(f"Starting experiment: {args.experiment_name}")
    logger.info(f"Configuration: {json.dumps(config, indent=2)}")
    
    try:
        # Set random seed
        np.random.seed(args.random_seed)
        torch.manual_seed(args.random_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(args.random_seed)
        
        # Setup sentiment provider
        sentiment_provider = setup_sentiment_provider(config, logger)
        
        # Setup environment
        env = setup_environment(config, sentiment_provider, logger)
        
        # Setup agent
        agent = setup_agent(env, config, logger)
        
        # Train model
        training_history = train_model(agent, config, logger)
        
        # Save results
        save_results(agent, training_history, config, logger)
        
        logger.info(f"Experiment {args.experiment_name} completed successfully!")
        
    except Exception as e:
        logger.error(f"Experiment failed: {str(e)}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()