"""
Comparison Script: Sentiment-Aware vs Baseline GBWM Models

This script trains and compares sentiment-aware and baseline models
to demonstrate the impact of market sentiment integration.
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
import matplotlib.pyplot as plt
import pandas as pd

# Add src to path for imports
sys.path.append(str(Path(__file__).parent.parent / 'src'))

from src.data.sentiment_provider import SentimentProvider
from src.environment.gbwm_env_sentiment import make_sentiment_gbwm_env
from src.models.sentiment_ppo_agent import SentimentAwarePPOAgent
from config.training_config import TrainingConfig


def setup_logging(log_dir: str, experiment_name: str) -> logging.Logger:
    """Setup logging configuration"""
    log_dir = Path(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    
    log_file = log_dir / f"{experiment_name}_comparison.log"
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    
    return logging.getLogger(__name__)


def create_training_config(args) -> TrainingConfig:
    """Create training configuration"""
    return TrainingConfig(
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        n_neurons=64,
        ppo_epochs=4,
        mini_batch_size=256,
        gamma=0.99,
        gae_lambda=0.95,
        clip_epsilon=0.5,
        entropy_coeff=0.01,
        max_grad_norm=0.5,
        time_horizon=16,
        device=args.device
    )


def train_model(
    model_name: str,
    sentiment_enabled: bool,
    env,
    config: TrainingConfig,
    args,
    logger: logging.Logger
) -> tuple:
    """Train a single model and return agent and history"""
    logger.info(f"Training {model_name} (sentiment_enabled={sentiment_enabled})")
    
    # Create agent
    agent = SentimentAwarePPOAgent(
        env=env,
        config=config,
        policy_type="standard",
        encoder_type="feature" if sentiment_enabled else "simple",
        sentiment_enabled=sentiment_enabled
    )
    
    # Calculate training iterations
    steps_per_batch = config.batch_size * 16
    total_iterations = max(1, args.timesteps // steps_per_batch)  # Ensure at least 1 iteration
    
    training_history = []
    
    logger.info(f"{model_name}: Starting {total_iterations} iterations")
    
    # Training loop
    for iteration in range(total_iterations):
        metrics = agent.train_iteration()
        training_history.append(metrics)
        
        # Logging every 10 iterations
        if iteration % 10 == 0:
            logger.info(
                f"{model_name} - Iter {iteration}: "
                f"Reward={metrics['mean_episode_reward']:.2f}, "
                f"GoalSuccess={metrics['mean_goal_success_rate']:.2%}"
            )
    
    logger.info(f"{model_name}: Training completed")
    return agent, training_history


def evaluate_model(
    agent: SentimentAwarePPOAgent,
    env,
    model_name: str,
    num_episodes: int,
    logger: logging.Logger
) -> dict:
    """Evaluate trained model"""
    logger.info(f"Evaluating {model_name} over {num_episodes} episodes")
    
    episode_rewards = []
    goal_success_rates = []
    episode_lengths = []
    portfolio_selections = []
    
    # Evaluation loop
    for episode in range(num_episodes):
        obs, _ = env.reset()
        episode_reward = 0
        episode_length = 0
        goals_taken = 0
        goals_available = 0
        
        done = False
        while not done:
            action = agent.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            
            episode_reward += reward
            episode_length += 1
            
            # Track goal statistics
            if info.get('goal_available', False):
                goals_available += 1
                if info.get('goal_taken', False):
                    goals_taken += 1
            
            # Track portfolio selections
            portfolio_selections.append(action[1])
            
            done = terminated or truncated
        
        episode_rewards.append(episode_reward)
        episode_lengths.append(episode_length)
        
        # Calculate goal success rate for this episode
        goal_success_rate = goals_taken / max(goals_available, 1)
        goal_success_rates.append(goal_success_rate)
    
    # Compute statistics
    evaluation_results = {
        'mean_episode_reward': np.mean(episode_rewards),
        'std_episode_reward': np.std(episode_rewards),
        'mean_goal_success_rate': np.mean(goal_success_rates),
        'std_goal_success_rate': np.std(goal_success_rates),
        'mean_episode_length': np.mean(episode_lengths),
        'portfolio_entropy': _calculate_portfolio_entropy(portfolio_selections),
        'most_selected_portfolio': int(np.bincount(portfolio_selections).argmax()),
        'episode_rewards': episode_rewards,
        'goal_success_rates': goal_success_rates
    }
    
    logger.info(
        f"{model_name} evaluation: "
        f"Reward={evaluation_results['mean_episode_reward']:.2f}±{evaluation_results['std_episode_reward']:.2f}, "
        f"GoalSuccess={evaluation_results['mean_goal_success_rate']:.2%}±{evaluation_results['std_goal_success_rate']:.2%}"
    )
    
    return evaluation_results


def _calculate_portfolio_entropy(selections: list) -> float:
    """Calculate entropy of portfolio selections"""
    if not selections:
        return 0.0
    
    counts = np.bincount(selections)
    probs = counts / len(selections)
    probs = probs[probs > 0]  # Remove zero probabilities
    
    return -np.sum(probs * np.log(probs))


def create_comparison_plots(
    baseline_history: list,
    sentiment_history: list,
    baseline_eval: dict,
    sentiment_eval: dict,
    output_dir: Path,
    logger: logging.Logger
):
    """Create comparison plots"""
    logger.info("Creating comparison plots")
    
    # Set up the plotting
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Sentiment vs Baseline GBWM Comparison', fontsize=16, fontweight='bold')
    
    # 1. Training Rewards
    ax1 = axes[0, 0]
    baseline_rewards = [m['mean_episode_reward'] for m in baseline_history]
    sentiment_rewards = [m['mean_episode_reward'] for m in sentiment_history]
    
    ax1.plot(baseline_rewards, label='Baseline', color='blue', alpha=0.8)
    ax1.plot(sentiment_rewards, label='Sentiment-Aware', color='red', alpha=0.8)
    ax1.set_xlabel('Iteration')
    ax1.set_ylabel('Mean Episode Reward')
    ax1.set_title('Training Progress: Episode Rewards')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. Goal Success Rates
    ax2 = axes[0, 1]
    baseline_goals = [m['mean_goal_success_rate'] for m in baseline_history]
    sentiment_goals = [m['mean_goal_success_rate'] for m in sentiment_history]
    
    ax2.plot(baseline_goals, label='Baseline', color='blue', alpha=0.8)
    ax2.plot(sentiment_goals, label='Sentiment-Aware', color='red', alpha=0.8)
    ax2.set_xlabel('Iteration')
    ax2.set_ylabel('Mean Goal Success Rate')
    ax2.set_title('Training Progress: Goal Success Rate')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 3. Evaluation Comparison (Bar Chart)
    ax3 = axes[1, 0]
    metrics = ['mean_episode_reward', 'mean_goal_success_rate', 'portfolio_entropy']
    baseline_values = [baseline_eval[m] for m in metrics]
    sentiment_values = [sentiment_eval[m] for m in metrics]
    
    x = np.arange(len(metrics))
    width = 0.35
    
    ax3.bar(x - width/2, baseline_values, width, label='Baseline', color='blue', alpha=0.7)
    ax3.bar(x + width/2, sentiment_values, width, label='Sentiment-Aware', color='red', alpha=0.7)
    
    ax3.set_xlabel('Metrics')
    ax3.set_ylabel('Values')
    ax3.set_title('Final Evaluation Comparison')
    ax3.set_xticks(x)
    ax3.set_xticklabels(['Episode Reward', 'Goal Success Rate', 'Portfolio Entropy'])
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # 4. Episode Reward Distribution
    ax4 = axes[1, 1]
    ax4.hist(baseline_eval['episode_rewards'], alpha=0.6, bins=20, label='Baseline', color='blue')
    ax4.hist(sentiment_eval['episode_rewards'], alpha=0.6, bins=20, label='Sentiment-Aware', color='red')
    ax4.set_xlabel('Episode Reward')
    ax4.set_ylabel('Frequency')
    ax4.set_title('Episode Reward Distribution')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save plot
    plot_path = output_dir / 'comparison_plots.png'
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info(f"Comparison plots saved: {plot_path}")


def generate_comparison_report(
    baseline_eval: dict,
    sentiment_eval: dict,
    baseline_history: list,
    sentiment_history: list,
    output_dir: Path,
    args,
    logger: logging.Logger
):
    """Generate detailed comparison report"""
    logger.info("Generating comparison report")
    
    # Calculate improvements
    reward_improvement = (
        (sentiment_eval['mean_episode_reward'] - baseline_eval['mean_episode_reward']) /
        abs(baseline_eval['mean_episode_reward']) * 100
    )
    
    goal_improvement = (
        (sentiment_eval['mean_goal_success_rate'] - baseline_eval['mean_goal_success_rate']) /
        baseline_eval['mean_goal_success_rate'] * 100
    )
    
    # Create comprehensive report
    report = {
        'experiment_info': {
            'timestamp': datetime.now().isoformat(),
            'num_goals': args.num_goals,
            'timesteps': args.timesteps,
            'batch_size': args.batch_size,
            'learning_rate': args.learning_rate,
            'eval_episodes': args.eval_episodes
        },
        'training_summary': {
            'baseline': {
                'final_reward': baseline_history[-1]['mean_episode_reward'],
                'final_goal_success': baseline_history[-1]['mean_goal_success_rate'],
                'total_iterations': len(baseline_history)
            },
            'sentiment': {
                'final_reward': sentiment_history[-1]['mean_episode_reward'],
                'final_goal_success': sentiment_history[-1]['mean_goal_success_rate'],
                'total_iterations': len(sentiment_history)
            }
        },
        'evaluation_results': {
            'baseline': baseline_eval,
            'sentiment': sentiment_eval
        },
        'performance_comparison': {
            'reward_improvement_percent': float(reward_improvement),
            'goal_success_improvement_percent': float(goal_improvement),
            'portfolio_entropy_ratio': float(sentiment_eval['portfolio_entropy'] / baseline_eval['portfolio_entropy']) if baseline_eval['portfolio_entropy'] != 0 else float('inf'),
            'reward_significance_p_value': _calculate_statistical_significance(
                baseline_eval['episode_rewards'],
                sentiment_eval['episode_rewards']
            )[0],
            'reward_significance': _calculate_statistical_significance(
                baseline_eval['episode_rewards'],
                sentiment_eval['episode_rewards']
            )[1]
        },
        'key_insights': {
            'sentiment_outperforms_reward': bool(sentiment_eval['mean_episode_reward'] > baseline_eval['mean_episode_reward']),
            'sentiment_outperforms_goals': bool(sentiment_eval['mean_goal_success_rate'] > baseline_eval['mean_goal_success_rate']),
            'sentiment_more_diverse_portfolios': bool(sentiment_eval['portfolio_entropy'] > baseline_eval['portfolio_entropy']),
            'reward_improvement_magnitude': float(abs(reward_improvement)),
            'goal_improvement_magnitude': float(abs(goal_improvement))
        }
    }
    
    # Save report
    report_path = output_dir / 'comparison_report.json'
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=2)
    
    # Create human-readable summary
    summary_text = f"""
SENTIMENT VS BASELINE GBWM COMPARISON REPORT
==========================================

Experiment Configuration:
- Goals: {args.num_goals}
- Training Timesteps: {args.timesteps:,}
- Evaluation Episodes: {args.eval_episodes}

Performance Results:
- Episode Reward: {sentiment_eval['mean_episode_reward']:.2f} vs {baseline_eval['mean_episode_reward']:.2f} 
  (Improvement: {reward_improvement:+.1f}%)
  
- Goal Success Rate: {sentiment_eval['mean_goal_success_rate']:.1%} vs {baseline_eval['mean_goal_success_rate']:.1%}
  (Improvement: {goal_improvement:+.1f}%)
  
- Portfolio Diversity: {sentiment_eval['portfolio_entropy']:.2f} vs {baseline_eval['portfolio_entropy']:.2f}
  (Entropy ratio: {sentiment_eval['portfolio_entropy'] / baseline_eval['portfolio_entropy']:.2f})

Key Findings:
- Sentiment-aware model {'outperforms' if report['key_insights']['sentiment_outperforms_reward'] else 'underperforms'} baseline in episode rewards
- Sentiment-aware model {'outperforms' if report['key_insights']['sentiment_outperforms_goals'] else 'underperforms'} baseline in goal success
- Sentiment-aware model shows {'higher' if report['key_insights']['sentiment_more_diverse_portfolios'] else 'lower'} portfolio diversity

Statistical Significance: {report['performance_comparison']['reward_significance']}
"""
    
    summary_path = output_dir / 'comparison_summary.txt'
    with open(summary_path, 'w') as f:
        f.write(summary_text)
    
    logger.info(f"Comparison report saved: {report_path}")
    logger.info(f"Comparison summary saved: {summary_path}")
    
    # Print summary to console
    print(summary_text)


def _calculate_statistical_significance(baseline_rewards: list, sentiment_rewards: list) -> tuple:
    """Calculate statistical significance using t-test
    
    Returns:
        tuple: (p_value, description) where p_value is float and description is str
    """
    try:
        from scipy import stats
        t_stat, p_value = stats.ttest_ind(sentiment_rewards, baseline_rewards)
        
        # Handle NaN case (identical distributions)
        if np.isnan(p_value):
            return 1.0, "Not significant (identical distributions)"
        
        if p_value < 0.001:
            return p_value, f"Highly significant (p < 0.001)"
        elif p_value < 0.01:
            return p_value, f"Very significant (p = {p_value:.3f})"
        elif p_value < 0.05:
            return p_value, f"Significant (p = {p_value:.3f})"
        else:
            return p_value, f"Not significant (p = {p_value:.3f})"
    except ImportError:
        return 1.0, "Statistical test not available (scipy required)"


def main():
    """Main comparison function"""
    parser = argparse.ArgumentParser(description="Compare Sentiment vs Baseline GBWM Models")
    
    # Experiment configuration
    parser.add_argument('--experiment_name', type=str,
                       default=f"comparison_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                       help='Name of the comparison experiment')
    parser.add_argument('--num_goals', type=int, default=4,
                       choices=[1, 2, 4, 8, 16],
                       help='Number of goals')
    
    # Training parameters
    parser.add_argument('--timesteps', type=int, default=500000,
                       help='Training timesteps per model')
    parser.add_argument('--batch_size', type=int, default=4800,
                       help='Batch size for training')
    parser.add_argument('--learning_rate', type=float, default=0.01,
                       help='Learning rate')
    
    # Evaluation parameters
    parser.add_argument('--eval_episodes', type=int, default=100,
                       help='Episodes for final evaluation')
    
    # System configuration
    parser.add_argument('--device', type=str, default='auto',
                       help='Device to use (auto, cuda, cpu)')
    parser.add_argument('--random_seed', type=int, default=42,
                       help='Random seed for reproducibility')
    
    # Output configuration
    parser.add_argument('--output_dir', type=str, default='./data/comparisons',
                       help='Output directory for comparison results')
    parser.add_argument('--log_dir', type=str, default='./logs',
                       help='Directory for log files')
    
    args = parser.parse_args()
    
    # Setup device
    if args.device == 'auto':
        args.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Create output directory
    output_dir = Path(args.output_dir) / args.experiment_name
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Setup logging
    logger = setup_logging(args.log_dir, args.experiment_name)
    logger.info(f"Starting comparison experiment: {args.experiment_name}")
    
    try:
        # Set random seeds
        np.random.seed(args.random_seed)
        torch.manual_seed(args.random_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(args.random_seed)
        
        # Setup sentiment provider
        logger.info("Setting up sentiment provider...")
        sentiment_provider = SentimentProvider(
            cache_dir='./data/sentiment',
            lookback_days=365
        )
        
        if not sentiment_provider.initialize():
            logger.error("Failed to initialize sentiment provider")
            sys.exit(1)
        
        # Create training configuration
        config = create_training_config(args)
        
        # Create environments
        logger.info("Creating environments...")
        
        # Sentiment-aware environment
        env_sentiment = make_sentiment_gbwm_env(
            num_goals=args.num_goals,
            sentiment_provider=sentiment_provider,
            sentiment_start_date='2015-01-01'
        )
        
        # Baseline environment (no sentiment)
        env_baseline = make_sentiment_gbwm_env(
            num_goals=args.num_goals,
            sentiment_provider=None  # No sentiment
        )
        
        # Train baseline model
        logger.info("=" * 50)
        baseline_agent, baseline_history = train_model(
            "Baseline", False, env_baseline, config, args, logger
        )
        
        # Train sentiment-aware model
        logger.info("=" * 50)
        sentiment_agent, sentiment_history = train_model(
            "Sentiment-Aware", True, env_sentiment, config, args, logger
        )
        
        # Evaluate both models
        logger.info("=" * 50)
        logger.info("Starting model evaluation...")
        
        baseline_eval = evaluate_model(
            baseline_agent, env_baseline, "Baseline", args.eval_episodes, logger
        )
        
        sentiment_eval = evaluate_model(
            sentiment_agent, env_sentiment, "Sentiment-Aware", args.eval_episodes, logger
        )
        
        # Create plots and reports
        logger.info("=" * 50)
        create_comparison_plots(
            baseline_history, sentiment_history,
            baseline_eval, sentiment_eval,
            output_dir, logger
        )
        
        generate_comparison_report(
            baseline_eval, sentiment_eval,
            baseline_history, sentiment_history,
            output_dir, args, logger
        )
        
        # Save models
        logger.info("Saving trained models...")
        baseline_agent.save(str(output_dir / "baseline_model.pth"))
        sentiment_agent.save(str(output_dir / "sentiment_model.pth"))
        
        logger.info(f"Comparison experiment completed successfully!")
        logger.info(f"Results saved in: {output_dir}")
        
    except Exception as e:
        logger.error(f"Comparison experiment failed: {str(e)}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()