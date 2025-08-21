"""
Model evaluation system for GBWM RL agents
"""

import torch
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
import logging
from pathlib import Path
import json

from src.models.ppo_agent import PPOAgent
from src.environment.gbwm_env import make_gbwm_env
from config.training_config import TrainingConfig


class GBWMEvaluator:
    """
    Comprehensive evaluation system for trained GBWM agents
    """

    def __init__(self,
                 model_path: str,
                 config: TrainingConfig = None,
                 device: str = None):
        """
        Initialize evaluator with trained model

        Args:
            model_path: Path to trained model
            config: Training configuration
            device: Device to run evaluation on
        """
        self.model_path = Path(model_path)
        self.config = config
        self.device = device or torch.device('cpu')

        # Setup logging FIRST
        self.logger = logging.getLogger(__name__)

        # Load trained agent
        self.agent = self._load_agent()

    def _load_agent(self) -> PPOAgent:
        """Load trained agent from checkpoint"""

        # First, load checkpoint to get config
        checkpoint = torch.load(self.model_path, map_location=self.device, weights_only=False)

        # Reconstruct config from saved dict
        if 'config_dict' in checkpoint:
            config_dict = checkpoint['config_dict']

            # Create config object
            from config.training_config import TrainingConfig
            config = TrainingConfig()

            # Update config with saved values
            for key, value in config_dict.items():
                if hasattr(config, key):
                    setattr(config, key, value)

            self.config = config
        elif self.config is None:
            # Fallback to default config
            from config.training_config import DEFAULT_TRAINING_CONFIG
            self.config = DEFAULT_TRAINING_CONFIG

        # Create dummy environment to initialize agent
        env = make_gbwm_env(num_goals=getattr(self.config, 'num_goals', 4))
        agent = PPOAgent(env=env, config=self.config, device=self.device)

        # Load trained weights manually
        agent.policy_net.load_state_dict(checkpoint['policy_state_dict'])
        agent.value_net.load_state_dict(checkpoint['value_state_dict'])

        # Load training state
        agent.total_timesteps = checkpoint.get('total_timesteps', 0)
        agent.iteration = checkpoint.get('iteration', 0)

        self.logger.info(f"Loaded trained agent from {self.model_path}")
        return agent

    def evaluate_policy(self,
                        num_goals: int = 4,
                        num_episodes: int = 10000,
                        deterministic: bool = True) -> Dict[str, Any]:
        """
        Evaluate policy performance over multiple episodes

        Args:
            num_goals: Number of goals for evaluation
            num_episodes: Number of episodes to run
            deterministic: Whether to use deterministic policy

        Returns:
            Evaluation metrics
        """
        self.logger.info(f"Evaluating policy over {num_episodes} episodes")

        # Create evaluation environment
        env = make_gbwm_env(num_goals=num_goals)

        # Collect metrics
        episode_rewards = []
        goal_success_rates = []
        final_wealths = []
        goals_taken_counts = []
        trajectory_summaries = []

        for episode in range(num_episodes):
            obs, _ = env.reset()
            episode_reward = 0

            # Run episode
            for step in range(16):
                action = self.agent.predict(obs, deterministic=deterministic)
                obs, reward, terminated, truncated, info = env.step(action)
                episode_reward += reward

                if terminated or truncated:
                    break

            # Collect episode metrics
            summary = env.get_trajectory_summary()

            episode_rewards.append(episode_reward)
            goal_success_rates.append(summary['goal_success_rate'])
            final_wealths.append(summary['final_wealth'])
            goals_taken_counts.append(len(summary['goals_taken']))
            trajectory_summaries.append(summary)

        # Compute statistics
        results = {
            'num_episodes': num_episodes,
            'num_goals': num_goals,
            'deterministic': deterministic,

            # Reward metrics
            'mean_reward': np.mean(episode_rewards),
            'std_reward': np.std(episode_rewards),
            'min_reward': np.min(episode_rewards),
            'max_reward': np.max(episode_rewards),
            'median_reward': np.median(episode_rewards),

            # Goal achievement metrics
            'mean_goal_success_rate': np.mean(goal_success_rates),
            'std_goal_success_rate': np.std(goal_success_rates),
            'mean_goals_taken': np.mean(goals_taken_counts),
            'goal_achievement_distribution': np.bincount(goals_taken_counts, minlength=num_goals + 1).tolist(),

            # Wealth metrics
            'mean_final_wealth': np.mean(final_wealths),
            'std_final_wealth': np.std(final_wealths),
            'median_final_wealth': np.median(final_wealths),

            # Additional metrics
            'total_utility_achieved': np.sum(episode_rewards),
            'perfect_score_rate': np.mean([r == sum(10 + t for t in env.goal_schedule) for r in episode_rewards])
        }

        self.logger.info(f"Evaluation completed: Mean reward = {results['mean_reward']:.2f}")
        return results

    def compare_with_benchmarks(self,
                                num_goals: int = 4,
                                num_episodes: int = 1000) -> Dict[str, Dict[str, Any]]:
        """
        Compare trained policy with benchmark strategies

        Args:
            num_goals: Number of goals
            num_episodes: Episodes per strategy

        Returns:
            Comparison results
        """
        from src.evaluation.benchmarks import (
            GreedyStrategy,
            BuyAndHoldStrategy,
            RandomStrategy
        )

        env = make_gbwm_env(num_goals=num_goals)

        strategies = {
            'trained_ppo': self.agent,
            'greedy': GreedyStrategy(),
            'buy_and_hold': BuyAndHoldStrategy(),
            'random': RandomStrategy()
        }

        results = {}

        for name, strategy in strategies.items():
            self.logger.info(f"Evaluating {name} strategy")

            episode_rewards = []
            goal_success_rates = []

            for episode in range(num_episodes):
                obs, _ = env.reset()
                episode_reward = 0

                for step in range(16):
                    if name == 'trained_ppo':
                        action = strategy.predict(obs, deterministic=True)
                    else:
                        action = strategy.get_action(obs, env)

                    obs, reward, terminated, truncated, info = env.step(action)
                    episode_reward += reward

                    if terminated or truncated:
                        break

                summary = env.get_trajectory_summary()
                episode_rewards.append(episode_reward)
                goal_success_rates.append(summary['goal_success_rate'])

            results[name] = {
                'mean_reward': np.mean(episode_rewards),
                'std_reward': np.std(episode_rewards),
                'mean_goal_success_rate': np.mean(goal_success_rates),
                'total_episodes': num_episodes
            }

        return results

    def analyze_single_trajectory(self,
                                  num_goals: int = 4,
                                  seed: int = 42) -> Dict[str, Any]:
        """
        Detailed analysis of a single trajectory

        Args:
            num_goals: Number of goals
            seed: Random seed for reproducibility

        Returns:
            Detailed trajectory analysis
        """
        env = make_gbwm_env(num_goals=num_goals)

        # Reset with seed
        obs, _ = env.reset(seed=seed)

        trajectory = []
        total_reward = 0

        for step in range(16):
            # Get action probabilities for analysis
            state_tensor = torch.FloatTensor(obs).unsqueeze(0).to(self.device)

            with torch.no_grad():
                goal_probs, portfolio_probs = self.agent.policy_net(state_tensor)
                value_estimate = self.agent.value_net(state_tensor)

            # Get actual action
            action = self.agent.predict(obs, deterministic=True)

            # Execute action
            next_obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward

            # Store step analysis
            step_analysis = {
                'step': step,
                'time': step,
                'wealth': obs[1] * env.config.max_wealth,  # Denormalize
                'goal_available': info.get('goal_available', False),
                'goal_probs': goal_probs.cpu().numpy().tolist(),
                'portfolio_probs': portfolio_probs.cpu().numpy().tolist(),
                'value_estimate': value_estimate.item(),
                'action_taken': action.tolist(),
                'reward': reward,
                'goal_taken': info.get('goal_taken', False)
            }
            trajectory.append(step_analysis)

            obs = next_obs

            if terminated or truncated:
                break

        summary = env.get_trajectory_summary()

        return {
            'trajectory': trajectory,
            'summary': summary,
            'total_reward': total_reward,
            'seed': seed
        }


def evaluate_trained_model(model_path: str,
                           num_goals: int = 4,
                           num_episodes: int = 10000) -> Dict[str, Any]:
    """
    Convenience function to evaluate a trained model

    Args:
        model_path: Path to trained model
        num_goals: Number of goals
        num_episodes: Number of evaluation episodes

    Returns:
        Evaluation results
    """
    evaluator = GBWMEvaluator(model_path)
    return evaluator.evaluate_policy(num_goals=num_goals, num_episodes=num_episodes)