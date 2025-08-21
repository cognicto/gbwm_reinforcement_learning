"""
Main training pipeline for GBWM PPO Agent
"""

import os
import time
import logging
from datetime import datetime
from typing import Dict, List, Optional, Any
import json

import torch
import numpy as np
from tqdm import tqdm

from src.models.ppo_agent import PPOAgent
from src.environment.gbwm_env import make_gbwm_env
from config.training_config import TrainingConfig, DEFAULT_TRAINING_CONFIG
from config.base_config import RESULTS_DIR, LOGS_DIR, MODELS_DIR


class GBWMTrainer:
    """
    Main trainer class for GBWM PPO agent

    Handles complete training pipeline including:
    - Environment setup
    - Agent initialization
    - Training loop with logging
    - Model checkpointing
    - Results saving
    """

    def __init__(self,
                 config: TrainingConfig = None,
                 experiment_name: str = None,
                 log_level: str = "INFO"):
        """
        Initialize trainer

        Args:
            config: Training configuration
            experiment_name: Name for this experiment
            log_level: Logging level
        """
        self.config = config or DEFAULT_TRAINING_CONFIG
        self.experiment_name = experiment_name or f"gbwm_ppo_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        # Setup logging
        self._setup_logging(log_level)

        # Create experiment directories
        self.experiment_dir = RESULTS_DIR / self.experiment_name
        self.experiment_dir.mkdir(exist_ok=True)

        self.checkpoint_dir = self.experiment_dir / "checkpoints"
        self.checkpoint_dir.mkdir(exist_ok=True)

        # Initialize environment and agent
        self.env = None
        self.agent = None

        # Training state
        self.training_history = []
        self.best_reward = -float('inf')

        self.logger.info(f"Trainer initialized for experiment: {self.experiment_name}")

    def _setup_logging(self, log_level: str):
        """Setup logging configuration"""

        # Create logs directory
        log_dir = LOGS_DIR / "training"
        log_dir.mkdir(exist_ok=True)

        # Configure logging
        log_file = log_dir / f"{self.experiment_name}.log"

        logging.basicConfig(
            level=getattr(logging, log_level.upper()),
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ]
        )

        self.logger = logging.getLogger(__name__)

    def setup_environment(self, num_goals: int = 4, data_mode: str = None, **env_kwargs):
        """
        Setup training environment with optional historical data support

        Args:
            num_goals: Number of financial goals
            data_mode: 'simulation' or 'historical' (if None, uses config.data_mode)
            **env_kwargs: Additional environment parameters
        """
        # Determine data mode
        if data_mode is None:
            data_mode = getattr(self.config, 'data_mode', 'simulation')
        
        self.logger.info(f"Setting up environment with {num_goals} goals in {data_mode} mode")

        # Calculate initial wealth using paper formula
        initial_wealth = self.config.get_initial_wealth(num_goals)

        # Setup historical data loader if needed
        historical_loader = None
        if data_mode == "historical":
            self.logger.info("Setting up historical data loader...")
            
            try:
                from src.data.historical_data_loader import create_historical_loader
                
                historical_data_path = getattr(self.config, 'historical_data_path', 'data/raw/market_data/')
                
                historical_loader = create_historical_loader(
                    data_path=historical_data_path,
                    start_date=getattr(self.config, 'historical_start_date', "2010-01-01"),
                    end_date=getattr(self.config, 'historical_end_date', "2023-12-31")
                )
                
                # Validate data availability
                stats = historical_loader.validate_data_quality()
                available_sequences = stats['available_16y_sequences']
                
                self.logger.info(f"Historical data loaded: {stats['total_periods']} periods")
                self.logger.info(f"Available {self.config.time_horizon}-year sequences: {available_sequences}")
                
                if available_sequences < self.config.batch_size:
                    self.logger.warning(
                        f"Limited historical sequences ({available_sequences}) vs batch size ({self.config.batch_size}). "
                        "Training will use repeated sequences."
                    )
                    
            except Exception as e:
                self.logger.error(f"Failed to setup historical data loader: {e}")
                raise RuntimeError(f"Historical mode setup failed: {e}")

        # Create environment
        self.env = make_gbwm_env(
            num_goals=num_goals,
            initial_wealth=initial_wealth,
            data_mode=data_mode,
            historical_loader=historical_loader,
            **env_kwargs
        )

        self.logger.info(f"Environment created with initial wealth: ${initial_wealth:,.0f}")
        self.logger.info(f"Goal schedule: {self.env.goal_schedule}")
        self.logger.info(f"Data mode: {self.env.data_mode}")

        # Update config with environment info
        self.config.num_goals = num_goals
        self.config.initial_wealth = initial_wealth
        self.config.data_mode = data_mode

    def setup_agent(self):
        """Setup PPO agent"""
        if self.env is None:
            raise ValueError("Environment must be setup before agent")

        self.logger.info("Setting up PPO agent")

        self.agent = PPOAgent(
            env=self.env,
            config=self.config,
            device=self.config.device
        )

        self.logger.info(f"Agent created on device: {self.config.device}")
        self.logger.info(f"Policy network parameters: {sum(p.numel() for p in self.agent.policy_net.parameters()):,}")
        self.logger.info(f"Value network parameters: {sum(p.numel() for p in self.agent.value_net.parameters()):,}")

    def train(self,
              num_goals: int = 4,
              total_timesteps: int = None,
              save_interval: int = None,
              eval_interval: int = None,
              data_mode: str = None) -> List[Dict[str, Any]]:
        """
        Run complete training with optional historical data support

        Args:
            num_goals: Number of goals for environment
            total_timesteps: Total timesteps to train
            save_interval: Save checkpoint every N iterations
            eval_interval: Evaluate every N iterations
            data_mode: 'simulation' or 'historical' (if None, uses config.data_mode)

        Returns:
            Training history
        """
        # Setup
        if self.env is None:
            self.setup_environment(num_goals=num_goals, data_mode=data_mode)

        if self.agent is None:
            self.setup_agent()

        # Training parameters
        if total_timesteps is None:
            total_timesteps = self.config.n_traj * self.config.time_horizon

        if save_interval is None:
            save_interval = self.config.save_interval

        if eval_interval is None:
            eval_interval = self.config.eval_interval

        total_iterations = total_timesteps // (self.config.batch_size * self.config.time_horizon)

        self.logger.info("=" * 60)
        self.logger.info("STARTING TRAINING")
        self.logger.info("=" * 60)
        self.logger.info(f"Total iterations: {total_iterations}")
        self.logger.info(f"Total timesteps: {total_timesteps:,}")
        self.logger.info(f"Batch size: {self.config.batch_size}")
        self.logger.info(f"Time horizon: {self.config.time_horizon}")
        self.logger.info("=" * 60)

        # Save initial config
        self._save_config()

        start_time = time.time()

        # Training loop with progress bar
        with tqdm(total=total_iterations, desc="Training Progress") as pbar:
            for iteration in range(total_iterations):
                # Training iteration
                metrics = self.agent.train_iteration()
                self.training_history.append(metrics)

                # Update progress bar
                pbar.set_postfix({
                    'Reward': f"{metrics['mean_episode_reward']:.2f}",
                    'PolicyLoss': f"{metrics['policy_loss']:.4f}",
                    'ValueLoss': f"{metrics['value_loss']:.4f}"
                })
                pbar.update(1)

                # Periodic logging
                if iteration % self.config.log_interval == 0:
                    self._log_training_progress(iteration, metrics)

                # Save checkpoint
                if save_interval > 0 and iteration % save_interval == 0:
                    self._save_checkpoint(iteration, metrics)

                # Evaluation
                if eval_interval > 0 and iteration % eval_interval == 0:
                    eval_metrics = self._evaluate_agent()
                    self._log_evaluation(iteration, eval_metrics)

        training_time = time.time() - start_time

        self.logger.info("=" * 60)
        self.logger.info("TRAINING COMPLETED")
        self.logger.info(f"Total training time: {training_time:.2f} seconds")
        if total_iterations > 0:
            self.logger.info(f"Time per iteration: {training_time / total_iterations:.2f} seconds")
        else:
            self.logger.info("No training iterations performed (timesteps too small)")
        self.logger.info("=" * 60)

        # Save final results
        self._save_final_results(training_time)

        return self.training_history

    def _log_training_progress(self, iteration: int, metrics: Dict[str, Any]):
        """Log training progress"""
        self.logger.info(
            f"Iteration {iteration:4d} | "
            f"Reward: {metrics['mean_episode_reward']:7.2f} | "
            f"PolicyLoss: {metrics['policy_loss']:8.4f} | "
            f"ValueLoss: {metrics['value_loss']:8.4f} | "
            f"LR: {metrics['policy_lr']:.6f} | "
            f"Episodes: {metrics['total_episodes']:6d}"
        )

    def _evaluate_agent(self, num_episodes: int = 100) -> Dict[str, float]:
        """
        Evaluate current agent performance

        Args:
            num_episodes: Number of episodes for evaluation

        Returns:
            Evaluation metrics
        """
        self.logger.debug(f"Evaluating agent over {num_episodes} episodes")

        total_rewards = []
        goal_success_rates = []
        final_wealths = []

        for _ in range(num_episodes):
            obs, _ = self.env.reset()
            episode_reward = 0

            for _ in range(self.config.time_horizon):
                action = self.agent.predict(obs, deterministic=True)
                obs, reward, terminated, truncated, info = self.env.step(action)
                episode_reward += reward

                if terminated or truncated:
                    break

            # Collect metrics
            summary = self.env.get_trajectory_summary()
            total_rewards.append(episode_reward)
            goal_success_rates.append(summary['goal_success_rate'])
            final_wealths.append(summary['final_wealth'])

        return {
            'eval_mean_reward': np.mean(total_rewards),
            'eval_std_reward': np.std(total_rewards),
            'eval_mean_goal_success': np.mean(goal_success_rates),
            'eval_mean_final_wealth': np.mean(final_wealths)
        }

    def _log_evaluation(self, iteration: int, eval_metrics: Dict[str, float]):
        """Log evaluation results"""
        self.logger.info(
            f"EVAL {iteration:4d} | "
            f"Reward: {eval_metrics['eval_mean_reward']:7.2f}±{eval_metrics['eval_std_reward']:.2f} | "
            f"Goals: {eval_metrics['eval_mean_goal_success']:5.1%} | "
            f"Wealth: ${eval_metrics['eval_mean_final_wealth']:,.0f}"
        )

    def _save_checkpoint(self, iteration: int, metrics: Dict[str, Any]):
        """Save training checkpoint"""
        checkpoint_path = self.checkpoint_dir / f"checkpoint_{iteration:04d}.pth"

        self.agent.save(checkpoint_path)

        # Save additional checkpoint info
        checkpoint_info = {
            'iteration': iteration,
            'metrics': metrics,
            'config': self.config.__dict__,
            'training_history': self.training_history
        }

        info_path = self.checkpoint_dir / f"checkpoint_{iteration:04d}_info.json"
        with open(info_path, 'w') as f:
            json.dump(checkpoint_info, f, indent=2, default=str)

        self.logger.debug(f"Checkpoint saved: {checkpoint_path}")

    def _save_config(self):
        """Save training configuration"""
        config_path = self.experiment_dir / "config.json"

        config_dict = {
            'training_config': self.config.__dict__,
            'experiment_name': self.experiment_name,
            'environment': {
                'num_goals': getattr(self.config, 'num_goals', None),
                'initial_wealth': getattr(self.config, 'initial_wealth', None),
                'time_horizon': self.config.time_horizon
            }
        }

        with open(config_path, 'w') as f:
            json.dump(config_dict, f, indent=2, default=str)

        self.logger.info(f"Configuration saved: {config_path}")

    def _save_final_results(self, training_time: float):
        """Save final training results"""
        results_path = self.experiment_dir / "training_results.json"

        final_metrics = self.training_history[-1] if self.training_history else {}

        results = {
            'experiment_name': self.experiment_name,
            'training_time_seconds': training_time,
            'total_iterations': len(self.training_history),
            'final_metrics': final_metrics,
            'best_reward': self.best_reward,
            'config': self.config.__dict__
        }

        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2, default=str)

        # Save training history
        history_path = self.experiment_dir / "training_history.json"
        with open(history_path, 'w') as f:
            json.dump(self.training_history, f, indent=2, default=str)

        # Save final model
        final_model_path = self.experiment_dir / "final_model.pth"
        self.agent.save(final_model_path)

        self.logger.info(f"Final results saved: {results_path}")
        self.logger.info(f"Final model saved: {final_model_path}")


def train_gbwm_agent(num_goals: int = 4,
                     config: TrainingConfig = None,
                     experiment_name: str = None,
                     data_mode: str = None) -> GBWMTrainer:
    """
    Convenience function to train GBWM agent with optional historical data support

    Args:
        num_goals: Number of financial goals
        config: Training configuration
        experiment_name: Experiment name
        data_mode: 'simulation' or 'historical' (if None, uses config.data_mode)

    Returns:
        Trained trainer object
    """
    trainer = GBWMTrainer(config=config, experiment_name=experiment_name)
    trainer.train(num_goals=num_goals, data_mode=data_mode)
    return trainer