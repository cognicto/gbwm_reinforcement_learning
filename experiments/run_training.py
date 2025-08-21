"""
Main training script for GBWM PPO Agent

Usage:
    python experiments/run_training.py --num_goals 4 --timesteps 800000
"""

import argparse
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.training.trainer import train_gbwm_agent
from config.training_config import TrainingConfig


def main():
    parser = argparse.ArgumentParser(description='Train GBWM PPO Agent')

    parser.add_argument('--num_goals', type=int, default=4,
                        choices=[1, 2, 4, 8, 16],
                        help='Number of financial goals')

    parser.add_argument('--timesteps', type=int, default=None,
                        help='Total training timesteps')

    parser.add_argument('--batch_size', type=int, default=4800,
                        help='Batch size (trajectories per update)')

    parser.add_argument('--learning_rate', type=float, default=0.01,
                        help='Initial learning rate')

    parser.add_argument('--clip_epsilon', type=float, default=0.50,
                        help='PPO clip parameter')

    parser.add_argument('--experiment_name', type=str, default=None,
                        help='Name for this experiment')

    parser.add_argument('--device', type=str, default='auto',
                        choices=['auto', 'cpu', 'cuda'],
                        help='Device to use for training')

    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')

    # Historical data arguments
    parser.add_argument('--data_mode', type=str, default='simulation',
                        choices=['simulation', 'historical'],
                        help='Training data mode: simulation (synthetic) or historical (real market data)')

    parser.add_argument('--historical_data_path', type=str, 
                        default='data/raw/market_data/',
                        help='Path to historical market data files')

    parser.add_argument('--historical_start_date', type=str, 
                        default='2010-01-01',
                        help='Start date for historical data (YYYY-MM-DD)')

    parser.add_argument('--historical_end_date', type=str, 
                        default='2023-12-31',
                        help='End date for historical data (YYYY-MM-DD)')

    args = parser.parse_args()

    # Create config
    config = TrainingConfig()

    # Override config with command line arguments
    if args.timesteps is not None:
        config.n_traj = args.timesteps // config.time_horizon

    config.batch_size = args.batch_size
    config.learning_rate = args.learning_rate
    config.clip_epsilon = args.clip_epsilon
    config.random_seed = args.seed

    if args.device != 'auto':
        config.device = args.device

    # Set data mode and historical parameters
    config.data_mode = args.data_mode
    if args.data_mode == 'historical':
        config.historical_data_path = args.historical_data_path
        config.historical_start_date = args.historical_start_date
        config.historical_end_date = args.historical_end_date

    # Set experiment name with data mode suffix
    experiment_name = args.experiment_name
    if experiment_name is None:
        data_suffix = "hist" if args.data_mode == "historical" else "sim"
        experiment_name = f"gbwm_{args.num_goals}goals_bs{args.batch_size}_lr{args.learning_rate}_{data_suffix}"

    print("🚀 Starting GBWM PPO Training")
    print("=" * 50)
    print(f"Number of goals: {args.num_goals}")
    print(f"Data mode: {config.data_mode}")
    print(f"Batch size: {config.batch_size}")
    print(f"Learning rate: {config.learning_rate}")
    print(f"Clip epsilon: {config.clip_epsilon}")
    print(f"Device: {config.device}")
    
    if config.data_mode == 'historical':
        print(f"Historical data path: {config.historical_data_path}")
        print(f"Date range: {config.historical_start_date} to {config.historical_end_date}")
    
    print(f"Experiment: {experiment_name}")
    print("=" * 50)

    # Train agent
    trainer = train_gbwm_agent(
        num_goals=args.num_goals,
        config=config,
        experiment_name=experiment_name,
        data_mode=args.data_mode
    )

    print("\n✅ Training completed successfully!")
    print(f"Results saved in: {trainer.experiment_dir}")


if __name__ == "__main__":
    main()