"""Quick test of training pipeline"""

import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.training.trainer import GBWMTrainer
from config.training_config import TrainingConfig


def test_training():
    """Test training with minimal configuration"""

    print("🧪 Testing GBWM Training Pipeline")
    print("=" * 50)

    # Minimal config for testing
    config = TrainingConfig()
    config.n_traj = 2000  # Much smaller for testing
    config.batch_size = 200  # Smaller batch
    config.time_horizon = 16
    config.num_goals = 4

    # Create trainer
    trainer = GBWMTrainer(
        config=config,
        experiment_name="test_training",
        log_level="DEBUG"
    )

    # Quick training run
    trainer.train(
        num_goals=4,
        total_timesteps=2000 * 16  # 2000 trajectories
    )

    print("✅ Training test completed!")
    print(f"Results in: {trainer.experiment_dir}")


if __name__ == "__main__":
    test_training()