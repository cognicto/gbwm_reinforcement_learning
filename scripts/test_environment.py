"""Test script for GBWM environment"""

import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from src.environment.gbwm_env import make_gbwm_env


def test_environment():
    """Test basic environment functionality"""

    print("🧪 Testing GBWM Environment")
    print("=" * 50)

    # Create environment
    env = make_gbwm_env(num_goals=4)

    print(f"Action space: {env.action_space}")
    print(f"Observation space: {env.observation_space}")
    print(f"Goal schedule: {env.goal_schedule}")
    print()

    # Test episode
    obs, info = env.reset()
    print(f"Initial observation: {obs}")
    print(f"Initial info: {info}")
    print()

    total_reward = 0
    for step in range(16):
        # Random action
        action = env.action_space.sample()

        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward

        print(f"Step {step + 1}: Action={action}, Reward={reward:.1f}, "
              f"Wealth=${info['wealth']:,.0f}")

        if terminated:
            break

    print()
    summary = env.get_trajectory_summary()
    print("📊 Episode Summary:")
    for key, value in summary.items():
        print(f"  {key}: {value}")

    print(f"\n✅ Environment test completed successfully!")


if __name__ == "__main__":
    test_environment()