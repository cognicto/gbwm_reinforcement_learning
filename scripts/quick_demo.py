"""
Quick demo script to showcase the complete GBWM system

This script demonstrates:
1. Training a small model
2. Evaluating its performance
3. Running inference
"""

import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
from pathlib import Path
import numpy as np

from src.training.trainer import GBWMTrainer
from src.evaluation.evaluator import GBWMEvaluator
from config.training_config import TrainingConfig


def quick_demo():
    """Run complete demo of GBWM system"""

    print("🎯 GBWM Complete System Demo")
    print("=" * 60)

    # Step 1: Quick training
    print("\n1️⃣ TRAINING PHASE")
    print("-" * 30)

    config = TrainingConfig()
    config.n_traj = 5000  # Smaller for demo
    config.batch_size = 500
    config.time_horizon = 16

    trainer = GBWMTrainer(
        config=config,
        experiment_name="quick_demo",
        log_level="INFO"
    )

    print("Training mini-model (this will take ~2 minutes)...")
    trainer.train(num_goals=4, total_timesteps=5000 * 16)

    # Step 2: Evaluation
    print("\n2️⃣ EVALUATION PHASE")
    print("-" * 30)

    model_path = trainer.experiment_dir / "final_model.pth"
    evaluator = GBWMEvaluator(str(model_path))

    # Quick evaluation
    eval_results = evaluator.evaluate_policy(num_goals=4, num_episodes=1000)

    print(f"✅ Evaluation Results:")
    print(f"   Mean Reward: {eval_results['mean_reward']:.2f}")
    print(f"   Goal Success Rate: {eval_results['mean_goal_success_rate']:.1%}")
    print(f"   Final Wealth: ${eval_results['mean_final_wealth']:,.0f}")

    # Benchmark comparison
    print("\n📊 Benchmark Comparison:")
    benchmark_results = evaluator.compare_with_benchmarks(num_goals=4, num_episodes=100)

    for strategy, results in benchmark_results.items():
        efficiency = results['mean_reward'] / benchmark_results['random']['mean_reward'] * 100
        print(f"   {strategy:15}: {results['mean_reward']:6.2f} ({efficiency:5.1f}% vs random)")

    # Step 3: Inference demo
    print("\n3️⃣ INFERENCE DEMO")
    print("-" * 30)

    # Demo scenarios
    scenarios = [
        {"name": "Young & Wealthy", "age": 27, "wealth": 250000},
        {"name": "Mid-career", "age": 33, "wealth": 180000},
        {"name": "Struggling", "age": 35, "wealth": 75000},
        {"name": "High Earner", "age": 38, "wealth": 600000}
    ]

    print("🤖 AI Advisor Recommendations:")
    print()

    for scenario in scenarios:
        age, wealth = scenario['age'], scenario['wealth']

        # Convert to model input
        time_step = age - 25
        normalized_time = time_step / 16
        normalized_wealth = wealth / 1000000

        observation = np.array([normalized_time, normalized_wealth], dtype=np.float32)

        # Get recommendation
        action = evaluator.agent.predict(observation, deterministic=True)
        goal_action, portfolio_action = action

        # Check if goal available
        goal_available = time_step in [4, 8, 12, 16]

        print(f"👤 {scenario['name']} (Age {age}, ${wealth:,})")

        if goal_available:
            goal_cost = 10 * (1.08 ** time_step)
            affordable = wealth >= goal_cost

            goal_decision = "🎯 TAKE GOAL" if goal_action == 1 else "⏳ WAIT"
            print(f"   Goal: {goal_decision} (Cost: ${goal_cost:,.0f}, Affordable: {affordable})")
        else:
            print(f"   Goal: 🚫 None available")

        risk_level = "Conservative" if portfolio_action < 5 else "Moderate" if portfolio_action < 10 else "Aggressive"
        print(f"   Portfolio: {risk_level} (#{portfolio_action + 1})")
        print()

    print("✅ Demo completed successfully!")
    print(f"\n📁 All results saved in: {trainer.experiment_dir}")
    print("\nTo run full training:")
    print("  python experiments/run_training.py --num_goals 4")
    print("\nTo evaluate a trained model:")
    print("  python experiments/run_evaluation.py --model_path path/to/model.pth")
    print("\nTo use interactive advisor:")
    print("  python experiments/run_inference.py --model_path path/to/model.pth")


if __name__ == "__main__":
    quick_demo()