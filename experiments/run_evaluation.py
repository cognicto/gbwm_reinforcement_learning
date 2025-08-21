"""
Evaluation script for trained GBWM models

Usage:
    python experiments/run_evaluation.py --model_path data/results/experiment_name/final_model.pth
"""

import argparse
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
from pathlib import Path

from src.evaluation.evaluator import GBWMEvaluator
from config.base_config import RESULTS_DIR


def main():
    parser = argparse.ArgumentParser(description='Evaluate trained GBWM model')

    parser.add_argument('--model_path', type=str, required=True,
                        help='Path to trained model file')

    parser.add_argument('--num_goals', type=int, default=4,
                        choices=[1, 2, 4, 8, 16],
                        help='Number of goals for evaluation')

    parser.add_argument('--num_episodes', type=int, default=10000,
                        help='Number of episodes for evaluation')

    parser.add_argument('--compare_benchmarks', action='store_true',
                        help='Compare with benchmark strategies')

    parser.add_argument('--analyze_trajectory', action='store_true',
                        help='Perform detailed trajectory analysis')

    parser.add_argument('--output_dir', type=str, default=None,
                        help='Directory to save evaluation results')

    args = parser.parse_args()

    # Validate model path
    model_path = Path(args.model_path)
    if not model_path.exists():
        print(f"❌ Model file not found: {model_path}")
        sys.exit(1)

    # Setup output directory
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        output_dir = model_path.parent / "evaluation"

    output_dir.mkdir(exist_ok=True)

    print("🔍 GBWM Model Evaluation")
    print("=" * 50)
    print(f"Model: {model_path}")
    print(f"Goals: {args.num_goals}")
    print(f"Episodes: {args.num_episodes}")
    print(f"Output: {output_dir}")
    print("=" * 50)

    # Initialize evaluator
    evaluator = GBWMEvaluator(str(model_path))

    # 1. Basic evaluation
    print("\n📊 Running basic evaluation...")
    eval_results = evaluator.evaluate_policy(
        num_goals=args.num_goals,
        num_episodes=args.num_episodes
    )

    print(f"✅ Mean Reward: {eval_results['mean_reward']:.2f} ± {eval_results['std_reward']:.2f}")
    print(f"✅ Goal Success Rate: {eval_results['mean_goal_success_rate']:.1%}")
    print(f"✅ Mean Final Wealth: ${eval_results['mean_final_wealth']:,.0f}")

    # Save basic results
    with open(output_dir / "evaluation_results.json", 'w') as f:
        json.dump(eval_results, f, indent=2, default=str)

    # 2. Benchmark comparison
    if args.compare_benchmarks:
        print("\n🏆 Comparing with benchmarks...")
        benchmark_results = evaluator.compare_with_benchmarks(
            num_goals=args.num_goals,
            num_episodes=min(1000, args.num_episodes)  # Smaller sample for benchmarks
        )

        print("\nBenchmark Comparison:")
        print("-" * 40)
        for strategy, results in benchmark_results.items():
            print(f"{strategy:15}: {results['mean_reward']:6.2f} ± {results['std_reward']:5.2f}")

        # Save benchmark results
        with open(output_dir / "benchmark_comparison.json", 'w') as f:
            json.dump(benchmark_results, f, indent=2, default=str)

    # 3. Trajectory analysis
    if args.analyze_trajectory:
        print("\n🔬 Analyzing sample trajectory...")
        trajectory_analysis = evaluator.analyze_single_trajectory(
            num_goals=args.num_goals,
            seed=42
        )

        print(f"✅ Sample trajectory total reward: {trajectory_analysis['total_reward']:.2f}")
        print(f"✅ Goals achieved: {len(trajectory_analysis['summary']['goals_taken'])}/{args.num_goals}")

        # Save trajectory analysis
        with open(output_dir / "trajectory_analysis.json", 'w') as f:
            json.dump(trajectory_analysis, f, indent=2, default=str)

    print(f"\n✅ Evaluation completed! Results saved in: {output_dir}")


if __name__ == "__main__":
    main()