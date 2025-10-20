"""
Compare Dynamic Programming vs Reinforcement Learning for GBWM

This script compares the performance of the DP algorithm (optimal solution)
with trained RL models.

Usage:
    python experiments/compare_dp_rl.py --dp_results data/results/dp_experiment --rl_model data/results/rl_experiment/final_model.pth
"""

import argparse
import sys
import os
import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, List, Tuple

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.algorithms.dynamic_programming import GBWMDynamicProgramming, DPConfig
from src.evaluation.evaluator import GBWMEvaluator
from src.environment.gbwm_env import make_gbwm_env
import torch


class DPRLComparator:
    """Compare Dynamic Programming and Reinforcement Learning approaches"""
    
    def __init__(self, dp_results_dir: str, rl_model_path: str = None):
        """
        Initialize comparator
        
        Args:
            dp_results_dir: Directory containing DP results
            rl_model_path: Path to trained RL model (optional)
        """
        self.dp_results_dir = Path(dp_results_dir)
        self.rl_model_path = Path(rl_model_path) if rl_model_path else None
        
        # Load DP results
        self._load_dp_results()
        
        # Initialize RL evaluator if model provided
        self.rl_evaluator = None
        if self.rl_model_path and self.rl_model_path.exists():
            self.rl_evaluator = GBWMEvaluator(str(self.rl_model_path))
        
    def _load_dp_results(self):
        """Load DP results from directory"""
        # Load configuration
        with open(self.dp_results_dir / "config.json", 'r') as f:
            self.dp_config = json.load(f)
        
        # Load results
        with open(self.dp_results_dir / "results.json", 'r') as f:
            self.dp_results = json.load(f)
        
        # Load arrays
        self.value_function = np.load(self.dp_results_dir / "value_function.npy")
        self.optimal_policy = np.load(self.dp_results_dir / "optimal_policy.npy")
        self.wealth_grid = np.load(self.dp_results_dir / "wealth_grid.npy")
        
        print(f"✅ Loaded DP results from {self.dp_results_dir}")
        print(f"   Optimal probability: {self.dp_results['theoretical_results']['optimal_probability']:.3f}")
    
    def _reconstruct_dp_algorithm(self) -> GBWMDynamicProgramming:
        """Reconstruct DP algorithm from saved configuration"""
        config = DPConfig(
            initial_wealth=self.dp_config['initial_wealth'],
            goal_wealth=self.dp_config['goal_wealth'],
            time_horizon=self.dp_config['time_horizon'],
            mu_min=self.dp_config['mu_min'],
            mu_max=self.dp_config['mu_max'],
            num_portfolios=self.dp_config['num_portfolios'],
            grid_density=self.dp_config['grid_density']
        )
        
        dp = GBWMDynamicProgramming(config)
        dp.value_function = self.value_function
        dp.policy = self.optimal_policy
        dp.wealth_grid = self.wealth_grid
        
        return dp
    
    def compare_success_rates(self, num_episodes: int = 10000) -> Dict:
        """Compare success rates between DP and RL"""
        
        results = {
            'dp_theoretical': self.dp_results['theoretical_results']['optimal_probability'],
            'dp_simulation': self.dp_results['simulation_validation']['success_rate'],
            'num_episodes': num_episodes
        }
        
        if self.rl_evaluator:
            # Evaluate RL model
            print(f"🔍 Evaluating RL model with {num_episodes:,} episodes...")
            
            # Determine number of goals from environment
            num_goals = 4  # Default - could be inferred from config
            
            rl_results = self.rl_evaluator.evaluate_policy(
                num_goals=num_goals,
                num_episodes=num_episodes
            )
            
            results['rl_mean_reward'] = rl_results['mean_reward']
            results['rl_std_reward'] = rl_results['std_reward']
            results['rl_goal_success_rate'] = rl_results['mean_goal_success_rate']
            results['rl_final_wealth'] = rl_results['mean_final_wealth']
            
            print(f"✅ RL evaluation completed")
        
        return results
    
    def compare_strategies(self, sample_points: int = 5) -> Dict:
        """Compare strategies at various wealth/time points"""
        
        if not self.rl_evaluator:
            return {"error": "No RL model provided for strategy comparison"}
        
        # Reconstruct DP algorithm for strategy queries
        dp = self._reconstruct_dp_algorithm()
        
        # Sample wealth and time points
        min_wealth = self.wealth_grid.min()
        max_wealth = self.wealth_grid.max()
        time_horizon = self.dp_config['time_horizon']
        
        wealth_samples = np.linspace(min_wealth, max_wealth, sample_points)
        time_samples = np.linspace(0, time_horizon - 1, sample_points, dtype=int)
        
        comparisons = []
        
        for t in time_samples:
            for w in wealth_samples:
                # Get DP strategy
                dp_portfolio_idx, dp_mu, dp_sigma = dp.get_optimal_strategy(w, t)
                
                # For RL strategy, we'd need to create environment and get action
                # This is more complex and requires the specific environment setup
                
                comparison = {
                    'time': int(t),
                    'wealth': float(w),
                    'dp_portfolio': int(dp_portfolio_idx),
                    'dp_return': float(dp_mu),
                    'dp_volatility': float(dp_sigma),
                    'dp_risk_level': "Conservative" if dp_sigma < 0.08 else "Moderate" if dp_sigma < 0.15 else "Aggressive"
                }
                
                comparisons.append(comparison)
        
        return {'strategy_comparisons': comparisons}
    
    def generate_comparison_report(self, output_dir: str = None) -> Dict:
        """Generate comprehensive comparison report"""
        
        if output_dir is None:
            output_dir = self.dp_results_dir / "comparison"
        else:
            output_dir = Path(output_dir)
        
        output_dir.mkdir(exist_ok=True)
        
        print("\n📊 Generating comparison report...")
        
        # 1. Success rate comparison
        success_comparison = self.compare_success_rates()
        
        # 2. Strategy comparison
        strategy_comparison = self.compare_strategies()
        
        # 3. Performance metrics
        performance_metrics = {
            'dp_solve_time': self.dp_results['theoretical_results']['solve_time'],
            'dp_grid_size': self.dp_results['grid_info']['grid_size'],
            'dp_wealth_range': self.dp_results['grid_info']['wealth_range']
        }
        
        # Compile full report
        report = {
            'comparison_type': 'Dynamic Programming vs Reinforcement Learning',
            'timestamp': str(pd.Timestamp.now()) if 'pd' in globals() else 'unknown',
            'dp_config': self.dp_config,
            'success_rates': success_comparison,
            'strategies': strategy_comparison,
            'performance': performance_metrics,
            'summary': self._generate_summary(success_comparison)
        }
        
        # Save report
        with open(output_dir / "comparison_report.json", 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        # Print summary
        self._print_summary(success_comparison)
        
        print(f"\n✅ Comparison report saved to: {output_dir}")
        
        return report
    
    def _generate_summary(self, success_comparison: Dict) -> Dict:
        """Generate summary of comparison"""
        summary = {
            'dp_theoretical_optimal': success_comparison['dp_theoretical'],
            'dp_simulation_validation': success_comparison['dp_simulation']
        }
        
        if 'rl_goal_success_rate' in success_comparison:
            summary['rl_performance'] = success_comparison['rl_goal_success_rate']
            summary['optimality_gap'] = success_comparison['dp_theoretical'] - success_comparison['rl_goal_success_rate']
            summary['rl_efficiency'] = success_comparison['rl_goal_success_rate'] / success_comparison['dp_theoretical']
        
        return summary
    
    def _print_summary(self, success_comparison: Dict):
        """Print comparison summary"""
        print("\n" + "="*60)
        print("🏆 DYNAMIC PROGRAMMING vs REINFORCEMENT LEARNING COMPARISON")
        print("="*60)
        
        print(f"\n📈 SUCCESS RATES:")
        print(f"   DP (Theoretical Optimal): {success_comparison['dp_theoretical']:.3f} ({success_comparison['dp_theoretical']:.1%})")
        print(f"   DP (Simulation):          {success_comparison['dp_simulation']:.3f} ({success_comparison['dp_simulation']:.1%})")
        
        if 'rl_goal_success_rate' in success_comparison:
            rl_rate = success_comparison['rl_goal_success_rate']
            dp_rate = success_comparison['dp_theoretical']
            efficiency = rl_rate / dp_rate if dp_rate > 0 else 0
            gap = dp_rate - rl_rate
            
            print(f"   RL (Learned Policy):      {rl_rate:.3f} ({rl_rate:.1%})")
            print(f"\n🎯 PERFORMANCE GAP:")
            print(f"   Optimality Gap:    {gap:.3f} ({gap:.1%})")
            print(f"   RL Efficiency:     {efficiency:.1%} of optimal")
            
            if efficiency >= 0.95:
                print("   🟢 RL achieves >95% of optimal performance")
            elif efficiency >= 0.90:
                print("   🟡 RL achieves 90-95% of optimal performance")
            else:
                print("   🔴 RL achieves <90% of optimal performance")
        
        print(f"\n⚡ COMPUTATIONAL PERFORMANCE:")
        print(f"   DP Solve Time:     {self.dp_results['theoretical_results']['solve_time']:.2f} seconds")
        print(f"   DP Grid Size:      {self.dp_results['grid_info']['grid_size']:,} points")
        
        if 'rl_mean_reward' in success_comparison:
            print(f"   RL Mean Reward:    {success_comparison['rl_mean_reward']:.2f}")


def main():
    parser = argparse.ArgumentParser(description='Compare DP and RL approaches for GBWM')
    
    parser.add_argument('--dp_results', type=str, required=True,
                        help='Directory containing DP results')
    
    parser.add_argument('--rl_model', type=str, default=None,
                        help='Path to trained RL model')
    
    parser.add_argument('--num_episodes', type=int, default=10000,
                        help='Number of episodes for RL evaluation')
    
    parser.add_argument('--output_dir', type=str, default=None,
                        help='Output directory for comparison results')
    
    args = parser.parse_args()
    
    # Validate inputs
    dp_dir = Path(args.dp_results)
    if not dp_dir.exists():
        print(f"❌ DP results directory not found: {dp_dir}")
        sys.exit(1)
    
    if args.rl_model:
        rl_model = Path(args.rl_model)
        if not rl_model.exists():
            print(f"❌ RL model not found: {rl_model}")
            sys.exit(1)
    else:
        print("⚠️  No RL model provided - will only analyze DP results")
        rl_model = None
    
    print("🔍 GBWM Algorithm Comparison")
    print("=" * 50)
    print(f"DP Results: {dp_dir}")
    if rl_model:
        print(f"RL Model: {rl_model}")
    print(f"Episodes: {args.num_episodes:,}")
    print("=" * 50)
    
    # Initialize comparator
    comparator = DPRLComparator(str(dp_dir), str(rl_model) if rl_model else None)
    
    # Generate comparison report
    report = comparator.generate_comparison_report(args.output_dir)
    
    print("\n✅ Comparison completed!")


if __name__ == "__main__":
    import pandas as pd  # For timestamp
    main()