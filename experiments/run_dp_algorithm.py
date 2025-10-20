"""
Run Dynamic Programming Algorithm for GBWM

This script runs the DP algorithm as described in the Das et al. (2019) paper
and saves results for comparison with RL approaches.

Usage:
    python experiments/run_dp_algorithm.py --initial_wealth 100000 --goal_wealth 200000
"""

import argparse
import sys
import os
import json
import time
import numpy as np
from pathlib import Path

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.algorithms.dynamic_programming import GBWMDynamicProgramming, DPConfig
from config.base_config import RESULTS_DIR


def main():
    parser = argparse.ArgumentParser(description='Run GBWM Dynamic Programming Algorithm')
    
    # Problem parameters
    parser.add_argument('--initial_wealth', type=float, default=100000,
                        help='Initial wealth ($)')
    parser.add_argument('--goal_wealth', type=float, default=200000,
                        help='Goal wealth ($)')
    parser.add_argument('--time_horizon', type=int, default=10,
                        help='Time horizon (years)')
    
    # Algorithm parameters
    parser.add_argument('--num_portfolios', type=int, default=15,
                        help='Number of portfolio choices (m)')
    parser.add_argument('--grid_density', type=float, default=3.0,
                        help='Wealth grid density (ρ_grid)')
    
    # Efficient frontier parameters
    parser.add_argument('--mu_min', type=float, default=0.0526,
                        help='Minimum expected return')
    parser.add_argument('--mu_max', type=float, default=0.0886,
                        help='Maximum expected return')
    
    # Output parameters
    parser.add_argument('--output_dir', type=str, default=None,
                        help='Output directory for results')
    parser.add_argument('--experiment_name', type=str, default=None,
                        help='Experiment name')
    
    # Simulation parameters
    parser.add_argument('--num_simulations', type=int, default=10000,
                        help='Number of Monte Carlo simulations for validation')
    
    args = parser.parse_args()
    
    # Setup experiment name and output directory
    if args.experiment_name is None:
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        args.experiment_name = f"dp_gbwm_{timestamp}"
    
    if args.output_dir is None:
        output_dir = Path(RESULTS_DIR) / args.experiment_name
    else:
        output_dir = Path(args.output_dir)
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("🧮 GBWM Dynamic Programming Algorithm")
    print("=" * 50)
    print(f"Initial Wealth: ${args.initial_wealth:,.0f}")
    print(f"Goal Wealth: ${args.goal_wealth:,.0f}")
    print(f"Time Horizon: {args.time_horizon} years")
    print(f"Portfolios: {args.num_portfolios}")
    print(f"Grid Density: {args.grid_density}")
    print(f"Output: {output_dir}")
    print("=" * 50)
    
    # Create configuration
    config = DPConfig(
        initial_wealth=args.initial_wealth,
        goal_wealth=args.goal_wealth,
        time_horizon=args.time_horizon,
        mu_min=args.mu_min,
        mu_max=args.mu_max,
        num_portfolios=args.num_portfolios,
        grid_density=args.grid_density
    )
    
    # Save configuration
    config_dict = {
        'algorithm': 'Dynamic Programming',
        'initial_wealth': config.initial_wealth,
        'goal_wealth': config.goal_wealth,
        'time_horizon': config.time_horizon,
        'mu_min': config.mu_min,
        'mu_max': config.mu_max,
        'sigma_min': config.sigma_min,
        'sigma_max': config.sigma_max,
        'num_portfolios': config.num_portfolios,
        'grid_density': config.grid_density,
        'eff_frontier_a': config.eff_frontier_a,
        'eff_frontier_b': config.eff_frontier_b,
        'eff_frontier_c': config.eff_frontier_c
    }
    
    with open(output_dir / "config.json", 'w') as f:
        json.dump(config_dict, f, indent=2)
    
    # Initialize and solve DP algorithm
    print("\n🚀 Initializing Dynamic Programming algorithm...")
    dp = GBWMDynamicProgramming(config)
    
    print("\n⚡ Solving Bellman equation...")
    start_time = time.time()
    V, policy = dp.solve()
    solve_time = time.time() - start_time
    
    # Get optimal probability
    optimal_prob = dp.get_optimal_probability()
    
    print(f"\n✅ Algorithm completed in {solve_time:.2f} seconds")
    print(f"✅ Optimal success probability: {optimal_prob:.3f} ({optimal_prob:.1%})")
    
    # Run Monte Carlo validation
    print(f"\n🎲 Running {args.num_simulations:,} Monte Carlo simulations...")
    sim_results = dp.simulate_trajectory(num_simulations=args.num_simulations, seed=42)
    
    print(f"✅ Simulation success rate: {sim_results['success_rate']:.3f} ({sim_results['success_rate']:.1%})")
    print(f"✅ Mean final wealth: ${sim_results['mean_final_wealth']:,.0f}")
    print(f"✅ Simulation vs Theory error: {abs(sim_results['success_rate'] - optimal_prob):.3f}")
    
    # Get policy summary
    print("\n📊 Extracting optimal strategy...")
    policy_summary = dp.get_policy_summary()
    
    # Get comprehensive results
    results = dp.get_results_summary()
    
    # Save all results
    print(f"\n💾 Saving results to {output_dir}...")
    
    # Save value function and policy (numpy arrays)
    np.save(output_dir / "value_function.npy", V)
    np.save(output_dir / "optimal_policy.npy", policy)
    np.save(output_dir / "wealth_grid.npy", dp.wealth_grid)
    
    # Save results summary
    with open(output_dir / "results.json", 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    # Save policy summary
    with open(output_dir / "policy_summary.json", 'w') as f:
        json.dump(policy_summary, f, indent=2, default=str)
    
    # Save simulation results
    sim_results_clean = {k: v for k, v in sim_results.items() if k != 'final_wealths'}  # Exclude large array
    with open(output_dir / "simulation_results.json", 'w') as f:
        json.dump(sim_results_clean, f, indent=2)
    
    # Show some example strategies
    print(f"\n📈 Example Optimal Strategies:")
    print("-" * 40)
    
    sample_wealths = [args.initial_wealth * 0.5, args.initial_wealth, args.initial_wealth * 1.5]
    sample_times = [0, args.time_horizon // 2, args.time_horizon - 1]
    
    for t in sample_times:
        print(f"Time {t}:")
        for w in sample_wealths:
            if t < args.time_horizon:
                portfolio_idx, mu, sigma = dp.get_optimal_strategy(w, t)
                risk_level = "Conservative" if sigma < 0.08 else "Moderate" if sigma < 0.15 else "Aggressive"
                print(f"  Wealth ${w:,.0f} → Portfolio {portfolio_idx} ({risk_level}): μ={mu:.1%}, σ={sigma:.1%}")
        print()
    
    print(f"✅ Dynamic Programming completed! Results saved in: {output_dir}")
    
    # Print comparison info
    print(f"\n💡 To compare with RL results, run:")
    print(f"   python experiments/compare_dp_rl.py --dp_results {output_dir}")


if __name__ == "__main__":
    main()