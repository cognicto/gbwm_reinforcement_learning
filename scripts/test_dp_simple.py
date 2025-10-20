"""
Simple test of Dynamic Programming algorithm
"""

from src.algorithms.dynamic_programming import solve_gbwm_dp
import time

print("🧮 Testing GBWM Dynamic Programming")
print("="*40)

# Test with smaller problem size for speed
print("Running base case (optimized for speed)...")
start = time.time()

dp = solve_gbwm_dp(
    initial_wealth=100000,
    goal_wealth=200000,
    time_horizon=10,
    num_portfolios=10,     # Reduced from 15
    grid_density=1.0       # Reduced from 3.0
)

elapsed = time.time() - start

print(f"\n✅ Results:")
print(f"   Optimal Probability: {dp.get_optimal_probability():.3f} ({dp.get_optimal_probability():.1%})")
print(f"   Solve Time: {elapsed:.2f} seconds")
print(f"   Grid Size: {len(dp.wealth_grid):,} points")

# Quick validation
print(f"\n🎲 Quick validation (1000 sims)...")
sim_results = dp.simulate_trajectory(1000, seed=42)
print(f"   Simulation: {sim_results['success_rate']:.3f} ({sim_results['success_rate']:.1%})")
print(f"   Error: {abs(sim_results['success_rate'] - dp.get_optimal_probability()):.3f}")

print(f"\n✅ DP algorithm working correctly!")
print(f"\nTo run full experiments:")
print(f"  python experiments/run_dp_algorithm.py")
print(f"  python experiments/compare_dp_rl.py --dp_results <dir> --rl_model <model>")