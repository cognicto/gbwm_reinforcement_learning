"""
Quick test for Dynamic Programming algorithm
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.algorithms.dynamic_programming import GBWMDynamicProgramming, DPConfig
import numpy as np
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)

def test_basic_dp():
    """Quick basic test"""
    print("🧮 Quick DP Test")
    print("="*30)
    
    # Create simple configuration
    config = DPConfig(
        initial_wealth=100000,
        goal_wealth=200000,
        time_horizon=5,  # Shorter horizon for quick test
        num_portfolios=5,  # Fewer portfolios
        grid_density=1.0   # Lower density
    )
    
    print(f"Config: ${config.initial_wealth:,.0f} → ${config.goal_wealth:,.0f} in {config.time_horizon} years")
    print(f"Portfolios: {config.num_portfolios}, Grid density: {config.grid_density}")
    
    # Initialize and test
    print("\n1. Initializing...")
    dp = GBWMDynamicProgramming(config)
    
    print(f"   ✅ Grid size: {len(dp.wealth_grid)} points")
    print(f"   ✅ Wealth range: ${dp.wealth_grid.min():,.0f} - ${dp.wealth_grid.max():,.0f}")
    print(f"   ✅ Portfolios: μ=[{dp.mu_array.min():.1%}, {dp.mu_array.max():.1%}]")
    
    # Test transition probabilities
    print("\n2. Testing transition probabilities...")
    probs = dp._compute_transition_probabilities(100000, 0.06, 0.10)
    print(f"   ✅ Transition probs sum: {np.sum(probs):.3f} (should be ~1.0)")
    
    # Solve
    print("\n3. Solving...")
    start_time = time.time()
    V, policy = dp.solve()
    solve_time = time.time() - start_time
    
    optimal_prob = dp.get_optimal_probability()
    
    print(f"   ✅ Solved in {solve_time:.2f} seconds")
    print(f"   ✅ Optimal probability: {optimal_prob:.3f} ({optimal_prob:.1%})")
    
    # Quick simulation validation  
    print("\n4. Quick validation...")
    sim_results = dp.simulate_trajectory(num_simulations=1000, seed=42)
    sim_prob = sim_results['success_rate']
    error = abs(sim_prob - optimal_prob)
    
    print(f"   ✅ Simulation: {sim_prob:.3f} ({sim_prob:.1%})")
    print(f"   ✅ Error: {error:.3f}")
    
    if error < 0.05:  # 5% tolerance for quick test
        print("   🟢 Validation passed!")
    else:
        print("   🟡 Large error - may need investigation")
    
    return dp

if __name__ == "__main__":
    import time
    try:
        dp = test_basic_dp()
        print("\n✅ Quick test completed successfully!")
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()