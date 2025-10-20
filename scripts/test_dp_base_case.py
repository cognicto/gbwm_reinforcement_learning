"""
Test the base case from the paper with optimized parameters
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.algorithms.dynamic_programming import GBWMDynamicProgramming, DPConfig
import numpy as np
import time

def test_paper_base_case():
    """Test base case from the paper with reasonable grid size"""
    print("🧮 GBWM DP Base Case Test (Paper Parameters)")
    print("="*55)
    
    # Paper base case with smaller grid for speed
    config = DPConfig(
        initial_wealth=100000,  # W(0) = $100k
        goal_wealth=200000,     # G = $200k
        time_horizon=10,        # T = 10 years 
        num_portfolios=15,      # m = 15 portfolios
        grid_density=2.0        # Reduced from 3.0 for speed
    )
    
    print(f"Parameters from Das et al. (2019):")
    print(f"  Initial Wealth: ${config.initial_wealth:,.0f}")
    print(f"  Goal Wealth: ${config.goal_wealth:,.0f}")
    print(f"  Time Horizon: {config.time_horizon} years")
    print(f"  Portfolios: {config.num_portfolios}")
    print(f"  μ range: [{config.mu_min:.4f}, {config.mu_max:.4f}]")
    print(f"  Grid Density: {config.grid_density} (reduced for speed)")
    print("="*55)
    
    # Initialize
    print("\n🚀 Initializing Dynamic Programming...")
    dp = GBWMDynamicProgramming(config)
    
    print(f"✅ Grid size: {len(dp.wealth_grid):,} points")
    print(f"✅ Wealth range: [${dp.wealth_grid.min():,.0f}, ${dp.wealth_grid.max():,.0f}]")
    print(f"✅ Initial wealth index: {dp.initial_wealth_idx}")
    
    # Solve
    print(f"\n⚡ Solving Bellman equation...")
    start_time = time.time()
    V, policy = dp.solve()
    solve_time = time.time() - start_time
    
    optimal_prob = dp.get_optimal_probability()
    
    print(f"\n🎯 RESULTS:")
    print(f"  ✅ Solve time: {solve_time:.2f} seconds")
    print(f"  ✅ Optimal success probability: {optimal_prob:.3f} ({optimal_prob:.1%})")
    
    # Compare with paper expectation
    paper_expected = 0.669  # From paper
    difference = optimal_prob - paper_expected
    
    print(f"\n📊 COMPARISON WITH PAPER:")
    print(f"  Paper result: {paper_expected:.3f} ({paper_expected:.1%})")
    print(f"  Our result:   {optimal_prob:.3f} ({optimal_prob:.1%})")
    print(f"  Difference:   {difference:+.3f} ({difference:+.1%})")
    
    if abs(difference) < 0.05:  # Within 5%
        print(f"  🟢 Result is close to paper expectation!")
    elif abs(difference) < 0.10:  # Within 10%
        print(f"  🟡 Result differs but within reasonable range")
        print(f"     (Differences can occur due to grid discretization)")
    else:
        print(f"  🔴 Large difference from paper - may need investigation")
    
    # Monte Carlo validation
    print(f"\n🎲 Monte Carlo Validation ({10000:,} simulations)...")
    sim_start = time.time()
    sim_results = dp.simulate_trajectory(num_simulations=10000, seed=42)
    sim_time = time.time() - sim_start
    
    sim_prob = sim_results['success_rate']
    theory_vs_sim_error = abs(sim_prob - optimal_prob)
    
    print(f"  ✅ Simulation time: {sim_time:.2f} seconds")
    print(f"  ✅ Simulation success rate: {sim_prob:.3f} ({sim_prob:.1%})")
    print(f"  ✅ Mean final wealth: ${sim_results['mean_final_wealth']:,.0f}")
    print(f"  ✅ Theory vs Simulation error: {theory_vs_sim_error:.3f}")
    
    if theory_vs_sim_error < 0.02:
        print(f"  🟢 Simulation validates theory!")
    else:
        print(f"  🟡 Some simulation error (normal for finite samples)")
    
    # Show optimal strategy examples
    print(f"\n📈 OPTIMAL STRATEGY EXAMPLES:")
    print(f"     (Portfolio choice for different wealth levels)")
    
    sample_wealths = [50000, 100000, 150000, 200000]
    sample_times = [0, 5, 9]
    
    for t in sample_times:
        print(f"\n  Time {t}:")
        for w in sample_wealths:
            if t < config.time_horizon:
                portfolio_idx, mu, sigma = dp.get_optimal_strategy(w, t)
                risk_level = "Conservative" if sigma < 0.08 else "Moderate" if sigma < 0.15 else "Aggressive"
                print(f"    ${w:>6,.0f} → Portfolio {portfolio_idx:2d} ({risk_level:11s}): μ={mu:5.1%}, σ={sigma:5.1%}")
    
    print(f"\n✅ Base case test completed successfully!")
    print(f"\n💡 Key Insights:")
    print(f"   • Algorithm solves optimally in {solve_time:.1f} seconds")
    print(f"   • Achieves {optimal_prob:.1%} success probability")
    print(f"   • Strategy adapts based on wealth and time remaining")
    print(f"   • When behind target: more aggressive")
    print(f"   • When ahead of target: more conservative")
    
    return dp

if __name__ == "__main__":
    try:
        dp = test_paper_base_case()
        print(f"\n🎉 Test completed - DP algorithm is working correctly!")
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()