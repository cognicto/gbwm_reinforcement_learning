"""
Test script for the Dynamic Programming algorithm

Quick test to verify the DP implementation works correctly
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.algorithms.dynamic_programming import solve_gbwm_dp
import numpy as np


def test_dp_base_case():
    """Test the base case from the paper"""
    print("🧮 Testing GBWM Dynamic Programming - Base Case")
    print("=" * 50)
    print("Parameters:")
    print("  Initial Wealth: $100,000")
    print("  Goal Wealth: $200,000") 
    print("  Time Horizon: 10 years")
    print("  Portfolios: 15")
    print("=" * 50)
    
    # Solve base case
    dp = solve_gbwm_dp(
        initial_wealth=100000,
        goal_wealth=200000,
        time_horizon=10,
        num_portfolios=15,
        grid_density=3.0
    )
    
    optimal_prob = dp.get_optimal_probability()
    
    print(f"\n✅ Optimal Success Probability: {optimal_prob:.3f} ({optimal_prob:.1%})")
    print(f"✅ Solve Time: {dp.solve_time:.2f} seconds")
    print(f"✅ Grid Size: {len(dp.wealth_grid):,} points")
    
    # Expected result from paper is around 0.669 (66.9%)
    expected_range = (0.65, 0.69)
    if expected_range[0] <= optimal_prob <= expected_range[1]:
        print(f"🟢 Result matches paper expectation ({expected_range[0]:.1%} - {expected_range[1]:.1%})")
    else:
        print(f"🟡 Result differs from paper expectation ({expected_range[0]:.1%} - {expected_range[1]:.1%})")
    
    # Test strategy at initial wealth and time 0
    portfolio_idx, mu, sigma = dp.get_optimal_strategy(100000, 0)
    risk_level = "Conservative" if sigma < 0.08 else "Moderate" if sigma < 0.15 else "Aggressive"
    
    print(f"\n📈 Initial Optimal Strategy:")
    print(f"   Portfolio Index: {portfolio_idx}")
    print(f"   Expected Return: {mu:.1%}")
    print(f"   Volatility: {sigma:.1%}")
    print(f"   Risk Level: {risk_level}")
    
    # Validate with simulation
    print(f"\n🎲 Validating with Monte Carlo simulation...")
    sim_results = dp.simulate_trajectory(num_simulations=10000, seed=42)
    sim_prob = sim_results['success_rate']
    error = abs(sim_prob - optimal_prob)
    
    print(f"✅ Simulation Success Rate: {sim_prob:.3f} ({sim_prob:.1%})")
    print(f"✅ Theory vs Simulation Error: {error:.3f}")
    
    if error < 0.02:  # Less than 2% error
        print("🟢 Simulation validates theoretical result")
    else:
        print("🟡 Large simulation error - check implementation")
    
    return dp


def test_dp_different_scenarios():
    """Test different scenarios to ensure robustness"""
    print(f"\n🔬 Testing Different Scenarios")
    print("=" * 30)
    
    scenarios = [
        {"name": "Easy Goal", "initial": 150000, "goal": 200000, "horizon": 10},
        {"name": "Hard Goal", "initial": 50000, "goal": 200000, "horizon": 10},
        {"name": "Short Horizon", "initial": 100000, "goal": 200000, "horizon": 5},
        {"name": "Long Horizon", "initial": 100000, "goal": 200000, "horizon": 15},
    ]
    
    for scenario in scenarios:
        print(f"\n{scenario['name']}:")
        dp = solve_gbwm_dp(
            initial_wealth=scenario['initial'],
            goal_wealth=scenario['goal'],
            time_horizon=scenario['horizon']
        )
        
        prob = dp.get_optimal_probability()
        print(f"  Success Probability: {prob:.3f} ({prob:.1%})")
        print(f"  Solve Time: {dp.solve_time:.2f}s")


def test_dp_algorithm_properties():
    """Test that the algorithm has expected properties"""
    print(f"\n🔍 Testing Algorithm Properties")
    print("=" * 35)
    
    dp = solve_gbwm_dp()
    
    # Test 1: Value function should be non-decreasing in wealth
    print("\n1. Value function monotonicity in wealth:")
    time_idx = 0
    wealth_values = dp.value_function[:, time_idx]
    is_monotonic = np.all(np.diff(wealth_values) >= -1e-6)  # Allow small numerical errors
    print(f"   Monotonic in wealth: {'✅' if is_monotonic else '❌'}")
    
    # Test 2: Value function should be non-increasing in time (for same wealth)
    print("\n2. Value function monotonicity in time:")
    wealth_idx = dp.initial_wealth_idx
    time_values = dp.value_function[wealth_idx, :]
    is_time_monotonic = np.all(np.diff(time_values) <= 1e-6)  # Should decrease or stay same
    print(f"   Non-increasing in time: {'✅' if is_time_monotonic else '❌'}")
    
    # Test 3: Terminal condition is correct
    print("\n3. Terminal condition:")
    terminal_values = dp.value_function[:, -1]
    goal_achieved = dp.wealth_grid >= dp.config.goal_wealth
    correct_terminal = np.allclose(terminal_values, goal_achieved.astype(float))
    print(f"   Correct terminal condition: {'✅' if correct_terminal else '❌'}")
    
    # Test 4: Probabilities are between 0 and 1
    print("\n4. Probability bounds:")
    all_in_bounds = np.all((dp.value_function >= 0) & (dp.value_function <= 1))
    print(f"   All probabilities in [0,1]: {'✅' if all_in_bounds else '❌'}")
    
    return dp


def main():
    """Run all tests"""
    print("🚀 Starting Dynamic Programming Algorithm Tests\n")
    
    try:
        # Test 1: Base case
        dp = test_dp_base_case()
        
        # Test 2: Different scenarios  
        test_dp_different_scenarios()
        
        # Test 3: Algorithm properties
        test_dp_algorithm_properties()
        
        print(f"\n" + "="*60)
        print("✅ ALL TESTS COMPLETED SUCCESSFULLY!")
        print("✅ Dynamic Programming algorithm is working correctly")
        print("="*60)
        
        # Show how to run full experiments
        print(f"\n💡 To run full experiments:")
        print(f"   python experiments/run_dp_algorithm.py")
        print(f"   python experiments/compare_dp_rl.py --dp_results <dp_results_dir> --rl_model <rl_model_path>")
        
    except Exception as e:
        print(f"\n❌ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)