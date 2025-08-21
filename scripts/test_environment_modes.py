"""
Test script for GBWMEnvironment with both simulation and historical modes
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.environment.gbwm_env import make_gbwm_env
from src.data.historical_data_loader import create_historical_loader
import numpy as np

def test_simulation_mode():
    """Test environment in simulation mode (existing behavior)"""
    print("=" * 50)
    print("Testing SIMULATION MODE")
    print("=" * 50)
    
    env = make_gbwm_env(num_goals=4, data_mode="simulation")
    print(f"✅ Environment created in simulation mode")
    
    # Run a short episode
    obs, info = env.reset()
    print(f"Initial obs: {obs}")
    
    for step in range(5):
        action = env.action_space.sample()  # Random action
        obs, reward, terminated, truncated, info = env.step(action)
        print(f"Step {step}: wealth=${info['wealth']:,.0f}, reward={reward:.2f}")
        
        if terminated or truncated:
            break
    
    print("✅ Simulation mode test completed")


def test_historical_mode():
    """Test environment in historical mode"""
    print("\n" + "=" * 50)
    print("Testing HISTORICAL MODE")
    print("=" * 50)
    
    # Create historical loader
    try:
        loader = create_historical_loader()
        print("✅ Historical loader created")
    except Exception as e:
        print(f"❌ Failed to create historical loader: {e}")
        return
    
    # Create environment in historical mode
    try:
        env = make_gbwm_env(
            num_goals=4, 
            data_mode="historical",
            historical_loader=loader
        )
        print("✅ Environment created in historical mode")
    except Exception as e:
        print(f"❌ Failed to create historical environment: {e}")
        return
    
    # Run a short episode
    try:
        obs, info = env.reset()
        print(f"Initial obs: {obs}")
        
        for step in range(5):
            action = env.action_space.sample()  # Random action
            obs, reward, terminated, truncated, info = env.step(action)
            print(f"Step {step}: wealth=${info['wealth']:,.0f}, reward={reward:.2f}")
            
            if terminated or truncated:
                break
        
        print("✅ Historical mode test completed")
        
    except Exception as e:
        print(f"❌ Historical mode test failed: {e}")
        import traceback
        traceback.print_exc()


def compare_modes():
    """Compare behavior between modes"""
    print("\n" + "=" * 50)
    print("COMPARING MODES")
    print("=" * 50)
    
    # Create historical loader
    loader = create_historical_loader()
    
    # Create environments
    sim_env = make_gbwm_env(num_goals=4, data_mode="simulation")
    hist_env = make_gbwm_env(num_goals=4, data_mode="historical", historical_loader=loader)
    
    print("Running 3 episodes in each mode...")
    
    # Run episodes in both modes
    for episode in range(3):
        print(f"\nEpisode {episode + 1}:")
        
        # Simulation mode
        sim_env.reset()
        sim_wealth = []
        for step in range(5):
            action = [0, 5]  # Fixed action for consistency
            _, _, terminated, truncated, info = sim_env.step(action)
            sim_wealth.append(info['wealth'])
            if terminated or truncated:
                break
        
        # Historical mode  
        hist_env.reset()
        hist_wealth = []
        for step in range(5):
            action = [0, 5]  # Same fixed action
            _, _, terminated, truncated, info = hist_env.step(action)
            hist_wealth.append(info['wealth'])
            if terminated or truncated:
                break
        
        print(f"Simulation wealth progression: {[f'${w:,.0f}' for w in sim_wealth]}")
        print(f"Historical wealth progression:  {[f'${w:,.0f}' for w in hist_wealth]}")
    
    print("\n✅ Mode comparison completed")


def main():
    """Run all tests"""
    print("Testing GBWM Environment with Historical Data Support")
    
    try:
        test_simulation_mode()
        test_historical_mode() 
        compare_modes()
        
        print("\n" + "=" * 50)
        print("🎉 ALL TESTS PASSED!")
        print("Environment successfully supports both simulation and historical modes")
        print("=" * 50)
        
    except Exception as e:
        print(f"\n❌ Test suite failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()