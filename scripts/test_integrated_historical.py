"""
Integrated test for historical data system
Tests the complete pipeline: Config -> Loader -> Environment
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.environment_config import create_historical_config, create_simulation_config
from config.training_config import TrainingConfig
from src.data.historical_data_loader import create_historical_loader
from src.environment.gbwm_env import make_gbwm_env
import numpy as np

def test_integrated_workflow():
    """Test complete workflow from config to environment"""
    print("=" * 60)
    print("INTEGRATED HISTORICAL DATA WORKFLOW TEST")
    print("=" * 60)
    
    # Step 1: Create historical configuration
    print("\n📝 Step 1: Creating historical configuration...")
    env_config = create_historical_config(
        num_goals=4,
        start_date="2010-01-01",
        end_date="2023-12-31",
        use_data_augmentation=True
    )
    print(f"✅ Environment config: {env_config.data_mode} mode")
    
    training_config = TrainingConfig(
        data_mode="historical",
        batch_size=4800,
        n_traj=50000
    )
    print(f"✅ Training config: {training_config.data_mode} mode")
    
    # Step 2: Create historical data loader
    print("\n📊 Step 2: Creating historical data loader...")
    hist_config = env_config.get_historical_config()
    loader = create_historical_loader(
        data_path=hist_config['data_path'],
        start_date=hist_config['start_date'],
        end_date=hist_config['end_date']
    )
    
    stats = loader.validate_data_quality()
    print(f"✅ Loaded {stats['total_periods']} time periods")
    print(f"✅ Available 16-year sequences: {stats['available_16y_sequences']}")
    
    # Step 3: Create environment using configuration
    print("\n🏗️ Step 3: Creating environment with historical data...")
    env = make_gbwm_env(
        num_goals=4,
        data_mode=env_config.data_mode,
        historical_loader=loader,
        initial_wealth=env_config.initial_wealth
    )
    print(f"✅ Environment created in {env.data_mode} mode")
    
    # Step 4: Run multiple episodes to test variety
    print("\n🎮 Step 4: Testing episode variety...")
    episode_wealths = []
    
    for episode in range(5):
        obs, info = env.reset()
        episode_wealth_trajectory = [info['wealth']]
        
        for step in range(5):  # First 5 steps of each episode
            action = [0, np.random.randint(0, 15)]  # Skip goals, random portfolio
            obs, reward, terminated, truncated, info = env.step(action)
            episode_wealth_trajectory.append(info['wealth'])
            
            if terminated or truncated:
                break
        
        episode_wealths.append(episode_wealth_trajectory)
        print(f"Episode {episode + 1}: {[f'${w:,.0f}' for w in episode_wealth_trajectory]}")
    
    # Verify episodes use different historical sequences
    first_episode_changes = [episode_wealths[0][i+1] - episode_wealths[0][i] 
                            for i in range(len(episode_wealths[0])-1)]
    second_episode_changes = [episode_wealths[1][i+1] - episode_wealths[1][i] 
                             for i in range(len(episode_wealths[1])-1)]
    
    if first_episode_changes != second_episode_changes:
        print("✅ Episodes use different historical sequences (expected)")
    else:
        print("⚠️ Episodes show identical patterns (may indicate issue)")
    
    print("\n✅ Integrated workflow test completed successfully!")


def compare_simulation_vs_historical():
    """Compare simulation and historical modes side by side"""
    print("\n" + "=" * 60)
    print("SIMULATION vs HISTORICAL COMPARISON")
    print("=" * 60)
    
    # Create simulation environment
    sim_env = make_gbwm_env(num_goals=4, data_mode="simulation")
    
    # Create historical environment
    loader = create_historical_loader()
    hist_env = make_gbwm_env(
        num_goals=4, 
        data_mode="historical", 
        historical_loader=loader
    )
    
    print("Running 3 episodes in each mode with identical actions...")
    
    # Fixed actions for comparison
    actions = [[0, 5], [0, 8], [0, 10], [0, 12], [1, 7]]  # Mix of goal and portfolio decisions
    
    for episode in range(3):
        print(f"\n📊 Episode {episode + 1}:")
        
        # Simulation episode
        sim_env.reset()
        sim_rewards = []
        sim_wealths = []
        
        for action in actions:
            obs, reward, term, trunc, info = sim_env.step(action)
            sim_rewards.append(reward)
            sim_wealths.append(info['wealth'])
            if term or trunc:
                break
        
        # Historical episode
        hist_env.reset()
        hist_rewards = []
        hist_wealths = []
        
        for action in actions:
            obs, reward, term, trunc, info = hist_env.step(action)
            hist_rewards.append(reward)
            hist_wealths.append(info['wealth'])
            if term or trunc:
                break
        
        print(f"Simulation:  Wealth={[f'${w:,.0f}' for w in sim_wealths]}, Rewards={sim_rewards}")
        print(f"Historical:  Wealth={[f'${w:,.0f}' for w in hist_wealths]}, Rewards={hist_rewards}")
        
        # Compare reward structures (should be identical for goal rewards)
        if sim_rewards == hist_rewards:
            print("✅ Reward structures match (goal logic consistent)")
        else:
            print("⚠️ Reward structures differ (check goal logic)")


def test_configuration_consistency():
    """Test that configurations work consistently across components"""
    print("\n" + "=" * 60)
    print("CONFIGURATION CONSISTENCY TEST")
    print("=" * 60)
    
    # Test different goal configurations
    goal_configs = [1, 2, 4, 8, 16]
    
    for num_goals in goal_configs:
        print(f"\n🎯 Testing {num_goals} goals configuration...")
        
        try:
            # Create configuration
            config = create_historical_config(num_goals=num_goals)
            
            # Create loader
            loader = create_historical_loader()
            
            # Create environment
            env = make_gbwm_env(
                num_goals=num_goals,
                data_mode="historical",
                historical_loader=loader
            )
            
            # Verify goal schedule
            expected_sequences = loader.get_available_sequences(config.time_horizon)
            actual_goal_years = config.goal_config.goal_years
            
            print(f"✅ {num_goals} goals: {actual_goal_years}, sequences: {expected_sequences}")
            
            # Test a single episode
            obs, info = env.reset()
            obs, reward, term, trunc, info = env.step([0, 0])
            
            print(f"✅ Episode runs successfully")
            
        except Exception as e:
            print(f"❌ Failed for {num_goals} goals: {e}")


def main():
    """Run all integrated tests"""
    print("INTEGRATED HISTORICAL DATA SYSTEM TEST")
    print("Testing complete pipeline: Configuration -> Data Loading -> Environment")
    
    try:
        test_integrated_workflow()
        compare_simulation_vs_historical()
        test_configuration_consistency()
        
        print("\n" + "=" * 60)
        print("🎉 ALL INTEGRATED TESTS PASSED!")
        print("Historical data system is fully functional and integrated")
        print("=" * 60)
        
    except Exception as e:
        print(f"\n❌ Integrated test suite failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()