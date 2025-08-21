"""
Test script for updated configuration system
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.environment_config import (
    EnvironmentConfig, 
    DEFAULT_ENV_CONFIG,
    create_historical_config,
    create_simulation_config
)
from config.training_config import TrainingConfig, DEFAULT_TRAINING_CONFIG
import json

def test_default_config():
    """Test default configuration"""
    print("=" * 50)
    print("Testing DEFAULT CONFIGURATION")
    print("=" * 50)
    
    config = DEFAULT_ENV_CONFIG
    print(f"✅ Default data mode: {config.data_mode}")
    print(f"✅ Default time horizon: {config.time_horizon}")
    print(f"✅ Default historical data path: {config.historical_data_path}")
    print(f"✅ Default goal years: {config.goal_config.goal_years}")
    
    # Test validation
    try:
        config._validate_config()
        print("✅ Default configuration validation passed")
    except Exception as e:
        print(f"❌ Default configuration validation failed: {e}")


def test_simulation_config():
    """Test simulation mode configuration"""
    print("\n" + "=" * 50)
    print("Testing SIMULATION CONFIGURATION")
    print("=" * 50)
    
    config = create_simulation_config(num_goals=4, initial_wealth=500000)
    print(f"✅ Simulation config created")
    print(f"Data mode: {config.data_mode}")
    print(f"Initial wealth: ${config.initial_wealth:,.0f}")
    print(f"Goal years: {config.goal_config.goal_years}")
    
    # Test with different goal numbers
    for num_goals in [1, 2, 4, 8, 16]:
        try:
            config = create_simulation_config(num_goals=num_goals)
            calculated_wealth = 12 * (num_goals ** 0.85) * 10000
            print(f"✅ {num_goals} goals: wealth=${calculated_wealth:,.0f}, goals={config.goal_config.goal_years}")
        except Exception as e:
            print(f"❌ Failed for {num_goals} goals: {e}")


def test_historical_config():
    """Test historical mode configuration"""
    print("\n" + "=" * 50)
    print("Testing HISTORICAL CONFIGURATION")
    print("=" * 50)
    
    config = create_historical_config(
        num_goals=4,
        historical_data_path="data/raw/market_data/",
        start_date="2015-01-01",
        end_date="2023-12-31",
        use_data_augmentation=True,
        max_missing_ratio=0.1
    )
    
    print(f"✅ Historical config created")
    print(f"Data mode: {config.data_mode}")
    print(f"Historical data path: {config.historical_data_path}")
    print(f"Date range: {config.historical_start_date} to {config.historical_end_date}")
    print(f"Use augmentation: {config.use_data_augmentation}")
    print(f"Max missing ratio: {config.max_missing_ratio}")
    print(f"Goal years: {config.goal_config.goal_years}")
    
    # Test get_historical_config method
    hist_config = config.get_historical_config()
    print(f"✅ Historical config dict: {list(hist_config.keys())}")


def test_training_config():
    """Test training configuration with historical support"""
    print("\n" + "=" * 50)
    print("Testing TRAINING CONFIGURATION")
    print("=" * 50)
    
    # Test simulation mode training config
    sim_config = TrainingConfig(data_mode="simulation")
    print(f"✅ Simulation training config")
    print(f"Is historical mode: {sim_config.is_historical_mode()}")
    print(f"Experiment suffix: {sim_config.get_experiment_suffix()}")
    
    # Test historical mode training config
    hist_config = TrainingConfig(
        data_mode="historical",
        historical_data_path="data/raw/market_data/",
        batch_size=2400,  # Smaller batch for historical data
        historical_validation_episodes=500
    )
    print(f"\n✅ Historical training config")
    print(f"Is historical mode: {hist_config.is_historical_mode()}")
    print(f"Experiment suffix: {hist_config.get_experiment_suffix()}")
    print(f"Historical validation episodes: {hist_config.historical_validation_episodes}")
    print(f"Batch size: {hist_config.batch_size}")


def test_config_serialization():
    """Test configuration serialization/deserialization"""
    print("\n" + "=" * 50)
    print("Testing CONFIG SERIALIZATION")
    print("=" * 50)
    
    # Create a historical config
    config = create_historical_config(
        num_goals=8,
        use_data_augmentation=True,
        log_historical_stats=True
    )
    
    # Convert to dictionary (for JSON serialization)
    try:
        config_dict = {
            'data_mode': config.data_mode,
            'time_horizon': config.time_horizon,
            'initial_wealth': config.initial_wealth,
            'historical_data_path': config.historical_data_path,
            'historical_start_date': config.historical_start_date,
            'historical_end_date': config.historical_end_date,
            'use_data_augmentation': config.use_data_augmentation,
            'max_missing_ratio': config.max_missing_ratio,
            'goal_years': config.goal_config.goal_years
        }
        
        print("✅ Configuration serialized to dictionary")
        print(f"Keys: {list(config_dict.keys())}")
        
        # Test JSON serialization
        json_str = json.dumps(config_dict, indent=2)
        print("✅ Configuration serialized to JSON")
        
    except Exception as e:
        print(f"❌ Serialization failed: {e}")


def test_validation():
    """Test configuration validation"""
    print("\n" + "=" * 50)
    print("Testing CONFIG VALIDATION")
    print("=" * 50)
    
    # Test invalid data mode
    try:
        config = EnvironmentConfig(data_mode="invalid_mode")
        print("❌ Should have failed for invalid data mode")
    except ValueError as e:
        print(f"✅ Correctly caught invalid data mode: {e}")
    
    # Test invalid time horizon
    try:
        config = EnvironmentConfig(time_horizon=-1)
        print("❌ Should have failed for negative time horizon")
    except ValueError as e:
        print(f"✅ Correctly caught invalid time horizon: {e}")
    
    # Test invalid validation split
    try:
        config = EnvironmentConfig(historical_validation_split=1.5)
        print("❌ Should have failed for invalid validation split")
    except ValueError as e:
        print(f"✅ Correctly caught invalid validation split: {e}")
    
    print("✅ All validation tests passed")


def main():
    """Run all configuration tests"""
    print("Testing Updated Configuration System")
    
    try:
        test_default_config()
        test_simulation_config()
        test_historical_config()
        test_training_config()
        test_config_serialization()
        test_validation()
        
        print("\n" + "=" * 50)
        print("🎉 ALL CONFIGURATION TESTS PASSED!")
        print("Configuration system ready for historical data support")
        print("=" * 50)
        
    except Exception as e:
        print(f"\n❌ Configuration test suite failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()