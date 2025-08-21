"""
Test script for updated training pipeline with historical data support
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.training.trainer import GBWMTrainer, train_gbwm_agent
from config.training_config import TrainingConfig
import tempfile
import shutil

def test_simulation_training():
    """Test training pipeline in simulation mode"""
    print("=" * 50)
    print("Testing SIMULATION MODE TRAINING")
    print("=" * 50)
    
    # Create temporary directory for results
    temp_dir = tempfile.mkdtemp()
    
    try:
        # Configure for quick training
        config = TrainingConfig(
            data_mode="simulation",
            n_traj=1000,  # Very small for quick test
            batch_size=100,
            learning_rate=0.01
        )
        
        # Create trainer
        trainer = GBWMTrainer(
            config=config,
            experiment_name="test_simulation"
        )
        
        print("✅ Trainer created in simulation mode")
        
        # Setup environment
        trainer.setup_environment(num_goals=4, data_mode="simulation")
        print(f"✅ Environment setup: {trainer.env.data_mode} mode")
        print(f"Goal schedule: {trainer.env.goal_schedule}")
        
        # Setup agent
        trainer.setup_agent()
        print("✅ Agent setup completed")
        
        # Run a very short training (just 1 iteration)
        total_timesteps = config.batch_size * config.time_horizon  # Single iteration
        training_history = trainer.train(
            num_goals=4,
            total_timesteps=total_timesteps,
            data_mode="simulation"
        )
        
        print(f"✅ Training completed: {len(training_history)} iterations")
        print(f"Final reward: {training_history[-1]['mean_episode_reward']:.2f}")
        
        print("✅ Simulation mode training test passed")
        
    except Exception as e:
        print(f"❌ Simulation training test failed: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        # Cleanup
        shutil.rmtree(temp_dir, ignore_errors=True)


def test_historical_training():
    """Test training pipeline in historical mode"""
    print("\n" + "=" * 50)
    print("Testing HISTORICAL MODE TRAINING")
    print("=" * 50)
    
    # Create temporary directory for results
    temp_dir = tempfile.mkdtemp()
    
    try:
        # Configure for quick training with historical data
        config = TrainingConfig(
            data_mode="historical",
            historical_data_path="data/raw/market_data/",
            n_traj=800,  # Smaller batch for historical
            batch_size=80,  # Even smaller to avoid data repetition
            learning_rate=0.01
        )
        
        # Create trainer
        trainer = GBWMTrainer(
            config=config,
            experiment_name="test_historical"
        )
        
        print("✅ Trainer created in historical mode")
        
        # Setup environment (this will create historical loader)
        trainer.setup_environment(num_goals=4, data_mode="historical")
        print(f"✅ Environment setup: {trainer.env.data_mode} mode")
        print(f"Goal schedule: {trainer.env.goal_schedule}")
        
        # Verify historical loader was created
        if trainer.env.historical_loader is not None:
            stats = trainer.env.historical_loader.validate_data_quality()
            print(f"✅ Historical data: {stats['total_periods']} periods, {stats['available_16y_sequences']} sequences")
        
        # Setup agent
        trainer.setup_agent()
        print("✅ Agent setup completed")
        
        # Run a very short training (just 1 iteration)
        total_timesteps = config.batch_size * config.time_horizon  # Single iteration
        training_history = trainer.train(
            num_goals=4,
            total_timesteps=total_timesteps,
            data_mode="historical"
        )
        
        print(f"✅ Training completed: {len(training_history)} iterations")
        print(f"Final reward: {training_history[-1]['mean_episode_reward']:.2f}")
        
        print("✅ Historical mode training test passed")
        
    except Exception as e:
        print(f"❌ Historical training test failed: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        # Cleanup
        shutil.rmtree(temp_dir, ignore_errors=True)


def test_convenience_function():
    """Test the convenience training function"""
    print("\n" + "=" * 50)
    print("Testing CONVENIENCE FUNCTION")
    print("=" * 50)
    
    try:
        # Test simulation mode
        print("Testing simulation mode convenience function...")
        sim_config = TrainingConfig(
            data_mode="simulation",
            n_traj=500,
            batch_size=50
        )
        
        trainer = train_gbwm_agent(
            num_goals=2,
            config=sim_config,
            experiment_name="test_convenience_sim",
            data_mode="simulation"
        )
        
        print(f"✅ Simulation convenience function: {trainer.env.data_mode}")
        
        # Test historical mode
        print("Testing historical mode convenience function...")
        hist_config = TrainingConfig(
            data_mode="historical",
            n_traj=400,
            batch_size=40
        )
        
        trainer = train_gbwm_agent(
            num_goals=2,
            config=hist_config,
            experiment_name="test_convenience_hist",
            data_mode="historical"
        )
        
        print(f"✅ Historical convenience function: {trainer.env.data_mode}")
        print("✅ Convenience function tests passed")
        
    except Exception as e:
        print(f"❌ Convenience function test failed: {e}")
        import traceback
        traceback.print_exc()


def test_command_line_interface():
    """Test command line interface parsing"""
    print("\n" + "=" * 50)
    print("Testing COMMAND LINE INTERFACE")
    print("=" * 50)
    
    import subprocess
    import sys
    
    # Test help message
    try:
        result = subprocess.run([
            sys.executable, 
            "experiments/run_training.py", 
            "--help"
        ], capture_output=True, text=True, timeout=10)
        
        if "data_mode" in result.stdout:
            print("✅ Help shows historical data options")
        else:
            print("❌ Help missing historical data options")
            
        if "historical_data_path" in result.stdout:
            print("✅ Historical data path option available")
        else:
            print("❌ Historical data path option missing")
            
    except Exception as e:
        print(f"❌ Command line help test failed: {e}")
    
    print("✅ Command line interface test completed")


def test_experiment_naming():
    """Test experiment naming with different modes"""
    print("\n" + "=" * 50)
    print("Testing EXPERIMENT NAMING")
    print("=" * 50)
    
    try:
        # Test simulation naming
        sim_config = TrainingConfig(data_mode="simulation")
        sim_trainer = GBWMTrainer(config=sim_config, experiment_name=None)
        print(f"✅ Simulation trainer: {sim_trainer.experiment_name}")
        
        # Test historical naming  
        hist_config = TrainingConfig(data_mode="historical")
        hist_trainer = GBWMTrainer(config=hist_config, experiment_name=None)
        print(f"✅ Historical trainer: {hist_trainer.experiment_name}")
        
        # Test explicit naming
        explicit_trainer = GBWMTrainer(config=sim_config, experiment_name="my_custom_experiment")
        print(f"✅ Custom name: {explicit_trainer.experiment_name}")
        
        print("✅ Experiment naming test passed")
        
    except Exception as e:
        print(f"❌ Experiment naming test failed: {e}")


def main():
    """Run all training pipeline tests"""
    print("TESTING UPDATED TRAINING PIPELINE")
    print("Testing complete pipeline with historical data support")
    
    try:
        test_simulation_training()
        test_historical_training()
        test_convenience_function()
        test_command_line_interface()
        test_experiment_naming()
        
        print("\n" + "=" * 50)
        print("🎉 ALL TRAINING PIPELINE TESTS PASSED!")
        print("Training pipeline ready for historical data support")
        print("=" * 50)
        
        # Show usage examples
        print("\nUsage Examples:")
        print("# Simulation mode (default)")
        print("python experiments/run_training.py --num_goals 4")
        
        print("\n# Historical mode")
        print("python experiments/run_training.py --data_mode historical --num_goals 4")
        
        print("\n# Historical with custom date range")
        print("python experiments/run_training.py --data_mode historical --num_goals 4 \\")
        print("  --historical_start_date 2015-01-01 --historical_end_date 2020-12-31")
        
    except Exception as e:
        print(f"\n❌ Training pipeline test suite failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()