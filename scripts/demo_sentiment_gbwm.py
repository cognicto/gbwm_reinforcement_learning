"""
Demonstration Script for Sentiment-Aware GBWM

This script provides a quick demonstration of the complete sentiment-aware
Goals-Based Wealth Management system.
"""

import os
import sys
from pathlib import Path
import numpy as np
import torch
import logging

# Add src to path for imports
sys.path.append(str(Path(__file__).parent.parent / 'src'))

from src.data.sentiment_provider import SentimentProvider
from src.environment.gbwm_env_sentiment import make_sentiment_gbwm_env
from src.models.sentiment_ppo_agent import SentimentAwarePPOAgent
from config.training_config import TrainingConfig
from config.sentiment_config import get_sentiment_config

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def demo_sentiment_data():
    """Demonstrate sentiment data fetching and processing"""
    print("=" * 60)
    print("DEMO 1: Sentiment Data System")
    print("=" * 60)
    
    # Create sentiment provider
    logger.info("Creating sentiment provider...")
    sentiment_provider = SentimentProvider(
        cache_dir='./demo_cache',
        lookback_days=30
    )
    
    # Initialize provider
    logger.info("Initializing sentiment data...")
    success = sentiment_provider.initialize()
    
    if not success:
        print("❌ Failed to initialize sentiment provider")
        return False
    
    print("✅ Sentiment provider initialized successfully")
    
    # Get sentiment features for recent dates
    from datetime import datetime, timedelta
    
    recent_date = datetime.now() - timedelta(days=5)
    features = sentiment_provider.get_sentiment_features(recent_date)
    
    print(f"📊 Sentiment features for {recent_date.date()}: {features}")
    print(f"   VIX Sentiment: {features[0]:.3f} (fear/greed indicator)")
    print(f"   VIX Momentum: {features[1]:.3f} (5-day change indicator)")
    
    # Get detailed sentiment info
    info = sentiment_provider.get_sentiment_info(recent_date)
    print(f"   Raw VIX: {info.get('vix_raw', 'N/A'):.1f}")
    print(f"   VIX Regime: {info.get('vix_regime', 'UNKNOWN')}")
    
    # Get statistics
    stats = sentiment_provider.get_statistics()
    if 'vix' in stats:
        vix_stats = stats['vix']
        print(f"   VIX Stats: mean={vix_stats.get('mean', 0):.1f}, "
              f"min={vix_stats.get('min', 0):.1f}, max={vix_stats.get('max', 0):.1f}")
    
    return True


def demo_environment():
    """Demonstrate sentiment-aware environment"""
    print("\n" + "=" * 60)
    print("DEMO 2: Sentiment-Aware Environment")
    print("=" * 60)
    
    # Create sentiment provider
    sentiment_provider = SentimentProvider(cache_dir='./demo_cache', lookback_days=30)
    if not sentiment_provider.initialize():
        print("❌ Failed to initialize sentiment provider for environment demo")
        return False
    
    # Create environments - with and without sentiment
    logger.info("Creating environments...")
    
    # Sentiment-aware environment
    env_sentiment = make_sentiment_gbwm_env(
        num_goals=4,
        sentiment_provider=sentiment_provider,
        sentiment_start_date='2020-01-01'
    )
    
    # Baseline environment (no sentiment)
    env_baseline = make_sentiment_gbwm_env(
        num_goals=4,
        sentiment_provider=None
    )
    
    print(f"✅ Environments created:")
    print(f"   Sentiment-aware: state_dim={env_sentiment.observation_space.shape[0]}")
    print(f"   Baseline: state_dim={env_baseline.observation_space.shape[0]}")
    
    # Test environment interactions
    print("\n📋 Testing environment interactions...")
    
    # Reset and test both environments
    obs_sentiment, info_sentiment = env_sentiment.reset()
    obs_baseline, info_baseline = env_baseline.reset()
    
    print(f"   Sentiment-aware initial state: {obs_sentiment}")
    print(f"   Baseline initial state: {obs_baseline}")
    
    # Take random actions
    action = env_sentiment.action_space.sample()
    print(f"   Random action: {action} (goal={action[0]}, portfolio={action[1]})")
    
    # Step both environments
    next_obs_s, reward_s, done_s, _, info_s = env_sentiment.step(action)
    next_obs_b, reward_b, done_b, _, info_b = env_baseline.step(action)
    
    print(f"   Sentiment-aware result: reward={reward_s:.2f}, next_state={next_obs_s}")
    print(f"   Baseline result: reward={reward_b:.2f}, next_state={next_obs_b}")
    
    # Show sentiment-specific info
    if 'sentiment_features' in info_s:
        print(f"   Sentiment info: {info_s['sentiment_features']}")
    
    return True


def demo_agent():
    """Demonstrate sentiment-aware agent"""
    print("\n" + "=" * 60)
    print("DEMO 3: Sentiment-Aware PPO Agent")
    print("=" * 60)
    
    # Create sentiment provider and environment
    sentiment_provider = SentimentProvider(cache_dir='./demo_cache', lookback_days=30)
    if not sentiment_provider.initialize():
        print("❌ Failed to initialize sentiment provider for agent demo")
        return False
    
    env = make_sentiment_gbwm_env(
        num_goals=4,
        sentiment_provider=sentiment_provider
    )
    
    # Create training configuration
    config = TrainingConfig(
        batch_size=10,  # Small for demo
        ppo_epochs=2,
        mini_batch_size=10,
        time_horizon=5,  # Short episodes for demo
        n_neurons=32,    # Small network for demo
        device='cpu'     # Force CPU for demo
    )
    
    # Create agent
    logger.info("Creating sentiment-aware agent...")
    agent = SentimentAwarePPOAgent(
        env=env,
        config=config,
        policy_type="standard",
        encoder_type="simple",  # Simple for demo
        sentiment_enabled=True
    )
    
    print(f"✅ Agent created: {agent.agent_config}")
    
    # Test action prediction
    print("\n🤖 Testing agent predictions...")
    obs, _ = env.reset()
    
    print(f"   Initial observation: {obs}")
    
    # Predict actions
    action_stochastic = agent.predict(obs, deterministic=False)
    action_deterministic = agent.predict(obs, deterministic=True)
    
    print(f"   Stochastic action: {action_stochastic}")
    print(f"   Deterministic action: {action_deterministic}")
    
    # Test a few training iterations
    print("\n🏋️ Testing training (3 mini-iterations)...")
    
    for i in range(3):
        # Collect a small batch
        batch_data = agent.collect_trajectories(num_trajectories=2)
        print(f"   Iteration {i+1}: collected {batch_data['states'].shape[0]} steps")
        
        # Update networks
        metrics = agent.update_policy(batch_data)
        print(f"      Policy loss: {metrics['policy_loss']:.4f}")
        print(f"      Value loss: {metrics['value_loss']:.4f}")
    
    # Get sentiment analysis
    sentiment_analysis = agent.get_sentiment_analysis()
    if sentiment_analysis['sentiment_enabled']:
        print(f"\n📈 Sentiment analysis available:")
        if 'vix_sentiment_stats' in sentiment_analysis:
            stats = sentiment_analysis['vix_sentiment_stats']
            print(f"   VIX sentiment range: [{stats['min']:.2f}, {stats['max']:.2f}]")
            print(f"   VIX sentiment mean: {stats['mean']:.2f}")
    
    return True


def demo_full_workflow():
    """Demonstrate a complete workflow"""
    print("\n" + "=" * 60)
    print("DEMO 4: Complete Workflow")
    print("=" * 60)
    
    print("🔄 Running complete sentiment-aware GBWM workflow...")
    
    # Configuration
    sentiment_config = get_sentiment_config("default")
    
    print(f"   Using sentiment config: {sentiment_config.model.encoder_type} encoder")
    print(f"   Sentiment enabled: {sentiment_config.experiment.sentiment_enabled}")
    
    # Setup sentiment provider with demo cache directory
    # Filter kwargs to only include valid SentimentProvider parameters
    valid_params = {'cache_dir', 'vix_weight', 'news_weight', 'long_term_vix_mean', 'lookback_days'}
    sentiment_kwargs = {
        k: v for k, v in sentiment_config.sentiment.__dict__.items() 
        if k in valid_params
    }
    sentiment_kwargs['cache_dir'] = './demo_cache'  # Override for demo
    
    sentiment_provider = SentimentProvider(**sentiment_kwargs)
    
    if not sentiment_provider.initialize():
        print("❌ Failed to setup sentiment provider")
        return False
    
    # Setup environment
    env = make_sentiment_gbwm_env(
        num_goals=2,  # Simplified for demo
        sentiment_provider=sentiment_provider
    )
    
    # Setup agent with configuration compatible with feature encoder
    config = TrainingConfig(
        batch_size=5,
        time_horizon=3,
        n_neurons=64,  # Match feature encoder output
        device='cpu'
    )
    
    agent = SentimentAwarePPOAgent(
        env=env,
        config=config,
        policy_type=sentiment_config.model.policy_type,
        encoder_type=sentiment_config.model.encoder_type,
        sentiment_enabled=sentiment_config.experiment.sentiment_enabled
    )
    
    print("✅ Complete system setup successful")
    
    # Run a mini training session
    print("\n🏃 Running mini training session (5 iterations)...")
    
    training_history = []
    for iteration in range(5):
        metrics = agent.train_iteration()
        training_history.append(metrics)
        
        print(f"   Iteration {iteration + 1}: "
              f"reward={metrics['mean_episode_reward']:.2f}, "
              f"goal_success={metrics['mean_goal_success_rate']:.1%}")
    
    print("✅ Training completed successfully")
    
    # Final evaluation
    print("\n📊 Final evaluation...")
    
    total_reward = 0
    total_episodes = 5
    
    for episode in range(total_episodes):
        obs, _ = env.reset()
        episode_reward = 0
        done = False
        
        while not done:
            action = agent.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            episode_reward += reward
            done = terminated or truncated
        
        total_reward += episode_reward
    
    avg_reward = total_reward / total_episodes
    print(f"   Average evaluation reward: {avg_reward:.2f}")
    
    # Save demonstration model
    demo_model_path = "./demo_model.pth"
    agent.save(demo_model_path)
    print(f"✅ Demo model saved: {demo_model_path}")
    
    return True


def cleanup_demo():
    """Clean up demo files"""
    import shutil
    
    cleanup_paths = [
        './demo_cache',
        './demo_model.pth',
        './demo_model_config.json'
    ]
    
    for path in cleanup_paths:
        if os.path.exists(path):
            try:
                if os.path.isdir(path):
                    shutil.rmtree(path)
                else:
                    os.remove(path)
                print(f"🧹 Cleaned up: {path}")
            except Exception as e:
                print(f"⚠️  Could not clean up {path}: {e}")


def main():
    """Main demonstration function"""
    print("🚀 Sentiment-Aware GBWM Demonstration")
    print("This demo shows the complete sentiment integration system.")
    print()
    
    try:
        # Set random seed for reproducibility
        np.random.seed(42)
        torch.manual_seed(42)
        
        # Run demonstrations
        success = True
        
        success &= demo_sentiment_data()
        success &= demo_environment()
        success &= demo_agent()
        success &= demo_full_workflow()
        
        if success:
            print("\n" + "=" * 60)
            print("🎉 ALL DEMONSTRATIONS COMPLETED SUCCESSFULLY!")
            print("=" * 60)
            print()
            print("The sentiment-aware GBWM system is working correctly.")
            print("Key capabilities demonstrated:")
            print("✅ VIX data fetching and sentiment feature extraction")
            print("✅ Sentiment-augmented environment (4D state space)")
            print("✅ Sentiment-aware PPO agent with feature encoding")
            print("✅ Complete training and evaluation workflow")
            print()
            print("Next steps:")
            print("• Run full training: python experiments/train_sentiment_gbwm.py")
            print("• Compare with baseline: python experiments/compare_sentiment_baseline.py")
            print("• Customize configuration in config/sentiment_config.py")
        else:
            print("\n❌ Some demonstrations failed. Check the logs above.")
        
    except Exception as e:
        print(f"\n💥 Demo failed with error: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        # Clean up demo files
        print("\n🧹 Cleaning up demo files...")
        cleanup_demo()


if __name__ == "__main__":
    main()