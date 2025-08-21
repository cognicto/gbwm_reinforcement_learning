"""Simple test to verify imports work"""

import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def test_imports():
    """Test all imports work correctly"""

    print("🧪 Testing Imports...")

    try:
        # Test environment
        from src.environment.gbwm_env import make_gbwm_env
        print("✅ Environment import successful")

        # Test models
        from src.models.policy_network import PolicyNetwork
        from src.models.value_network import ValueNetwork
        print("✅ Models import successful")

        # Test config
        from config.training_config import TrainingConfig
        print("✅ Config import successful")

        # Test utils
        from src.utils.data_utils import compute_gae
        print("✅ Utils import successful")

        # Test basic environment creation
        env = make_gbwm_env(num_goals=4)
        obs, _ = env.reset()
        print(f"✅ Environment creation successful: obs shape = {obs.shape}")

        # Test networks
        import torch
        policy_net = PolicyNetwork()
        value_net = ValueNetwork()

        # Test forward pass
        state = torch.FloatTensor(obs).unsqueeze(0)
        goal_probs, portfolio_probs = policy_net(state)
        value = value_net(state)

        print(f"✅ Networks forward pass successful")
        print(f"   Goal probs shape: {goal_probs.shape}")
        print(f"   Portfolio probs shape: {portfolio_probs.shape}")
        print(f"   Value shape: {value.shape}")

        print("\n🎉 All imports and basic functionality working!")
        return True

    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = test_imports()
    if success:
        print("\n✅ Ready to proceed with training!")
    else:
        print("\n❌ Please fix errors before proceeding")