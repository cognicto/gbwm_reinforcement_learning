"""
Convert old model format to new safe format

Usage:
    python scripts/convert_old_models.py --input_path old_model.pth --output_path new_model.pth
"""

import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import torch
from pathlib import Path


def convert_model(input_path: str, output_path: str):
    """Convert old model format to new safe format"""

    print(f"Converting {input_path} to {output_path}")

    try:
        # Load old model with weights_only=False to allow the config object
        checkpoint = torch.load(input_path, map_location='cpu', weights_only=False)

        # Extract config as dict
        if 'config' in checkpoint:
            config_obj = checkpoint['config']
            config_dict = {
                'n_traj': getattr(config_obj, 'n_traj', 50000),
                'batch_size': getattr(config_obj, 'batch_size', 4800),
                'learning_rate': getattr(config_obj, 'learning_rate', 0.01),
                'clip_epsilon': getattr(config_obj, 'clip_epsilon', 0.50),
                'n_neurons': getattr(config_obj, 'n_neurons', 64),
                'ppo_epochs': getattr(config_obj, 'ppo_epochs', 4),
                'mini_batch_size': getattr(config_obj, 'mini_batch_size', 256),
                'gamma': getattr(config_obj, 'gamma', 0.99),
                'gae_lambda': getattr(config_obj, 'gae_lambda', 0.95),
                'entropy_coeff': getattr(config_obj, 'entropy_coeff', 0.01),
                'value_loss_coeff': getattr(config_obj, 'value_loss_coeff', 0.5),
                'max_grad_norm': getattr(config_obj, 'max_grad_norm', 0.5),
                'time_horizon': getattr(config_obj, 'time_horizon', 16),
                'num_goals': getattr(config_obj, 'num_goals', 4),
                'num_portfolios': getattr(config_obj, 'num_portfolios', 15),
                'initial_wealth_base': getattr(config_obj, 'initial_wealth_base', 12.0),
                'wealth_scaling': getattr(config_obj, 'wealth_scaling', 0.85),
                'device': str(getattr(config_obj, 'device', 'cpu')),
                'random_seed': getattr(config_obj, 'random_seed', 42),
                'log_interval': getattr(config_obj, 'log_interval', 10),
                'save_interval': getattr(config_obj, 'save_interval', 50),
                'eval_interval': getattr(config_obj, 'eval_interval', 20)
            }
        else:
            # Use default config
            config_dict = {
                'n_traj': 50000,
                'batch_size': 4800,
                'learning_rate': 0.01,
                'clip_epsilon': 0.50,
                'n_neurons': 64,
                'ppo_epochs': 4,
                'mini_batch_size': 256,
                'gamma': 0.99,
                'gae_lambda': 0.95,
                'entropy_coeff': 0.01,
                'value_loss_coeff': 0.5,
                'max_grad_norm': 0.5,
                'time_horizon': 16,
                'num_goals': 4,
                'num_portfolios': 15,
                'initial_wealth_base': 12.0,
                'wealth_scaling': 0.85,
                'device': 'cpu',
                'random_seed': 42,
                'log_interval': 10,
                'save_interval': 50,
                'eval_interval': 20
            }

        # Convert training metrics to serializable format
        training_metrics_dict = {}
        if 'training_metrics' in checkpoint:
            for key, value in checkpoint['training_metrics'].items():
                if hasattr(value, '__iter__'):
                    training_metrics_dict[key] = list(value)
                else:
                    training_metrics_dict[key] = value

        # Create new checkpoint
        new_checkpoint = {
            'policy_state_dict': checkpoint['policy_state_dict'],
            'value_state_dict': checkpoint['value_state_dict'],
            'policy_optimizer_state_dict': checkpoint.get('policy_optimizer_state_dict', {}),
            'value_optimizer_state_dict': checkpoint.get('value_optimizer_state_dict', {}),
            'config_dict': config_dict,  # Save as dict instead of object
            'training_metrics': training_metrics_dict,
            'total_timesteps': checkpoint.get('total_timesteps', 0),
            'iteration': checkpoint.get('iteration', 0)
        }

        # Save new format
        torch.save(new_checkpoint, output_path)

        print("✅ Conversion successful!")
        print(f"New model saved at: {output_path}")

    except Exception as e:
        print(f"❌ Conversion failed: {e}")
        return False

    return True


def main():
    parser = argparse.ArgumentParser(description='Convert old model format to new safe format')

    parser.add_argument('--input_path', type=str, required=True,
                        help='Path to old model file')

    parser.add_argument('--output_path', type=str, default=None,
                        help='Path for converted model (default: adds _converted suffix)')

    args = parser.parse_args()

    input_path = Path(args.input_path)

    if not input_path.exists():
        print(f"❌ Input file not found: {input_path}")
        return

    if args.output_path:
        output_path = Path(args.output_path)
    else:
        output_path = input_path.parent / f"{input_path.stem}_converted{input_path.suffix}"

    convert_model(str(input_path), str(output_path))


if __name__ == "__main__":
    main()