# Goals-Based Wealth Management with Reinforcement Learning

![Python](https://img.shields.io/badge/python-3.8+-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)

This project implements the paper **'Reinforcement learning for Multiple Goals
in Goals-Based Wealth Management'** by Das et al., using Proximal Policy
Optimization (PPO) to solve multi-objective financial planning problems.

## 🎯 Project Overview

### Problem
Traditional wealth management focuses on single-goal optimization (usually retirement).
This project tackles **Goals-Based Wealth Management (GBWM)** - optimizing multiple
competing financial goals with different timing, costs, and importance levels.

### Solution
- **Method**: Proximal Policy Optimization (PPO) with custom multi-discrete action space
- **Performance**: Achieves 94-98% of optimal dynamic programming solutions
- **Speed**: 32-second training time vs. hours for traditional methods
- **Scalability**: Linear scaling with number of goals (vs. exponential for DP)

### Key Innovation
- **2D Action Space**: Simultaneous goal timing decisions + portfolio allocation choices
- **Multi-head Policy Network**: Coordinated learning for different decision types
- **Real-time Optimization**: Practical deployment for robo-advisory applications

## 🚀 Quick Start

### Installation
```bash
# Clone the repository
git clone <repository-url>
cd gbwm_reinforcement_learning

# Install in development mode
pip install -e .
```

### Basic Usage
```bash
# Train the PPO agent
python experiments/run_training.py

# Evaluate trained model
python experiments/run_evaluation.py
```

## 📁 Project Structure

```
gbwm_reinforcement_learning/
├── config/                 # Configuration files
├── src/
│   ├── environment/        # GBWM environment implementation
│   ├── models/            # Neural network architectures (PPO)
│   ├── training/          # Training pipeline & data collection
│   ├── evaluation/        # Performance evaluation & benchmarks
│   └── utils/            # Helper functions & utilities
├── experiments/           # Experiment scripts
├── notebooks/            # Jupyter notebooks for analysis
├── tests/               # Unit tests
└── data/               # Data storage
```

## 🔬 Key Features

- **Custom Gymnasium Environment** for Goals-Based Wealth Management
- **Multi-head Policy Network** for coordinated decision making
- **PPO Implementation** with clipped surrogate objective
- **NSGA-II Hyperparameter Optimization** for multi-objective tuning

## 📊 Expected Results

| Goals | RL Efficiency | Runtime (seconds) |
|-------|---------------|-------------------|
| 1     | 97.4%         | 32-33            |
| 2     | 94.8%         | 32-33            |
| 4     | 96.1%         | 32-33            |
| 8     | 97.8%         | 32-33            |
| 16    | 97.8%         | 32-33            |

## 🧪 Testing

```bash
# Run all tests
pytest tests/

# Run with coverage
pytest tests/ --cov=src --cov-report=html
```

## 📄 License

This project is licensed under the MIT License.

## 📚 Citation

If you use this code in your research, please cite the original paper by Das et al.

---

⭐ **Star this repository if you find it helpful!**



# Train with 4 goals (paper default)
python experiments/run_training.py --num_goals 4

# Train with different configurations
python experiments/run_training.py --num_goals 8 --batch_size 4800 --learning_rate 0.01

# Train with custom experiment name
python experiments/run_training.py --num_goals 4 --experiment_name "paper_replication"






1. Project structure 
gbwm_reinforcement_learning/
├── config/                   # Configuration files
│   ├── base_config.py       # Project paths and basic settings
│   ├── training_config.py   # Training hyperparameters
│   └── environment_config.py # Environment parameters
├── src/
│   ├── environment/         # GBWM environment implementation
│   ├── models/             # Neural networks (Policy + Value)
│   ├── training/           # Training pipeline
│   ├── evaluation/         # Performance evaluation (MISSING)
│   └── utils/             # Helper functions
├── experiments/            # Training scripts
├── data/results/          # Saved models and results
└── scripts/              # Test scripts

2. How the code works : 
Step 1: Environment setup 

# src/environment/gbwm_env.py
# Creates a Gymnasium environment that simulates:
# - 16-year investment horizon
# - Multiple financial goals at different times
# - Portfolio choices with realistic returns
# - Wealth evolution through geometric Brownian motion

Step 2: Neural Networks 
# src/models/policy_network.py - The "Brain" that makes decisions
class PolicyNetwork:
    # Input: [time, wealth] (normalized)
    # Output: [goal_probabilities, portfolio_probabilities]
    # Architecture: Shared backbone → Two specialized heads
    
# src/models/value_network.py - Evaluates how good situations are
class ValueNetwork:
    # Input: [time, wealth]
    # Output: Expected future utility (single number)

Step 3: PPO Agent 

# src/models/ppo_agent.py - The complete learning system
class PPOAgent:
    # Collects 4,800 trajectories (16 years each = 76,800 decisions)
    # Learns from these experiences using PPO algorithm
    # Updates both policy and value networks

Step 4: Training pipeline 
# src/training/trainer.py - Orchestrates everything
class GBWMTrainer:
    # Sets up environment and agent
    # Runs training loop
    # Saves models and results
    # Logs progress


3. What happens during training 
# Training Loop (simplified)
for iteration in range(10):  # 10 PPO updates
    
    # 1. COLLECT DATA (76,800 decisions)
    batch_data = agent.collect_trajectories(4800)  # 4,800 episodes
    
    # 2. LEARN FROM DATA
    for epoch in range(4):  # Use same data 4 times
        for mini_batch in batches:
            # Update policy: "Do more of what worked well"
            # Update value: "Predict outcomes more accurately"
    
    # 3. REPEAT WITH BETTER POLICY



Complete Usage guide : 
1. Training models :

# Train with paper settings (4 goals)
python experiments/run_training.py --num_goals 4

# Train different configurations
python experiments/run_training.py --num_goals 8 --batch_size 4800 --learning_rate 0.01

# Quick training for testing
python experiments/run_training.py --num_goals 4 --timesteps 100000 --experiment_name "quick_test"

2. Evaluating Trained Models
# Basic evaluation
python experiments/run_evaluation.py --model_path data/results/experiment_name/final_model.pth --num_goals 4

# Full evaluation with benchmarks
python experiments/run_evaluation.py \
    --model_path data/results/gbwm_4goals_bs4800_lr0.01/final_model_safe.pth \
    --num_goals 4 \
    --num_episodes 10000 \
    --compare_benchmarks \
    --analyze_trajectory

3. Interactive Inference (AI Financial Advisor)

# Interactive mode - chat with the AI advisor
python experiments/run_inference.py --model_path data/results/gbwm_4goals_bs4800_lr0.01/final_model_safe.pth

# Batch inference on multiple scenarios
python experiments/run_inference.py \
    --model_path data/results/gbwm_4goals_bs4800_lr0.01/final_model_safe.pth \
    --mode batch


4. Quick demo : 
# See the complete system in action
python scripts/quick_demo.py

python scripts/convert_old_models.py \
    --input_path data/results/gbwm_8goals_bs4800_lr0.01/final_model.pth \    
    --output_path data/results/gbwm_8goals_bs4800_lr0.01/final_model_safe.pth
Converting data/results/gbwm_8goals_bs4800_lr0.01/final_model.pth to data/results/gbwm_8goals_bs4800_lr0.01/final_model_safe.pth





python scripts/setup_data_structure.py

python scripts/organize_models.py
# Use production model
python experiments/run_evaluation.py \
    --model_path data/models/production/gbwm_4goals_bs4800_lr0.01.pth

# Use for inference
python experiments/run_inference.py \
    --model_path data/models/production/gbwm_4goals_bs4800_lr0.01.pth



 python experiments/run_training.py --data_mode historical --num_goals 2 --timesteps 1000 --experiment_name "test_historical_quick"
 
  Quick Start:

  # Traditional simulation mode (unchanged)
  python experiments/run_training.py --num_goals 4

  # New historical data mode
  python experiments/run_training.py --data_mode historical --num_goals 4

  # Historical with custom parameters
  python experiments/run_training.py --data_mode historical --num_goals 4 \
    --historical_start_date 2015-01-01 --historical_end_date 2020-12-31



python experiments/run_evaluation.py --model_path data/results/gbwm_4goals_bs4800_lr0.01_hist/final_model.pth --num_goals 4

