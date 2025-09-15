# Goals-Based Wealth Management with Reinforcement Learning

![Python](https://img.shields.io/badge/python-3.8+-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)

This project implements the paper **'Reinforcement learning for Multiple Goals
in Goals-Based Wealth Management'** by Das et al., using Proximal Policy
Optimization (PPO) to solve multi-objective financial planning problems.

## 🔄 Recent Updates

### Historical Data Bug Fix (v1.1)
- **Fixed critical time scale mismatch**: Historical data now uses proper annual periods instead of monthly
- **Extended data coverage**: 54 years of synthetic annual market data (1970-2023) 
- **Improved training diversity**: 39 unique 16-year sequences for historical mode
- **Enhanced realism**: Proper bond duration modeling and correlation structures
- **Backward compatibility**: All existing simulation mode functionality preserved

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

# Setup data structure
python scripts/setup_data_structure.py
```

### Basic Usage
```bash
# Train with simulation data (default)
python experiments/run_training.py --num_goals 4

# Train with historical market data
python experiments/run_training.py --num_goals 4 --data_mode historical

# Evaluate trained model
python experiments/run_evaluation.py --model_path data/results/experiment_name/final_model_safe.pth --num_goals 4

# Interactive financial advisor
python experiments/run_inference.py --model_path data/models/production/gbwm_4goals_bs4800_lr0.01.pth
```

## 📁 Project Structure

```
gbwm_reinforcement_learning/
├── config/                   # Configuration files
│   ├── base_config.py       # Project paths and basic settings
│   ├── training_config.py   # Training hyperparameters
│   └── environment_config.py # Environment parameters
├── src/
│   ├── environment/         # GBWM environment implementation
│   │   └── gbwm_env.py     # Custom Gymnasium environment
│   ├── models/             # Neural networks (Policy + Value)
│   │   ├── policy_network.py   # Multi-head decision network
│   │   ├── value_network.py    # State value estimation
│   │   └── ppo_agent.py       # Complete PPO implementation
│   ├── training/           # Training pipeline
│   │   └── trainer.py      # Training orchestration
│   ├── evaluation/         # Performance evaluation
│   ├── data/              # Historical data processing
│   │   └── historical_data_loader.py  # Market data management
│   └── utils/             # Helper functions
├── experiments/            # Training and evaluation scripts
├── data/                  # Data storage
│   ├── raw/              # Raw market data
│   ├── processed/        # Processed portfolio parameters
│   ├── models/           # Trained model artifacts
│   └── results/          # Training results and logs
├── scripts/              # Utility scripts
└── docs/                # Documentation
```

## 🔬 Key Features

### Core Components
- **Custom Gymnasium Environment** for Goals-Based Wealth Management
- **Multi-head Policy Network** for coordinated decision making
- **PPO Implementation** with clipped surrogate objective
- **Historical Data Integration** with 54 years of market data

### Data Modes
1. **Simulation Mode**: Synthetic data using Geometric Brownian Motion
2. **Historical Mode**: Real market patterns from 1970-2023

### Training Capabilities
- **Scalable**: Linear scaling with number of goals (1, 2, 4, 8, 16)
- **Fast**: ~32 seconds training time regardless of goal count
- **Robust**: 39 unique historical sequences for diverse market exposure

## 📊 Expected Results

| Goals | RL Efficiency | Runtime (seconds) | Historical Sequences |
|-------|---------------|-------------------|---------------------|
| 1     | 97.4%         | 32-33            | 39                 |
| 2     | 94.8%         | 32-33            | 39                 |
| 4     | 96.1%         | 32-33            | 39                 |
| 8     | 97.8%         | 32-33            | 39                 |
| 16    | 97.8%         | 32-33            | 39                 |

## 💻 Usage Examples

### 1. Training Models

```bash
# Train with paper settings (4 goals, simulation mode)
python experiments/run_training.py --num_goals 4

# Train with historical market data
python experiments/run_training.py --num_goals 4 --data_mode historical

# Train different configurations
python experiments/run_training.py --num_goals 8 --batch_size 4800 --learning_rate 0.01

# Quick training for testing
python experiments/run_training.py --num_goals 4 --timesteps 100000 --experiment_name "quick_test"

# Historical mode with custom date range
python experiments/run_training.py --data_mode historical --num_goals 4 \
    --historical_start_date 1980-01-01 --historical_end_date 2020-12-31
```

### 2. Evaluating Trained Models

```bash
# Basic evaluation
python experiments/run_evaluation.py \
    --model_path data/results/experiment_name/final_model_safe.pth \
    --num_goals 4

# Full evaluation with benchmarks
python experiments/run_evaluation.py \
    --model_path data/results/gbwm_4goals_bs4800_lr0.01/final_model_safe.pth \
    --num_goals 4 \
    --num_episodes 10000 \
    --compare_benchmarks \
    --analyze_trajectory
```

### 3. Interactive Financial Advisor

```bash
# Interactive mode - chat with the AI advisor
python experiments/run_inference.py \
    --model_path data/models/production/gbwm_4goals_bs4800_lr0.01.pth

# Batch inference on multiple scenarios
python experiments/run_inference.py \
    --model_path data/models/production/gbwm_4goals_bs4800_lr0.01.pth \
    --mode batch
```

### 4. Utility Scripts

```bash
# Setup data structure and generate synthetic market data
python scripts/setup_data_structure.py

# Test the system with a quick demo
python scripts/quick_demo.py

# Test historical data loader
python scripts/test_historical_loader.py

# Convert training model to inference-ready format
python scripts/convert_old_models.py \
    --input_path data/results/experiment_name/final_model.pth \
    --output_path data/results/experiment_name/final_model_safe.pth
```

## 🏗️ How the System Works

### 1. Environment Setup
```python
# src/environment/gbwm_env.py
# Creates a Gymnasium environment that simulates:
# - 16-year investment horizon
# - Multiple financial goals at different times
# - Portfolio choices with realistic returns
# - Wealth evolution through market dynamics
```

### 2. Neural Networks
```python
# src/models/policy_network.py - The "Brain" that makes decisions
class PolicyNetwork:
    # Input: [time, wealth] (normalized)
    # Output: [goal_probabilities, portfolio_probabilities]
    # Architecture: Shared backbone → Two specialized heads
    
# src/models/value_network.py - Evaluates how good situations are
class ValueNetwork:
    # Input: [time, wealth]
    # Output: Expected future utility (single number)
```

### 3. PPO Agent
```python
# src/models/ppo_agent.py - The complete learning system
class PPOAgent:
    # Collects 4,800 trajectories (16 years each = 76,800 decisions)
    # Learns from these experiences using PPO algorithm
    # Updates both policy and value networks
```

### 4. Training Pipeline
```python
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
```

## 📈 Historical Data Details

### Data Coverage
- **Time Period**: 1970-2023 (54 years)
- **Frequency**: Annual returns
- **Sequences**: 39 unique 16-year windows
- **Asset Classes**: US Bonds, US Stocks, International Stocks

### Data Generation Process
1. **Stock Returns**: Geometric Brownian Motion with realistic parameters
2. **Bond Returns**: Treasury yields with duration effects
3. **Correlations**: Based on academic literature (Das et al. 2024)
4. **Portfolio Construction**: Efficient frontier weights across risk levels

### Training with Historical Data
- Each episode randomly selects one of 39 historical sequences
- 4,800 episodes per batch → ~123 uses of each sequence
- Exposes agent to diverse market conditions (bull markets, crashes, recoveries)

## 🧪 Testing

```bash
# Run all tests
pytest tests/

# Run with coverage
pytest tests/ --cov=src --cov-report=html

# Test historical data functionality
python scripts/test_historical_loader.py

# Test training pipeline
python scripts/test_training.py
```

## 📄 License

This project is licensed under the MIT License.

## 📚 Citation

If you use this code in your research, please cite the original paper by Das et al.

---

⭐ **Star this repository if you find it helpful!**