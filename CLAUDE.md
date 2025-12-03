# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This repository implements Goals-Based Wealth Management (GBWM) using both **Dynamic Programming** (optimal theoretical solution) and **Proximal Policy Optimization** (PPO) reinforcement learning. The system solves single-goal and multi-objective financial planning, with RL achieving 94-98% of optimal DP solutions while scaling linearly with the number of goals.

## Common Commands

### Training Models
```bash
# Train with paper defaults (4 goals, simulation mode)
python experiments/run_training.py --num_goals 4

# Train with historical market data (recommended for realistic scenarios)
python experiments/run_training.py --num_goals 4 --data_mode historical

# Custom training configuration
python experiments/run_training.py --num_goals 8 --batch_size 4800 --learning_rate 0.01

# Quick test training
python experiments/run_training.py --num_goals 4 --timesteps 100000 --experiment_name "quick_test"

# Paper replication
python experiments/run_training.py --num_goals 4 --experiment_name "paper_replication"

# Historical mode with custom date range (1980-2020)
python experiments/run_training.py --data_mode historical --num_goals 4 \
    --historical_start_date 1980-01-01 --historical_end_date 2020-12-31
```

### Model Evaluation
```bash
# Basic evaluation
python experiments/run_evaluation.py --model_path data/models/production/gbwm_4goals_bs4800_lr0.01.pth --num_goals 4

# Full evaluation with benchmarks
python experiments/run_evaluation.py \
    --model_path data/results/gbwm_4goals_bs4800_lr0.01/final_model_safe.pth \
    --num_goals 4 \
    --num_episodes 10000 \
    --compare_benchmarks \
    --analyze_trajectory
```

### Interactive Inference (AI Financial Advisor)
```bash
# Interactive mode
python experiments/run_inference.py --model_path data/models/production/gbwm_4goals_bs4800_lr0.01.pth

# Batch processing
python experiments/run_inference.py --model_path {model_path} --mode batch
```

### Dynamic Programming (Theoretical Optimal)
```bash
# Quick DP test (optimized for speed)
python test_dp_simple.py

# Paper base case replication (W(0)=$100k → G=$200k in 10 years)
python scripts/test_dp_base_case.py

# Full DP algorithm test suite (exact paper parameters)
python scripts/test_dp_algorithm.py

# Run DP with custom parameters
python experiments/run_dp_algorithm.py --initial_wealth 150000 --goal_wealth 300000 --time_horizon 15

# Run complete DP experiment
python experiments/run_dp_algorithm.py --initial_wealth 100000 --goal_wealth 200000 --experiment_name "dp_baseline"

# Compare DP vs RL performance
python experiments/compare_dp_rl.py --dp_results data/results/dp_baseline --rl_model data/results/gbwm_4goals_bs4800_lr0.01/final_model_safe.pth
```

### Development & Testing
```bash
# Install in development mode
pip install -e .

# Setup data directories
python scripts/setup_data_structure.py

# Quick system demo
python scripts/quick_demo.py

# Test training pipeline
python scripts/test_training.py

# Run tests
pytest tests/ --cov=src --cov-report=html

# Code formatting
black src/ experiments/ scripts/ tests/
```

## Architecture Overview

### Core Components

#### Dynamic Programming (Theoretical Optimal)
1. **GBWMDynamicProgramming** (`src/algorithms/dynamic_programming.py`): Optimal strategy solver
   - Implements Das et al. (2019) paper algorithm exactly
   - Bellman equation backward recursion with efficient frontier portfolios
   - Calculates V(Wi,t) value function and μ*(Wi,t) optimal policy
   - Provides theoretical benchmark: 66.9% success for base case ($100k → $200k in 10 years)

#### Reinforcement Learning (Scalable Learning)
1. **GBWMEnvironment** (`src/environment/gbwm_env.py`): Custom Gymnasium environment
   - State: `[normalized_time, normalized_wealth]` (2D continuous)
   - Actions: `[goal_decision, portfolio_choice]` (multi-discrete: binary + 15 portfolios)
   - Simulates 16-year investment horizon with geometric Brownian motion

2. **Multi-Head PolicyNetwork** (`src/models/policy_network.py`): Coordinated decision making
   - Shared backbone: 2 → 64 → 64 features
   - Goal head: Skip/take goal probabilities
   - Portfolio head: 15 portfolio selection probabilities

3. **PPOAgent** (`src/models/ppo_agent.py`): Complete RL training system
   - Collects 4,800 trajectories per batch (76,800 decisions)
   - Uses PPO with clipped surrogate objective
   - Implements Generalized Advantage Estimation (GAE)

4. **GBWMTrainer** (`src/training/trainer.py`): Training orchestration
   - Manages complete training pipeline
   - Handles checkpointing, logging, and evaluation
   - Organizes results in `/data/results/{experiment_name}/`

### Training Pipeline Flow
```
Data Collection → PPO Learning → Model Update
    ↓                ↓              ↓
4,800 episodes → 4 PPO epochs → LR decay
(16 years each)   (256 mini-batch)
    ↓                ↓              ↓  
76,800 decisions → GAE advantages → Repeat 10x
```

### Key Algorithms
- **Wealth Evolution**: Geometric Brownian motion with portfolio-specific μ/σ
- **Goal Dynamics**: Cost = 10 × 1.08^t, Utility = 10 + t
- **Portfolio Selection**: 15 efficient frontier portfolios (5.26%-8.86% return, 3.74%-19.54% volatility)
- **PPO**: ε=0.50 clipping, entropy bonus, GAE with λ=0.95

## Configuration System

### Training Hyperparameters (`config/training_config.py`)
```python
batch_size: 4800      # Trajectories per PPO update (paper setting)
learning_rate: 0.01   # Initial LR with linear decay to 0
clip_epsilon: 0.50    # PPO clipping parameter
n_neurons: 64         # Hidden layer size
time_horizon: 16      # Investment period (years)
```

### Environment Parameters (`config/environment_config.py`)
```python
goal_years: [4,8,12,16]    # When goals become available
initial_wealth: 120000     # Base wealth (scaled by num_goals^0.85)
goal_cost_growth: 1.08     # 8% annual cost inflation
```

### Experiment Naming Convention
Models saved as: `gbwm_{num_goals}goals_bs{batch_size}_lr{learning_rate}`
- Results: `/data/results/{experiment_name}/`
- Production models: `/data/models/production/`

## Historical Data Mode

### Two Training Modes Available
1. **Simulation Mode** (default): Synthetic data using Geometric Brownian Motion
2. **Historical Mode** (new): 54 years of market data patterns (1970-2023)

### Historical Data Specifications
- **Coverage**: 54 years of annual market data (1970-2023)
- **Sequences**: 39 unique 16-year windows for training diversity
- **Asset Classes**: US Bonds, US Stocks, International Stocks
- **Realism**: Proper duration modeling, correlation structures from academic literature

### Episode Generation with Historical Data
- 4,800 episodes per batch randomly sample from 39 sequences
- Each sequence gets used ~123 times per batch
- Provides exposure to diverse market conditions:
  - Bull markets (1980s-1990s)
  - Financial crises (2008-2009) 
  - High inflation periods (late 1970s)
  - Recent market conditions (2020s)

### Historical Data Bug Fix (v1.1)
Fixed critical time scale mismatch where historical data was incorrectly using monthly periods instead of annual. The system now properly:
- Uses annual time steps matching the 16-year model horizon
- Generates 54 years of realistic annual market data
- Provides sufficient sequences (39) for meaningful training diversity

## Data Organization

### Model Files
- `final_model.pth`: Training checkpoint (includes optimizer state)
- `final_model_safe.pth`: Inference-ready model (weights only)
- `config.json`: Complete experiment configuration
- `training_history.json`: Metrics from all training iterations

### Portfolio Parameters (`data/processed/portfolio_parameters/`)
Pre-calculated efficient frontier portfolios with risk/return profiles:
- Portfolio 0: 5.26% return, 3.74% volatility (conservative)
- Portfolio 14: 8.86% return, 19.54% volatility (aggressive)

### Results Structure
```
data/results/{experiment_name}/
├── final_model_safe.pth     # Use for inference
├── config.json              # Full experiment config  
├── training_history.json    # Training metrics
├── evaluation/              # Benchmark results
│   ├── evaluation_results.json
│   └── trajectory_analysis.json
└── checkpoints/            # Training checkpoints
    ├── checkpoint_0000.pth
    └── checkpoint_0000_info.json
```

## Key Usage Patterns

### Typical Development Workflow
1. **Quick validation**: `python scripts/test_training.py`
2. **Full training**: `python experiments/run_training.py --num_goals 4`
3. **Evaluation**: Use `final_model_safe.pth` for inference
4. **Deployment**: Move successful models to `/data/models/production/`

### Performance Expectations
Based on paper replication:
- Training time: ~32 seconds (independent of goal count)
- RL efficiency vs optimal DP: 94-98%
- Scalability: Linear with number of goals (1, 2, 4, 8, 16 supported)
- Historical sequences: 39 unique 16-year market patterns for diverse training

### Common Model Loading
```python
# For inference
model_path = "data/results/experiment_name/final_model_safe.pth"

# For continued training  
model_path = "data/results/experiment_name/final_model.pth"

# Production deployment
model_path = "data/models/production/gbwm_4goals_bs4800_lr0.01.pth"
```

## System Requirements

- Python 3.8+, PyTorch 2.0+, Gymnasium 0.29+
- Training: ~32 seconds on modern CPU
- Memory: ~8GB recommended for large batch training
- GPU: Optional (training is fast on CPU)

The codebase enables rapid experimentation with multi-objective financial planning while maintaining research reproducibility and production deployment capabilities.