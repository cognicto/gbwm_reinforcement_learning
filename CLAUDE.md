# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This repository implements Goals-Based Wealth Management (GBWM) using Proximal Policy Optimization (PPO) reinforcement learning. The system solves multi-objective financial planning with competing goals, achieving 94-98% of optimal solutions while scaling linearly with the number of goals.

## Common Commands

### Training Models
```bash
# Train with paper defaults (4 goals)
python experiments/run_training.py --num_goals 4

# Custom training configuration
python experiments/run_training.py --num_goals 8 --batch_size 4800 --learning_rate 0.01

# Quick test training
python experiments/run_training.py --num_goals 4 --timesteps 100000 --experiment_name "quick_test"

# Paper replication
python experiments/run_training.py --num_goals 4 --experiment_name "paper_replication"
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