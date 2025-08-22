# GBWM PPO Training and Evaluation Design - Complete Simulation Example

## Overview

This document illustrates the complete training and evaluation cycle of the Goals-Based Wealth Management (GBWM) PPO agent through a detailed simulation example. We follow one complete training run from initialization to evaluation.

## System Architecture

```
┌─────────────────────────────────────────────────────────────────────────────────────┐
│                          GBWM PPO SYSTEM ARCHITECTURE                              │
├─────────────────────────────────────────────────────────────────────────────────────┤
│                                                                                     │
│  ┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐                │
│  │ Configuration   │    │ Environment     │    │ Neural Networks │                │
│  │                 │    │                 │    │                 │                │
│  │ • Training      │    │ • GBWM State    │    │ • Policy Net    │                │
│  │ • Environment   │    │ • Goal Schedule │    │ • Value Net     │                │
│  │ • Model         │    │ • Portfolio     │    │ • Optimizers    │                │
│  │                 │    │   Universe      │    │                 │                │
│  └─────────────────┘    └─────────────────┘    └─────────────────┘                │
│           │                       │                       │                        │
│           └───────────────────────┼───────────────────────┘                        │
│                                   │                                                │
│  ┌─────────────────────────────────┼─────────────────────────────────────────────┐  │
│  │                        TRAINING PIPELINE                                      │  │
│  │                                 │                                             │  │
│  │  ┌─────────────┐    ┌─────────────┐    ┌─────────────┐    ┌─────────────┐   │  │
│  │  │ Data        │    │ PPO         │    │ Evaluation  │    │ Model       │   │  │
│  │  │ Collection  │───▶│ Learning    │───▶│ & Testing   │───▶│ Saving      │   │  │
│  │  │             │    │             │    │             │    │             │   │  │
│  │  └─────────────┘    └─────────────┘    └─────────────┘    └─────────────┘   │  │
│  └─────────────────────────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────────────────────────┘
```

## Training Configuration Example

```python
# Complete training configuration for our example
training_config = {
    # PPO Hyperparameters
    'n_traj': 500,                    # 500 training iterations  
    'batch_size': 2048,               # 2048 experiences per iteration
    'learning_rate': 0.0003,          # Initial learning rate
    'clip_epsilon': 0.2,              # PPO clipping parameter
    'ppo_epochs': 4,                  # Train on each batch 4 times
    'mini_batch_size': 256,           # Mini-batch size (2048/256 = 8 mini-batches)
    'gamma': 0.99,                    # Discount factor
    'gae_lambda': 0.95,               # GAE lambda parameter
    'entropy_coeff': 0.01,            # Entropy regularization
    'max_grad_norm': 0.5,             # Gradient clipping
    
    # Network Architecture
    'hidden_dim': 64,                 # Hidden layer size
    'goal_action_dim': 2,             # Skip/Take goal
    'portfolio_action_dim': 15,       # 15 portfolios on efficient frontier
    
    # Environment Configuration
    'time_horizon': 10,               # 10-year investment period
    'initial_wealth': 500000,         # $500k starting wealth
    'max_wealth': 2000000,            # $2M wealth cap for normalization
    'goal_years': [2, 4, 6, 8],       # Goals available at years 2, 4, 6, 8
    'data_mode': 'simulation'         # Use GBM simulation
}
```

## Example Training Episode Walkthrough

### Episode Setup
```python
# Initialize episode
episode_id = 1247  # Example episode during training iteration 85
initial_state = {
    'time': 0,
    'wealth': 500000,
    'goals_available': [2, 4, 6, 8],
    'goals_taken': [],
    'total_utility': 0.0
}

# Normalized state for neural networks
normalized_state = [0.0, 1.0]  # [time/10, wealth/500000]
```

### Step-by-Step Episode Progression

#### Step 0: Time=0, Wealth=$500k
```python
# Environment State
state = [0.0, 1.0]  # [normalized_time, normalized_wealth]
goal_available = False  # No goal at time 0

# Policy Network Forward Pass
policy_net_output = {
    'goal_probs': [0.8, 0.2],     # 80% skip, 20% take (no goal available anyway)
    'portfolio_probs': [0.02, 0.05, 0.08, 0.12, 0.18, 0.22, 0.15, 0.10, 0.05, 0.02, 0.01, 0.0, 0.0, 0.0, 0.0]
}

# Action Sampling
action = [0, 5]  # Skip goal (forced), choose portfolio 5 (moderate-aggressive)
log_prob = log(0.8) + log(0.22) = -0.223 + (-1.514) = -1.737

# Value Network Prediction
value_estimate = 28.5  # "Expect 28.5 total utility from this state"

# Environment Step
portfolio_return = 1.08  # +8% return (GBM simulation)
new_wealth = 500000 * 1.08 = 540000
reward = 0.0  # No goal to take

# Store Experience
experience_0 = {
    'state': [0.0, 1.0],
    'action': [0, 5],
    'reward': 0.0,
    'log_prob': -1.737,
    'value': 28.5,
    'done': False
}
```

#### Step 2: Time=2, Wealth=$583k (Goal Available!)
```python
# Environment State  
state = [0.2, 1.166]  # [2/10, 583000/500000]
goal_available = True
goal_cost = 80000    # $80k goal cost
goal_utility = 14.0  # 10 base + 2*2 time bonus

# Policy Network Forward Pass (after some training)
policy_net_output = {
    'goal_probs': [0.35, 0.65],   # 35% skip, 65% take (learned goals are valuable)
    'portfolio_probs': [0.05, 0.08, 0.12, 0.15, 0.20, 0.18, 0.12, 0.06, 0.03, 0.01, 0.0, 0.0, 0.0, 0.0, 0.0]
}

# Action Sampling
action = [1, 4]  # Take goal, choose portfolio 4 (moderate)
log_prob = log(0.65) + log(0.20) = -0.431 + (-1.609) = -2.040

# Value Network Prediction
value_estimate = 22.3  # "Expect 22.3 total utility from this state"

# Environment Step
wealth_after_goal = 583000 - 80000 = 503000  # Pay goal cost
portfolio_return = 1.06  # +6% return
new_wealth = 503000 * 1.06 = 533180
reward = 14.0  # Goal utility achieved

# Store Experience
experience_2 = {
    'state': [0.2, 1.166],
    'action': [1, 4],
    'reward': 14.0,
    'log_prob': -2.040,
    'value': 22.3,
    'done': False
}
```

#### Step 4: Time=4, Wealth=$487k (Goal Available)
```python
# Environment State
state = [0.4, 0.974]  # [4/10, 487000/500000]
goal_available = True
goal_cost = 100000   # $100k goal cost  
goal_utility = 18.0  # 10 base + 2*4 time bonus

# Policy Network Forward Pass
policy_net_output = {
    'goal_probs': [0.45, 0.55],   # 45% skip, 55% take
    'portfolio_probs': [0.08, 0.12, 0.18, 0.22, 0.20, 0.12, 0.06, 0.02, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
}

# Action Sampling  
action = [0, 6]  # Skip goal (can't afford comfortably), moderate-aggressive portfolio
log_prob = log(0.45) + log(0.06) = -0.798 + (-2.813) = -3.611

# Value Network Prediction
value_estimate = 15.8  # "Expect 15.8 total utility from this state"

# Environment Step  
portfolio_return = 1.12  # +12% good return
new_wealth = 487000 * 1.12 = 545440
reward = 0.0  # Skipped goal

# Store Experience
experience_4 = {
    'state': [0.4, 0.974],
    'action': [0, 6],
    'reward': 0.0,
    'log_prob': -3.611,
    'value': 15.8,
    'done': False
}
```

#### Step 6: Time=6, Wealth=$612k (Goal Available)
```python
# Environment State
state = [0.6, 1.224]  # [6/10, 612000/500000]
goal_available = True
goal_cost = 120000   # $120k goal cost
goal_utility = 22.0  # 10 base + 2*6 time bonus

# Policy Network Forward Pass
policy_net_output = {
    'goal_probs': [0.25, 0.75],   # 25% skip, 75% take (high wealth, affordable)
    'portfolio_probs': [0.06, 0.10, 0.15, 0.22, 0.25, 0.15, 0.05, 0.02, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
}

# Action Sampling
action = [1, 3]  # Take goal, conservative-moderate portfolio 
log_prob = log(0.75) + log(0.22) = -0.288 + (-1.514) = -1.802

# Value Network Prediction
value_estimate = 25.1  # "Expect 25.1 total utility from this state"

# Environment Step
wealth_after_goal = 612000 - 120000 = 492000
portfolio_return = 1.04  # +4% conservative return
new_wealth = 492000 * 1.04 = 511680
reward = 22.0  # Goal utility achieved

# Store Experience
experience_6 = {
    'state': [0.6, 1.224],
    'action': [1, 3],
    'reward': 22.0,
    'log_prob': -1.802,
    'value': 25.1,
    'done': False
}
```

#### Step 8: Time=8, Wealth=$524k (Final Goal)
```python
# Environment State
state = [0.8, 1.048]  # [8/10, 524000/500000]
goal_available = True
goal_cost = 140000   # $140k goal cost
goal_utility = 26.0  # 10 base + 2*8 time bonus

# Policy Network Forward Pass
policy_net_output = {
    'goal_probs': [0.15, 0.85],   # 15% skip, 85% take (final opportunity)
    'portfolio_probs': [0.12, 0.18, 0.25, 0.20, 0.15, 0.08, 0.02, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
}

# Action Sampling
action = [1, 2]  # Take goal, conservative portfolio
log_prob = log(0.85) + log(0.25) = -0.163 + (-1.386) = -1.549

# Value Network Prediction  
value_estimate = 26.8  # "Expect 26.8 total utility from this state"

# Environment Step
wealth_after_goal = 524000 - 140000 = 384000
portfolio_return = 1.03  # +3% conservative return
new_wealth = 384000 * 1.03 = 395520
reward = 26.0  # Goal utility achieved

# Store Experience
experience_8 = {
    'state': [0.8, 1.048],
    'action': [1, 2],
    'reward': 26.0,
    'log_prob': -1.549,
    'value': 26.8,
    'done': False
}
```

#### Step 10: Episode End
```python
# Final steps (9, 10) - no more goals available
# Agent focuses on wealth preservation
# Episode terminates at step 10

# Final Episode Summary
episode_summary = {
    'total_steps': 10,
    'final_wealth': 415000,
    'goals_taken': [2, 6, 8],  # Took 3 out of 4 available goals
    'total_utility': 14.0 + 22.0 + 26.0 = 62.0,
    'episode_reward': 62.0
}
```

## PPO Learning Phase

### Batch Processing
```python
# After collecting 2048 experiences from multiple episodes
batch_data = {
    'states': torch.tensor([
        [0.0, 1.0],    # Experience 0
        [0.2, 1.166],  # Experience 2  
        [0.4, 0.974],  # Experience 4
        [0.6, 1.224],  # Experience 6
        [0.8, 1.048],  # Experience 8
        # ... 2043 more experiences
    ]),
    'actions': torch.tensor([
        [0, 5], [1, 4], [0, 6], [1, 3], [1, 2], # ... more actions
    ]),
    'old_log_probs': torch.tensor([
        -1.737, -2.040, -3.611, -1.802, -1.549, # ... more log_probs
    ]),
    'rewards': torch.tensor([
        0.0, 14.0, 0.0, 22.0, 26.0, # ... more rewards
    ]),
    'values': torch.tensor([
        28.5, 22.3, 15.8, 25.1, 26.8, # ... more values
    ]),
    'dones': torch.tensor([
        False, False, False, False, False, # ... more dones
    ])
}
```

### GAE Computation Example
```python
# Compute advantages for our example experiences
# Working backwards from episode end

# Step 8: Final goal (terminal advantage)
advantage_8 = reward_8 + gamma * 0 - value_8  # No future value (episode ends soon)
advantage_8 = 26.0 + 0.99 * 0 - 26.8 = -0.8

# Step 6: Middle goal  
advantage_6 = reward_6 + gamma * value_7 - value_6
advantage_6 = 22.0 + 0.99 * 18.2 - 25.1 = 14.9  # Much better than expected

# Step 4: Skipped goal
advantage_4 = reward_4 + gamma * value_5 - value_4  
advantage_4 = 0.0 + 0.99 * 20.1 - 15.8 = 4.1  # Skipping paid off (enabled future goals)

# Step 2: First goal
advantage_2 = reward_2 + gamma * value_3 - value_2
advantage_2 = 14.0 + 0.99 * 19.5 - 22.3 = 11.0  # Good decision

# Step 0: Initial portfolio choice
advantage_0 = reward_0 + gamma * value_1 - value_0
advantage_0 = 0.0 + 0.99 * 25.8 - 28.5 = -3.0  # Slightly worse than expected
```

### Policy Update Example  
```python
# Mini-batch processing (256 samples from 2048)
mini_batch = batch_data[0:256]  # First mini-batch

# Forward pass with current policy (after 85 iterations of training)
new_log_probs, entropy = policy_net.evaluate_actions(
    mini_batch['states'], 
    mini_batch['actions']
)

# Example: Experience 2 (take goal at time=2)
old_log_prob_2 = -2.040  # From data collection
new_log_prob_2 = -1.850  # Current policy is more confident
advantage_2 = 11.0       # Action was good

# PPO computation
ratio_2 = exp(new_log_prob_2 - old_log_prob_2) = exp(-1.850 - (-2.040)) = exp(0.19) = 1.21
surr1_2 = ratio_2 * advantage_2 = 1.21 * 11.0 = 13.31
surr2_2 = clip(ratio_2, 0.8, 1.2) * advantage_2 = 1.2 * 11.0 = 13.20
policy_objective_2 = min(surr1_2, surr2_2) = 13.20  # Clipped for safety

# Policy loss (negative because we maximize objective)
policy_loss = -mean(policy_objectives) = -8.75

# Backpropagation
policy_loss.backward()  # Compute gradients
clip_grad_norm_(policy_net.parameters(), max_norm=0.5)  # Clip gradients
policy_optimizer.step()  # Update weights
```

### Value Update Example
```python
# Value network training on same mini-batch
target_returns = advantages + old_values  # Reconstruct actual returns
target_returns_2 = 11.0 + 22.3 = 33.3   # What actually happened

# Current value prediction
new_value_2 = value_net(state_2) = 24.8  # Updated prediction

# Value loss (MSE)
value_loss = MSE(new_values, target_returns)
value_loss_2 = (24.8 - 33.3)² = 72.25

# Backpropagation  
value_loss.backward()
value_optimizer.step()
```

## Training Progress Tracking

### Iteration 85 Metrics
```python
iteration_85_metrics = {
    'policy_loss': -8.75,
    'value_loss': 4.32,
    'mean_episode_reward': 58.3,
    'mean_episode_length': 10.0,
    'goal_success_rate': 0.73,  # 73% of available goals taken
    'mean_advantage': 2.1,
    'explained_variance': 0.85,
    'learning_rate': 0.000255,  # Decayed from 0.0003
    'total_timesteps': 173600   # 85 * 2048
}
```

### Learning Progression
```python
# Training progression over 500 iterations
training_history = {
    'iterations': [0, 50, 100, 150, 200, 250, 300, 350, 400, 450, 500],
    'mean_rewards': [12.5, 28.3, 42.1, 54.7, 61.2, 65.8, 68.1, 69.4, 70.2, 70.6, 70.8],
    'goal_success_rates': [0.25, 0.45, 0.62, 0.71, 0.76, 0.79, 0.81, 0.82, 0.83, 0.83, 0.83],
    'value_losses': [25.3, 12.1, 6.8, 4.2, 3.1, 2.5, 2.1, 1.9, 1.8, 1.7, 1.7],
    'explained_variances': [0.15, 0.42, 0.65, 0.78, 0.84, 0.87, 0.89, 0.90, 0.91, 0.91, 0.91]
}
```

## Model Evaluation

### Evaluation Configuration
```python
evaluation_config = {
    'num_episodes': 1000,
    'deterministic': True,    # Use argmax instead of sampling
    'random_seed': 42,
    'env_config': training_config['env_config']
}
```

### Evaluation Episode Example
```python
# Evaluation episode with trained policy (deterministic)
eval_episode = {
    'states': [
        [0.0, 1.0],   # Time=0, Wealth=$500k
        [0.2, 1.16],  # Time=2, Wealth=$580k
        [0.4, 1.05],  # Time=4, Wealth=$525k
        [0.6, 1.18],  # Time=6, Wealth=$590k
        [0.8, 1.12],  # Time=8, Wealth=$560k
    ],
    
    'deterministic_actions': [
        [0, 5],  # Skip (no goal), moderate-aggressive
        [1, 4],  # Take goal, moderate
        [1, 3],  # Take goal, conservative-moderate
        [1, 2],  # Take goal, conservative
        [1, 1],  # Take goal, very conservative
    ],
    
    'rewards': [0.0, 15.0, 18.0, 22.0, 26.0],
    'total_utility': 81.0,
    'final_wealth': 485000
}
```

### Benchmark Comparison
```python
# Evaluation against benchmark strategies
benchmark_results = {
    'trained_ppo': {
        'mean_reward': 70.8,
        'std_reward': 12.3,
        'goal_success_rate': 0.83,
        'final_wealth_mean': 520000,
        'episodes': 1000
    },
    
    'greedy_goal_taking': {
        'mean_reward': 45.2,
        'std_reward': 18.7,
        'goal_success_rate': 0.95,  # Takes all affordable goals
        'final_wealth_mean': 380000,  # Lower due to aggressive goal taking
        'episodes': 1000
    },
    
    'buy_and_hold': {
        'mean_reward': 28.4,
        'std_reward': 22.1,
        'goal_success_rate': 0.35,   # Only affordable goals
        'final_wealth_mean': 780000,  # High wealth, low utility
        'episodes': 1000
    },
    
    'random_policy': {
        'mean_reward': 18.9,
        'std_reward': 15.4,
        'goal_success_rate': 0.42,
        'final_wealth_mean': 450000,
        'episodes': 1000
    }
}
```

## Model Persistence

### Saving Trained Model
```python
# Model checkpoint after 500 iterations
checkpoint = {
    'iteration': 500,
    'policy_state_dict': policy_net.state_dict(),
    'value_state_dict': value_net.state_dict(),
    'optimizer_states': {
        'policy': policy_optimizer.state_dict(),
        'value': value_optimizer.state_dict()
    },
    'training_config': training_config,
    'training_history': training_history,
    'evaluation_results': benchmark_results,
    'model_metadata': {
        'total_timesteps': 1024000,  # 500 * 2048
        'training_time_hours': 3.2,
        'convergence_iteration': 420,
        'best_eval_reward': 71.1
    }
}

# Save to file
torch.save(checkpoint, 'models/gbwm_ppo_trained_500iter.pt')
```

## Key Design Insights

### 1. **Goal-Portfolio Coordination**
The agent learns sophisticated strategies:
- **High wealth + Early time**: Aggressive portfolios acceptable
- **Low wealth + Goal available**: Conservative portfolios to preserve options
- **Late time + Valuable goal**: Conservative portfolios to ensure affordability

### 2. **Temporal Strategy**
- **Early goals**: Skip low-utility goals for growth
- **Middle goals**: Balance utility vs. future opportunities  
- **Late goals**: Prioritize high-utility goals with wealth preservation

### 3. **Risk Management**
- **Wealth buffer**: Maintain cushion for goal opportunities
- **Portfolio adaptation**: Adjust risk based on goal proximity and wealth level
- **Utility maximization**: Balance immediate vs. future goal opportunities

### 4. **Learning Efficiency**
- **PPO stability**: Clipped updates prevent policy collapse
- **GAE credit assignment**: Proper attribution of rewards to actions
- **Value function**: Accurate state evaluation improves advantage estimation

## Performance Characteristics

### Training Efficiency
- **Convergence**: ~400 iterations for policy stabilization
- **Sample efficiency**: 820k timesteps for near-optimal performance
- **Computational cost**: ~3 hours on modern hardware

### Final Performance
- **Goal success rate**: 83% of available goals taken optimally
- **Utility achievement**: 70.8 average utility (vs. 18.9 random baseline)
- **Wealth management**: Maintains healthy wealth while maximizing utility
- **Strategy sophistication**: Context-dependent goal and portfolio decisions

This design demonstrates how PPO can learn complex, multi-objective financial strategies through experience, balancing competing objectives of wealth growth and goal achievement in a dynamic environment.