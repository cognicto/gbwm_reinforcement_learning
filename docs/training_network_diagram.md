# Goals-Based Wealth Management PPO Training Network Architecture

## Complete System Block Diagram

```
                    GBWM PPO TRAINING SYSTEM ARCHITECTURE
    ═══════════════════════════════════════════════════════════════════════════════════════════

                        DATA SOURCES & PREPROCESSING
    ┌─────────────────────────────────────────────────────────────────────────────────────────┐
    │                           RAW MARKET DATA                                                │
    │  ┌─────────────────┐    ┌─────────────────┐    ┌─────────────────────────────────────┐  │
    │  │  S&P 500 CSV    │    │ Treasury Bonds  │    │   Efficient Frontier Parameters   │  │
    │  │ Date,Close,Vol  │    │ Date,10Y,2Y     │    │  Portfolio weights & returns       │  │
    │  │ 2010-2023       │    │ Yields %        │    │  15 portfolios: Conservative       │  │
    │  │ 168 records     │    │ 168 records     │    │  to Aggressive (5.26%-8.86%)      │  │
    │  └─────────────────┘    └─────────────────┘    └─────────────────────────────────────┘  │
    │           │                       │                              │                       │
    │           ▼                       ▼                              ▼                       │
    │  ┌─────────────────────────────────────────────────────────────────────────────────────┐  │
    │  │                    HISTORICAL DATA LOADER                                            │  │
    │  │  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────────────────────────┐  │  │
    │  │  │ Asset Returns   │  │Portfolio Weights│  │    Portfolio Return Matrix          │  │  │
    │  │  │ US Stocks: pct  │  │[Bonds,US,Intl]  │  │     Shape: (15, 168)               │  │  │
    │  │  │ Bonds: yield/12 │  │Portfolio 0:     │  │ 15 portfolios × 168 monthly returns│  │  │
    │  │  │ Intl: synthetic │  │[0.7, 0.25, 0.05]│  │ Available sequences: 153           │  │  │
    │  │  └─────────────────┘  │Portfolio 14:    │  └─────────────────────────────────────┘  │  │
    │  │                       │[0.07,0.59,0.34] │                                          │  │
    │  │                       └─────────────────┘                                          │  │
    │  └─────────────────────────────────────────────────────────────────────────────────────┘  │
    └─────────────────────────────────────────────────────────────────────────────────────────┘
                                              │
                                              ▼
    ┌─────────────────────────────────────────────────────────────────────────────────────────┐
    │                             ENVIRONMENT SYSTEM                                           │
    │  ┌─────────────────────────────────────────────────────────────────────────────────────┐  │
    │  │                         GBWM ENVIRONMENT                                            │  │
    │  │                                                                                     │  │
    │  │  State Space: [normalized_time, normalized_wealth]                                  │  │
    │  │  • time ∈ [0,1]: current_year / 16                                                 │  │
    │  │  • wealth ∈ [0,1]: current_wealth / max_wealth                                     │  │
    │  │                                                                                     │  │
    │  │  Action Space: [goal_decision, portfolio_choice]                                   │  │
    │  │  • goal_decision ∈ {0,1}: skip=0, take=1                                          │  │
    │  │  • portfolio_choice ∈ {0,1,2,...,14}: 15 portfolio options                       │  │
    │  │                                                                                     │  │
    │  │  ┌─────────────────┐    ┌─────────────────┐    ┌─────────────────────────────┐    │  │
    │  │  │ Goal Dynamics   │    │ Wealth Evolution│    │   Reward Function           │    │  │
    │  │  │ Years: [4,8,12,16]   │ Mode: Historical│    │ Goal Utility = 10 + year    │    │  │
    │  │  │ Cost: 10k×1.08^t│    │ or Simulation   │    │ Only when goal taken        │    │  │
    │  │  │ Utility: 10+year│    │ Historical: use │    │ Max possible: 54 points     │    │  │
    │  │  │                 │    │ real returns    │    │ (10+4)+(10+8)+(10+12)+(10+16)│    │  │
    │  │  └─────────────────┘    └─────────────────┘    └─────────────────────────────┘    │  │
    │  └─────────────────────────────────────────────────────────────────────────────────────┘  │
    └─────────────────────────────────────────────────────────────────────────────────────────┘
                                              │
                                              ▼
    ┌─────────────────────────────────────────────────────────────────────────────────────────  ┐
    │                          DATA COLLECTION PHASE                                            │
    │                                                                                           │
    │  ┌─────────────────────────────────────────────────────────────────────────────────────┐  │
    │  │                    TRAJECTORY COLLECTION LOOP                                       │  │
    │  │                      (4,800 episodes × 16 steps)                                    │  │
    │  │                                                                                     │  │
    │  │  For episode in range(4800):                                                        │  │
    │  │    obs, _ = env.reset()  # Load random historical sequence                          │  │
    │  │    for step in range(16):                                                           │  │
    │  │      ┌─────────────────────────────────────────────────────────────────────────┐    │  │
    │  │      │                    FORWARD PASS                                         │    │  │
    │  │      │                                                                         │    │  │
    │  │      │  Input: obs = [time_norm, wealth_norm]                                  │    │  │
    │  │      │  Example: [0.25, 0.045] = [4/16, $450k/$10M]                            │    │  │
    │  │      │                             │                                           │    │  │
    │  │      │                             ▼                                           │    │  │
    │  │      │                    state_tensor                                         │    │  │
    │  │      │                   Shape: (1, 2)                                         │    │  │
    │  │      │                                                                         │    │  │
    │  │      │              ┌─────────────────┬─────────────────┐                      │    │  │
    │  │      │              ▼                 ▼                 ▼                      │    │  │
    │  │      │    ┌─────────────────┐    ┌─────────────────┐    ┌──────────────┐       │    │  │
    │  │      │    │ POLICY NETWORK  │    │ VALUE NETWORK   │    │ ACTION SAMPLE│       │    │  │
    │  │      │    │                 │    │                 │    │              │       │    │  │
    │  │      │    │ Input: (1,2)    │    │ Input: (1,2)    │    │ Categorical  │       │    │  │
    │  │      │    │ ┌─────────────┐ │    │ ┌─────────────┐ │    │ Sampling     │       │    │  │
    │  │      │    │ │Shared Backbone     │ │Layer1: 2→64 │ │    │              │       │    │  │
    │  │      │    │ │ Layer1: 2→64│ │    │ │ReLU         │ │    │              │       │    │  │
    │  │      │    │ │ Layer2:64→64│ │    │ │Layer2:64→64 │ │    │              │       │    │  │
    │  │      │    │ │ ReLU        │ │    │ │ReLU         │ │    │              │       │    │  │
    │  │      │    │ └─────────────┘ │    │ │Layer3:64→1  │ │    │              │       │    │  │
    │  │      │    │        │        │    │ └─────────────┘ │    │              │       │    │  │
    │  │      │    │        ▼        │    │                 │    │              │       │    │  │
    │  │      │    │ ┌─────┐  ┌─────┐│    │ Output: (1,1)   │    │              │       │    │  │
    │  │      │    │ │Goal │  │Port ││    │ Value Estimate  │    │              │       │    │  │
    │  │      │    │ │Head │  │Head ││    │ Example: 18.4   │    │              │       │    │  │
    │  │      │    │ │64→2 │  │64→15││    │                 │    │              │       │    │  │
    │  │      │    │ │Soft │  │Soft ││    └─────────────────┘    │              │       │    │  │
    │  │      │    │ │max  │  │max  ││                           │              │       │    │  │
    │  │      │    │ └─────┘  └─────┘│                           │              │       │    │  │
    │  │      │    │    │        │   │                           │              │       │    │  │
    │  │      │    │    ▼        ▼   │                           │              │       │    │  │
    │  │      │    │ [0.85,    [0.05,│                           │              │       │    │  │
    │  │      │    │  0.15]   0.08,..│                           │              │       │    │  │
    │  │      │    │ Goal     Portfolio                          │              │       │    │  │
    │  │      │    │ Probs    Probs  │                           │              │       │    │  │
    │  │      │    │ (1,2)    (1,15) │                           │              │       │    │  │
    │  │      │    └─────────────────┘                           └──────────────┘       │    │  │
    │  │      │              │                                                          │    │  │
    │  │      │              ▼                                                          │    │  │
    │  │      │    ┌─────────────────┐                                                  │    │  │
    │  │      │    │ Action Sampling │                                                  │    │  │
    │  │      │    │ goal_action = 0 │                                                  │    │  │
    │  │      │    │ port_action = 7 │                                                  │    │  │
    │  │      │    │ actions = [0,7] │                                                  │    │  │
    │  │      │    │ log_prob = -2.1 │                                                  │    │  │
    │  │      │    └─────────────────┘                                                  │    │  │
    │  │      └─────────────────────────────────────────────────────────────────────────┘    │  │
    │  │                             │                                                       │  │
    │  │                             ▼                                                       │  │
    │  │      ┌─────────────────────────────────────────────────────────────────────────┐    │  │
    │  │      │                   ENVIRONMENT STEP                                      │    │  │
    │  │      │                                                                         │    │  │
    │  │      │  action = [0, 7] → env.step(action)                                     │    │  │
    │  │      │                                                                         │    │  │
    │  │      │  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐          │    │  │
    │  │      │  │ Goal Processing │  │ Wealth Evolution│  │ Reward & State   │         │    │  │
    │  │      │  │ action[0] = 0   │  │ Portfolio: 7    │  │ reward = 0.0     │         │    │  │
    │  │      │  │ Skip goal       │  │ Historical      │  │ next_obs = [0.31,│         │    │  │
    │  │      │  │ (not available) │  │ return: +1.2%   │  │ 0.047]           │         │    │  │
    │  │      │  │ No cost/reward  │  │ New wealth:     │  │ done = False     │         │    │  │
    │  │      │  │                 │  │ $470k           │  │                  │         │    │  │
    │  │      │  └─────────────────┘  └─────────────────┘  └─────────────────┘      │    │  │
    │  │      └─────────────────────────────────────────────────────────────────────────┘    │  │
    │  │                                                                                     │  │
    │  │      # Store experience: (state, action, reward, log_prob, value, next_state)      │  │
    │  │                                                                                     │  │
    │  └─────────────────────────────────────────────────────────────────────────────────────┘  │
    │                                                                                         │
    │  Result: 76,800 experiences = 4,800 episodes × 16 steps                                │
    │                                                                                         │
    │  batch_data = {                                                                         │
    │    'states': tensor(76800, 2),      # All normalized [time, wealth] states           │
    │    'actions': tensor(76800, 2),     # All [goal_decision, portfolio_choice] actions  │
    │    'rewards': tensor(76800),        # All rewards (0 for no goal, utility if taken)  │
    │    'old_log_probs': tensor(76800),  # Log probabilities from policy during collection│
    │    'values': tensor(76800),         # Value estimates from value network             │
    │    'dones': tensor(76800)           # Episode termination flags                      │
    │  }                                                                                      │
    └─────────────────────────────────────────────────────────────────────────────────────────┘
                                              │
                                              ▼
    ┌─────────────────────────────────────────────────────────────────────────────────────────┐
    │                         ADVANTAGE COMPUTATION PHASE                                       │
    │                                                                                         │
    │  ┌─────────────────────────────────────────────────────────────────────────────────────┐  │
    │  │              GENERALIZED ADVANTAGE ESTIMATION (GAE)                                 │  │
    │  │                                                                                     │  │
    │  │  Input: rewards, values, dones, γ=0.99, λ=0.95                                     │  │
    │  │                                                                                     │  │
    │  │  For each episode (working backwards):                                              │  │
    │  │    for t in reversed(range(16)):                                                   │  │
    │  │      if t == 15 or done[t]:                                                        │  │
    │  │        next_value = 0                                                              │  │
    │  │      else:                                                                          │  │
    │  │        next_value = values[t+1]                                                    │  │
    │  │                                                                                     │  │
    │  │      # TD Error                                                                    │  │
    │  │      δ[t] = reward[t] + γ × next_value - values[t]                                │  │
    │  │                                                                                     │  │
    │  │      # GAE                                                                         │  │
    │  │      A[t] = δ[t] + γ × λ × A[t+1]                                                 │  │
    │  │                                                                                     │  │
    │  │      # Returns for Value Function                                                  │  │
    │  │      R[t] = A[t] + values[t]                                                       │  │
    │  │                                                                                     │  │
    │  │  Example for one episode:                                                          │  │
    │  │    rewards = [0, 0, 14, 0, 0, 0, 18, 0, 0, 0, 0, 0, 0, 0, 0, 0]                 │  │
    │  │    values  = [15,16,14,13,12,11,10, 9, 8, 7, 6, 5, 4, 3, 2, 0]                  │  │
    │  │    advantages = [2.1, 1.8, 5.2, -0.7, -1.1, -0.8, 8.5, -0.4, ...]              │  │
    │  │    returns = [17.1, 17.8, 19.2, 12.3, 10.9, 10.2, 18.5, 8.6, ...]              │  │
    │  │                                                                                     │  │
    │  │  # Normalize advantages for stable training                                        │  │
    │  │  advantages_norm = (advantages - mean(advantages)) / (std(advantages) + 1e-8)     │  │
    │  │                                                                                     │  │
    │  └─────────────────────────────────────────────────────────────────────────────────────┘  │
    └─────────────────────────────────────────────────────────────────────────────────────────┘
                                              │
                                              ▼
    ┌─────────────────────────────────────────────────────────────────────────────────────────┐
    │                              PPO LEARNING PHASE                                          │
    │                                                                                         │
    │  ┌─────────────────────────────────────────────────────────────────────────────────────┐  │
    │  │                           PPO UPDATE LOOP                                           │  │
    │  │                     (4 epochs × 300 mini-batches)                                   │  │
    │  │                                                                                     │  │
    │  │  for epoch in range(4):  # Use same data 4 times                                   │  │
    │  │    indices = torch.randperm(76800)  # Shuffle                                       │  │
    │  │    for start in range(0, 76800, 256):  # Mini-batches                              │  │
    │  │      mini_batch = batch_data[indices[start:start+256]]                             │  │
    │  │                                                                                     │  │
    │  │      ┌─────────────────────────────────────────────────────────────────────────┐    │  │
    │  │      │                      FORWARD PASS                                       │    │  │
    │  │      │                                                                         │    │  │
    │  │      │  mb_states = mini_batch['states']        # Shape: (256, 2)             │    │  │
    │  │      │  mb_actions = mini_batch['actions']      # Shape: (256, 2)             │    │  │
    │  │      │  mb_advantages = mini_batch['advantages'] # Shape: (256,)              │    │  │
    │  │      │  mb_returns = mini_batch['returns']      # Shape: (256,)               │    │  │
    │  │      │  mb_old_log_probs = mini_batch['old_log_probs'] # Shape: (256,)        │    │  │
    │  │      │                                                                         │    │  │
    │  │      │                              │                                          │    │  │
    │  │      │                              ▼                                          │    │  │
    │  │      │           ┌─────────────────────────────────────┐                      │    │  │
    │  │      │           │        CURRENT POLICY NETWORK       │                      │    │  │
    │  │      │           │                                     │                      │    │  │
    │  │      │           │  Input: mb_states (256, 2)         │                      │    │  │
    │  │      │           │                │                   │                      │    │  │
    │  │      │           │                ▼                   │                      │    │  │
    │  │      │           │  ┌─────────────────────────────────┐ │                      │    │  │
    │  │      │           │  │     Shared Backbone             │ │                      │    │  │
    │  │      │           │  │  Linear(2, 64) → ReLU          │ │                      │    │  │
    │  │      │           │  │  Linear(64, 64) → ReLU         │ │                      │    │  │
    │  │      │           │  │  Output: (256, 64)             │ │                      │    │  │
    │  │      │           │  └─────────────────────────────────┘ │                      │    │  │
    │  │      │           │                │                   │                      │    │  │
    │  │      │           │       ┌────────┴────────┐          │                      │    │  │
    │  │      │           │       ▼                 ▼          │                      │    │  │
    │  │      │           │  ┌─────────┐      ┌─────────────┐  │                      │    │  │
    │  │      │           │  │Goal Head│      │Portfolio Head  │                      │    │  │
    │  │      │           │  │Lin(64,2)│      │Lin(64,15)   │  │                      │    │  │
    │  │      │           │  │Softmax  │      │Softmax      │  │                      │    │  │
    │  │      │           │  │(256,2)  │      │(256,15)     │  │                      │    │  │
    │  │      │           │  └─────────┘      └─────────────┘  │                      │    │  │
    │  │      │           │      │                  │         │                      │    │  │
    │  │      │           │      ▼                  ▼         │                      │    │  │
    │  │      │           │  goal_probs         port_probs    │                      │    │  │
    │  │      │           │  [0.82,0.18]       [0.05,0.08,..] │                      │    │  │
    │  │      │           │                                     │                      │    │  │
    │  │      │           └─────────────────────────────────────┘                      │    │  │
    │  │      │                              │                                          │    │  │
    │  │      │                              ▼                                          │    │  │
    │  │      │           ┌─────────────────────────────────────┐                      │    │  │
    │  │      │           │        LOG PROB COMPUTATION         │                      │    │  │
    │  │      │           │                                     │                      │    │  │
    │  │      │           │  Extract probs for taken actions:   │                      │    │  │
    │  │      │           │  goal_log_probs = log(goal_probs.   │                      │    │  │
    │  │      │           │    gather(1, mb_actions[:,0]))      │                      │    │  │
    │  │      │           │  port_log_probs = log(port_probs.   │                      │    │  │
    │  │      │           │    gather(1, mb_actions[:,1]))      │                      │    │  │
    │  │      │           │                                     │                      │    │  │
    │  │      │           │  new_log_probs = goal_log_probs +   │                      │    │  │
    │  │      │           │                  port_log_probs     │                      │    │  │
    │  │      │           │  Shape: (256,)                      │                      │    │  │
    │  │      │           │  Example: [-2.15, -1.92, -2.07,..]  │                      │    │  │
    │  │      │           └─────────────────────────────────────┘                      │    │  │
    │  │      └─────────────────────────────────────────────────────────────────────────┘    │  │
    │  │                                                                                     │  │
    │  │      ┌─────────────────────────────────────────────────────────────────────────┐    │  │
    │  │      │                    PPO LOSS COMPUTATION                                 │    │  │
    │  │      │                                                                         │    │  │
    │  │      │  # Probability Ratio                                                   │    │  │
    │  │      │  ratio = exp(new_log_probs - mb_old_log_probs)                         │    │  │
    │  │      │  Shape: (256,)                                                         │    │  │
    │  │      │  Example: [1.05, 0.92, 1.31, 0.87, ...]                              │    │  │
    │  │      │                                                                         │    │  │
    │  │      │  # PPO Clipped Surrogate Objective                                     │    │  │
    │  │      │  surr1 = ratio × mb_advantages                    # Unclipped          │    │  │
    │  │      │  surr2 = clamp(ratio, 0.5, 1.5) × mb_advantages  # Clipped (ε=0.5)   │    │  │
    │  │      │                                                                         │    │  │
    │  │      │  # Policy Loss (negative to maximize)                                  │    │  │
    │  │      │  policy_loss = -mean(min(surr1, surr2))                               │    │  │
    │  │      │  Example: 0.234                                                        │    │  │
    │  │      │                                                                         │    │  │
    │  │      │  # Entropy Bonus (encourage exploration)                               │    │  │
    │  │      │  goal_entropy = -sum(goal_probs × log(goal_probs + 1e-8))             │    │  │
    │  │      │  port_entropy = -sum(port_probs × log(port_probs + 1e-8))             │    │  │
    │  │      │  entropy = goal_entropy + port_entropy                                 │    │  │
    │  │      │  entropy_loss = -0.01 × mean(entropy)                                 │    │  │
    │  │      │                                                                         │    │  │
    │  │      │  # Total Policy Loss                                                   │    │  │
    │  │      │  total_policy_loss = policy_loss + entropy_loss                       │    │  │
    │  │      │  Example: 0.234 + (-0.015) = 0.219                                   │    │  │
    │  │      └─────────────────────────────────────────────────────────────────────────┘    │  │
    │  │                              │                                                     │  │
    │  │                              ▼                                                     │  │
    │  │      ┌─────────────────────────────────────────────────────────────────────────┐    │  │
    │  │      │                   VALUE NETWORK LOSS                                   │    │  │
    │  │      │                                                                         │    │  │
    │  │      │  Input: mb_states (256, 2)                                             │    │  │
    │  │      │                │                                                       │    │  │
    │  │      │                ▼                                                       │    │  │
    │  │      │  ┌─────────────────────────────────────────────┐                      │    │  │
    │  │      │  │           VALUE NETWORK                     │                      │    │  │
    │  │      │  │  Linear(2, 64) → ReLU                      │                      │    │  │
    │  │      │  │  Linear(64, 64) → ReLU                     │                      │    │  │
    │  │      │  │  Linear(64, 1)                             │                      │    │  │
    │  │      │  │  Output: current_values (256, 1)           │                      │    │  │
    │  │      │  │  Example: [17.2, 16.8, 19.5, 12.9, ...]   │                      │    │  │
    │  │      │  └─────────────────────────────────────────────┘                      │    │  │
    │  │      │                │                                                       │    │  │
    │  │      │                ▼                                                       │    │  │
    │  │      │  # Value Loss (MSE between predicted and target returns)              │    │  │
    │  │      │  value_loss = mse_loss(current_values.squeeze(), mb_returns)          │    │  │
    │  │      │  target_returns = [18.0, 18.0, 20.0, 13.2, ...]                     │    │  │
    │  │      │  predicted_vals = [17.2, 16.8, 19.5, 12.9, ...]                     │    │  │
    │  │      │  value_loss = mean((predicted - target)²) = 1.23                     │    │  │
    │  │      └─────────────────────────────────────────────────────────────────────────┘    │  │
    │  └─────────────────────────────────────────────────────────────────────────────────────┘  │
    └─────────────────────────────────────────────────────────────────────────────────────────┘
                                              │
                                              ▼
    ┌─────────────────────────────────────────────────────────────────────────────────────────┐
    │                            BACKPROPAGATION PHASE                                         │
    │                                                                                         │
    │  ┌─────────────────────────────────────────────────────────────────────────────────────┐  │
    │  │                           POLICY NETWORK UPDATE                                     │  │
    │  │                                                                                     │  │
    │  │  # Zero gradients                                                                   │  │
    │  │  policy_optimizer.zero_grad()                                                       │  │
    │  │                                                                                     │  │
    │  │  # Backward pass - compute gradients                                               │  │
    │  │  total_policy_loss.backward()                                                      │  │
    │  │                                                                                     │  │
    │  │  # Gradients flow backwards through network:                                       │  │
    │  │  ∂L/∂W_portfolio_head ← ∂L/∂portfolio_probs ← ∂L/∂shared_features                │  │
    │  │  ∂L/∂W_goal_head ← ∂L/∂goal_probs ← ∂L/∂shared_features                          │  │
    │  │  ∂L/∂W_shared_layer2 ← ∂L/∂shared_features                                        │  │
    │  │  ∂L/∂W_shared_layer1 ← ∂L/∂shared_layer2_output                                   │  │
    │  │                                                                                     │  │
    │  │  Example gradient magnitudes:                                                      │  │
    │  │  shared_backbone.0.weight.grad: [-0.023, 0.041, -0.015, ...]                     │  │
    │  │  goal_head.weight.grad: [0.012, -0.034, 0.018, ...]                              │  │
    │  │  portfolio_head.weight.grad: [0.003, -0.018, 0.007, ...]                         │  │
    │  │                                                                                     │  │
    │  │  # Gradient clipping (prevent exploding gradients)                                │  │
    │  │  nn.utils.clip_grad_norm_(policy_net.parameters(), max_grad_norm=0.5)             │  │
    │  │                                                                                     │  │
    │  │  # Adam optimizer weight update                                                    │  │
    │  │  policy_optimizer.step()  # LR = 0.01 initially, decays linearly                  │  │
    │  │                                                                                     │  │
    │  │  For each parameter θ:                                                             │  │
    │  │    m_t = β₁ × m_{t-1} + (1-β₁) × ∇θ     # Momentum (β₁=0.9)                      │  │
    │  │    v_t = β₂ × v_{t-1} + (1-β₂) × ∇θ²    # Adaptive LR (β₂=0.999)                 │  │
    │  │    m̂_t = m_t / (1-β₁^t)                  # Bias correction                         │  │
    │  │    v̂_t = v_t / (1-β₂^t)                  # Bias correction                         │  │
    │  │    θ_t = θ_{t-1} - α × m̂_t / (√v̂_t + ε) # Weight update                          │  │
    │  │                                                                                     │  │
    │  └─────────────────────────────────────────────────────────────────────────────────────┘  │
    │                                                                                         │
    │  ┌─────────────────────────────────────────────────────────────────────────────────────┐  │
    │  │                            VALUE NETWORK UPDATE                                     │  │
    │  │                                                                                     │  │
    │  │  # Zero gradients                                                                   │  │
    │  │  value_optimizer.zero_grad()                                                        │  │
    │  │                                                                                     │  │
    │  │  # Backward pass                                                                    │  │
    │  │  value_loss.backward()                                                             │  │
    │  │                                                                                     │  │
    │  │  # Gradients flow backwards:                                                       │  │
    │  │  ∂L/∂W_output ← ∂MSE/∂predicted_values                                            │  │
    │  │  ∂L/∂W_layer2 ← ∂L/∂layer2_output                                                 │  │
    │  │  ∂L/∂W_layer1 ← ∂L/∂layer1_output                                                 │  │
    │  │                                                                                     │  │
    │  │  # Gradient clipping                                                               │  │
    │  │  nn.utils.clip_grad_norm_(value_net.parameters(), max_grad_norm=0.5)              │  │
    │  │                                                                                     │  │
    │  │  # Adam optimizer weight update                                                    │  │
    │  │  value_optimizer.step()                                                            │  │
    │  │                                                                                     │  │
    │  └─────────────────────────────────────────────────────────────────────────────────────┘  │
    └─────────────────────────────────────────────────────────────────────────────────────────┘
                                              │
                                              ▼
    ┌─────────────────────────────────────────────────────────────────────────────────────────┐
    │                             LEARNING RATE DECAY                                          │
    │                                                                                         │
    │  ┌─────────────────────────────────────────────────────────────────────────────────────┐  │
    │  │                          SCHEDULER UPDATES                                          │  │
    │  │                                                                                     │  │
    │  │  # Linear decay from initial LR to 0 over training                                 │  │
    │  │  policy_scheduler.step()                                                            │  │
    │  │  value_scheduler.step()                                                             │  │
    │  │                                                                                     │  │
    │  │  # Example progression:                                                             │  │
    │  │  Iteration 0: policy_lr = 0.010, value_lr = 0.010                                 │  │
    │  │  Iteration 1: policy_lr = 0.009, value_lr = 0.009                                 │  │
    │  │  Iteration 2: policy_lr = 0.008, value_lr = 0.008                                 │  │
    │  │  ...                                                                               │  │
    │  │  Iteration 10: policy_lr = 0.000, value_lr = 0.000                                │  │
    │  │                                                                                     │  │
    │  └─────────────────────────────────────────────────────────────────────────────────────┘  │
    └─────────────────────────────────────────────────────────────────────────────────────────┘
                                              │
                                              ▼
    ┌─────────────────────────────────────────────────────────────────────────────────────────┐
    │                              TRAINING METRICS                                            │
    │                                                                                         │
    │  ┌─────────────────────────────────────────────────────────────────────────────────────┐  │
    │  │                         ITERATION SUMMARY                                           │  │
    │  │                                                                                     │  │
    │  │  Training Metrics (after 1 complete iteration):                                    │  │
    │  │                                                                                     │  │
    │  │  ┌─────────────────────────────────────────────────────────────────────────────┐    │  │
    │  │  │                      PERFORMANCE METRICS                                   │    │  │
    │  │  │  mean_episode_reward: 24.7    # Average utility per episode                │    │  │
    │  │  │  mean_episode_length: 16.0    # Steps per episode (always 16)             │    │  │
    │  │  │  total_episodes: 4800         # Episodes collected this iteration          │    │  │
    │  │  │  total_timesteps: 76800       # Total environment steps                    │    │  │
    │  │  └─────────────────────────────────────────────────────────────────────────────┘    │  │
    │  │                                                                                     │  │
    │  │  ┌─────────────────────────────────────────────────────────────────────────────┐    │  │
    │  │  │                       LOSS METRICS                                         │    │  │
    │  │  │  policy_loss: 0.219           # PPO policy loss                           │    │  │
    │  │  │  value_loss: 1.23             # Value function MSE loss                   │    │  │
    │  │  │  entropy: 2.87                # Policy entropy (exploration)             │    │  │
    │  │  │  kl_divergence: 0.023         # KL between old and new policy            │    │  │
    │  │  └─────────────────────────────────────────────────────────────────────────────┘    │  │
    │  │                                                                                     │  │
    │  │  ┌─────────────────────────────────────────────────────────────────────────────┐    │  │
    │  │  │                     LEARNING RATES                                         │    │  │
    │  │  │  policy_lr: 0.009             # Current policy learning rate              │    │  │
    │  │  │  value_lr: 0.009              # Current value learning rate               │    │  │
    │  │  └─────────────────────────────────────────────────────────────────────────────┘    │  │
    │  │                                                                                     │  │
    │  │  ┌─────────────────────────────────────────────────────────────────────────────┐    │  │
    │  │  │                    IMPROVEMENT INDICATORS                                   │    │  │
    │  │  │  Before: policy_loss = 2.34, value_loss = 5.67, reward = 12.3             │    │  │
    │  │  │  After:  policy_loss = 0.219, value_loss = 1.23, reward = 24.7           │    │  │
    │  │  │  Improvement: Policy ↓90%, Value ↓78%, Reward ↑101%                       │    │  │
    │  │  └─────────────────────────────────────────────────────────────────────────────┘    │  │
    │  └─────────────────────────────────────────────────────────────────────────────────────┘  │
    └─────────────────────────────────────────────────────────────────────────────────────────┘

                              REPEAT FOR NEXT ITERATION
                          (Total: 10 iterations for paper settings)
                              
    Final Result: Trained policy network that achieves 94-98% of optimal 
                 dynamic programming solution while running in real-time
```

## Network Architecture Details

### Policy Network Structure:
```
Input: [time_norm, wealth_norm] ∈ ℝ²
  ↓
Shared Backbone:
  Linear(2, 64) + ReLU + Linear(64, 64) + ReLU
  ↓
Multi-Head Output:
  ├─ Goal Head: Linear(64, 2) + Softmax → [P(skip), P(take)]
  └─ Portfolio Head: Linear(64, 15) + Softmax → [P(port₀), ..., P(port₁₄)]

Total Parameters: 5,457 (Policy) + 4,417 (Value) = 9,874 parameters
```

### Training Hyperparameters:
```
Batch Size: 4,800 episodes × 16 steps = 76,800 experiences
PPO Epochs: 4 (reuse data 4 times)
Mini-batch Size: 256
Clip Epsilon: 0.5
Learning Rate: 0.01 → 0.0 (linear decay)
GAE Lambda: 0.95
Discount Factor: 0.99
Entropy Coefficient: 0.01
Max Grad Norm: 0.5
```

This diagram shows the complete data flow from raw market data through neural network processing to final weight updates, capturing every transformation and computation in the GBWM PPO training system! 🎯