
  Complete PPO Training Cycle: From Raw Data to Weight Updates

  Training Cycle Overview

  Raw Environment Data → Data Collection → Forward Pass → Loss Computation → Backpropagation → Weight Updates

  Let me walk through one complete training iteration with actual data examples.

  ---
  Phase 1: Data Collection (collect_trajectories())

  Environment State Generation

  Raw Environment State:
  # Episode starts
  initial_wealth = 389881.0  # For 4 goals
  current_time = 0
  max_wealth = 10000000.0    # Normalization constant

  # Raw observation from environment
  raw_obs = [current_time, initial_wealth]  # [0, 389881.0]

  # Normalized state for neural network
  normalized_state = [
      current_time / 16,              # 0/16 = 0.0 (time progress)
      initial_wealth / max_wealth     # 389881/10000000 = 0.039 (wealth ratio)
  ]

  Input to Neural Network:
  state_tensor = torch.FloatTensor([0.0, 0.039]).unsqueeze(0)
  # Shape: (1, 2) - batch_size=1, features=2
  # Values: [[0.0, 0.039]]

  Forward Pass Through Policy Network

  Shared Backbone:

  # Input: [0.0, 0.039]
  # Layer 1: Linear(2, 64) + ReLU
  shared_layer1 = ReLU(Linear(state_tensor))  # Shape: (1, 64)
  # Example output: [0.12, 0.0, 0.45, 0.23, ..., 0.67]  # 64 values

  # Layer 2: Linear(64, 64) + ReLU  
  shared_features = ReLU(Linear(shared_layer1))  # Shape: (1, 64)
  # Example output: [0.34, 0.78, 0.0, 0.89, ..., 0.12]  # 64 values

  Goal Decision Head:

  # Goal head: Linear(64, 2) + Softmax
  goal_logits = Linear_goal(shared_features)  # Shape: (1, 2)
  # Example logits: [1.2, -0.8]  # [skip_logit, take_logit]

  goal_probs = Softmax(goal_logits)  # Shape: (1, 2)
  # Example: [0.86, 0.14]  # [skip_prob=86%, take_prob=14%]

  Portfolio Decision Head:

  # Portfolio head: Linear(64, 15) + Softmax
  portfolio_logits = Linear_portfolio(shared_features)  # Shape: (1, 15)
  # Example logits: [-0.1, 0.3, 0.8, 1.2, 0.5, ..., -0.4]  # 15 values

  portfolio_probs = Softmax(portfolio_logits)  # Shape: (1, 15)
  # Example: [0.05, 0.08, 0.12, 0.18, 0.09, ..., 0.04]  # 15 probabilities summing to 1.0

  Action Sampling

  # Sample actions from probability distributions
  goal_dist = Categorical(goal_probs)
  portfolio_dist = Categorical(portfolio_probs)

  goal_action = goal_dist.sample()        # e.g., 0 (skip goal)
  portfolio_action = portfolio_dist.sample()  # e.g., 7 (moderate portfolio)

  # Roll 1: random_num = 0.2  → 0.2 < 0.7 → action = 0 (skip)
  # Roll 2: random_num = 0.8  → 0.8 > 0.7 → action = 1 (take)  
  # Roll 3: random_num = 0.5  → 0.5 < 0.7 → action = 0 (skip)
  # Roll 4: random_num = 0.9  → 0.9 > 0.7 → action = 1 (take)
  # Roll 5: random_num = 0.1  → 0.1 < 0.7 → action = 0 (skip)


  actions = torch.tensor([goal_action, portfolio_action])  # [0, 7]

  # Compute log probabilities for PPO
  goal_log_prob = goal_dist.log_prob(goal_action)        # e.g., -0.15
  portfolio_log_prob = portfolio_dist.log_prob(portfolio_action)  # e.g., -1.89
  total_log_prob = goal_log_prob + portfolio_log_prob    # e.g., -2.04

    # If we took action 0 (skip goal) and the probabilities were:
  goal_probs = [0.8, 0.2]  # 80% skip, 20% take
  goal_action = 0          # We chose to skip

  # log_prob tells us the likelihood of our choice
  goal_log_prob = goal_dist.log_prob(goal_action)
  # This calculates: log(0.8) = -0.223


  Value Network Forward Pass

  # Same input state
  value_input = torch.FloatTensor([0.0, 0.039]).unsqueeze(0)  # Shape: (1, 2)

  # Value network: 2 → 64 → 64 → 1
  value_layer1 = ReLU(Linear(value_input))        # Shape: (1, 64)
  value_layer2 = ReLU(Linear(value_layer1))       # Shape: (1, 64)  
  value_estimate = Linear(value_layer2)           # Shape: (1, 1)

  # Example value estimate
  value_estimate = 15.7  # Expected future utility

  Environment Step

  # Execute action in environment
  action_array = [0, 7]  # [goal_decision, portfolio_choice]
  next_obs, reward, done, info = env.step(action_array)


  Complete env.step(action_array) Example

  Input: Action Array

  action_array = np.array([1, 7])  # [goal_action, portfolio_action]
  #                         ↑  ↑
  #                      take  portfolio_7 (moderate-aggressive)

  What Happens Inside env.step()

  Before the step:
  # Current environment state
  current_time = 2          # Year 2 of 10-year horizon
  current_wealth = 450000   # $450,000
  goals_taken = [0]         # Already took goal at time 0
  total_utility = 25.0      # Utility accumulated so far

  Step 1: Execute Goal Decision

  goal_action = 1  # Take goal

  # Check if goal is available at time=2
  goal_available = True  # Let's say a goal is available

  # Goal parameters at time=2
  goal_cost = 50000      # Costs $50,000
  goal_utility = 20.0    # Gives 20 utility points

  # Check if can afford: $450,000 >= $50,000 ✓
  if current_wealth >= goal_cost:
      wealth_after_goal = 450000 - 50000 = 400000  # $400,000 remaining
      reward = 20.0  # Utility gained
      goals_taken.append(2)  # Add to goals taken list
      total_utility += 20.0  # Update total utility

  Step 2: Evolve Portfolio

  portfolio_action = 7  # Choose portfolio 7 (moderate-aggressive)

  # Portfolio 7 characteristics (from efficient frontier)
  portfolio_7_mean = 0.08    # 8% expected annual return  
  portfolio_7_std = 0.15     # 15% volatility

  # Wealth evolution (using Geometric Brownian Motion)
  wealth = 400000  # After goal cost

  # Generate random market movement
  random_shock = np.random.normal(0, 1)  # e.g., 0.5 (positive market day)

  # Calculate portfolio return
  drift = 0.08 - 0.5 * (0.15 ** 2) = 0.08 - 0.01125 = 0.06875
  diffusion = 0.15 * 0.5 = 0.075
  portfolio_return = np.exp(0.06875 + 0.075) = np.exp(0.14375) = 1.155

  # New wealth
  new_wealth = 400000 * 1.155 = 462000  # $462,000

  Step 3: Update Environment State

  # Update time and wealth
  current_time = 2 + 1 = 3
  current_wealth = 462000

  # Check if episode is done
  terminated = (current_time >= 10)  # False, still have 7 years left
  truncated = False  # GBWM doesn't use truncation

  Step 4: Create Observation

  # Normalize state for neural network
  normalized_time = 3 / 10 = 0.3        # 30% through episode
  normalized_wealth = 462000 / 500000 = 0.924  # 92.4% of initial wealth

  next_obs = np.array([0.3, 0.924])  # [normalized_time, normalized_wealth]

  Step 5: Create Info Dictionary

  info = {
      'time': 3,
      'wealth': 462000,
      'goal_available': True,      # Was goal available when we acted?
      'goal_taken': True,          # Did we take the goal?
      'goals_taken_so_far': 2,     # Total goals taken (time 0 + time 2)
      'total_utility': 45.0,       # 25.0 + 20.0
      'portfolio_choice': 7        # Which portfolio we chose
  }

  Complete Return Values

  next_obs, reward, terminated, truncated, info = env.step([1, 7])

  # next_obs = np.array([0.3, 0.924])   # New normalized state
  # reward = 20.0                       # Utility gained from taking goal
  # terminated = False                  # Episode not finished
  # truncated = False                   # No truncation
  # info = {...}                        # Dictionary with episode details

  Visual Timeline

  Before step:  Time=2, Wealth=$450k, Utility=25
  Action:       [Take Goal, Portfolio 7]
  Goal Effect:  Wealth=$450k → $400k, Utility=25 → 45, Reward=+20
  Portfolio:    $400k × 1.155 return → $462k
  After step:   Time=3, Wealth=$462k, Utility=45

  Returns: obs=[0.3, 0.924], reward=20.0, done=False, info={...}


  # Example results
  reward = 0.0              # No goal taken (goal not available at step 0)
  next_obs = [0.0625, 0.041]  # [1/16, new_wealth/max_wealth] after portfolio evolution

  Trajectory Data Storage

  After collecting 4,800 episodes × 16 steps each = 76,800 experiences:

  batch_data = {
      'states': torch.FloatTensor([
          [0.000, 0.039],    # Step 0, Episode 0
          [0.063, 0.041],    # Step 1, Episode 0  
          [0.125, 0.043],    # Step 2, Episode 0
          ...
          [0.875, 0.067],    # Step 14, Episode 4799
          [1.000, 0.072]     # Step 15, Episode 4799
      ]),  # Shape: (76800, 2)

      'actions': torch.LongTensor([
          [0, 7],   # Episode 0, Step 0
          [0, 5],   # Episode 0, Step 1
          [1, 12],  # Episode 0, Step 2 (took goal)
          ...
      ]),  # Shape: (76800, 2)

      'rewards': torch.FloatTensor([
          0.0,   # Step 0 (no goal)
          0.0,   # Step 1 (no goal)
          14.0,  # Step 2 (took goal, utility = 10 + 4)
          ...
      ]),  # Shape: (76800,)

      'old_log_probs': torch.FloatTensor([
          -2.04,  # Step 0 log probability
          -1.87,  # Step 1 log probability
          -2.15,  # Step 2 log probability
          ...
      ]),  # Shape: (76800,)

      'values': torch.FloatTensor([
          15.7,   # Value estimate for step 0
          16.2,   # Value estimate for step 1
          14.8,   # Value estimate for step 2
          ...
      ])  # Shape: (76800,)
  }

  ---
  Phase 2: Advantage Computation (GAE)

  Generalized Advantage Estimation

  # Example for one episode (16 steps)
  rewards = [0.0, 0.0, 14.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
  values = [15.7, 16.2, 14.8, 13.9, 12.1, 11.3, 10.8, 9.2, 8.1, 7.5, 6.8, 5.9, 4.2, 3.1, 1.8, 0.0]
  dones = [False] * 15 + [True]  # Episode ends at step 15

  # GAE computation (γ=0.99, λ=0.95)
  advantages = []
  gae = 0

  for t in reversed(range(16)):
      if t == 15:
          next_value = 0  # Terminal state
      else:
          next_value = values[t + 1]

      # TD error
      delta = rewards[t] + 0.99 * next_value - values[t]

      # GAE
      gae = delta + 0.99 * 0.95 * gae
      advantages.append(gae)

  advantages.reverse()

  # Example advantages
  advantages = [2.3, 1.8, 5.2, -0.7, -1.1, -0.8, -0.6, -0.4, -0.2, 0.1, 0.3, 0.5, 0.8, 1.2, 1.8, 0.0]

  # Returns (for value function targets)  
  returns = [adv + value for adv, value in zip(advantages, values)]
  returns = [18.0, 18.0, 20.0, 13.2, 11.0, 10.5, 10.2, 8.8, 7.9, 7.6, 7.1, 6.4, 5.0, 4.3, 3.6, 0.0]

  # Normalize advantages
  advantages_normalized = (advantages - mean(advantages)) / (std(advantages) + 1e-8)

  ---
  Phase 3: PPO Update (4 epochs)

  Mini-batch Processing

  # Configuration
  ppo_epochs = 4
  mini_batch_size = 256
  batch_size = 76800

  for epoch in range(4):  # Use same data 4 times
      # Shuffle data
      indices = torch.randperm(76800)

      for start in range(0, 76800, 256):  # Mini-batches of 256
          end = min(start + 256, 76800)
          mb_indices = indices[start:end]

          # Extract mini-batch
          mb_states = batch_data['states'][mb_indices]      # Shape: (256, 2)
          mb_actions = batch_data['actions'][mb_indices]    # Shape: (256, 2)
          mb_old_log_probs = batch_data['old_log_probs'][mb_indices]  # Shape: (256,)
          mb_advantages = advantages_normalized[mb_indices]  # Shape: (256,)
          mb_returns = returns[mb_indices]                  # Shape: (256,)

  Forward Pass (Current Policy)

  # Policy network forward pass
  goal_probs, portfolio_probs = policy_net(mb_states)  # Shapes: (256, 2), (256, 15)

  # Extract log probabilities for taken actions
  goal_actions = mb_actions[:, 0]       # Shape: (256,)
  portfolio_actions = mb_actions[:, 1]  # Shape: (256,)

  goal_log_probs = torch.log(goal_probs.gather(1, goal_actions.unsqueeze(1))).squeeze()
  portfolio_log_probs = torch.log(portfolio_probs.gather(1, portfolio_actions.unsqueeze(1))).squeeze()
  new_log_probs = goal_log_probs + portfolio_log_probs  # Shape: (256,)

  # Entropy for exploration bonus
  goal_entropy = -(goal_probs * torch.log(goal_probs + 1e-8)).sum(dim=1)
  portfolio_entropy = -(portfolio_probs * torch.log(portfolio_probs + 1e-8)).sum(dim=1)
  entropy = goal_entropy + portfolio_entropy  # Shape: (256,)

  PPO Loss Computation

  # Probability ratio
  ratio = torch.exp(new_log_probs - mb_old_log_probs)  # Shape: (256,)

  # Example ratios: [1.05, 0.92, 1.31, 0.87, ...]

  # PPO clipped surrogate objective
  surr1 = ratio * mb_advantages                        # Unclipped objective
  surr2 = torch.clamp(ratio, 0.5, 1.5) * mb_advantages  # Clipped (ε=0.5)

  # Policy loss (negative because we want to maximize)
  policy_loss = -torch.min(surr1, surr2).mean()       # Scalar

  # Entropy bonus
  entropy_loss = -0.01 * entropy.mean()               # Encourage exploration

  # Total policy loss
  total_policy_loss = policy_loss + entropy_loss      # e.g., 0.234

  # Example calculation:
  # ratio = [1.05, 0.92, 1.31, 0.87]
  # advantages = [2.1, -1.3, 0.8, -0.6]
  # surr1 = [2.21, 1.20, 1.05, -0.52] 
  # surr2 = [2.1, 1.20, 1.2, -0.6]    # ratio clamped to [0.5, 1.5]
  # min = [2.1, 1.20, 1.05, -0.6]
  # policy_loss = -mean([2.1, 1.20, 1.05, -0.6]) = -0.94

  Value Network Loss

  # Value network forward pass
  current_values = value_net(mb_states)  # Shape: (256, 1)

  # Value loss (MSE between predicted and actual returns)
  value_loss = F.mse_loss(current_values.squeeze(), mb_returns)  # Scalar

  # Example: 
  # predicted_values = [17.2, 16.8, 19.5, 12.9, ...]
  # target_returns =   [18.0, 18.0, 20.0, 13.2, ...]
  # value_loss = mean((predicted - target)^2) = 1.23

  ---
  Phase 4: Backpropagation & Weight Updates

  Policy Network Gradients

  # Zero gradients
  policy_optimizer.zero_grad()

  # Backward pass
  total_policy_loss.backward()  # Compute gradients

  # Gradient clipping
  torch.nn.utils.clip_grad_norm_(policy_net.parameters(), max_grad_norm=0.5)

  # Example gradients for policy network weights:
  # shared_backbone.0.weight.grad: Tensor of shape (64, 2) with values [-0.023, 0.041, ...]
  # goal_head.weight.grad: Tensor of shape (2, 64) with values [0.012, -0.034, ...]
  # portfolio_head.weight.grad: Tensor of shape (15, 64) with values [0.003, -0.018, ...]

  # Update weights
  policy_optimizer.step()  # Adam optimizer applies gradients

  Value Network Gradients

  # Zero gradients
  value_optimizer.zero_grad()

  # Backward pass
  value_loss.backward()  # Compute gradients

  # Gradient clipping
  torch.nn.utils.clip_grad_norm_(value_net.parameters(), max_grad_norm=0.5)

  # Example gradients for value network weights:
  # layer1.weight.grad: Tensor of shape (64, 2) with values [0.008, -0.015, ...]
  # layer2.weight.grad: Tensor of shape (64, 64) with values [-0.006, 0.011, ...]
  # output.weight.grad: Tensor of shape (1, 64) with values [0.021, -0.009, ...]

  # Update weights
  value_optimizer.step()  # Adam optimizer applies gradients

  Adam Weight Updates

  # For each parameter θ in the network:
  # Adam maintains moving averages of gradients

  # Example for one weight in shared backbone:
  old_weight = 0.234
  gradient = -0.023
  learning_rate = 0.01

  # Adam update (simplified):
  m_t = 0.9 * m_t_1 + 0.1 * gradient           # Momentum
  v_t = 0.999 * v_t_1 + 0.001 * gradient**2    # Adaptive learning rate
  m_hat = m_t / (1 - 0.9**t)                   # Bias correction
  v_hat = v_t / (1 - 0.999**t)                 # Bias correction

  new_weight = old_weight - learning_rate * m_hat / (sqrt(v_hat) + 1e-8)
  new_weight = 0.234 - 0.01 * adjusted_gradient = 0.236  # Weight increased

  ---
  Complete Iteration Summary

  Data Flow Summary:

  Input:  76,800 experiences from environment interaction
         ↓
  Processing: GAE computation, advantage normalization
         ↓
  Training: 4 epochs × 300 mini-batches × 256 samples = 307,200 gradient updates
         ↓
  Output: Updated policy & value network weights

  Network Changes:

  # Before iteration
  policy_net.goal_head.weight[0,0] = 0.234
  value_net.output.weight[0,0] = -0.456

  # After iteration  
  policy_net.goal_head.weight[0,0] = 0.236  # Slightly increased
  value_net.output.weight[0,0] = -0.451     # Slightly adjusted

  # Improvement metrics
  policy_loss_reduction: 2.34 → 2.28  # Better action selection
  value_loss_reduction: 1.23 → 1.18   # Better value estimates
  mean_reward_improvement: 24.3 → 24.7  # Better performance

  Learning Progress:

  Iteration 0: Random policy, high loss, low rewards
  Iteration 1: Slight improvement in goal timing
  Iteration 2: Better portfolio selection in volatile markets
  ...
  Iteration 10: Near-optimal policy matching greedy strategy

  This complete cycle repeats 10 times (for paper settings), with the policy gradually improving from random actions to optimal financial decision-making,
  learning to time goals perfectly and select appropriate portfolios based on market conditions! 




Backpropagation


Step 1: Policy Network Loss Calculation



⏺ Policy Network Example

  # Example mini-batch data
  mb_states = torch.tensor([[0.3, 0.924], [0.4, 0.856], [0.5, 0.798]])  # 3 states
  mb_actions = torch.tensor([[1, 7], [0, 3], [1, 12]])  # 3 actions taken
  mb_old_log_probs = torch.tensor([-2.1, -1.8, -2.5])  # Original action probabilities
  mb_advantages = torch.tensor([2.3, -0.8, 1.5])  # How good actions were

  # Step 1: Get new probabilities with current policy
  new_log_probs, entropy = policy_net.evaluate_actions(mb_states, mb_actions)
  # new_log_probs = [-1.9, -1.9, -2.2]  # Updated probabilities

  # Step 2: Compute probability ratios
  ratio = torch.exp(new_log_probs - mb_old_log_probs)
  # ratio[0] = exp(-1.9 - (-2.1)) = exp(0.2) = 1.22  # Action became MORE likely
  # ratio[1] = exp(-1.9 - (-1.8)) = exp(-0.1) = 0.90  # Action became LESS likely  
  # ratio[2] = exp(-2.2 - (-2.5)) = exp(0.3) = 1.35  # Action became MORE likely

  # Step 3: PPO clipped objective (prevents huge policy changes)
  clip_epsilon = 0.2  # Clip to [0.8, 1.2] range
  surr1 = ratio * mb_advantages
  # surr1 = [1.22 * 2.3, 0.90 * (-0.8), 1.35 * 1.5] = [2.81, -0.72, 2.03]

  surr2 = torch.clamp(ratio, 0.8, 1.2) * mb_advantages
  # surr2 = [1.2 * 2.3, 0.90 * (-0.8), 1.2 * 1.5] = [2.76, -0.72, 1.8]

  # Step 4: Take minimum (conservative update)
  policy_objective = torch.min(surr1, surr2).mean()
  # policy_objective = mean([2.76, -0.72, 1.8]) = 1.28

  # Step 5: Policy loss (negative because we want to maximize objective)
  policy_loss = -policy_objective = -1.28

  # Step 6: Add entropy bonus (encourages exploration)
  entropy_loss = -0.01 * entropy.mean()  # entropy_coeff = 0.01
  # entropy_loss = -0.01 * 1.5 = -0.015

  # Step 7: Total loss
  total_policy_loss = policy_loss + entropy_loss = -1.28 + (-0.015) = -1.295

  Step 2: Policy Network Backpropagation

  # Clear previous gradients
  policy_optimizer.zero_grad()

  # Compute gradients
  total_policy_loss.backward()  # This is where the magic happens!

  # What happens inside backward():
  # ∂loss/∂w₁ = -0.23  # Gradient for weight 1
  # ∂loss/∂w₂ = +0.15  # Gradient for weight 2  
  # ∂loss/∂w₃ = -0.08  # Gradient for weight 3
  # ... (gradients for all 10,000+ parameters)

  # Clip gradients (prevent exploding gradients)
  nn.utils.clip_grad_norm_(policy_net.parameters(), max_grad_norm=0.5)

  # Update weights
  policy_optimizer.step()  # w_new = w_old - learning_rate * gradient

  Concrete Weight Update Example

  # Before update
  goal_head_weight_old = torch.tensor([0.45, -0.23])  # Weight connecting to "take goal"

  # Gradient computed by backprop  
  gradient = torch.tensor([-0.12, +0.08])  # Loss decreases if we change weights this way

  # Weight update (Adam optimizer)
  learning_rate = 0.0003
  goal_head_weight_new = goal_head_weight_old - learning_rate * gradient
  goal_head_weight_new = [0.45, -0.23] - 0.0003 * [-0.12, +0.08]
  goal_head_weight_new = [0.45036, -0.23024]

  # Result: Network now slightly more likely to recommend "take goal" in similar situations

  Step 3: Value Network Loss & Backpropagation



⏺ Value Network Example

  # Same mini-batch states
  mb_states = torch.tensor([[0.3, 0.924], [0.4, 0.856], [0.5, 0.798]])
  mb_returns = torch.tensor([22.5, 15.3, 18.7])  # Actual returns (advantages + old_values)

  # Step 1: Get current value predictions
  new_values = value_net(mb_states)
  # new_values = [20.1, 16.8, 17.2]  # Current network predictions

  # Step 2: Compute Mean Squared Error loss
  value_loss = MSE(new_values, mb_returns)
  # value_loss = mean([(20.1-22.5)², (16.8-15.3)², (17.2-18.7)²])
  # value_loss = mean([5.76, 2.25, 2.25]) = 3.42

  # Step 3: Backpropagation
  value_optimizer.zero_grad()
  value_loss.backward()

  # What happens inside backward():
  # Network learns: "When I see state [0.3, 0.924], predict closer to 22.5, not 20.1"
  # ∂loss/∂w₁ = +0.18  # Increase this weight
  # ∂loss/∂w₂ = -0.09  # Decrease this weight
  # ... (gradients for all value network parameters)

  # Step 4: Update weights
  value_optimizer.step()

  Step 4: Multiple Epochs & Mini-batches

  PPO repeats this process multiple times for better learning:

  # PPO configuration
  ppo_epochs = 4          # Train on same data 4 times
  mini_batch_size = 256   # Process 256 samples at once
  batch_size = 2048       # Total collected samples

  # Training loop
  for epoch in range(ppo_epochs):  # 4 epochs
      for mini_batch in shuffle_and_split(batch_data, mini_batch_size):  # 8 mini-batches
          # Policy update (as shown above)
          policy_loss = compute_and_update_policy(mini_batch)

          # Value update (as shown above)  
          value_loss = compute_and_update_value(mini_batch)

  # Total updates per iteration: 4 epochs × 8 mini-batches = 32 updates each network

  Step 5: Learning Rate Scheduling

  # Learning rates decay over time for stable convergence
  initial_lr = 0.0003
  current_iteration = 150
  total_iterations = 1000

  # Linear decay
  lr_factor = 1.0 - (current_iteration / total_iterations)  # 0.85
  current_lr = initial_lr * lr_factor = 0.0003 * 0.85 = 0.000255

  # Both optimizers use this reduced learning rate
  policy_optimizer.param_groups[0]['lr'] = current_lr
  value_optimizer.param_groups[0]['lr'] = current_lr

  Visualization: Weight Changes Over Time

  # Policy Network Goal Head Weights (example trajectory)
  Iteration 0:   [0.45, -0.23, 0.12, -0.08]
  Iteration 50:  [0.48, -0.19, 0.15, -0.05]  # Learning to favor goal-taking
  Iteration 100: [0.52, -0.15, 0.18, -0.02]  # Stronger goal-taking preference  
  Iteration 150: [0.54, -0.13, 0.19, -0.01]  # Converging to optimal policy

  # Value Network Final Layer Weights (example trajectory)  
  Iteration 0:   [2.1, -1.8]
  Iteration 50:  [2.3, -1.6]  # Better at predicting future utility
  Iteration 100: [2.4, -1.5]  # More accurate value estimates
  Iteration 150: [2.45, -1.4] # Stable, accurate predictions

  Error Signals Driving Learning

  Policy Network learns from:
  - Positive advantages: "That action was better than expected → increase probability"
  - Negative advantages: "That action was worse than expected → decrease probability"

  Value Network learns from:
  - Prediction errors: "I predicted 15.7 but actual return was 22.5 → adjust predictions upward"

  Complete Training Iteration Summary

  # One complete PPO iteration:
  1. Collect 2048 experiences using current policy
  2. Compute advantages and returns
  3. For 4 epochs:
     a. Shuffle data into 8 mini-batches of 256 samples
     b. For each mini-batch:
        - Update policy network (32 parameter updates total)
        - Update value network (32 parameter updates total)
  4. Decay learning rates
  5. Repeat for next iteration

  # Result: Both networks gradually improve their performance
  # - Policy becomes better at selecting profitable actions
  # - Value becomes better at predicting future utility

  This backpropagation phase is where the actual "learning" happens - the networks adjust their internal parameters based on experience to make better decisions
  and predictions in future episodes.