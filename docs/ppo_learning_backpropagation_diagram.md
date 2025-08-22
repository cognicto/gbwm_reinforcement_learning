# PPO Learning and Backpropagation Block Diagram

## Overview: Complete PPO Training Cycle with Backpropagation

```
┌─────────────────────────────────────────────────────────────────────────────────────┐
│                           PPO TRAINING ITERATION CYCLE                             │
└─────────────────────────────────────────────────────────────────────────────────────┘

Phase 1: Data Collection
┌─────────────────────────────────────────────────────────────────────────────────────┐
│                              EXPERIENCE COLLECTION                                 │
├─────────────────────────────────────────────────────────────────────────────────────┤
│                                                                                     │
│  ┌─────────────┐    ┌─────────────────┐    ┌─────────────────┐    ┌─────────────┐  │
│  │ GBWM        │    │ Policy Network  │    │ Value Network   │    │ Environment │  │
│  │ Environment │────│ (Actor)         │────│ (Critic)        │    │ Step        │  │
│  │             │    │                 │    │                 │    │             │  │
│  │ State:      │    │ Input: [0.3,0.9]│    │ Input: [0.3,0.9]│    │ Action:     │  │
│  │ [0.3, 0.9]  │    │                 │    │                 │    │ [1, 7]      │  │
│  │             │    │ ┌─────────────┐ │    │ ┌─────────────┐ │    │             │  │
│  │ Time: 3/10  │    │ │Goal Head    │ │    │ │Hidden Layer │ │    │ Reward:     │  │
│  │ Wealth: 90% │    │ │[0.4, 0.6]   │ │    │ │[64 neurons] │ │    │ 15.0        │  │
│  │             │    │ └─────────────┘ │    │ └─────────────┘ │    │             │  │
│  └─────────────┘    │                 │    │                 │    │ Next State: │  │
│                     │ ┌─────────────┐ │    │ ┌─────────────┐ │    │ [0.4, 0.95] │  │
│                     │ │Portfolio    │ │    │ │Value Output │ │    │             │  │
│                     │ │Head         │ │    │ │V(s) = 12.5  │ │    │ Done: False │  │
│                     │ │[0.05,0.08,..]│ │    │ └─────────────┘ │    │             │  │
│                     │ └─────────────┘ │    │                 │    │             │  │
│                     │                 │    │                 │    │             │  │
│                     │ Action: [1, 7]  │    │                 │    │             │  │
│                     │ old_log_prob:   │    │                 │    │             │  │
│                     │ -2.31           │    │                 │    │             │  │
│                     └─────────────────┘    └─────────────────┘    └─────────────┘  │
│                                                                                     │
│  Collect 2048 timesteps × Store: [state, action, reward, old_log_prob, value]      │
└─────────────────────────────────────────────────────────────────────────────────────┘

Phase 2: Advantage Computation
┌─────────────────────────────────────────────────────────────────────────────────────┐
│                         GENERALIZED ADVANTAGE ESTIMATION                           │
├─────────────────────────────────────────────────────────────────────────────────────┤
│                                                                                     │
│  ┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐                │
│  │ Collected Data  │    │ GAE Computation │    │ Training Batch  │                │
│  │                 │    │                 │    │                 │                │
│  │ rewards:        │    │ For t in T-1...0│    │ states: (2048,2)│                │
│  │ [0,0,15,0,12,..]│───▶│                 │───▶│ actions:(2048,2)│                │
│  │                 │    │ δt = rt + γV(t+1│    │ old_log_probs:  │                │
│  │ values:         │    │      - V(t)     │    │ (2048,)         │                │
│  │ [12.5,10.3,8.7.]│    │                 │    │                 │                │
│  │                 │    │ At = δt + γλAt+1│    │ advantages:     │                │
│  │ dones:          │    │                 │    │ (2048,)         │                │
│  │ [F,F,F,F,T,...] │    │ Returns = A + V │    │                 │                │
│  │                 │    │                 │    │ returns: (2048,)│                │
│  └─────────────────┘    └─────────────────┘    └─────────────────┘                │
│                                                                                     │
│  Example GAE calculation:                                                           │
│  δ₂ = 15 + 0.99×10.3 - 8.7 = 16.497                                               │
│  A₂ = 16.497 + 0.99×0.95×A₃ = 16.497 + future_advantage                          │
│  Return₂ = A₂ + V₂ = advantage₂ + 8.7                                             │
└─────────────────────────────────────────────────────────────────────────────────────┘

Phase 3: PPO Updates (4 epochs × 8 mini-batches = 32 updates)
┌─────────────────────────────────────────────────────────────────────────────────────┐
│                              PPO LEARNING PHASE                                    │
├─────────────────────────────────────────────────────────────────────────────────────┤
│                                                                                     │
│  ┌─────────────────────────────────────────────────────────────────────────────────┐ │
│  │                           MINI-BATCH PROCESSING                                │ │
│  │                                                                                 │ │
│  │  ┌─────────────┐         ┌─────────────────┐         ┌─────────────────┐      │ │
│  │  │ Mini-batch  │         │ Current Policy  │         │ Current Value   │      │ │
│  │  │ (256 samples│    ────▶│ Network         │    ────▶│ Network         │      │ │
│  │  │ from 2048)  │         │                 │         │                 │      │ │
│  │  │             │         │ Same states:    │         │ Same states:    │      │ │
│  │  │ states:     │         │ [0.3, 0.9]      │         │ [0.3, 0.9]      │      │ │
│  │  │ [0.3, 0.9]  │         │                 │         │                 │      │ │
│  │  │             │         │ NEW weights     │         │ NEW weights     │      │ │
│  │  │ actions:    │         │ (after learning)│         │ (after learning)│      │ │
│  │  │ [1, 7]      │         │                 │         │                 │      │ │
│  │  │             │         │ ┌─────────────┐ │         │ ┌─────────────┐ │      │ │
│  │  │ old_log_prob│         │ │Goal Head    │ │         │ │Hidden Layers│ │      │ │
│  │  │ -2.31       │         │ │[0.25, 0.75] │ │         │ │[ReLU, ReLU] │ │      │ │
│  │  │             │         │ └─────────────┘ │         │ └─────────────┘ │      │ │
│  │  │ advantages: │         │                 │         │                 │      │ │
│  │  │ [2.1, -0.8, │         │ ┌─────────────┐ │         │ ┌─────────────┐ │      │ │
│  │  │  1.5, ...]  │         │ │Portfolio    │ │         │ │Value Output │ │      │ │
│  │  │             │         │ │Head         │ │         │ │new_values   │ │      │ │
│  │  │ returns:    │         │ │[0.08,0.12,.]│ │         │ │[13.2, 9.8,.]│ │      │ │
│  │  │ [14.6, 9.5, │         │ └─────────────┘ │         │ └─────────────┘ │      │ │
│  │  │  11.2, ...] │         │                 │         │                 │      │ │
│  │  │             │         │ new_log_prob:   │         │                 │      │ │
│  │  └─────────────┘         │ -1.89           │         │                 │      │ │
│  │                          └─────────────────┘         └─────────────────┘      │ │
│  └─────────────────────────────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────────────────────────┘

Phase 4: Loss Computation & Backpropagation
┌─────────────────────────────────────────────────────────────────────────────────────┐
│                            LOSS COMPUTATION PHASE                                  │
├─────────────────────────────────────────────────────────────────────────────────────┤
│                                                                                     │
│  ┌─────────────────────────────────────────────────────────────────────────────────┐ │
│  │                              POLICY LOSS                                       │ │
│  │                                                                                 │ │
│  │  ┌─────────────┐    ┌─────────────┐    ┌─────────────┐    ┌─────────────┐    │ │
│  │  │ Ratio       │    │ PPO Clipped │    │ Entropy     │    │ Total Policy│    │ │
│  │  │ Calculation │    │ Objective   │    │ Regularizer │    │ Loss        │    │ │
│  │  │             │    │             │    │             │    │             │    │ │
│  │  │ ratio = exp │    │ surr1 =     │    │ entropy =   │    │ policy_loss │    │ │
│  │  │ (new_log -  │    │ ratio × adv │    │ -Σ(p×log(p))│    │ = -min(surr1│    │ │
│  │  │  old_log)   │    │             │    │             │    │    surr2)   │    │ │
│  │  │             │    │ surr2 =     │    │ Goal entropy│    │   - ε×entropy│    │ │
│  │  │ ratio = exp │    │ clip(ratio, │    │ Portfolio   │    │             │    │ │
│  │  │ (-1.89-(-2.3│    │ 0.8,1.2)×adv│    │ entropy     │    │ Loss: -1.24 │    │ │
│  │  │ = 1.51      │    │             │    │             │    │             │    │ │
│  │  └─────────────┘    └─────────────┘    └─────────────┘    └─────────────┘    │ │
│  │                                                                                 │ │
│  │  Example calculation:                                                           │ │
│  │  advantage = 2.1, ratio = 1.51                                                 │ │
│  │  surr1 = 1.51 × 2.1 = 3.17                                                    │ │
│  │  surr2 = clip(1.51, 0.8, 1.2) × 2.1 = 1.2 × 2.1 = 2.52                     │ │
│  │  objective = min(3.17, 2.52) = 2.52 (clipping prevents excessive update)      │ │
│  └─────────────────────────────────────────────────────────────────────────────────┘ │
│                                                                                     │
│  ┌─────────────────────────────────────────────────────────────────────────────────┐ │
│  │                               VALUE LOSS                                       │ │
│  │                                                                                 │ │
│  │  ┌─────────────┐              ┌─────────────┐              ┌─────────────┐    │ │
│  │  │ Value       │              │ Mean Squared│              │ Value Loss  │    │ │
│  │  │ Predictions │              │ Error       │              │ Result      │    │ │
│  │  │             │              │             │              │             │    │ │
│  │  │ new_values: │    ─────▶    │ value_loss =│    ─────▶    │ Loss: 3.85  │    │ │
│  │  │ [13.2, 9.8, │              │ MSE(new_val,│              │             │    │ │
│  │  │  11.1, ...] │              │     returns)│              │ Teaches     │    │ │
│  │  │             │              │             │              │ network to  │    │ │
│  │  │ returns:    │              │ MSE([13.2,  │              │ predict     │    │ │
│  │  │ [14.6, 9.5, │              │ 9.8, 11.1], │              │ better      │    │ │
│  │  │  11.2, ...] │              │ [14.6, 9.5, │              │ values      │    │ │
│  │  │             │              │  11.2])     │              │             │    │ │
│  │  └─────────────┘              └─────────────┘              └─────────────┘    │ │
│  └─────────────────────────────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────────────────────────┘

Phase 5: Backpropagation & Weight Updates
┌─────────────────────────────────────────────────────────────────────────────────────┐
│                         BACKPROPAGATION & OPTIMIZATION                             │
├─────────────────────────────────────────────────────────────────────────────────────┤
│                                                                                     │
│  ┌─────────────────────────────────────────────────────────────────────────────────┐ │
│  │                          POLICY NETWORK BACKPROP                               │ │
│  │                                                                                 │ │
│  │  ┌─────────────┐    ┌─────────────┐    ┌─────────────┐    ┌─────────────┐    │ │
│  │  │ Loss        │    │ Gradient    │    │ Gradient    │    │ Weight      │    │ │
│  │  │ Computation │    │ Computation │    │ Clipping    │    │ Update      │    │ │
│  │  │             │    │             │    │             │    │             │    │ │
│  │  │ policy_loss │    │ ∂L/∂W =     │    │ clip_grad_  │    │ W_new =     │    │ │
│  │  │ = -1.24     │    │ chain rule  │    │ norm(grads, │    │ W_old -     │    │ │
│  │  │             │    │ through     │    │ max_norm=0.5│    │ lr × grad   │    │ │
│  │  │ policy_loss.│    │ network     │    │ )           │    │             │    │ │
│  │  │ backward()  │    │             │    │             │    │ Goal head:  │    │ │
│  │  │             │    │ Goal Head:  │    │ Prevents    │    │ [0.48→0.482]│    │ │
│  │  │ Automatic   │    │ ∂L/∂W = -0.12│    │ exploding  │    │             │    │ │
│  │  │ differentiat│    │             │    │ gradients   │    │ Portfolio:  │    │ │
│  │  │ ion through │    │ Portfolio:  │    │             │    │ [0.31→0.307]│    │ │
│  │  │ computation │    │ ∂L/∂W = +0.08│    │ Scale down │    │             │    │ │
│  │  │ graph       │    │             │    │ if too large│    │ lr=0.0003   │    │ │
│  │  └─────────────┘    └─────────────┘    └─────────────┘    └─────────────┘    │ │
│  └─────────────────────────────────────────────────────────────────────────────────┘ │
│                                                                                     │
│  ┌─────────────────────────────────────────────────────────────────────────────────┐ │
│  │                           VALUE NETWORK BACKPROP                               │ │
│  │                                                                                 │ │
│  │  ┌─────────────┐    ┌─────────────┐    ┌─────────────┐    ┌─────────────┐    │ │
│  │  │ Loss        │    │ Gradient    │    │ Gradient    │    │ Weight      │    │ │
│  │  │ Computation │    │ Computation │    │ Clipping    │    │ Update      │    │ │
│  │  │             │    │             │    │             │    │             │    │ │
│  │  │ value_loss  │    │ ∂L/∂W =     │    │ clip_grad_  │    │ W_new =     │    │ │
│  │  │ = 3.85      │    │ 2×(pred-    │    │ norm(grads, │    │ W_old -     │    │ │
│  │  │             │    │ target)×∂pred│    │ max_norm=0.5│    │ lr × grad   │    │ │
│  │  │ value_loss. │    │ /∂W         │    │ )           │    │             │    │ │
│  │  │ backward()  │    │             │    │             │    │ Hidden1:    │    │ │
│  │  │             │    │ For MSE:    │    │ Stabilizes  │    │ [2.1→2.097] │    │ │
│  │  │ Computes    │    │ Simple      │    │ training    │    │             │    │ │
│  │  │ gradients   │    │ quadratic   │    │             │    │ Hidden2:    │    │ │
│  │  │ w.r.t. all  │    │ derivative  │    │ Prevents    │    │ [-1.8→-1.79]│    │ │
│  │  │ value net   │    │             │    │ instability │    │             │    │ │
│  │  │ parameters  │    │             │    │             │    │ Output:     │    │ │
│  │  │             │    │             │    │             │    │ [0.5→0.502] │    │ │
│  │  └─────────────┘    └─────────────┘    └─────────────┘    └─────────────┘    │ │
│  └─────────────────────────────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────────────────────────┘

Phase 6: Learning Rate Scheduling & Iteration Complete
┌─────────────────────────────────────────────────────────────────────────────────────┐
│                        OPTIMIZATION SCHEDULING                                     │
├─────────────────────────────────────────────────────────────────────────────────────┤
│                                                                                     │
│  ┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐                │
│  │ Learning Rate   │    │ Metric Tracking │    │ Next Iteration  │                │
│  │ Decay           │    │                 │    │ Preparation     │                │
│  │                 │    │ Policy Loss:    │    │                 │                │
│  │ Current iter:   │    │ -1.24 → -1.31   │    │ Collect new     │                │
│  │ 150/1000        │    │                 │    │ 2048 samples    │                │
│  │                 │    │ Value Loss:     │    │ with updated    │                │
│  │ Decay factor:   │    │ 3.85 → 3.12     │    │ policy          │                │
│  │ 1-(150/1000)    │    │                 │    │                 │                │
│  │ = 0.85          │    │ Mean Reward:    │    │ Improved policy │                │
│  │                 │    │ 12.3 → 14.7     │    │ should lead to  │                │
│  │ New LR:         │    │                 │    │ better actions  │                │
│  │ 0.0003 × 0.85   │    │ Goal Success:   │    │ and higher      │                │
│  │ = 0.000255      │    │ 45% → 52%       │    │ rewards         │                │
│  │                 │    │                 │    │                 │                │
│  └─────────────────┘    └─────────────────┘    └─────────────────┘                │
└─────────────────────────────────────────────────────────────────────────────────────┘

Learning Progress Visualization
┌─────────────────────────────────────────────────────────────────────────────────────┐
│                           WEIGHT EVOLUTION OVER TIME                               │
├─────────────────────────────────────────────────────────────────────────────────────┤
│                                                                                     │
│  Policy Network Goal Head Weight (Connection to "Take Goal" output):               │
│                                                                                     │
│  Iteration:    0      50     100    150    200    250    300 (Converged)           │
│  Weight:    0.45 → 0.48 → 0.52 → 0.54 → 0.55 → 0.556 → 0.557                     │
│  Meaning:   Random  Learning  Prefers  Strong   Very   Stable  Optimal             │
│             policy  goals    goals    goal     strong  policy  policy              │
│                    are good  more     bias     goal              learned           │
│                                      bias                                           │
│                                                                                     │
│  Value Network Prediction Accuracy:                                                │
│                                                                                     │
│  Iteration:    0      50     100    150    200    250    300                       │
│  MSE Loss:   12.5 → 8.3  → 5.1  → 3.8  → 2.9  → 2.1  → 1.8                      │
│  Meaning:   Poor   Better Decent  Good   Very   Excellent Near-                   │
│             predict predict predict predict good   predict  perfect                │
│             ions   ions   ions   ions   predict predict  predict                  │
│                                          ions   ions    ions                      │
│                                                                                     │
│  Result: Agent learns sophisticated goal-portfolio coordination strategy           │
└─────────────────────────────────────────────────────────────────────────────────────┘
```

## Key Insights from the Block Diagram

### 1. **Dual Network Training**
- Policy and Value networks train simultaneously but independently
- Each has its own loss function, gradients, and optimizer
- Policy learns "what to do", Value learns "how good is this state"

### 2. **PPO Clipping Mechanism**  
- Prevents excessive policy updates that could destabilize training
- Ratio clipping ensures conservative, stable learning
- Balances exploration with exploitation

### 3. **Gradient Flow**
- Automatic differentiation computes gradients through entire network
- Gradient clipping prevents exploding gradients
- Adam optimizer adapts learning rates per parameter

### 4. **Mini-batch Processing**
- 2048 experiences split into 8 mini-batches of 256
- Each mini-batch processed 4 times (epochs)
- Total: 32 weight updates per training iteration

### 5. **Learning Convergence**
- Weights gradually shift toward optimal policy
- Value predictions become more accurate over time  
- System learns goal-portfolio coordination through experience

This diagram shows how PPO transforms raw experiences into learned behavior through systematic gradient-based optimization.




  🔙 Phase 4: Backpropagation & Weight Updates

  This is where the neural networks actually learn by adjusting their weights based on the training data collected.

  Overview: Two Separate Networks Learning

  # PPO trains two networks simultaneously:
  1. Policy Network (Actor): Learns better action selection
  2. Value Network (Critic): Learns better state value prediction

  # Each network has its own:
  - Loss function
  - Optimizer
  - Backpropagation process


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
  total_policy_loss.backward()  # 

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

 Value Network Example

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
  












⏺ Conceptual Meaning of PPO Components

  1. old_log_prob vs new_log_prob - Policy Evolution Tracking

  Conceptual Meaning:
  - old_log_prob: "How confident was I in this action when I originally took it?"
  - new_log_prob: "How confident am I in this same action now, after learning?"

  GBWM Example

  # Scenario: Time=3, Wealth=$450k, Goal available (cost=$80k, utility=15)
  state = [0.3, 0.9]  # [normalized_time, normalized_wealth]
  action = [1, 8]     # [take_goal, aggressive_portfolio]

  # Week 1: When action was originally taken
  old_policy_probs = {
      'goal': [0.6, 0.4],     # 60% skip, 40% take
      'portfolio': [0.05, 0.08, 0.12, 0.15, 0.18, 0.15, 0.12, 0.08, 0.07, ...]
  }
  old_log_prob = log(0.4) + log(0.07) = -0.916 + (-2.659) = -3.575
  # "I was somewhat uncertain about taking this goal with aggressive portfolio"

  # Week 2: After seeing results and learning
  # Agent took goal, market crashed (-15%), goal was still valuable
  new_policy_probs = {
      'goal': [0.3, 0.7],     # 30% skip, 70% take (learned goals are valuable!)
      'portfolio': [0.08, 0.12, 0.18, 0.22, 0.15, 0.10, 0.08, 0.04, 0.03, ...]
  }
  new_log_prob = log(0.7) + log(0.04) = -0.357 + (-3.219) = -3.576
  # "I'm now more confident about taking goals, less confident about aggressive portfolios"

  Ratio Calculation:
  ratio = exp(new_log_prob - old_log_prob)
  ratio = exp(-3.576 - (-3.575)) = exp(-0.001) = 0.999
  # "My confidence in this exact action stayed almost the same"


⏺ 2. GAE (Generalized Advantage Estimation) - "How Good Was This Decision?"

  Conceptual Meaning:
  GAE answers: "Given what I thought would happen vs. what actually happened, was this action better or worse than expected?"

  The Advantage Concept

  # Basic advantage formula
  advantage = actual_outcome - expected_outcome

  # If advantage > 0: "Action was better than expected → do it more"
  # If advantage < 0: "Action was worse than expected → do it less"
  # If advantage ≈ 0: "Action was about as expected → no change needed"

  GBWM GAE Example

  # Scenario: Agent at Time=2, Wealth=$400k
  state = [0.2, 0.8]
  action = [1, 5]  # Take goal, moderate portfolio

  # Value network's expectation
  value_estimate = 18.5  # "I expect 18.5 total utility from this state"

  # What actually happened over the episode
  actual_rewards = [12, 0, 0, 15, 0, 0, 8, 0]  # Goals taken at times 2, 5, 8
  discounted_return = 12 + 0.99^3 * 15 + 0.99^6 * 8 = 12 + 14.55 + 7.53 = 34.08

  # Raw advantage calculation
  raw_advantage = 34.08 - 18.5 = 15.58
  # "This action led to much better outcomes than expected!"

  # GAE smoothing (considers temporal credit assignment)
  # "Was the good outcome due to this specific action, or later actions?"
  gae_advantage = 12.3  # GAE reduces the advantage, attributing some success to later decisions

  Why GAE Matters for Portfolio Selection

  # Without GAE: Raw temporal difference
  # Problem: Late rewards get incorrectly attributed to early actions

  # Example problem scenario:
  # Time 2: Choose conservative portfolio → Small gains
  # Time 8: Take valuable goal → Large reward
  # Without GAE: Conservative portfolio gets credit for the goal reward!

  # With GAE: Proper credit assignment
  # Time 2: Conservative portfolio → Modest positive advantage
  # Time 8: Goal taking → Large positive advantage
  # Each decision gets appropriate credit for its actual contribution


⏺ 3. How PPO Optimizes Portfolio Selection with Respect to Goals

  The Core Optimization Challenge:
  "Given my current wealth and time remaining, which portfolio should I choose to maximize my ability to take valuable future goals while still growing 
  wealth?"

  The Learning Process

  Step 1: Exploration Phase
  # Early training - agent tries random combinations
  Episode 1: [Skip goal, Aggressive portfolio] → Wealth grows but missed utility
  Episode 2: [Take goal, Conservative portfolio] → Got utility but slow growth
  Episode 3: [Take goal, Aggressive portfolio] → Got utility, wealth volatile
  Episode 4: [Skip goal, Conservative portfolio] → Steady growth, missed opportunities

  Step 2: Pattern Recognition
  # Agent starts recognizing patterns through advantages:

  # Pattern 1: "When I have high wealth relative to goal cost, aggressive portfolios work"
  state = [0.3, 1.2]  # Time=3, Wealth=120% of initial
  action = [1, 12]    # Take goal, very aggressive portfolio
  advantage = +8.5    # "This worked well → increase probability"

  # Pattern 2: "When wealth is marginal, conservative portfolios preserve goal options"  
  state = [0.6, 0.9]  # Time=6, Wealth=90% of initial
  action = [1, 2]     # Take goal, conservative portfolio
  advantage = +3.2    # "This worked → increase probability"

  # Pattern 3: "Skipping early goals for portfolio growth can pay off"
  state = [0.1, 1.0]  # Time=1, Wealth=100%
  action = [0, 8]     # Skip goal, moderate-aggressive portfolio
  advantage = +5.1    # "This enabled taking more valuable later goals"

  Step 3: Policy Refinement
  # Agent develops sophisticated strategies:

  def learned_strategy(time, wealth, goal_cost, goal_utility):
      wealth_ratio = wealth / goal_cost
      time_remaining = 1.0 - time

      if wealth_ratio > 2.0 and goal_utility > 10:
          # "I can afford to take goal and be aggressive"
          return take_goal=True, portfolio=aggressive

      elif wealth_ratio > 1.5 and time_remaining < 0.3:
          # "Goal is affordable and time is running out"
          return take_goal=True, portfolio=moderate

      elif wealth_ratio < 1.2 and time_remaining > 0.5:
          # "Can't afford goal, focus on growth for future opportunities"
          return take_goal=False, portfolio=aggressive

      else:
          # "Borderline case, be conservative"
          return take_goal=True, portfolio=conservative


⏺ 4. Concrete GBWM Optimization Examples

  Example 1: Learning Goal-Portfolio Coordination

  Before Training:
  # State: Time=4, Wealth=$480k, Goal available (cost=$100k, utility=20)
  # Random policy probabilities:
  goal_probs = [0.5, 0.5]      # 50/50 random choice
  portfolio_probs = [0.067] * 15  # Equal probability for all portfolios

  # Action taken: [1, 14] - Take goal, maximum risk portfolio
  # Outcome: Goal achieved (+20 utility), but portfolio crashed (-30%), 
  #          leaving only $266k for remaining 6 years
  # Final episode utility: 20 (no more goals affordable)
  # Advantage: -15.3 (much worse than expected)

  After Training:
  # Same state: Time=4, Wealth=$480k, Goal available (cost=$100k, utility=20)
  # Learned policy probabilities:
  goal_probs = [0.2, 0.8]      # 80% take goal (learned goals are valuable)
  portfolio_probs = [0.02, 0.05, 0.12, 0.18, 0.25, 0.20, 0.12, 0.04, 0.02, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
                  # Conservative ←→ Moderate ←→ Aggressive (learned to avoid extremes when goals matter)

  # Action taken: [1, 4] - Take goal, moderate portfolio  
  # Outcome: Goal achieved (+20 utility), portfolio gained 8%, 
  #          leaving $410k for remaining 6 years
  # Final episode utility: 20 + 18 + 12 = 50 (two more goals taken)
  # Advantage: +12.7 (much better than expected)

  Example 2: Wealth Preservation Strategy

  Learning Scenario:
  # Multiple episodes teach wealth preservation:

  # Episode A: High wealth, aggressive portfolio
  state = [0.5, 1.4]  # Time=5, Wealth=$700k (high)
  action = [1, 11]    # Take goal, aggressive portfolio
  outcome = "Goal taken, portfolio volatile but high wealth cushions risk"
  advantage = +6.2

  # Episode B: Low wealth, aggressive portfolio  
  state = [0.5, 0.8]  # Time=5, Wealth=$400k (low)
  action = [1, 11]    # Take goal, aggressive portfolio  
  outcome = "Goal taken, portfolio crashed, no future goals possible"
  advantage = -8.9

  # Episode C: Low wealth, conservative portfolio
  state = [0.5, 0.8]  # Time=5, Wealth=$400k (low)
  action = [1, 2]     # Take goal, conservative portfolio
  outcome = "Goal taken, steady growth enables one more goal"
  advantage = +3.4

  # Learning result: Agent learns wealth-dependent portfolio selection

  Example 3: Temporal Goal Strategy

  Early Episode Learning:
  # Agent learns when to skip vs. take goals:

  # Scenario 1: Early goal, modest utility
  time = 2, goal_utility = 10, wealth = $500k
  learned_action = [0, 7]  # Skip goal, moderate-aggressive growth
  reasoning = "Early goal with low utility - invest for bigger future goals"

  # Scenario 2: Late goal, high utility
  time = 8, goal_utility = 25, wealth = $600k
  learned_action = [1, 3]  # Take goal, conservative preservation
  reasoning = "Valuable late goal - preserve wealth to ensure I can afford it"

  # Scenario 3: Final opportunity
  time = 9, goal_utility = 15, wealth = $450k
  learned_action = [1, 1]  # Take goal, very conservative
  reasoning = "Last chance for utility - minimize risk"

  The Optimization Feedback Loop

  # How the system continuously improves:

  1. **Experience Collection**: Try action combinations in various states
     ↓
  2. **Advantage Calculation**: Measure if outcomes were better/worse than expected
     ↓
  3. **Policy Update**: Increase probability of good combinations, decrease bad ones
     ↓
  4. **Value Update**: Improve predictions of future utility for better advantages
     ↓
  5. **Repeat**: Collect new experiences with improved policy

  Key Insights:
  - old_log_prob vs new_log_prob tracks how the agent's confidence in decisions evolves
  - GAE ensures credit assignment - portfolio choices get credit for wealth effects, goal choices get credit for utility effects
  - PPO optimization gradually shifts the agent toward goal-aware portfolio strategies that maximize long-term utility rather than just wealth growth

  The result is an agent that learns sophisticated coordination between goal timing and portfolio risk management.


⏺  why log probabilities are calculated twice and how they differ.

  Two Different Neural Networks, Two Different Log Probabilities

  The Key Insight

  The same state is evaluated by two different versions of the policy network:
  1. Old Policy: The network weights when action was originally taken
  2. New Policy: The network weights after some learning has occurred

  Timeline Breakdown

  Phase 1: Data Collection (Forward Pass)

  # Week 1: Collecting training data
  state = torch.tensor([0.3, 0.924])  # Time=3, Wealth=$462k

  # Policy network weights at this moment
  policy_weights_v1 = {
      'goal_head': [0.45, -0.23],
      'portfolio_head': [0.12, -0.08, 0.31, ...]
  }

  # Forward pass with old policy
  old_goal_probs, old_portfolio_probs = policy_net_v1.forward(state)
  # old_goal_probs = [0.6, 0.4]      # 60% skip, 40% take
  # old_portfolio_probs = [0.05, 0.08, 0.12, 0.15, 0.20, 0.18, ...]

  # Sample action
  action = [1, 5]  # Take goal, portfolio 5

  # Calculate log probability with OLD weights
  old_log_prob = log(0.4) + log(0.18) = -0.916 + (-1.715) = -2.631
  # Store this in training buffer

  Phase 2: PPO Learning (Weeks Later)

  # Week 3: Using collected data for training
  # Policy network has been updated multiple times since data collection

  # Policy network weights NOW (after learning)
  policy_weights_v2 = {
      'goal_head': [0.48, -0.19],      # Weights have changed!
      'portfolio_head': [0.15, -0.05, 0.28, ...]  # Different from v1
  }

  # Forward pass with NEW policy on SAME state
  new_goal_probs, new_portfolio_probs = policy_net_v2.forward(state)
  # new_goal_probs = [0.3, 0.7]      # 30% skip, 70% take (learned goals are good!)
  # new_portfolio_probs = [0.08, 0.12, 0.15, 0.20, 0.15, 0.12, ...]  # Different distribution

  # Calculate log probability for SAME action with NEW weights
  new_log_prob = log(0.7) + log(0.12) = -0.357 + (-2.120) = -2.477
  # This is different from old_log_prob!

  Visual Comparison

  # SAME STATE: [0.3, 0.924]
  # SAME ACTION: [1, 5] (take goal, portfolio 5)

  # OLD POLICY (Week 1):
  old_goal_probs =      [0.6,  0.4 ]  # Less confident about taking goals
  old_portfolio_probs = [... , 0.18, ...]  # Portfolio 5 had 18% probability
  old_log_prob = -2.631

  # NEW POLICY (Week 3):  
  new_goal_probs =      [0.3,  0.7 ]  # More confident about taking goals
  new_portfolio_probs = [... , 0.12, ...]  # Portfolio 5 now has 12% probability  
  new_log_prob = -2.477

  # RATIO CALCULATION:
  ratio = exp(new_log_prob - old_log_prob)
  ratio = exp(-2.477 - (-2.631)) = exp(0.154) = 1.167
  # "New policy is 16.7% more confident in this action combination"

  Why This Matters for PPO

  PPO's Core Question:

  "How much has my policy changed since I took this action?"

  # Ratio interpretation:
  if ratio > 1.0:
      # "I'm now MORE confident in this action"
      # "My policy has shifted to favor this action"

  if ratio < 1.0:
      # "I'm now LESS confident in this action"  
      # "My policy has shifted away from this action"

  if ratio ≈ 1.0:
      # "My confidence hasn't changed much"
      # "Policy is stable for this state-action pair"

  PPO Clipping Protection

  # Without ratio tracking, bad things happen:
  advantage = +5.0  # Action was very good
  # If new policy became 1000x more confident: huge update!
  # Agent might become overconfident and stop exploring

  # With PPO clipping:
  if ratio > 1.2:  # Clip to prevent excessive updates
      clipped_ratio = 1.2
  elif ratio < 0.8:
      clipped_ratio = 0.8
  else:
      clipped_ratio = ratio

  # Conservative update protects against policy collapse

  Concrete GBWM Example

  Learning Scenario: "Goals are More Valuable Than I Thought"

  # Training batch contains multiple experiences like:
  # State: [0.4, 0.9], Action: [1, 6], Advantage: +8.2  # Taking goals was great!
  # State: [0.6, 0.8], Action: [1, 4], Advantage: +6.1  # Taking goals was great!
  # State: [0.2, 1.1], Action: [0, 8], Advantage: -3.4  # Skipping goals was bad!

  # OLD POLICY (before seeing these experiences):
  for state in [[0.4, 0.9], [0.6, 0.8], [0.2, 1.1]]:
      old_goal_probs = [0.65, 0.35]  # Generally preferred skipping goals
      old_log_prob_take = log(0.35) = -1.05

  # NEW POLICY (after learning from experiences):  
  for state in [[0.4, 0.9], [0.6, 0.8], [0.2, 1.1]]:
      new_goal_probs = [0.25, 0.75]  # Now prefers taking goals!
      new_log_prob_take = log(0.75) = -0.29

  # POLICY SHIFT MEASUREMENT:
  ratio = exp(-0.29 - (-1.05)) = exp(0.76) = 2.14
  # "Policy is now 214% more confident about taking goals"

  # PPO CLIPPING:
  clipped_ratio = min(2.14, 1.2) = 1.2  # Prevent excessive update
  # "Limit the policy change to 20% increase per update for stability"

  Summary

  Two different log probabilities because:
  1. old_log_prob: Calculated with network weights from data collection time
  2. new_log_prob: Calculated with current network weights (after learning)

  Their ratio tells PPO:
  - How much the policy has changed
  - Whether to increase or decrease action probabilities
  - How big an update to make (with clipping for safety)

  This is the core mechanism that allows PPO to learn from experience while maintaining training stability.

> block diagram of ppo learning and bakcpropagation


     ```
     ┌─────────────────────────────────────────────────────────────────────────────────────┐
     │                           PPO TRAINING ITERATION CYCLE                             │
     └─────────────────────────────────────────────────────────────────────────────────────┘
     Phase 1: Data Collection
     … +263 lines (ctrl+r to expand)

⏺ I've created a comprehensive block diagram that shows the complete PPO learning and backpropagation process. The diagram covers:

  6 Main Phases:

  1. Experience Collection - How the agent interacts with GBWM environment
  2. Advantage Computation - GAE calculation for training signals
  3. PPO Updates - Mini-batch processing with current vs old policy
  4. Loss Computation - Policy and Value loss calculations
  5. Backpropagation - Gradient computation and weight updates
  6. Learning Scheduling - Learning rate decay and progress tracking

  Key Technical Details:

  - Dual Network Architecture: Separate policy and value networks with independent training
  - PPO Clipping: Ratio clipping mechanism preventing excessive policy changes
  - Mini-batch Processing: 2048 samples → 8 mini-batches × 4 epochs = 32 updates
  - Gradient Management: Automatic differentiation, clipping, and Adam optimization
  - Weight Evolution: Shows how network parameters evolve toward optimal policy

  GBWM-Specific Examples:

  - State representations: [normalized_time, normalized_wealth]
  - Action spaces: Goal decisions [skip, take] and portfolio choices [0-14]
  - Reward signals: Utility from goal achievement
  - Learning progression: From random to sophisticated goal-portfolio coordination


⏺ Excellent question! Let me explain what happens in the very first iteration of PPO training, showing how old_log_prob and new_log_prob can differ even on
  the first update.

  First PPO Iteration: Why Log Probabilities Differ

  The Key Insight

  Even in the first iteration, the policy network gets updated multiple times during the PPO learning phase, so the "old" and "new" versions exist within the
   same iteration.

  Timeline Breakdown

  Step 1: Data Collection (Beginning of First Iteration)

  # Initialize random policy network weights
  policy_weights_initial = {
      'goal_head_weights': [0.1, -0.05],      # Random initialization
      'portfolio_head_weights': [0.02, 0.08, -0.03, 0.12, ...]  # Random values
  }

  # State: Time=2, Wealth=$450k
  state = torch.tensor([0.2, 0.9])  # [normalized_time, normalized_wealth]

  # Forward pass with initial weights
  goal_logits = goal_head(shared_features) = [0.3, 0.7]
  portfolio_logits = portfolio_head(shared_features) = [0.1, 0.4, 0.2, 0.8, 0.3, ...]

  # Apply softmax to get probabilities
  goal_probs = softmax([0.3, 0.7]) = [0.45, 0.55]      # 45% skip, 55% take
  portfolio_probs = softmax([0.1, 0.4, 0.2, 0.8, 0.3, ...]) = [0.067, 0.089, 0.073, 0.134, 0.081, ...]

  # Sample action using these probabilities
  action = [1, 3]  # Take goal, choose portfolio 3

  # Calculate old_log_prob (stored for later)
  old_log_prob = log(0.55) + log(0.134) = -0.598 + (-2.009) = -2.607

  Step 2: Collect Full Batch

  # Collect 2048 experiences using the SAME initial policy weights
  # All experiences use the same policy, so they're all "old" relative to future updates

  experiences = []
  for i in range(2048):
      state = env.get_state()
      action, log_prob = policy_net.get_action_and_log_prob(state)  # Same weights
      reward, next_state = env.step(action)

      experiences.append({
          'state': state,
          'action': action,
          'old_log_prob': log_prob,  # This becomes the "old" reference
          'reward': reward
      })

  Step 3: Compute Advantages (Still First Iteration)

  # Process the 2048 experiences to compute advantages
  advantages, returns = compute_gae(rewards, values, dones, gamma=0.99, gae_lambda=0.95)

  # Our example state now has:
  # state = [0.2, 0.9]
  # action = [1, 3] 
  # old_log_prob = -2.607  (from data collection)
  # advantage = +3.2       (action was better than expected)
  # return = 18.5          (actual discounted future reward)

  Step 4: PPO Learning Begins (Still First Iteration!)

  Mini-batch 1, Epoch 1:
  # Extract our example from mini-batch
  mini_batch = {
      'states': torch.tensor([[0.2, 0.9], ...]),
      'actions': torch.tensor([[1, 3], ...]),
      'old_log_probs': torch.tensor([-2.607, ...]),  # From data collection
      'advantages': torch.tensor([3.2, ...]),
      'returns': torch.tensor([18.5, ...])
  }

  # Forward pass with SAME initial weights (no updates yet)
  new_log_probs, entropy = policy_net.evaluate_actions(mini_batch['states'], mini_batch['actions'])
  # new_log_probs[0] = -2.607  (identical to old_log_prob)

  # Compute PPO loss and update weights
  ratio = exp(-2.607 - (-2.607)) = exp(0) = 1.0  # No difference yet
  policy_loss = compute_ppo_loss(ratio, advantages)
  policy_loss.backward()
  optimizer.step()  # FIRST WEIGHT UPDATE!

  # Weights are now different!
  policy_weights_after_update1 = {
      'goal_head_weights': [0.102, -0.048],  # Slightly changed
      'portfolio_head_weights': [0.022, 0.082, -0.028, 0.118, ...]
  }

  Mini-batch 2, Epoch 1 (Same First Iteration!):
  # Process same data again, but with updated weights
  mini_batch = {
      'states': torch.tensor([[0.2, 0.9], ...]),      # Same state
      'actions': torch.tensor([[1, 3], ...]),         # Same action
      'old_log_probs': torch.tensor([-2.607, ...]),   # Still from original collection
      'advantages': torch.tensor([3.2, ...]),         # Same advantages
      'returns': torch.tensor([18.5, ...])            # Same returns
  }

  # Forward pass with UPDATED weights
  new_goal_probs, new_portfolio_probs = policy_net.forward(state)
  # Due to weight changes:
  new_goal_probs = [0.44, 0.56]         # Slightly different from [0.45, 0.55]
  new_portfolio_probs = [0.069, 0.087, 0.074, 0.136, ...]  # Slightly different

  # Calculate new_log_prob with updated network
  new_log_prob = log(0.56) + log(0.136) = -0.580 + (-1.994) = -2.574

  # Now they differ!
  ratio = exp(new_log_prob - old_log_prob)
  ratio = exp(-2.574 - (-2.607)) = exp(0.033) = 1.034
  # "New policy is 3.4% more confident in this action"

  Visual Timeline of First Iteration

  Time:     0ms    50ms    100ms   150ms   200ms   250ms   300ms
  Event:    Init   Collect PPO     PPO     PPO     PPO     End
            Policy  2048    Update  Update  Update  Update  Iter
                    samples #1      #2      #3      #4      

  Weights:  W₀      W₀      W₁      W₂      W₃      W₄

  old_log_prob: Always -2.607 (from initial collection with W₀)
  new_log_prob:         -2.607  -2.574  -2.541  -2.508  (changes with W₁,W₂,W₃,W₄)

  Concrete Example with Numbers

  # State: [0.2, 0.9] - Time=20%, Wealth=90%
  # Action: [1, 3] - Take goal, Portfolio 3

  # Data Collection (W₀):
  old_log_prob = -2.607

  # PPO Update #1 (W₀ → W₁):
  new_log_prob = -2.607  # Same as old, ratio = 1.0
  # Small weight update based on advantage = +3.2

  # PPO Update #2 (W₁ → W₂):  
  new_log_prob = -2.574  # Slightly higher confidence, ratio = 1.034
  # Advantage > 0, so increase probability of this action

  # PPO Update #3 (W₂ → W₃):
  new_log_prob = -2.541  # Even higher confidence, ratio = 1.068
  # Continue increasing probability

  # PPO Update #4 (W₃ → W₄):
  new_log_prob = -2.508  # Highest confidence, ratio = 1.104
  # Policy has learned this action is good in this state

  Why This Happens in First Iteration

  1. Multiple Mini-batches: 2048 samples split into 8 mini-batches
  2. Multiple Epochs: Each mini-batch processed 4 times
  3. Progressive Updates: 8 × 4 = 32 weight updates in one iteration
  4. Same Data, Different Network: Each update uses same data but updated weights

  The Learning Effect

  # By end of first iteration, policy has learned:
  # "In state [0.2, 0.9] with action [1, 3], I got advantage +3.2"
  # "This means taking goals with moderate portfolios is better than I initially thought"
  # "I should increase the probability of similar actions in similar states"

  # Result: Even after just one iteration, policy is measurably different
  initial_confidence = exp(-2.607) = 0.074  # 7.4% probability
  final_confidence = exp(-2.508) = 0.081     # 8.1% probability  
  # Improvement: +9.5% relative increase in action probability

  This demonstrates how PPO learns incrementally within each iteration, using the same collected data multiple times with progressively updated policy
  weights.

