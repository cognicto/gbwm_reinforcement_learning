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