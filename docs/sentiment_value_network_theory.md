# Sentiment-Aware Value Network: Theoretical Foundation and Conceptual Framework

**Technical Documentation**  
**Version:** 1.0  
**Date:** November 30, 2025  
**Focus:** Deep Theoretical Analysis of Sentiment-Aware Critic Networks

---

## Table of Contents

1. [Theoretical Foundation](#theoretical-foundation)
2. [Value Function Theory in Regime-Switching Environments](#value-function-theory-in-regime-switching-environments)
3. [Mathematical Framework for Sentiment Integration](#mathematical-framework-for-sentiment-integration)
4. [Temporal Decomposition Theory](#temporal-decomposition-theory)
5. [Regime-Dependent Value Estimation](#regime-dependent-value-estimation)
6. [Information-Theoretic Analysis](#information-theoretic-analysis)
7. [Learning Dynamics and Convergence](#learning-dynamics-and-convergence)
8. [Uncertainty Quantification Theory](#uncertainty-quantification-theory)
9. [Behavioral Finance Integration](#behavioral-finance-integration)
10. [Theoretical Validation Framework](#theoretical-validation-framework)

---

## Theoretical Foundation

### The Fundamental Challenge

Traditional value function estimation in reinforcement learning assumes **environmental stationarity** - that the expected future rewards from any given state remain constant over time. However, financial markets exhibit **regime-switching behavior** where identical fundamental states (time, wealth) can have vastly different values depending on market sentiment conditions.

**Core Problem**: In sentiment-driven environments, the value function becomes **regime-dependent**:
```
V^π(s_t) ≠ constant across market regimes
```

This violates the basic stationarity assumption underlying most value function approximation methods.

### Regime-Dependent Value Theory

**Definition**: A regime-dependent value function acknowledges that expected cumulative rewards vary systematically with market conditions:

```
V^π(s_t, R_t) = E[∑_{τ=t}^T γ^(τ-t) r_τ | s_t, R_t, π]
```

Where:
- `s_t` = fundamental state (time, wealth)
- `R_t` = market regime (fear, normal, greed)
- `r_τ` = regime-dependent rewards

**Key Insight**: The same wealth level at the same time can have different values depending on whether the market is in a fear regime (potential for recovery) versus a greed regime (potential for correction).

### Theoretical Decomposition

The sentiment-aware value function can be decomposed into **additive components**:

```
V^π(s_t, R_t) = V^π_base(s_base) + V^π_sentiment(s_sentiment, R_t) + V^π_interaction(s_base, s_sentiment)
```

Where:
- **Base Value**: `V^π_base(s_base)` captures fundamental wealth accumulation dynamics
- **Sentiment Premium**: `V^π_sentiment(s_sentiment, R_t)` represents regime-dependent adjustments
- **Interaction Value**: `V^π_interaction` captures cross-effects between fundamental and sentiment states

This decomposition enables **modular learning** where each component can be understood and optimized separately.

---

## Value Function Theory in Regime-Switching Environments

### Bellman Equation Extension

The classical Bellman equation:
```
V^π(s) = E[r(s,a) + γV^π(s') | π]
```

Must be extended to account for regime transitions:
```
V^π(s,R) = E[r(s,a,R) + γE[V^π(s',R') | R] | π]
```

**Regime Transition Model**: The expectation over future regimes requires modeling regime dynamics:
```
P(R_{t+1} | R_t, sentiment_features_t)
```

This introduces **temporal dependencies** in regime evolution that the value function must capture.

### Fixed Point Theory

**Existence and Uniqueness**: Under standard assumptions (bounded rewards, discount factor γ < 1), the regime-dependent Bellman operator has a unique fixed point.

**Convergence Properties**: The value function converges to this fixed point, but the convergence rate depends on:
1. **Regime stability**: How frequently regimes change
2. **Regime persistence**: How long each regime lasts
3. **Transition predictability**: How well regime changes can be anticipated

### Non-Stationarity Challenges

**Temporal Inconsistency**: Value estimates may become inconsistent when:
1. **Regime distributions shift**: Training vs. deployment regime frequencies differ
2. **Regime definitions evolve**: VIX thresholds for fear/greed change over time
3. **Structural breaks**: Market dynamics fundamentally change

**Theoretical Solution**: **Adaptive value estimation** that continuously updates regime models based on recent observations.

---

## Mathematical Framework for Sentiment Integration

### State Space Augmentation Theory

**Original State Space**: `S = {(t, W) : t ∈ [0,T], W ∈ ℝ+}`

**Augmented State Space**: `S' = {(t, W, S_vix, M_vix) : t ∈ [0,T], W ∈ ℝ+, S_vix ∈ [-1,1], M_vix ∈ [-1,1]}`

**Dimensionality Analysis**:
- **Benefit**: Richer state representation enables regime-aware decisions
- **Cost**: Increased sample complexity O(|S'|) vs. O(|S|)
- **Trade-off**: The additional dimensions must provide sufficient value to justify the complexity

### Sentiment Feature Mathematics

**VIX Sentiment Transformation**:
```
S_vix(t) = tanh((VIX(t) - μ_long_term) / σ_normalization)
```

**Properties**:
- **Bounded**: S_vix ∈ [-1, 1] ensures numerical stability
- **Centered**: Zero corresponds to long-term VIX average
- **Symmetric**: Equal sensitivity to fear and greed extremes

**VIX Momentum Calculation**:
```
M_vix(t) = tanh((VIX(t) - VIX(t-δ)) / (ε × VIX(t-δ)))
```

Where δ = lookback period, ε = normalization factor

**Properties**:
- **Mean-reverting**: Captures temporary deviations from trend
- **Bounded**: Prevents explosive momentum values
- **Scale-invariant**: Normalized by current VIX level

### Value Function Approximation Theory

**Linear Decomposition**:
```
V^π(s) = α₀ + α₁·t + α₂·W + α₃·S_vix + α₄·M_vix + ∑ᵢⱼ αᵢⱼ·fᵢ·fⱼ
```

Where fᵢ represents individual features and αᵢⱼ captures interaction effects.

**Neural Network Approximation**:
```
V^π(s) = f_θ(s) = universal_approximator(heterogeneous_encoder(s))
```

**Theoretical Justification**: Universal approximation theorem guarantees that sufficiently large networks can approximate any continuous value function to arbitrary accuracy.

---

## Temporal Decomposition Theory

### Multi-Horizon Value Estimation

Financial decisions involve **multiple time horizons** with different sentiment sensitivity:

**Short-term Horizon** (1-4 years):
- **High sentiment sensitivity**: Current market conditions strongly influence near-term returns
- **Volatility dominance**: VIX effects are most pronounced
- **Behavioral factors**: Fear and greed have immediate impact

**Medium-term Horizon** (5-8 years):
- **Moderate sentiment sensitivity**: Some mean reversion dampens current conditions
- **Transition effects**: Regime changes become more likely
- **Mixed influences**: Both fundamental and sentiment factors matter

**Long-term Horizon** (9+ years):
- **Low sentiment sensitivity**: Fundamental wealth dynamics dominate
- **Mean reversion**: Markets tend toward long-term equilibrium
- **Fundamental factors**: Company performance, economic growth drive returns

### Mathematical Formulation

**Horizon-Weighted Value Function**:
```
V^π_total(s) = ∑ʰ₌₁ᴴ wₕ(t) · V^π_h(s)
```

Where:
- `V^π_h(s)` = h-step value function
- `wₕ(t)` = time-dependent horizon weights
- `H` = maximum horizon (investment period length)

**Weight Function Properties**:
```
wₕ(t) = softmax(αₕ + βₕ·(T-t))
```

- **Adaptive**: Weights change based on time remaining
- **Normalized**: ∑wₕ = 1 ensures proper value scaling
- **Learnable**: αₕ, βₕ parameters adapt during training

### Temporal Consistency Theory

**Consistency Requirement**: Value estimates across horizons should be **temporally coherent**:
```
V^π_h(s,t) ≈ r(s,π) + γ·V^π_{h-1}(s',t+1)
```

**Violation Consequences**: Inconsistent estimates lead to:
- **Policy instability**: Conflicting value signals
- **Suboptimal decisions**: Incorrect action ranking
- **Training instability**: Oscillating loss functions

**Regularization Solution**: **Temporal consistency loss** encourages coherent multi-horizon estimates.

---

## Regime-Dependent Value Estimation

### Regime Classification Theory

**Continuous Regime Representation**: Rather than discrete regime switching, use **soft regime probabilities**:

```
P(R_t = fear | s_t) = σ(w_fear · sentiment_features + b_fear)
P(R_t = greed | s_t) = σ(w_greed · sentiment_features + b_greed)
P(R_t = normal | s_t) = 1 - P(fear) - P(greed)
```

**Advantages**:
- **Smooth transitions**: Avoids discontinuous value jumps
- **Uncertainty representation**: Captures ambiguous market conditions
- **Gradient flow**: Enables end-to-end learning of regime definitions

### Regime-Specific Value Heads

**Theoretical Motivation**: Different regimes require **specialized value estimation**:

**Fear Regime Characteristics**:
- **Elevated uncertainty**: Higher variance in return expectations
- **Mean reversion tendency**: High volatility often precedes recovery
- **Risk premium**: Fear creates buying opportunities

**Greed Regime Characteristics**:
- **Overconfidence**: Lower perceived risk than actual risk
- **Momentum effects**: Trends tend to continue short-term
- **Correction potential**: Overvaluation increases crash risk

**Normal Regime Characteristics**:
- **Equilibrium conditions**: Standard risk-return relationships
- **Predictable patterns**: Historical relationships hold
- **Balanced expectations**: Neither extreme optimism nor pessimism

### Value Aggregation Theory

**Weighted Combination**:
```
V^π(s) = ∑_R P(R|s) · V^π_R(s)
```

**Properties**:
- **Probabilistic consistency**: Weights sum to one
- **Regime specialization**: Each head optimizes for specific conditions
- **Smooth interpolation**: Gradual transitions between regimes

**Alternative Formulations**:

**Gating Mechanism**:
```
V^π(s) = gate_R(s) ⊙ V^π_R(s)
```

Where ⊙ represents element-wise multiplication and gate_R produces regime-specific activation patterns.

**Hierarchical Composition**:
```
V^π(s) = V^π_base(s) + ∑_R P(R|s) · ΔV^π_R(s)
```

Where ΔV^π_R represents regime-specific adjustments to a base value estimate.

---

## Information-Theoretic Analysis

### Mutual Information Decomposition

The value network must capture relationships between state components and expected returns:

```
I(Value; State) = I(Value; Time) + I(Value; Wealth) + I(Value; Sentiment) - Redundancy_terms
```

**Component Analysis**:

**Temporal Information**: `I(Value; Time)`
- **Low entropy**: Time progression is deterministic
- **High predictive value**: Goal urgency increases with time
- **Simple patterns**: Mostly linear or exponential relationships

**Wealth Information**: `I(Value; Wealth)`
- **Medium entropy**: Wealth follows stochastic process
- **High predictive value**: Wealth directly enables goal achievement
- **Complex patterns**: Nonlinear risk preferences and affordability thresholds

**Sentiment Information**: `I(Value; Sentiment)`
- **High entropy**: Market sentiment is highly variable
- **Medium predictive value**: Important but secondary to fundamentals
- **Moderate complexity**: Regime-switching with mean reversion

### Information Bottleneck for Value Networks

**Objective**: Learn minimal sufficient representations for value estimation:
```
minimize β·I(S; Ψ) - I(Ψ; V*)
```

Where:
- `Ψ` = learned state representation
- `V*` = true value function
- `β` = compression-prediction trade-off

**Optimal Representation**: The bottleneck solution provides the most compressed representation that preserves value-relevant information.

### Entropy Regularization Effects

**Value Function Smoothness**: Entropy regularization encourages **smooth value landscapes**:
- **Reduces overfitting**: Prevents sharp value discontinuities
- **Improves generalization**: More robust to unseen states
- **Stabilizes learning**: Reduces gradient variance

---

## Learning Dynamics and Convergence

### Temporal Difference Learning with Regimes

**Standard TD Error**:
```
δ_t = r_t + γV^π(s_{t+1}) - V^π(s_t)
```

**Regime-Aware TD Error**:
```
δ_t^regime = r_t + γE[V^π(s_{t+1}, R_{t+1}) | R_t] - V^π(s_t, R_t)
```

**Key Differences**:
- **Expectation over regimes**: Must account for regime transition probabilities
- **Regime persistence**: Current regime influences future regime likelihood
- **Transition timing**: Regime changes create prediction challenges

### Convergence Theory

**Classical Convergence Conditions**:
1. **Robbins-Monro conditions**: Learning rates satisfy ∑αₜ = ∞, ∑αₜ² < ∞
2. **Bounded rewards**: |r(s,a)| ≤ R_max
3. **Discount factor**: 0 ≤ γ < 1

**Regime-Dependent Extensions**:

**Condition 1 - Regime Balance**: Training must observe sufficient examples from all regimes:
```
min_R (frequency_of_regime_R) ≥ ε > 0
```

**Condition 2 - Regime Stability**: Regime transition probabilities must be stationary:
```
P(R_{t+1}|R_t, S_t) = P(R_{t+1}|R_t, S_t') for similar S_t, S_t'
```

**Condition 3 - Approximation Capacity**: Network must have sufficient capacity to represent regime-specific patterns.

### Learning Phase Dynamics

**Phase 1 - Regime Discovery (0-25% training)**:
- **Objective**: Identify distinct behavioral patterns across regimes
- **Dynamics**: High value prediction errors during regime transitions
- **Indicators**: Large TD errors, unstable value estimates

**Phase 2 - Regime Specialization (25-60% training)**:
- **Objective**: Develop regime-specific value estimation capabilities
- **Dynamics**: Improved within-regime accuracy, persistent cross-regime errors
- **Indicators**: Decreasing within-regime TD errors, stable regime classification

**Phase 3 - Transition Modeling (60-85% training)**:
- **Objective**: Learn smooth transitions between regimes
- **Dynamics**: Focus on boundary conditions and transition periods
- **Indicators**: Reduced transition-period prediction errors

**Phase 4 - Fine-tuning (85-100% training)**:
- **Objective**: Optimize overall value estimation accuracy
- **Dynamics**: Minor adjustments to regime boundaries and interaction effects
- **Indicators**: Stable TD errors across all regimes and transitions

---

## Uncertainty Quantification Theory

### Sources of Uncertainty

**Aleatoric Uncertainty** (irreducible):
- **Market randomness**: Fundamental unpredictability of financial returns
- **Regime stochasticity**: Random timing of regime transitions
- **External shocks**: Unpredictable events affecting markets

**Epistemic Uncertainty** (reducible with more data):
- **Model uncertainty**: Limited network capacity or architecture choices
- **Parameter uncertainty**: Uncertainty in learned weights
- **Regime uncertainty**: Ambiguity in current regime classification

### Ensemble Methods Theory

**Theoretical Foundation**: Multiple independent value estimators provide **uncertainty quantification**:

```
V̂_ensemble(s) = (1/K) ∑ᵏ₌₁ᴷ V^π_k(s)
Uncertainty(s) = Var({V^π_k(s)}ᵏ₌₁ᴷ)
```

**Benefits**:
- **Uncertainty estimates**: Standard deviation indicates confidence
- **Robustness**: Averaging reduces individual model errors
- **Risk awareness**: High uncertainty signals need for caution

**Theoretical Guarantees**: Under independence assumptions, ensemble variance decreases as O(1/K).

### Bayesian Neural Networks for Value Estimation

**Theoretical Framework**: Treat network weights as random variables with posterior distributions:

```
p(θ|D) ∝ p(D|θ)p(θ)
```

**Value Distribution**: Value estimates become distributions rather than point estimates:
```
p(V(s)|D) = ∫ p(V(s)|θ)p(θ|D)dθ
```

**Practical Benefits**:
- **Principled uncertainty**: Bayesian framework provides theoretical foundation
- **Calibrated confidence**: Uncertainty estimates are well-calibrated
- **Decision support**: Can incorporate uncertainty into decision making

---

## Behavioral Finance Integration

### Prospect Theory in Value Functions

**Standard Utility Theory**: Linear value in probabilities and outcomes
```
U(lottery) = ∑ᵢ pᵢu(xᵢ)
```

**Prospect Theory Extensions**:

**Reference Dependence**: Values are relative to reference points
```
v(x) = v(x - reference_point)
```

**Loss Aversion**: Losses weigh more heavily than equivalent gains
```
v(x) = { x^α         if x ≥ 0
        { -λ(-x)^β   if x < 0
```

Where λ > 1 represents loss aversion coefficient.

**Probability Weighting**: Systematic distortions in probability perception
```
w(p) = p^γ / (p^γ + (1-p)^γ)^(1/γ)
```

### Sentiment-Dependent Behavioral Parameters

**Dynamic Reference Points**: Reference points shift with market conditions:
```
reference_point(S_t) = base_reference + sentiment_adjustment(S_t)
```

**Regime-Dependent Loss Aversion**: Fear increases loss aversion:
```
λ(S_t) = λ_base + λ_sentiment · max(0, S_t)
```

**Probability Weighting Sensitivity**: Market conditions affect probability perception:
```
γ(S_t) = γ_base + γ_sentiment · S_t
```

### Mental Accounting in Multi-Goal Settings

**Theoretical Framework**: Individuals maintain separate mental accounts for different goals:

**Security Account** (High VIX):
- **Purpose**: Capital preservation during market stress
- **Risk tolerance**: Very low
- **Time horizon**: Short-term focus

**Growth Account** (Low VIX):
- **Purpose**: Wealth accumulation during favorable conditions
- **Risk tolerance**: High
- **Time horizon**: Long-term focus

**Balanced Account** (Normal VIX):
- **Purpose**: Standard goal progression
- **Risk tolerance**: Moderate
- **Time horizon**: Goal-dependent

**Value Function Implications**: Different accounts have different value functions, weighted by market regime probabilities.

---

## Theoretical Validation Framework

### Theoretical Predictions

**Hypothesis 1 - Regime-Dependent Value Ranking**:
Identical states should have different value estimates across regimes:
```
V^π(s | fear_regime) ≠ V^π(s | greed_regime) for many states s
```

**Testable Prediction**: Value estimates should show statistically significant differences across regimes when controlling for fundamental state variables.

**Hypothesis 2 - Temporal Sentiment Decay**:
Sentiment influence should decay with time horizon:
```
∂V^π/∂sentiment_features ∝ 1/time_horizon
```

**Testable Prediction**: Gradient magnitudes with respect to sentiment should be inversely related to remaining time.

**Hypothesis 3 - Mean Reversion in Value Estimates**:
Extreme sentiment should increase estimated future value:
```
V^π(s | extreme_sentiment) > V^π(s | normal_sentiment) when controlling for current wealth
```

**Testable Prediction**: High VIX periods should increase value estimates due to mean reversion expectations.

### Empirical Testing Framework

**Controlled Experiments**:

**Regime Isolation Tests**: Compare value estimates for identical fundamental states across different sentiment conditions.

**Temporal Sensitivity Analysis**: Measure how value gradient magnitudes change with remaining time to goals.

**Cross-Regime Generalization**: Test value network performance on regimes not seen during training.

**Behavioral Consistency Checks**: Verify that learned patterns align with established behavioral finance theories.

### Expected Theoretical Relationships

**Value-Sentiment Correlation**: 
- **Fear regime**: Negative correlation (high fear → higher expected future returns)
- **Greed regime**: Positive correlation (low fear → current valuations more reliable)

**Temporal Pattern**: 
- **Short horizon**: Strong sentiment effects
- **Long horizon**: Weak sentiment effects

**Wealth Interaction**: 
- **Low wealth**: High sensitivity to sentiment (limited buffer)
- **High wealth**: Low sensitivity to sentiment (sufficient buffer)

**Goal Proximity Effects**: 
- **Near goals**: High sensitivity (immediate relevance)
- **Distant goals**: Low sensitivity (time for mean reversion)

---

## Conclusion

The sentiment-aware value network represents a **fundamental advancement** in financial reinforcement learning theory by:

### Theoretical Contributions

1. **Regime-Dependent Value Theory**: Formalizes how market conditions affect expected future rewards
2. **Temporal Decomposition Framework**: Provides mathematical foundation for multi-horizon value estimation
3. **Uncertainty Quantification Methods**: Establishes principled approaches to confidence estimation
4. **Behavioral Integration**: Incorporates established psychological and behavioral finance principles
5. **Convergence Analysis**: Extends classical RL convergence theory to regime-switching environments

### Practical Implications

1. **Market-Adaptive Decision Making**: Enables policies that respond appropriately to changing market conditions
2. **Risk-Aware Planning**: Provides uncertainty estimates for robust decision making
3. **Behavioral Realism**: Captures human-like biases and preferences in automated systems
4. **Empirical Validation**: Offers testable hypotheses for validating theoretical predictions

### Future Research Directions

1. **Dynamic Regime Models**: Extend to time-varying regime definitions
2. **Multi-Asset Sentiment**: Incorporate sentiment from multiple market indicators
3. **Non-Linear Regime Interactions**: Explore complex regime dependencies
4. **Real-Time Adaptation**: Develop online learning methods for changing market conditions

This theoretical framework provides the **mathematical foundation** and **conceptual understanding** necessary for developing sophisticated sentiment-aware value estimation systems that can navigate the complexities of real financial markets while maintaining theoretical rigor and empirical validity.