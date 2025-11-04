# Soft Actor-Critic (SAC) Algorithm Explained

This document provides a comprehensive explanation of the Soft Actor-Critic (SAC) algorithm used in this project, including all parameters and training metrics.

## Table of Contents
- [What is SAC?](#what-is-sac)
- [Key Concepts](#key-concepts)
- [Network Architecture](#network-architecture)
- [Hyperparameters](#hyperparameters)
- [Training Metrics](#training-metrics)
- [Interpreting Results](#interpreting-results)

---

## What is SAC?

**Soft Actor-Critic (SAC)** is a state-of-the-art off-policy reinforcement learning algorithm designed for continuous action spaces. It combines three powerful ideas:

1. **Actor-Critic Architecture**: Separates policy (actor) from value estimation (critic)
2. **Maximum Entropy Reinforcement Learning**: Encourages exploration by maximizing both reward and entropy
3. **Twin Q-Networks**: Reduces overestimation bias using two critic networks

### Why SAC for CarRacing?

- **Continuous Control**: Naturally handles continuous actions (steering, gas, brake)
- **Sample Efficient**: Learns from off-policy data using replay buffer
- **Stable**: Automatic entropy tuning prevents premature convergence
- **Robust**: Works well across diverse environments

---

## Key Concepts

### 1. Actor-Critic Framework

- **Actor (Policy Network)**: Decides what action to take
  - Outputs: Mean and log-std of Gaussian distribution
  - Samples actions using reparameterization trick
  - Goal: Maximize expected return + entropy

- **Critic (Value Network)**: Evaluates how good actions are
  - Two Q-networks (twin critics) to reduce overestimation
  - Estimates Q(s, a): expected return from state s taking action a
  - Goal: Accurately predict future rewards

### 2. Maximum Entropy Objective

SAC maximizes the objective:

```
J(π) = E[Σ r(s,a) + α·H(π(·|s))]
```

Where:
- `r(s,a)` = reward at state s, action a
- `α` = entropy coefficient (temperature parameter)
- `H(π(·|s))` = policy entropy (measure of randomness)

**Benefits:**
- Encourages exploration (high entropy = more random actions)
- Prevents collapse to suboptimal deterministic policy
- Makes policy more robust to perturbations

### 3. Twin Q-Networks (Clipped Double Q-Learning)

Uses two Q-networks and takes the minimum:

```
Q_target = min(Q1(s',a'), Q2(s',a'))
```

**Why?** Single Q-network tends to overestimate values, leading to instability. Taking the minimum provides a conservative estimate.

### 4. Automatic Entropy Tuning

Instead of manually setting α, SAC learns it automatically:

```
α_loss = -log_α · (log_prob + target_entropy)
target_entropy = -action_dim  # For 3D actions: -3.0
```

This adjusts α to maintain a target entropy level, balancing exploration vs exploitation.

---

## Network Architecture

### Vector Mode (36D State)

**Actor (VectorActor)**:
```
Input (36D) → FC(256) → ReLU → FC(256) → ReLU → FC(256) → ReLU
            ├─→ FC(3) → mean
            └─→ FC(3) → log_std (clamped to [-20, 2])
```

**Critic (VectorCritic)** (×2 networks):
```
Input (36D state + 3D action = 39D) → FC(256) → ReLU
→ FC(256) → ReLU → FC(256) → ReLU → FC(256) → ReLU → FC(1) → Q-value
```

### Visual Mode (4×96×96 RGB)

**Actor (VisualActor)**:
```
Input (4×96×96) → Conv(32, 8×8, stride 4) → ReLU
→ Conv(64, 4×4, stride 2) → ReLU
→ Conv(64, 3×3, stride 1) → ReLU
→ Flatten → FC(512) → ReLU
            ├─→ FC(3) → mean
            └─→ FC(3) → log_std
```

**Critic (VisualCritic)** (×2 networks):
```
State: Input (4×96×96) → Conv layers → Flatten → 4096D
Action: Input (3D) → concatenate with state features → FC(512) → ReLU
→ FC(256) → ReLU → FC(1) → Q-value
```

---

## Hyperparameters

### Learning Rates

| Parameter | Default | Description |
|-----------|---------|-------------|
| `lr_actor` | 3e-4 | Actor network learning rate |
| `lr_critic` | 3e-4 | Critic networks learning rate |
| `lr_alpha` | 3e-4 | Entropy coefficient learning rate |

**Why separate learning rates?** Allows independent tuning of actor, critic, and entropy learning speeds.

### Core SAC Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `gamma` | 0.99 | Discount factor for future rewards (0.99 = value rewards 100 steps ahead) |
| `tau` | 0.005 | Soft target network update rate (0.005 = 0.5% update per step) |
| `alpha_init` | 0.2 | Initial entropy coefficient (auto-tuned during training) |
| `auto_entropy_tuning` | True | Automatically adjust alpha to maintain target entropy |
| `target_entropy` | -3.0 | Target entropy = -action_dim (for 3D action space) |

### Training Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `batch_size` | 256 | Number of samples per update |
| `buffer_size` | 1,000,000 | Replay buffer capacity |
| `learning_starts` | 1000 | Steps before learning begins (collect diverse data first) |

### Action Bounds

| Action | Min | Max | Description |
|--------|-----|-----|-------------|
| Steering | -1.0 | +1.0 | Left (-1) to Right (+1) |
| Gas | 0.0 | 1.0 | No gas (0) to Full throttle (1) |
| Brake | 0.0 | 1.0 | No brake (0) to Full brake (1) |

---

## Training Metrics

### Episode Metrics

| Metric | Description | Good Values |
|--------|-------------|-------------|
| `episode` | Episode number | N/A |
| `total_steps` | Cumulative steps across all episodes | Increases linearly |
| `episode_steps` | Steps in current episode | 500+ (longer = better) |
| `reward` | Total episode reward | 500+ (max ~900) |
| `elapsed_time_sec` | Training time in seconds | N/A |
| `avg_reward_100` | Rolling average of last 100 episodes | Increasing trend |

### Actor Metrics

| Metric | Description | Interpretation |
|--------|-------------|----------------|
| `actor_loss` | Policy gradient loss | **Negative values** are normal! Actor maximizes Q - α·log_prob, so loss = -(Q - α·log_prob). Lower magnitude = policy converging |
| `mean_log_prob` | Average log probability of actions | **Negative** (probabilities < 1). Closer to 0 = more confident/deterministic. More negative = more exploratory |

**Example**: `actor_loss = -6.76, mean_log_prob = -2.05`
- Negative actor_loss is expected (maximizing objective, not minimizing)
- log_prob = -2.05 means actions have probability ~0.13 (e^-2.05), indicating some exploration

### Critic Metrics

| Metric | Description | Interpretation |
|--------|-------------|----------------|
| `critic_1_loss` | Q-network 1 MSE loss | Lower = better Q-value predictions. High early in training (learning), should decrease |
| `critic_2_loss` | Q-network 2 MSE loss | Similar to critic_1_loss. Should track closely |
| `mean_q1` | Average Q-value from critic 1 | Estimated expected return. Should match actual rewards. Can be negative initially |
| `mean_q2` | Average Q-value from critic 2 | Should be similar to mean_q1 |

**Example**: `critic_1_loss = 0.44, critic_2_loss = 0.44, mean_q1 = 6.96, mean_q2 = 6.96`
- Low critic losses (0.44) = good value predictions
- Positive Q-values (6.96) = agent expects positive future returns
- Q1 ≈ Q2 = twin networks agree on value estimates

### Entropy Metrics

| Metric | Description | Interpretation |
|--------|-------------|----------------|
| `alpha` | Entropy coefficient | **Decreases** over time. High α = more exploration, low α = more exploitation |
| `alpha_loss` | Entropy tuning loss | Adjusts alpha to maintain target entropy. Fluctuates during training |

**Example**: `alpha = 0.84 → 0.06`
- Alpha starts high (~0.84) for exploration
- Decreases to ~0.06 as policy becomes more confident
- Automatic tuning balances exploration/exploitation

**Target Entropy**: -3.0 (negative of action dimensions)
- Agent tries to maintain entropy around this level
- Prevents policy from becoming too deterministic too quickly

---

## Interpreting Results

### Example Training Progression

From the test run (`logs/training_metrics.csv`):

| Episode | Reward | Alpha | Actor Loss | Critic Loss | Mean Q | Log Prob |
|---------|--------|-------|------------|-------------|--------|----------|
| 1 | -1780 | 0.84 | -6.76 | 0.44 | 6.96 | -2.05 |
| 10 | -582 | 0.06 | 20.32 | 60.63 | -5.02 | 2.58 |
| 20 | 14 | 0.02 | 5.16 | 29.35 | -11.44 | 2.86 |
| 25 | -590 | 0.02 | 1.12 | 25.65 | -23.74 | 4.23 |

**Analysis**:

1. **Reward**: -1780 → 14 → -590 (fluctuates, but improving on average)
2. **Alpha**: 0.84 → 0.02 (decreased, less exploration over time)
3. **Actor Loss**: -6.76 → 5.16 (magnitude decreasing, policy stabilizing)
4. **Critic Loss**: 0.44 → 60.63 → 25.65 (increases as learning begins, should eventually decrease)
5. **Mean Q**: 6.96 → -23.74 (Q-values adjust to match actual rewards)
6. **Log Prob**: -2.05 → 4.23 (becoming more positive = more deterministic actions)

### Healthy Training Signs

✅ **Good**:
- Reward trend increasing over 100+ episodes
- Alpha decreasing (0.8 → 0.01-0.2 range)
- Critic losses stabilizing after initial spike
- Q-values correlating with actual rewards
- Episode length increasing (reaching 500+ steps)

⚠️ **Warning Signs**:
- Reward stuck at low values (-100) for 200+ episodes
- Alpha drops to near-zero too quickly (< 0.001)
- Critic losses exploding (> 1000)
- Q-values diverging wildly from rewards
- Episodes terminating early (< 150 steps) consistently

### Evaluation Metrics

During evaluation (`evaluate=True`):
- Agent uses **deterministic policy** (mean action, no sampling)
- More consistent performance
- Should achieve higher rewards than training episodes
- Use to track true policy improvement

**Example from logs**:
```
Evaluation at episode 20
  Mean: -45.47 | Std: 15.16  (5 episodes)

Final evaluation (10 episodes)
  Mean: 18.02 | Std: 43.61
```

- First eval: -45.47 (early in training)
- Final eval: +18.02 (improved after 25 episodes)
- High std dev (43.61) indicates some variability

---

## Advanced Topics

### Reparameterization Trick

SAC samples actions using:
```python
action = mean + std * noise
where noise ~ N(0, 1)
```

This allows gradients to flow through the sampling operation, enabling the actor to learn from Q-values.

### Tanh Squashing

Raw actions are unbounded Gaussian samples. They're squashed using tanh:
```python
action_bounded = tanh(action_unbounded)
```

This ensures actions stay within [-1, 1] for steering and [0, 1] for gas/brake.

### Soft Target Updates

Instead of copying target networks every N steps (hard update), SAC uses soft updates:
```python
target_params = tau * params + (1 - tau) * target_params
```

With tau=0.005, target networks slowly track main networks, providing stability.

---

## References

- [Original SAC Paper](https://arxiv.org/abs/1801.01290): Haarnoja et al., 2018
- [Automatic Entropy Tuning](https://arxiv.org/abs/1812.05905): Haarnoja et al., 2018
- [Spinning Up in Deep RL - SAC](https://spinningup.openai.com/en/latest/algorithms/sac.html)

---

## Quick Reference Card

**Training command**:
```bash
python train.py --episodes 2000 --batch-size 256 --learning-starts 1000
```

**Watch agent**:
```bash
python watch_agent.py --checkpoint checkpoints/best_model.pt --episodes 5
```

**Key metrics to watch**:
- `reward`: Should trend upward (target: 500+)
- `alpha`: Should decrease (0.8 → 0.01-0.2)
- `episode_steps`: Should increase (target: 500+)
- Evaluation mean: Track true policy performance

**Typical training timeline** (vector mode, MPS):
- Episodes 1-50: Exploration (-100 to 100 rewards)
- Episodes 50-200: Learning (100 to 400 rewards)
- Episodes 200-500: Refinement (400 to 700 rewards)
- Episodes 500+: Mastery (700+ rewards)

---

*Generated for CarRacing-v3 SAC Project - November 2025*
