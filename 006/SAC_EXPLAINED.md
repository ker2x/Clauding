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
target_entropy = -action_dim  # For 2D actions: -2.0
```

This adjusts α to maintain a target entropy level, balancing exploration vs exploitation.

---

## Network Architecture

### Vector Mode (36D State)

**Actor (VectorActor)**:
```
Input (36D) → FC(256) → LeakyReLU → FC(256) → LeakyReLU → FC(256) → LeakyReLU
            ├─→ FC(3) → mean
            └─→ FC(3) → log_std (clamped to [-20, 2])
```

**Critic (VectorCritic)** (×2 networks):
```
Input (36D state + 3D action = 39D) → FC(256) → LeakyReLU
→ FC(256) → LeakyReLU → FC(256) → LeakyReLU → FC(256) → LeakyReLU → FC(1) → Q-value
```

**Note**: Uses LeakyReLU (slope=0.01) instead of ReLU to prevent dead neurons and improve gradient flow.

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
| `gamma` | 0.99 | Discount factor for future rewards |
| `tau` | 0.005 | Soft target network update rate |
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

**Note**: The policy outputs actions in [-1, 1] range using tanh squashing. The environment internally converts the 3D continuous actions to the appropriate ranges.

---

## Hyperparameter Effects and Tuning

This section explains what happens when you change each hyperparameter and how to tune them for better performance.

### Learning Rates

#### `lr_actor` (Default: 3e-4)

**What it controls**: How fast the policy (actor) network learns.

**Effects of changing**:
- **Too high** (>1e-3):
  - Policy updates too aggressively
  - Unstable training (rewards fluctuate wildly)
  - May "forget" good behaviors quickly
  - Actor loss oscillates dramatically
  - **Symptom**: Policy keeps changing direction, never stabilizes

- **Too low** (<1e-5):
  - Policy learns very slowly
  - Takes forever to improve
  - May get stuck in local optima
  - **Symptom**: Rewards plateau early, agent doesn't improve after initial phase

**When to increase**:
- Rewards are improving but very slowly
- You have a large replay buffer and stable critics
- Training is taking too long

**When to decrease**:
- Policy is unstable (rewards jump around)
- Actor loss is exploding
- Training diverges after initial progress

#### `lr_critic` (Default: 3e-4)

**What it controls**: How fast the Q-networks (critics) learn to estimate values.

**Effects of changing**:
- **Too high** (>1e-3):
  - Q-value estimates become unstable
  - Critic losses oscillate or explode
  - Poor gradient signals to actor
  - Training becomes chaotic
  - **Symptom**: `critic_1_loss` and `critic_2_loss` > 100 and unstable

- **Too low** (<1e-5):
  - Critic learns slowly to predict returns
  - Actor gets poor value estimates
  - Slow convergence overall
  - **Symptom**: Q-values don't match actual rewards even after 100+ episodes

**When to increase**:
- Critic losses are decreasing too slowly
- Q-values are very inaccurate (far from actual rewards)
- You're using a large batch size (512+)

**When to decrease**:
- Critic losses are exploding (>1000)
- Q-values are diverging wildly
- Training becomes unstable after working initially

**Critical balance**: Actor and critic learning rates should usually be similar. If one is 10x the other, you may get instability.

#### `lr_alpha` (Default: 3e-4)

**What it controls**: How fast the entropy coefficient (temperature) adapts.

**Effects of changing**:
- **Too high** (>1e-3):
  - Alpha oscillates rapidly
  - Exploration/exploitation balance is unstable
  - May converge too quickly to deterministic policy

- **Too low** (<1e-5):
  - Alpha adapts very slowly
  - May stay too high (over-exploring) or too low (under-exploring)
  - Suboptimal exploration-exploitation tradeoff

**When to adjust**:
- If alpha drops to near-zero too quickly (< 0.001 within 50 episodes): increase `lr_alpha`
- If alpha oscillates wildly: decrease `lr_alpha`
- Generally, this is the least sensitive hyperparameter—3e-4 works well

### Core SAC Parameters

#### `gamma` (Discount Factor, Default: 0.99)

**What it controls**: How much the agent values future rewards vs immediate rewards.

**Mathematical meaning**: A reward R in N steps is worth `gamma^N * R` today.

**Effects of changing**:
- **Higher** (0.995-0.999):
  - Agent plans further ahead
  - Better for tasks requiring long-term strategy
  - Slower learning (more variance in returns)
  - **Use when**: Track completion requires strategic planning over many steps

- **Lower** (0.90-0.98):
  - Agent focuses on immediate rewards
  - Faster learning (less variance)
  - May develop myopic behavior (doesn't look ahead)
  - **Use when**: Immediate feedback is most important, or episodes are short

**For CarRacing**:
- 0.99 is a good default (episodes can be 200-1000 steps)
- 0.995 might improve strategic planning (better cornering)
- 0.98 might speed up early learning but hurt final performance

**Symptom**: If agent learns to collect early checkpoints but fails to complete track, try increasing gamma.

#### `tau` (Soft Update Coefficient, Default: 0.005)

**What it controls**: How fast target networks track the main networks.

**Mathematical meaning**: `target = tau * main + (1-tau) * target`
- tau=0.005 means target moves 0.5% toward main each update
- Equivalent to averaging over ~200 updates

**Effects of changing**:
- **Higher** (0.01-0.05):
  - Target networks update faster
  - Less stable (target is "chasing a moving target")
  - Can speed up learning if main networks are stable
  - **Risk**: Unstable training, oscillating Q-values

- **Lower** (0.001-0.003):
  - Target networks update very slowly
  - More stable training
  - Slower learning (stale targets)
  - **Risk**: Q-values lag behind actual performance

**When to increase**:
- Training is very stable but slow
- Q-values are too conservative (much lower than actual rewards)

**When to decrease**:
- Training is unstable
- Critic losses are oscillating
- You're using high learning rates

**Rule of thumb**: tau should be inversely proportional to update frequency. More updates = lower tau needed.

#### `alpha` (Entropy Coefficient)

**What it controls**: Exploration vs exploitation balance (auto-tuned, but can be set manually).

**Mathematical meaning**: Penalty for deterministic actions. Higher alpha = prefer diverse actions.

**Effects of changing initial value** (with auto-tuning enabled):
- **Higher initial** (0.5-1.0):
  - More random exploration early
  - Slower to converge to good policy
  - Better final performance (explores more)
  - **Use when**: Task requires extensive exploration

- **Lower initial** (0.05-0.15):
  - Less random exploration early
  - Faster convergence
  - Risk of getting stuck in local optima
  - **Use when**: Good initialization or curriculum learning

**If auto-tuning disabled**:
- **Fixed high** (0.2-0.5): Perpetual exploration, never fully exploits
- **Fixed low** (0.01-0.05): Quickly becomes deterministic, may get stuck

**Monitoring alpha**:
- **Good**: Starts 0.2-0.8, gradually decreases to 0.01-0.1
- **Bad**: Drops to <0.001 quickly (over-exploitation)
- **Bad**: Stays >0.5 for 200+ episodes (over-exploration)

#### `target_entropy` (Default: -action_dim = -3.0)

**What it controls**: Target level of randomness in the policy.

**Mathematical meaning**: Negative of action dimensionality is a heuristic that works well.

**Effects of changing**:
- **More negative** (-4 to -5):
  - Encourages more exploration
  - Policy stays more stochastic longer
  - **Use when**: Complex exploration needed

- **Less negative** (-1 to -2):
  - Encourages earlier determinism
  - Faster convergence to deterministic policy
  - **Risk**: Premature convergence

**Default of -3.0 for 3D actions is usually optimal.** Only change if you have specific exploration needs.

### Training Parameters

#### `batch_size` (Default: 256)

**What it controls**: Number of samples used in each gradient update.

**Effects of changing**:
- **Larger** (512, 1024):
  - **Pros**:
    - More stable gradients (less noise)
    - Better GPU utilization
    - Smoother training curves
  - **Cons**:
    - Slower updates (fewer updates per second)
    - Requires more memory
    - May need higher learning rates
  - **When to use**: Stable environment, good GPU, want smooth training

- **Smaller** (64, 128):
  - **Pros**:
    - More updates per second
    - Works with less memory
    - Can escape local optima easier (more noise)
  - **Cons**:
    - Noisy gradients
    - Less stable training
    - May need lower learning rates
  - **When to use**: Limited memory, want faster iteration

**Recommended**: Larger batches (256-512) for stability with vector observations

**Critical**: Batch size should be << buffer size. If batch_size = 256, buffer should be at least 10,000.

#### `buffer_size` (Default: 100,000)

**What it controls**: How many past experiences are stored for training.

**Effects of changing**:
- **Larger** (200k-1M):
  - **Pros**:
    - More diverse training data
    - Better sample efficiency
    - Reduces overfitting to recent experiences
  - **Cons**:
    - More memory usage (~1GB for 1M vector obs)
    - Slower sampling (more data to sample from)
    - May include outdated experiences
  - **When to use**: You have memory, want best sample efficiency

- **Smaller** (10k-50k):
  - **Pros**:
    - Less memory
    - Faster sampling
    - Only trains on recent (relevant) data
  - **Cons**:
    - Less diversity
    - May overfit to recent experiences
    - Worse sample efficiency
  - **When to use**: Limited memory, or environment changes over time

**Memory estimates**:
- Vector (71D state): ~1GB for 1M transitions

**Practical guideline**:
- Use largest buffer that fits in RAM
- For 16GB RAM: 1-2M transitions
- For 32GB RAM: 2-4M transitions

#### `learning_starts` (Default: 5,000)

**What it controls**: How many random steps to collect before training begins.

**Purpose**: Fill buffer with diverse experiences before learning.

**Effects of changing**:
- **Higher** (10k-20k):
  - **Pros**:
    - More diverse initial data
    - Better initial Q-value estimates
    - More stable early training
  - **Cons**:
    - Longer before seeing any learning
    - May collect too much bad data
  - **When to use**: Complex environment, want stable start

- **Lower** (1k-2k):
  - **Pros**:
    - See learning sooner
    - Less time wasted on random actions
  - **Cons**:
    - Less diverse initial data
    - Q-values may be poor initially
  - **When to use**: Simple environment, want fast iteration

**Rule of thumb**: `learning_starts` should be at least 2-5× `batch_size`
- batch_size=256 → learning_starts ≥ 1000
- Ensures enough diversity for initial training
- Recommended: 2k-5k for vector observations

---

## Hyperparameter Troubleshooting Guide

This section helps diagnose and fix common training issues by adjusting hyperparameters.

### Problem: Training is unstable (rewards jump around wildly)

**Symptoms**:
- Reward swings from 500 to -500 between episodes
- Critic losses explode (>1000)
- Q-values diverge dramatically

**Likely causes & fixes**:

1. **Learning rates too high**
   - **Fix**: Reduce `lr_actor` and `lr_critic` by 2-5x (try 1e-4 or 5e-5)
   - **Check**: If critic_loss stabilizes, this was the issue

2. **Tau too high**
   - **Fix**: Reduce `tau` to 0.001 or 0.002
   - **Why**: Target networks changing too fast

3. **Batch size too small**
   - **Fix**: Increase `batch_size` to 512
   - **Why**: Noisy gradients from small batches

4. **Alpha oscillating**
   - **Fix**: Reduce `lr_alpha` to 1e-4
   - **Check**: Monitor alpha value, should decrease smoothly

### Problem: Agent learns very slowly or doesn't improve

**Symptoms**:
- Rewards stay around -100 for 200+ episodes
- Q-values don't change much
- Alpha stays very high (>0.5)

**Likely causes & fixes**:

1. **Learning rates too low**
   - **Fix**: Increase `lr_actor` and `lr_critic` to 5e-4 or 1e-3
   - **Check**: Training should accelerate within 20-30 episodes

2. **Not enough exploration**
   - **Fix**: Increase `learning_starts` to 10k (collect more diverse data)
   - **Alternative**: Increase initial alpha to 0.5

3. **Buffer too small**
   - **Fix**: Increase `buffer_size` to 200k or more
   - **Why**: Not enough diversity in training data

4. **Gamma too low**
   - **Fix**: Increase `gamma` from 0.99 to 0.995
   - **Why**: Agent too short-sighted, needs long-term planning

5. **Batch size too large**
   - **Fix**: Reduce `batch_size` to 128
   - **Why**: Updates too conservative, needs more update frequency

### Problem: Agent learns quickly then performance degrades

**Symptoms**:
- Rewards reach 400-500, then drop to 100-200
- Actor loss increases after decreasing
- Q-values become pessimistic (much lower than rewards)

**Likely causes & fixes**:

1. **Buffer too small** (catastrophic forgetting)
   - **Fix**: Increase `buffer_size` to 200k+
   - **Why**: Old experiences being overwritten, agent forgets

2. **Learning rates too high** (overshooting)
   - **Fix**: Reduce `lr_actor` to 1e-4 after initial learning
   - **Why**: Policy overshoots good solutions

3. **Alpha too low** (premature convergence)
   - **Fix**: Increase `target_entropy` to -2.0 (more exploration)
   - **Check**: Alpha should stay >0.01

4. **Tau too high** (unstable targets)
   - **Fix**: Reduce `tau` to 0.002
   - **Why**: Target networks not providing stable baseline

### Problem: Agent gets stuck in local optimum

**Symptoms**:
- Learns basic behavior (moves forward) but never improves
- Rewards plateau at 100-300
- Alpha drops to near-zero quickly (<0.001)
- Policy becomes deterministic early

**Likely causes & fixes**:

1. **Insufficient exploration**
   - **Fix 1**: Increase initial `alpha` to 0.5-0.8
   - **Fix 2**: Increase `target_entropy` to -2.0 (encourage exploration)
   - **Fix 3**: Increase `learning_starts` to 10k (more diverse initial data)

2. **Gamma too low**
   - **Fix**: Increase `gamma` to 0.995
   - **Why**: Agent needs to value long-term rewards more

3. **Buffer size too small**
   - **Fix**: Increase `buffer_size` to 500k+
   - **Why**: Need more diverse experiences to escape local optimum

### Problem: Training is too slow (wall-clock time)

**Symptoms**:
- Training takes >10 hours for 1000 episodes
- Low GPU/CPU utilization

**Performance fixes**:

1. **Reduce batch size** (more updates/second)
   - **Fix**: Try batch_size=128 instead of 256
   - **Tradeoff**: Slightly less stable, but 2x faster updates

2. **Reduce buffer size** (faster sampling)
   - **Fix**: Try buffer_size=50k instead of 100k
   - **Tradeoff**: Less diversity, but faster sampling

3. **Configure CPU threading** (for CPU training)
   - **Fix**: Set `OMP_NUM_THREADS=<num_cores>` environment variable
   - **Why**: PyTorch defaults to using only 1-2 threads

### Problem: Critic losses explode

**Symptoms**:
- `critic_1_loss` and `critic_2_loss` > 1000
- Q-values become extremely large (>10000) or small (<-10000)
- Training crashes or NaN values appear

**Likely causes & fixes**:

1. **Learning rate too high**
   - **Fix**: Immediately reduce `lr_critic` to 1e-4 or 5e-5
   - **Urgent**: This can cause complete training failure

2. **Tau too high**
   - **Fix**: Reduce `tau` to 0.001
   - **Why**: Target network instability

3. **Gamma too high**
   - **Fix**: Reduce `gamma` to 0.98
   - **Why**: Value estimates compound errors over long horizons

4. **Gradient clipping needed** (code modification)
   - **Fix**: Add gradient clipping to critic optimization
   - **In code**: `torch.nn.utils.clip_grad_norm_(critic.parameters(), 1.0)`

### Problem: Alpha drops to zero too quickly

**Symptoms**:
- Alpha <0.001 within first 50 episodes
- Policy becomes deterministic early
- Agent gets stuck in local optimum

**Likely causes & fixes**:

1. **lr_alpha too high**
   - **Fix**: Reduce `lr_alpha` to 1e-4 or 5e-5
   - **Why**: Alpha optimization too aggressive

2. **Target entropy too low** (too easy to satisfy)
   - **Fix**: Increase `target_entropy` to -2.0 or -2.5
   - **Why**: Allows higher entropy before penalty

3. **Q-values too high initially**
   - **Fix**: Increase `learning_starts` to collect better initial data
   - **Why**: Optimistic Q-values make alpha drop prematurely

### Problem: Actor loss becomes positive and large

**Symptoms**:
- Actor loss changes from negative (-5) to positive (+20)
- Policy performance degrades
- Q-values are very negative

**Likely causes & fixes**:

1. **Q-values are underestimated** (pessimistic critics)
   - **Fix**: Check if rewards are positive but Q-values negative
   - **Solution**: Increase `lr_critic` to improve Q-estimates faster

2. **Actor learning too fast**
   - **Fix**: Reduce `lr_actor` by 2-5x
   - **Why**: Actor outpacing critic improvements

**Note**: Negative actor loss is normal! Actor maximizes Q - α·log_prob, so loss = -(Q - α·log_prob) is typically negative.

### Hyperparameter Configuration Recipes

Here are tested configurations for different scenarios:

#### Fast Prototyping (Quick Feedback)
```python
lr_actor = 5e-4
lr_critic = 5e-4
gamma = 0.98
batch_size = 128
buffer_size = 50000
learning_starts = 2000
```
**Use when**: Testing ideas, want to see results quickly
**Tradeoff**: Less stable, may not reach peak performance

#### Stable Training (Best Final Performance)
```python
lr_actor = 3e-4
lr_critic = 3e-4
gamma = 0.99
batch_size = 256
buffer_size = 200000  # or larger if RAM allows
learning_starts = 5000
tau = 0.005
```
**Use when**: Training final model, have time
**Tradeoff**: Slower but more reliable

#### Maximum Exploration (Escaping Local Optima)
```python
lr_actor = 3e-4
lr_critic = 3e-4
alpha_init = 0.8  # High initial exploration
target_entropy = -2.0  # Encourage staying stochastic
learning_starts = 10000  # More diverse initial data
buffer_size = 500000  # Large buffer
```
**Use when**: Agent stuck at low performance
**Tradeoff**: Slower convergence, but better final policy

#### Conservative (Avoid Instability)
```python
lr_actor = 1e-4
lr_critic = 1e-4
gamma = 0.99
batch_size = 512  # Large batches = stable gradients
tau = 0.002  # Slow target updates
```
**Use when**: Training keeps crashing or diverging
**Tradeoff**: Very slow learning, but reliable

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
| `alpha` | Entropy coefficient | Decreases over time. High α = more exploration, low α = more exploitation |
| `alpha_loss` | Entropy tuning loss | Adjusts alpha to maintain target entropy |

**Target Entropy**: -3.0 (negative of 3D action space)
- Alpha starts high (~0.2-0.8) for exploration
- Decreases to ~0.01-0.2 as policy converges
- Prevents premature convergence to deterministic policy

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
