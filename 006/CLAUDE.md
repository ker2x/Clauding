# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a **Soft Actor-Critic (SAC)** reinforcement learning agent for CarRacing-v3 using **continuous action space** and a **custom 2D physics engine** (no Box2D dependency). The project implements maximum entropy RL with automatic entropy tuning, twin Q-networks, and uses **vector-based (67D) state representation** for training.

### Key Features

- **Parallel Selection Training**: Primary training method using N independent agents with evolutionary selection
- **Custom 2D Physics**: Clean, interpretable physics simulation with Magic Formula tires
- **Soft Actor-Critic**: State-of-the-art continuous control algorithm
- **Vector Mode**: 67D state representation (car state + track geometry + lookahead waypoints)
- **Clean Architecture**: Simplified codebase focused on vector mode for optimal performance

### Current Reward Structure

All configurable in `env/car_racing.py` (lines 64-71):

**Sparse rewards:**
- +100 per checkpoint (15 checkpoints total)
- +1000 for completing a full lap

**Dense rewards:**
- +0.1 per m/s forward velocity per frame
- -2.0 per frame step penalty

**Penalties:**
- -1.0 per wheel off-track (when >2 wheels off)
- -100 + episode termination when all 4 wheels off

## Virtual Environment

This project uses a **shared virtual environment** in the parent directory:

```bash
source ../.venv/bin/activate
```

## Training Methods

### Primary Method: Parallel Selection Training (RECOMMENDED)

Train N independent agents in parallel with evolutionary selection:

```bash
# Train with 8 parallel agents (RECOMMENDED)
python train_selection_parallel.py --num-agents 8 --episodes 2000

# Train with elite preservation (top 2 agents survive)
python train_selection_parallel.py --num-agents 8 --elite-count 2

# Quick test with fewer agents
python train_selection_parallel.py --num-agents 4 --episodes 500

# Resume from latest generation
python train_selection_parallel.py --num-agents 8 --resume checkpoints_selection_parallel/latest_generation.pt
```

**How it works:**
- N agents train simultaneously on separate CPU cores
- Every M episodes: synchronize, evaluate all agents, select best performer
- Clone winner to other agents (with optional elite preservation)
- Checkpoint saved after every tournament
- Provides evolutionary pressure with N× sample collection

**Tournament Strategies:**
- `--elite-count 1` (default): Winner-takes-all, maximum selection pressure
- `--elite-count 2+`: Elite preservation, maintains diversity

**Advantages:**
- True parallel execution (N× CPU utilization)
- Evolutionary selection pressure improves convergence
- Sample efficient (N× data collection)
- Wall-clock speedup: ~N× compared to single agent
- Automatic checkpoint saving every tournament

### Alternative Training Method

**Standard single-agent training:**
```bash
python train.py --episodes 2000
```

## Common Commands

### Setup and Testing
```bash
# Install dependencies
pip install -r requirements.txt

# Verify installation
python test_setup.py
```

### Watching Agents
```bash
# Watch random agent (baseline)
python watch_random_agent.py --episodes 3

# Watch trained agent
python watch_agent.py --checkpoint checkpoints_selection_parallel/best_model.pt --episodes 5

# Play as human
python play_human.py
```

## Architecture Overview

### State Representation

**Vector Mode (67D):**
- Car state (11D): position, velocity, angle, wheel contacts, progress
- Track segment (5D): distance to center, angle, curvature, segment info
- Lookahead waypoints (20×2 = 40D): future waypoints in car coordinates
- Obstacles/competitors (11D): nearest opponent information
- Fast training, no rendering required

### SAC Algorithm

**Components:**
1. **Actor**: Stochastic policy (Gaussian) outputting continuous actions
2. **Twin Critics**: Two Q-networks to reduce overestimation
3. **Automatic Entropy Tuning**: Learns exploration-exploitation balance

**Actions:** `[steering, acceleration]`
- steering ∈ [-1, 1]
- acceleration ∈ [-1, 1] (negative = brake, positive = gas)

### Network Architecture

**Vector Mode:**
- Actor: 67D → FC(256)×3 → 2D action (mean, log_std)
- Critic: 67D + 2D action → FC(512)×4 → Q-value

## File Structure

```
006/
├── env/
│   ├── car_racing.py       # Modified CarRacing-v3 with custom physics
│   └── car_dynamics.py     # Custom 2D car physics (Magic Formula)
│
├── sac_agent.py            # SAC implementation (vector mode only)
├── preprocessing.py         # Environment factory function
│
├── train_selection_parallel.py  # PRIMARY: Parallel selection training
├── train.py                     # Standard single-agent training
│
├── watch_agent.py          # Visualize trained agent
├── watch_random_agent.py   # Baseline random agent
├── play_human.py           # Human playable mode
├── test_setup.py           # Verify installation
│
├── checkpoints_selection_parallel/  # Saved models (parallel selection)
├── logs_selection_parallel/         # Training logs
│
└── [Documentation files]
```

## Key Hyperparameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--num-agents` | 8 | Number of parallel agents (selection training) |
| `--selection-frequency` | 50 | Episodes between selection tournaments |
| `--eval-episodes` | 10 | Episodes per tournament evaluation |
| `--elite-count` | 1 | Top N agents preserved (1=winner-takes-all) |
| `--learning-starts` | 5000 | Steps before training begins |
| `--lr-actor` | 3e-4 | Actor learning rate |
| `--lr-critic` | 3e-4 | Critic learning rate |
| `--gamma` | 0.99 | Discount factor |
| `--tau` | 0.005 | Target network update rate |
| `--buffer-size` | 1000000 | Replay buffer capacity |
| `--batch-size` | 256 | Training batch size |

## Training Metrics

### Episode Metrics
- `reward`: Total episode reward (target: 500+ for good performance)
- `episode_steps`: Steps in episode (longer = better, 500+ ideal)
- `avg_reward_100`: Rolling average over 100 episodes

### SAC Metrics
- `actor_loss`: **Can be negative** (maximizing objective, normal behavior)
- `critic_1_loss`, `critic_2_loss`: Should decrease and stabilize
- `alpha`: Entropy coefficient (decreases over time: 0.8 → 0.01-0.2)
- `mean_q1`, `mean_q2`: Q-value estimates (should correlate with rewards)

### Healthy Training Pattern
1. Alpha decreases from ~0.8 to 0.01-0.2
2. Critic losses spike initially, then stabilize
3. Rewards trend upward over 100+ episodes
4. Episode length increases (reaching 500+ steps)
5. Best agent performance improves with each selection

## Device Handling

**For parallel selection training:** Use CPU (required for multiprocessing)
```bash
python train_selection_parallel.py --device cpu  # Default
```

**For single-agent training:**
- Vector mode: Use `--device cpu` (fastest for small MLPs) or `--device auto` (auto-detects)

## Checkpoint Format

Checkpoints contain:
- Network weights (actor, critics, targets)
- Optimizer states
- `state_dim`: 67 (vector dimension)
- `action_dim`: 2
- Entropy tuning parameters (if enabled)

**Parallel selection training saves:**
- `generation_N.pt`: Winner from generation N (historical record)
- `latest_generation.pt`: Most recent tournament winner (easy resume)
- `best_model.pt`: Best reward ever achieved (only updated on improvement)

## Debugging Patterns

### Agent Doesn't Learn
1. Check alpha convergence (should decrease to 0.01-0.2)
2. Verify learning started at `learning_starts` steps
3. Check Q-values correlate with actual rewards
4. Increase training (needs 1M+ steps for good performance)

### Training Unstable
1. Reduce learning rates (`--lr-actor 1e-4 --lr-critic 1e-4`)
2. Reduce tau (`--tau 0.002`)
3. Increase batch size (`--batch-size 512`)

### Selection Tournament Issues
- Agents should synchronize at checkpoint episodes
- Winner should be cloned to non-elite positions
- Check logs for "GENERATION N: Selection Tournament" messages
- If timeout occurs, missing agents assigned -inf reward
- Evaluation has 2500 step limit per episode (prevents infinite loops)

## Reward Tuning

Edit constants at top of `env/car_racing.py:64-71`:

**Increase time pressure:**
```python
STEP_PENALTY = 3.0  # Increase from 2.0
```

**Reduce off-track penalty:**
```python
OFFTRACK_PENALTY = 0.5  # Decrease from 1.0
OFFTRACK_THRESHOLD = 3  # Allow 3 wheels off
```

**Increase progress reward:**
```python
PROGRESS_REWARD_SCALE = 6000.0  # Increase from 4000.0
```

## Training Timeline

| Phase | Episodes | Expected Behavior |
|-------|----------|-------------------|
| Exploration | 1-50 | Random actions, negative rewards |
| Learning | 50-200 | Basic control, some track following |
| Improvement | 200-500 | Consistent track following |
| Mastery | 500-1000+ | Good racing, 500+ rewards, full laps |

**With parallel selection (8 agents):** Expect faster convergence due to evolutionary pressure and increased sample collection.

## Additional Documentation

- `README.md`: User-facing documentation and quick start
- `SAC_EXPLAINED.md`: Deep dive into SAC algorithm
- `TRAINING_COMPARISON.md`: Comparison of training methods
- `logs_selection_parallel/training.log`: Training progress

## References

- Haarnoja et al., 2018: "Soft Actor-Critic: Off-Policy Maximum Entropy Deep RL"
- Haarnoja et al., 2019: "Soft Actor-Critic Algorithms and Applications"

---

*Last updated: 2025 - Project 006 - Vector Mode Only*
