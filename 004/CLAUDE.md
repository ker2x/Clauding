# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a **Soft Actor-Critic (SAC)** reinforcement learning agent for CarRacing-v3 using **continuous action space** (no discretization). Project 004 is a fork of Project 003 - both projects share the same implementation. The project implements maximum entropy RL with automatic entropy tuning, twin Q-networks, and supports both vector-based (36D track geometry) and visual-based (96×96 images) state representations.

## Recent Changes

**Physics Engine Rework & Code Cleanup**:
- The custom 2D car physics engine (`car_dynamics.py`) has been completely reworked with refined Pacejka tire modeling
- Physics code is clean and well-documented with no dead code or debug markers
- Environment code (`car_racing.py`) has been cleaned up with improved comments
- Exception handling improved: replaced bare `except:` clauses with specific exception types

**Key Architecture Principle**: SAC uses separate actor (policy) and critic (value) networks with automatic entropy tuning via a learned alpha parameter. The algorithm maintains twin Q-networks to reduce overestimation bias and uses soft target updates for stability.

## Virtual Environment

This project uses a **shared virtual environment** located in the parent directory (`../.venv`). Always activate it before running any commands:

```bash
source ../.venv/bin/activate
```

## Common Commands

### Setup and Testing
```bash
# Install dependencies
pip install -r requirements.txt

# Verify installation and GPU availability
python test_setup.py
```

### Training
```bash
# Standard training (2000 episodes, vector mode - RECOMMENDED)
python train.py

# Quick test run (25 episodes)
python train.py --episodes 25 --learning-starts 500 --eval-frequency 20

# Custom hyperparameters
python train.py --episodes 1000 --learning-starts 5000 --batch-size 256 --lr-actor 3e-4

# Resume from checkpoint
python train.py --resume checkpoints/best_model.pt --episodes 1000

# Train on specific device (auto/cpu/cuda/mps)
python train.py --device mps

# Visual mode (NOT recommended for training - very slow)
python train.py --state-mode visual --episodes 200
```

### Watching Agents
```bash
# Watch random agent (baseline)
python watch_random_agent.py --episodes 3

# Watch trained agent (auto-detects state mode from checkpoint)
python watch_agent.py --checkpoint checkpoints/best_model.pt --episodes 5

# Watch without rendering (just compute rewards)
python watch_agent.py --checkpoint checkpoints/best_model.pt --no-render --episodes 10

# Adjust display FPS
python watch_agent.py --checkpoint checkpoints/best_model.pt --fps 60
```

## High-Level Architecture

### State Mode Architecture Pattern

The codebase implements a **dual-mode architecture** where the same SAC algorithm can operate on two fundamentally different state representations:

1. **Vector Mode (RECOMMENDED)**: 36D state = car state (11D) + track segment info (5D) + lookahead waypoints (20D)
   - Uses MLP networks (VectorActor, VectorCritic)
   - 3-5x faster training
   - No rendering required during training
   - Full track geometry information

2. **Visual Mode**: 96×96 RGB images with frame stacking (4 frames)
   - Uses CNN networks (VisualActor, VisualCritic)
   - Requires full rendering
   - Too slow for training, useful for visualization

**Critical Pattern**: The agent's network architecture is determined by `state_mode` and saved in checkpoints. When loading a checkpoint, the `watch_agent.py` script **auto-detects** the state mode from the checkpoint and creates matching environment and networks. Never hardcode state mode when loading checkpoints.

### SAC Algorithm Flow

1. **Data Collection**: Agent interacts with environment using stochastic policy (samples from Gaussian)
2. **Experience Replay**: Store (s, a, r, s', done) tuples in replay buffer
3. **Learning** (after `learning_starts` steps):
   - **Critic Update**: Minimize TD error using twin Q-networks with entropy term
   - **Actor Update**: Maximize Q-value and policy entropy using reparameterization trick
   - **Alpha Update**: Adjust entropy coefficient to maintain target entropy (-action_dim)
   - **Soft Target Update**: Slowly update target networks (tau=0.005)

### Network Architecture Pattern

**Actor Networks** output action distributions:
- Input: State (36D vector or 4×96×96 images)
- Output: `(mean, log_std)` for Gaussian policy
- Action sampling: `action = tanh(mean + std * noise)` where noise ~ N(0,1)
- Log probability includes tanh correction for bounded actions

**Critic Networks** estimate Q-values:
- Input: State + Action concatenated
- Output: Single Q-value scalar
- Twin critics: Two independent networks, use minimum for target computation

**Target Networks**: Soft-updated copies of critics for stable learning

### Critical Implementation Details

#### Checkpoint State Mode Detection
When loading a checkpoint, the agent must match the architecture it was trained with. The pattern in `watch_agent.py` lines 120-146:

```python
# Load checkpoint to detect state mode
checkpoint = torch.load(args.checkpoint, map_location='cpu')

# Auto-detect from checkpoint
if 'state_mode' in checkpoint:
    state_mode = checkpoint['state_mode']
else:
    # Fallback: detect from network architecture
    actor_state = checkpoint['actor']
    if 'conv1.weight' in actor_state:
        state_mode = 'visual'
    else:
        state_mode = 'vector'

# Use detected mode for environment AND agent creation
env = make_carracing_env(..., state_mode=state_mode)
agent = SACAgent(..., state_mode=state_mode)
```

**Why This Matters**: Mismatching state mode causes architecture errors (trying to load CNN weights into MLP or vice versa).

#### Action Space Handling
Actions are continuous `[steering, acceleration]`:
- `steering`: -1.0 (full left) to +1.0 (full right)
- `acceleration`: -1.0 (full brake) to +1.0 (full gas)
- Raw policy outputs unbounded Gaussian samples
- Tanh squashing bounds actions: `action = tanh(z)` where z ~ N(mean, std)
- Log probability correction: `log_prob = log_normal(z) - log(1 - tanh²(z))`
- This allows gradient flow while ensuring bounded actions

#### Entropy Tuning
Alpha (entropy coefficient) is automatically learned:
- Target entropy = -action_dim (for 2D actions: -2.0)
- Alpha is parameterized as `exp(log_alpha)` to ensure positivity
- Loss: `alpha_loss = -log_alpha * (log_prob + target_entropy).detach()`
- Alpha typically decreases during training (0.8 → 0.01-0.2)

## File Structure and Responsibilities

```
003/ and 004/ (identical structure)
├── env/
│   ├── car_racing.py       # Modified CarRacing-v3 with vector state mode
│   │                       # Key: _create_vector_state() returns 36D state
│   └── car_dynamics.py     # Car physics simulation
│
├── sac_agent.py            # SAC implementation (440 lines)
│   ├── VectorActor/Critic  # MLP networks for 36D state
│   ├── VisualActor/Critic  # CNN networks for images
│   ├── ReplayBuffer        # Experience replay
│   └── SACAgent            # Main agent class with update() logic
│
├── preprocessing.py         # Environment wrappers
│   ├── GrayscaleWrapper    # RGB → grayscale (visual mode only)
│   ├── NormalizeObservation # [0,255] → [0,1] (visual mode only)
│   ├── FrameStack          # Stack 4 frames (visual mode only)
│   ├── RewardShaper        # Penalize short episodes (-50 for <150 steps)
│   └── make_carracing_env() # Factory function, applies wrappers based on state_mode
│
├── train.py                # Training script (620 lines)
│   ├── Training loop with evaluation
│   ├── Logging: CSV metrics, plots, human-readable logs
│   └── Checkpointing: best_model.pt, checkpoint_epN.pt, final_model.pt
│
├── watch_agent.py          # Visualize trained agent
│   └── CRITICAL: Auto-detects state_mode from checkpoint (lines 120-146)
│
└── watch_random_agent.py   # Baseline random agent
```

## State Mode Terminology

**Important**: Project 002 used "snapshot" mode for 36D track geometry. In projects 003 and 004, this has been renamed to "vector" mode:

- ✅ **Vector mode** = 36D track geometry (current terminology)
- ❌ **Snapshot mode** = Old name, do not use
- ✅ **Visual mode** = 96×96 RGB images (unchanged)

The old 11D basic vector mode from project 002 has been removed as it provided insufficient information.

## Training Metrics Interpretation

### Episode Metrics (CSV: `logs/training_metrics.csv`)
- `reward`: Episode total reward (target: 500+ for good performance)
- `episode_steps`: Steps in episode (longer is better, 500+ indicates completing laps)
- `avg_reward_100`: Rolling average (should trend upward)

### Actor Metrics
- `actor_loss`: Policy gradient loss (**can be negative**, this is normal! Actor maximizes Q - α·log_prob)
- `mean_log_prob`: Average action log probability (negative values expected, closer to 0 = more deterministic)

### Critic Metrics
- `critic_1_loss`, `critic_2_loss`: Q-value prediction MSE (should decrease and stabilize)
- `mean_q1`, `mean_q2`: Average Q-values (should correlate with actual rewards)

### Entropy Metrics
- `alpha`: Entropy coefficient (**decreases over time**: 0.8 → 0.01-0.2)
- `alpha_loss`: Entropy tuning loss (fluctuates, adjusts alpha to target entropy)
- Target entropy = -3.0 (negative of action dimensions)

**Healthy Training Pattern**:
1. Alpha decreases from ~0.8 to 0.01-0.2
2. Critic losses spike initially, then stabilize
3. Rewards trend upward over 100+ episodes
4. Episode length increases (reaching 500+ steps)

## Device Handling

The code auto-detects available compute:
1. CUDA (NVIDIA GPU) - preferred
2. MPS (Apple Silicon) - good
3. CPU - fallback (slow)

Override with `--device` flag: `python train.py --device mps`

## Checkpoint Format

Checkpoints (.pt files) contain:
```python
{
    'actor': actor.state_dict(),
    'critic_1': critic_1.state_dict(),
    'critic_2': critic_2.state_dict(),
    'critic_target_1': critic_target_1.state_dict(),
    'critic_target_2': critic_target_2.state_dict(),
    'actor_optimizer': actor_optimizer.state_dict(),
    'critic_1_optimizer': critic_1_optimizer.state_dict(),
    'critic_2_optimizer': critic_2_optimizer.state_dict(),
    'log_alpha': log_alpha,  # If auto_entropy_tuning
    'alpha_optimizer': alpha_optimizer.state_dict(),  # If auto_entropy_tuning
    'state_mode': 'vector' or 'visual',  # Critical for loading
    'action_dim': 2
}
```

**When resuming training**: `--resume` loads networks and optimizers to continue from checkpoint.

**When watching**: Script must detect `state_mode` to create matching environment/networks.

## Continuous Action Space

Unlike project 002 (discrete 9 actions), this uses a 2D continuous action space:
- `steering` ∈ [-1.0, 1.0] (negative = left, positive = right)
- `acceleration` ∈ [-1.0, 1.0] (negative = brake, positive = gas)

Actions are sampled from Gaussian distribution and bounded with tanh. No discretization or ActionDiscretizer class exists in this project.

## Reward Shaping

Default reward shaping applies -50 penalty for episodes ending in <150 steps. This prevents agents from exploiting stationary termination as a "safe" strategy. Controlled by:
- `reward_shaping=True` (default)
- `min_episode_steps=150`
- `short_episode_penalty=-50.0`

## Early Termination

Environment terminates episode if car remains stationary for 100 frames (after 50 minimum steps). This speeds up training ~3x by skipping stuck/crashed episodes. Controlled by:
- `terminate_stationary=True` (default)
- `stationary_patience=100`
- `stationary_min_steps=50`

## Typical Training Timeline (Vector Mode, MPS/CUDA)

| Episodes | Steps | Time | Expected Behavior |
|----------|-------|------|-------------------|
| 1-50 | 0-100k | ~20 min | Random exploration, negative rewards |
| 50-200 | 100k-500k | ~1-2 hours | Learning basic control, some track following |
| 200-500 | 500k-1M | ~3-5 hours | Consistent track following, improving speed |
| 500-1000 | 1M-2M | ~6-10 hours | Good racing behavior, 500+ rewards |
| 1000+ | 2M+ | 10+ hours | Mastery, 700+ rewards, full laps |

## Debugging Patterns

### Agent Doesn't Learn
1. Check alpha convergence: Should decrease to 0.01-0.2, not stay at 0.8
2. Verify learning started: Look for "Learning started" message at `learning_starts` steps
3. Check Q-values: Should correlate with actual rewards (not diverge)
4. Increase training: CarRacing needs 1M+ steps for good performance

### Architecture Mismatch Errors
```
RuntimeError: Error(s) in loading state_dict for VisualActor:
    Missing key(s) in state_dict: "conv1.weight"...
```
**Cause**: Loading checkpoint trained in one state_mode into agent created with different mode.

**Fix**: Ensure `watch_agent.py` auto-detects state_mode from checkpoint (see lines 120-146).

### Training Too Slow
1. Verify vector mode: `--state-mode vector` (default)
2. Check device: Should show "cuda" or "mps", not "cpu"
3. Reduce batch size if memory limited: `--batch-size 128`

## Additional Documentation

- `README.md`: User-facing documentation, quick start guide
- `SAC_EXPLAINED.md`: Deep dive into SAC algorithm, parameters, and metrics interpretation
- `logs/training.log`: Human-readable training progress during runs
- `logs/training_metrics.csv`: Complete episode-by-episode metrics for analysis

## References

SAC algorithm based on:
- Haarnoja et al., 2018: "Soft Actor-Critic: Off-Policy Maximum Entropy Deep Reinforcement Learning with a Stochastic Actor"
- Haarnoja et al., 2019: "Soft Actor-Critic Algorithms and Applications" (automatic entropy tuning)
