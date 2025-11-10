# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a **Soft Actor-Critic (SAC)** reinforcement learning agent for CarRacing-v3 using **continuous action space** (no discretization) and a **custom 2D physics engine** (removed Box2D dependency). Project 005 evolves from Project 004 with enhanced physics simulation and improved reward structure. The project implements maximum entropy RL with automatic entropy tuning, twin Q-networks, and supports both vector-based (36D track geometry) and visual-based (96×96 images) state representations.

### Current Reward Structure (Quick Reference)

All configurable at top of `env/car_racing.py` (lines 64-71):

**Sparse (main objective):**
- `+100` per checkpoint (15 checkpoints total = 1500 points max)
- `+1000` for completing a full lap (generous bonus for finishing)

**Dense (speed incentive):**
- `+0.1` per m/s of forward velocity per frame (encourages faster driving)
- `-2.0` per frame step penalty (increased to strongly favor fast lap times)

**Dense (constraints):**
- `-1.0` per wheel off-track (only when >2 wheels off, so 2 wheels off = OK)
- `-100 + episode ends` when all 4 wheels off

## Recent Changes

**Current Implementation Highlights**:
- **Custom 2D physics engine**: Removed Box2D dependency for cleaner, more interpretable physics simulation
- **Checkpoint-based rewards**: 15 checkpoints × 100 points + 1000 lap completion = 2500 max score
- **Forward velocity bonus**: +0.1 per m/s per frame actively encourages faster driving
- **Increased step penalty**: -2.0 per frame strongly favors fast lap completion times
- **Polygon-based collision detection**: Accurate wheel-on-track detection using ray casting
- **Configurable reward parameters**: All constants at top of `car_racing.py:64-71` for easy tuning
- **Performance optimized**: Spatial partitioning for collision detection (~580 steps/sec)
- **Verbose mode**: `--verbose` flag provides detailed timing diagnostics for debugging

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

# Device selection (see Device Handling section for guidance)
python train.py --device mps          # Apple Silicon GPU
python train.py --device cpu          # CPU only

# Visual mode with MPS (10x faster than CPU for convolutions)
python train.py --state-mode visual --device mps --episodes 200

# Verbose mode for performance diagnostics
python train.py --verbose             # Shows timing every 10 env steps / 10 agent updates
python train.py --verbose --state-mode visual --device mps  # Debug visual mode performance
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

1. **Vector Mode (RECOMMENDED for most use cases)**: 36D state = car state (11D) + track segment info (5D) + lookahead waypoints (20D)
   - Uses MLP networks (VectorActor, VectorCritic)
   - Fast training on CPU (~2-3ms per update)
   - No rendering required during training
   - Full track geometry information
   - **Device**: Use CPU explicitly (`--device cpu`)

2. **Visual Mode (practical with GPU)**: 96×96 RGB images with frame stacking (4 frames)
   - Uses CNN networks (VisualActor, VisualCritic)
   - Requires full rendering (state creation)
   - **Performance**: 10x faster on MPS/CUDA vs CPU
     - MPS: ~73ms per update (practical for training)
     - CPU: ~740ms per update (thermal throttling on laptops)
   - **Device**: Use MPS/CUDA explicitly (`--device mps`)

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

**Actor**: State (36D or 4×96×96) → Gaussian policy (mean, log_std) → 2D action
**Critic**: State + Action → Q-value (twin networks for stability)
**Target Networks**: Soft-updated copies of critics

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
Actions are continuous 2D: `[steering, acceleration]`
- `steering`: -1.0 (left) to +1.0 (right)
- `acceleration`: -1.0 (brake) to +1.0 (gas)
- Policy outputs Gaussian, bounded with tanh: `action = tanh(mean + std * noise)`
- Log probability includes tanh correction for bounded actions

#### Entropy Tuning
Alpha (entropy coefficient) is learned automatically:
- Target entropy = -2.0 (negative of action dimensions)
- Alpha decreases during training (0.8 → 0.01-0.2)

## File Structure and Responsibilities

```
005/
├── env/
│   ├── car_racing.py       # Modified CarRacing-v3 with custom 2D physics (no Box2D)
│   │                       # Key: _create_vector_state() returns 36D state
│   └── car_dynamics.py     # Custom 2D car physics simulation (Magic Formula tires)
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
- `alpha`: Entropy coefficient (decreases over time: 0.8 → 0.01-0.2)
- `alpha_loss`: Entropy tuning loss (fluctuates, adjusts alpha to target)
- Target entropy = -2.0 (negative of 2D action space)

**Healthy Training Pattern**:
1. Alpha decreases from ~0.8 to 0.01-0.2
2. Critic losses spike initially, then stabilize
3. Rewards trend upward over 100+ episodes
4. Episode length increases (reaching 500+ steps)

## Device Handling

### Device Selection Strategy

**The optimal device depends on the state mode:**

#### Vector Mode (36D input, small MLPs):
- **CPU**: Fast ✅ **RECOMMENDED**
- **MPS/CUDA**: Slower (GPU transfer overhead not worth it for tiny MLPs)

#### Visual Mode (4×96×96 input, CNN):
- **MPS/CUDA**: 10x faster ✅ **RECOMMENDED** (massive GPU parallelism for convolutions)
- **CPU**: Very slow, subject to thermal throttling on laptops

### Auto-Detection Behavior

The code auto-detects available compute:
1. CUDA (NVIDIA GPU) - preferred
2. MPS (Apple Silicon) - good
3. CPU - fallback

Override with `--device` flag: `python train.py --device cpu`

### Performance Benchmarks (MacBook Air M1)

**Vector Mode:**
- CPU: ~2-3ms per agent update
- MPS: ~5-10ms per agent update (slower due to overhead)

**Visual Mode:**
- CPU: ~740ms per agent update (conv1: 27ms, thermal throttling issues)
- MPS: ~73ms per agent update (conv1: 0.05ms) → **10x faster**

### Recommendations

```bash
# Vector mode: explicit CPU for best performance
python train.py --state-mode vector --device cpu

# Visual mode: explicit MPS for best performance
python train.py --state-mode visual --device mps

# Auto mode: will choose suboptimally for vector mode (picks MPS)
python train.py  # Not recommended - manually specify device
```

### Thermal Throttling on MacBook Air

MacBook Air has passive cooling and will thermal throttle during extended visual mode training on CPU:
- Initial runs: fast (CPU at full speed when cool)
- After ~5-10 minutes: 2-3x slower (CPU throttled to prevent overheating)
- Solution: Use MPS instead, which distributes thermal load better

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

This project uses 2D continuous actions (no discretization):
- `steering` ∈ [-1.0, 1.0] (left to right)
- `acceleration` ∈ [-1.0, 1.0] (brake to gas)

Actions sampled from Gaussian, bounded with tanh.

## Reward Structure Tuning

All reward parameters are configured at the top of `env/car_racing.py` (lines 64-71). No code diving required!

### Current Configuration (Default)

```python
NUM_WAYPOINTS = 20            # 20 waypoints dividing track (~15 tiles each for 300-tile track)
WAYPOINT_REWARD = 200.0       # 200 points per waypoint (4000 total)
LAP_COMPLETION_REWARD = 500.0 # 500 bonus for completing a full lap
STEP_PENALTY = 0.5            # -0.5 per frame (mild time pressure - keeps rewards positive)
OFFTRACK_PENALTY = 2.0        # -2.0 per wheel off track
OFFTRACK_THRESHOLD = 2        # Allow 2 wheels off (aggressive lines OK)
WAYPOINT_DISTANCE_THRESHOLD = 5.0  # Distance in meters to "reach" waypoint
```

**Reward Math:**
- Per waypoint (50-100 frames): +200 - (0.5 × 50-100) = **+175 to +150** ✓
- Full lap (1000-2000 frames): +4000 + 500 - (0.5 × 1000-2000) = **+4000 to +3500** ✓
- Rewards always positive when making progress!

### Tuning Guide

**If agent struggles to reach waypoints** (too sparse):
```python
NUM_WAYPOINTS = 25-30         # More waypoints = smaller steps
WAYPOINT_REWARD = 150-160     # Adjust to keep similar ratio
# or
WAYPOINT_DISTANCE_THRESHOLD = 7.0  # Easier to reach waypoints
```

**If agent needs stronger intermediate guidance** (still too sparse):
```python
NUM_WAYPOINTS = 40-50         # Very dense waypoints
WAYPOINT_REWARD = 80-100      # Adjust to keep similar ratio
```

**If agent drives too slowly** (needs more time pressure):
```python
STEP_PENALTY = 1.0            # Increase time penalty (from 0.5)
LAP_COMPLETION_REWARD = 1000.0  # Higher lap bonus encourages speed
```

**If agent drives too aggressively off-track** (crashes too often):
```python
STEP_PENALTY = 0.5          # Reduce time pressure (from 1.0)
OFFTRACK_PENALTY = 5.0      # Increase off-track penalty (from 2.0)
```

**If agent drives recklessly off-track** (too lenient):
```python
OFFTRACK_THRESHOLD = 0      # Penalize ANY wheel off track
# or
OFFTRACK_PENALTY = 2.0      # Increase penalty severity
```

**If agent is too cautious on corners** (too strict):
```python
OFFTRACK_THRESHOLD = 3      # Allow 3 wheels off
# or
OFFTRACK_PENALTY = 0.5      # Reduce penalty severity
```

### Experimentation Workflow

1. **Change constants** at top of `env/car_racing.py`
2. **Train agent** with new configuration
3. **Compare results** using training metrics
4. **Iterate** based on observed behavior

No code changes required - just tweak the constants!

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
1. **Check state mode + device combination**:
   - Vector mode: Use `--device cpu` (fastest)
   - Visual mode: Use `--device mps` or `--device cuda` (10x faster than CPU)
2. **Use verbose mode to diagnose**: `python train.py --verbose`
3. Reduce batch size if memory limited: `--batch-size 128`

### Verbose Mode Performance Diagnostics

Use `--verbose` flag to get detailed timing information for debugging performance issues:

```bash
python train.py --verbose --state-mode visual --device mps
```

**Environment Timing** (printed every 10 steps):
- Physics step time
- Collision detection time
- State creation time (rendering for visual mode)
- Total step time

**Agent Update Timing** (printed every 10 updates):
- Sample batch time
- Target network forward pass
- Critic 1/2 forward passes with **layer-level breakdown**:
  - conv1, conv2, conv3 timing (visual mode)
  - FC layer timing
- Actor update time
- Total update time
- CPU diagnostics: thread count, CPU usage, memory

**Example Output**:
```
SAC UPDATE 400 TIMING:
  Critic 1 forward:     5.98 ms  <<< WATCH THIS
    ├─ conv1:           0.05 ms  ← Individual layer timing
    ├─ conv2:           0.05 ms
    ├─ conv3:           0.04 ms
    └─ FC layers:       0.03 ms
  CPU DIAGNOSTICS:
    PyTorch threads:    4
    Memory usage:       410.9 MB
```

**What to Look For**:
- **Slow conv layers** (>20ms on CPU): Thermal throttling or need GPU
- **High total update time** (>500ms): Wrong device for state mode
- **Many PyTorch threads on CPU**: Try reducing with `torch.set_num_threads(2)`

**Timing Data Storage**:
Timing history is stored in `agent.layer_timings` list for post-training analysis.

## Additional Documentation

- `README.md`: User-facing documentation, quick start guide
- `SAC_EXPLAINED.md`: Deep dive into SAC algorithm, parameters, and metrics interpretation
- `logs/training.log`: Human-readable training progress during runs
- `logs/training_metrics.csv`: Complete episode-by-episode metrics for analysis

## References

SAC algorithm based on:
- Haarnoja et al., 2018: "Soft Actor-Critic: Off-Policy Maximum Entropy Deep Reinforcement Learning with a Stochastic Actor"
- Haarnoja et al., 2019: "Soft Actor-Critic Algorithms and Applications" (automatic entropy tuning)
