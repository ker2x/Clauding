# CarRacing-v3 SAC Agent (Project 005)

Soft Actor-Critic (SAC) implementation for CarRacing-v3 with **continuous action space** and **custom 2D physics engine** (no Box2D dependency).

**Note**: Project 005 is an evolution of Project 004, featuring a custom physics engine that removes the Box2D dependency. This is an improved version of earlier projects, using:
- **Continuous actions** instead of discretized actions
- **SAC** (Soft Actor-Critic) instead of DDQN
- **Vector mode** (36D track geometry) for fast training
- **Visual mode** (96×96 frames) available for watching

## Overview

This project implements SAC for the CarRacing-v3 environment, which features a **continuous action space** (steering, gas, brake). Unlike project 002 which discretized actions for DQN, this project uses the continuous action space directly with SAC.

### Key Features

- **Custom 2D Physics Engine**: Removed Box2D dependency for cleaner, more interpretable physics simulation
- **Soft Actor-Critic (SAC)**: State-of-the-art off-policy RL algorithm for continuous control
- **Continuous Actions**: Native support for `[steering, gas, brake]` without discretization
- **Twin Q-Networks**: Reduces Q-value overestimation bias
- **Automatic Entropy Tuning**: Learns optimal exploration-exploitation balance
- **Multiple State Modes**: Vector mode (RECOMMENDED) and visual mode
- **Experience Replay**: Stores and samples past experiences for stable learning
- **Frame Preprocessing**: Grayscale conversion, normalization, and frame stacking (visual mode)
- **Early Termination**: Stationary car detection for 3x training speedup
- **Enhanced Reward Shaping**: Forward velocity bonus, lap completion bonus, and increased step penalty for faster lap times
- **Checkpointing**: Save and resume training at any point

## Environment

**CarRacing-v3** (Gymnasium)
- **Task**: Control a racing car to complete laps on randomly generated tracks
- **Observation**: 36D vector (track geometry + lookahead) or 96×96×3 RGB images
- **Action Space**: Continuous `[steering, acceleration]`
  - steering: [-1.0, 1.0] (left to right)
  - acceleration: [-1.0, 1.0] (brake to gas)
- **Reward**: +100 per checkpoint (15 total), +1000 lap completion, +0.1 per m/s forward velocity, -2.0 per frame, -1.0 per wheel off-track (when >2 wheels off)

## Setup

This project uses a shared virtual environment in the parent directory.

### 1. Activate Virtual Environment

```bash
source ../.venv/bin/activate
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Verify Setup

```bash
python test_setup.py
```

You should see:
```
🎉 ALL TESTS PASSED!
Your environment is ready for training.
```

## Quick Start

### Train an Agent

```bash
# Basic training (2000 episodes, vector mode - RECOMMENDED)
python train.py

# Custom training
python train.py --episodes 1000 --learning-starts 5000

# Use visual mode (slow, not recommended for training)
python train.py --episodes 200 --state-mode visual

# Resume from checkpoint
python train.py --resume checkpoints/best_model.pt
```

**Note**: Training uses fast vector mode by default (3-5x speedup vs visual). Watch scripts automatically use visual mode.

### Watch Random Agent (Baseline)

```bash
python watch_random_agent.py --episodes 3
```

### Watch Trained Agent

```bash
python watch_agent.py --checkpoint checkpoints/best_model.pt --episodes 5
```

### Play as Human

Want to try driving the car yourself? Use the human playable script:

```bash
python play_human.py --episodes 1
```

**Controls:**
- **Steering**: A/D or Arrow Left/Right
- **Gas**: W or Arrow Up
- **Brake**: S or Arrow Down
- **Reset Action**: SPACE (coast to neutral)
- **Reset Episode**: R
- **Quit**: Q or ESC

This is a great way to understand how difficult the task is and why 500+ reward (completing laps) is impressive for a trained agent!

## How It Works

### 1. Continuous Action Space

Unlike project 002 which discretized actions, this project uses continuous actions:

**Action**: `[steering, acceleration]`
- steering ∈ [-1.0, 1.0] (left to right)
- acceleration ∈ [-1.0, 1.0] (brake to gas)

SAC learns a stochastic Gaussian policy that allows fine-grained control.

### 2. State Representation

The agent can use two state representations:

#### Vector Mode (RECOMMENDED - Default for Training!)
Returns a 36-dimensional compact state vector:
- **Car state** (11): `[x, y, vx, vy, angle, angular_vel, wheel_contacts[4], track_progress]`
- **Track segment** (5): `[dist_to_center, angle_diff, curvature, dist_along_segment, segment_length]`
- **Lookahead waypoints** (20): 10 waypoints × (x, y) in car-relative coordinates
- No rendering required
- Uses MLP network optimized for 36D input
- 3-5x faster training than visual mode
- Agent learns proper racing behavior
- Low memory usage

#### Visual Mode (For Watching Only)
Raw frames undergo several transformations:
1. **RGB → Grayscale**: Reduces channels from 3 to 1 (preserves track boundaries)
2. **Native Resolution**: Uses CarRacing's native 96×96 resolution (no resize needed)
3. **Normalize**: [0, 255] → [0, 1] (better for neural networks)
4. **Frame Stacking**: Stack 4 consecutive frames to capture motion/velocity

Final shape: **(4, 96, 96)** - 4 stacked 96×96 grayscale frames
- Uses CNN architecture
- Full rendering for visualization
- Too slow for training

**Recommendation**: Use vector mode for training (default). Visual mode is automatically used for watching.

### 3. Soft Actor-Critic Algorithm

SAC is an off-policy, maximum entropy RL algorithm:

**Key Components:**
1. **Actor (Policy)**: Learns a stochastic policy π(a|s) that maximizes both reward and entropy
2. **Twin Critics**: Two Q-networks Q₁, Q₂ to reduce overestimation bias
3. **Entropy Coefficient**: Automatically tuned α parameter that balances exploration vs exploitation

**Update Rules:**
```
# Critic update: minimize TD error
Q_target = r + γ(min(Q₁(s', a'), Q₂(s', a')) - α log π(a'|s'))

# Actor update: maximize Q-value and entropy
J(π) = E[α log π(a|s) - Q(s,a)]

# Alpha update: match target entropy
J(α) = E[-α(log π(a|s) + H_target)]
```

### 4. Network Architecture

#### Vector Mode (RECOMMENDED - Default!)
**Actor**: 36D input → FC(256)×3 → 2D action (mean, log_std)
**Critic**: 36D state + 2D action → FC(256)×4 → Q-value

#### Visual Mode (For Watching Only)
**Actor**: 4×96×96 frames → Conv layers → FC(512) → 2D action
**Critic**: Similar CNN + action input → Q-value

## Training Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--episodes` | 2000 | Number of episodes to train |
| `--learning-starts` | 5000 | Steps before training begins |
| `--lr-actor` | 3e-4 | Actor learning rate |
| `--lr-critic` | 3e-4 | Critic learning rate |
| `--gamma` | 0.99 | Discount factor |
| `--tau` | 0.005 | Target network soft update rate |
| `--buffer-size` | 100000 | Replay buffer capacity |
| `--batch-size` | 256 | Training batch size |
| `--state-mode` | `vector` | State: `vector` (RECOMMENDED) or `visual` (slow) |

## Training Timeline

CarRacing is more challenging than Atari games. With vector mode (RECOMMENDED):

| Steps | Time (Vector Mode) | Expected Behavior |
|-------|-------------------|-------------------|
| 50k-100k | ~10-20 min | Mostly random exploration |
| 200k-500k | ~30-90 min | Learning basic control |
| 500k-1M | ~1.5-3 hours | Learning to stay on track |
| 1M-2M | ~3-6 hours | Improving racing strategy |
| 2M+ | ~6+ hours | Strong performance |

**Important**:
- Vector mode is **3-5x faster** than visual mode (default for training)
- Times assume Apple Silicon (MPS) or CUDA GPU

## File Structure

```
.
├── README.md                     # This file
├── requirements.txt              # Python dependencies
├── .gitignore                   # Git ignore rules
│
├── env/
│   ├── __init__.py              # Package init
│   ├── car_racing.py            # CarRacing environment with vector state
│   └── car_dynamics.py          # Car physics simulation
├── preprocessing.py              # Frame preprocessing & environment wrappers
├── sac_agent.py                 # SAC agent (actor-critic networks)
├── train.py                     # Main training script
│
├── watch_agent.py               # Visualize trained agent
├── watch_random_agent.py        # Visualize random agent (baseline)
├── play_human.py                # Play as human (keyboard control)
├── test_setup.py                # Verify installation
│
├── checkpoints/                 # Saved model checkpoints
└── logs/                        # Training logs and plots
```

## Key Differences from Project 002 (DDQN)

- **Algorithm**: DDQN (value-based) → SAC (actor-critic with entropy)
- **Action Space**: 9 discrete actions → 2D continuous [steering, acceleration]
- **State Modes**: Renamed "snapshot" → "vector" (36D), removed 11D mode
- **Exploration**: ε-greedy → Maximum entropy with auto-tuning
- **Network**: Single Q + target → Twin Q + actor + learned alpha

## Tips for Better Performance

1. **Use Vector Mode**: Default for training (3-5x faster than visual, full track information)
2. **Train Longer**: CarRacing requires 1M-2M+ steps for good performance
3. **Monitor Entropy**: Alpha should converge to a stable value indicating good exploration
4. **Check Q-Values**: Monitor Q-values to ensure they're not diverging
5. **Resume Training**: Don't start from scratch if alpha is still adapting
6. **GPU Acceleration**: Training on Apple Silicon (MPS) or CUDA is much faster

## Troubleshooting

### "Agent looks random after training"
- **Cause**: Not enough training steps or alpha too high
- **Fix**: Train for at least 1M steps, check alpha convergence

### "Agent goes off track immediately"
- **Cause**: Insufficient training or poor hyperparameters
- **Fix**: Train for at least 1M steps, verify vector mode is being used

### "Training is very slow"
- **Cause 1**: Using visual mode instead of vector mode
- **Fix 1**: Ensure `--state-mode vector` (or omit, it's the default)
- **Cause 2**: Running on CPU instead of GPU
- **Fix 2**: Verify MPS/CUDA is available with `test_setup.py`

### "Import errors / pygame or opencv missing"
- **Cause**: Required dependencies not installed
- **Fix**: `pip install -r requirements.txt` or individually: `pip install pygame opencv-python`

## References

- **SAC Paper**: [Soft Actor-Critic: Off-Policy Maximum Entropy Deep Reinforcement Learning with a Stochastic Actor](https://arxiv.org/abs/1801.01290) (Haarnoja et al., 2018)
- **SAC with Auto-Tuning**: [Soft Actor-Critic Algorithms and Applications](https://arxiv.org/abs/1812.05905) (Haarnoja et al., 2019)
- **Gymnasium Docs**: [CarRacing-v3](https://gymnasium.farama.org/environments/box2d/car_racing/)

## License

This is an educational project for learning deep reinforcement learning.

## Acknowledgments

Built with:
- [Gymnasium](https://gymnasium.farama.org/) - Reinforcement learning environments
- [PyTorch](https://pytorch.org/) - Deep learning framework
- [OpenCV](https://opencv.org/) - Image processing and visualization
