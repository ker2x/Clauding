# CarRacing-v3 SAC Agent (Project 004)

Soft Actor-Critic (SAC) implementation for CarRacing-v3 with **continuous action space** - no discretization.

**Note**: Project 004 is a fork of Project 003. Both projects share the same SAC implementation. This is an improved version of project 002, using:
- **Continuous actions** instead of discretized actions
- **SAC** (Soft Actor-Critic) instead of DDQN
- **Vector mode** (36D track geometry) for fast training
- **Visual mode** (96×96 frames) available for watching

## Overview

This project implements SAC for the CarRacing-v3 environment, which features a **continuous action space** (steering, gas, brake). Unlike project 002 which discretized actions for DQN, this project uses the continuous action space directly with SAC.

### Key Features

- **Soft Actor-Critic (SAC)**: State-of-the-art off-policy RL algorithm for continuous control
- **Continuous Actions**: Native support for `[steering, gas, brake]` without discretization
- **Twin Q-Networks**: Reduces Q-value overestimation bias
- **Automatic Entropy Tuning**: Learns optimal exploration-exploitation balance
- **Multiple State Modes**: Vector mode (RECOMMENDED) and visual mode
- **Experience Replay**: Stores and samples past experiences for stable learning
- **Frame Preprocessing**: Grayscale conversion, normalization, and frame stacking (visual mode)
- **Early Termination**: Stationary car detection for 3x training speedup
- **Continuous Reward Shaping**: Speed bonus and progressive off-track penalties
- **Checkpointing**: Save and resume training at any point

## Environment

**CarRacing-v3** (Gymnasium)
- **Task**: Control a racing car to complete laps on randomly generated tracks
- **Observation**: 36D vector (track geometry + lookahead) or 96×96×3 RGB images
- **Action Space**: Continuous `[steering, gas, brake]`
  - steering: [-1.0, 1.0] (left to right)
  - gas: [0.0, 1.0]
  - brake: [0.0, 1.0]
- **Reward**: +1000/N per track tile visited, -0.1 per frame, +0.02×speed bonus, -5 per wheel off-track

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

Unlike project 002 which discretized actions into 9 discrete choices, this project uses the native continuous action space:

**Action**: `[steering, gas, brake]`
- steering ∈ [-1.0, 1.0] (continuous)
- gas ∈ [0.0, 1.0] (continuous)
- brake ∈ [0.0, 1.0] (continuous)

SAC learns a stochastic policy that outputs a Gaussian distribution over actions, allowing for fine-grained control.

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

#### Vector Actor (RECOMMENDED - Default!)
```
Input: (36,) state vector
  ↓
FC1: 256 neurons → ReLU
  ↓
FC2: 256 neurons → ReLU
  ↓
FC3: 256 neurons → ReLU
  ↓
Mean: 3 neurons (action means)
Log_Std: 3 neurons (action log stds)
  ↓
Gaussian policy: π(a|s) = N(mean, exp(log_std))
```

#### Vector Critic (RECOMMENDED - Default!)
```
Input: (36,) state + (3,) action
  ↓
FC1: 256 neurons → ReLU
  ↓
FC2: 256 neurons → ReLU
  ↓
FC3: 256 neurons → ReLU
  ↓
FC4: 1 neuron (Q-value)
```

#### Visual Actor (Too Slow - For Watching Only)
```
Input: (4, 96, 96) stacked frames
  ↓
Conv1: 32 filters, 8×8, stride 4 → ReLU
  ↓
Conv2: 64 filters, 4×4, stride 2 → ReLU
  ↓
Conv3: 64 filters, 3×3, stride 1 → ReLU → Flatten
  ↓
FC1: 512 neurons → ReLU
  ↓
Mean: 3 neurons (action means)
Log_Std: 3 neurons (action log stds)
```

**Note**: Visual critic follows similar CNN architecture.

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

### Algorithm
- **002 (DDQN)**: Discrete action space (9 actions), value-based learning
- **003/004 (SAC)**: Continuous action space (infinite actions), actor-critic learning

### Action Space
- **002**: Discretized into 9 actions (3 steering × 3 gas/brake)
- **003/004**: Native continuous actions `[steering ∈ [-1,1], gas ∈ [0,1], brake ∈ [0,1]]`

### State Modes
- **002**: "snapshot" mode (36D vector), "vector" mode (11D, too limited), "visual" mode
- **003/004**: "vector" mode (36D, renamed from snapshot), "visual" mode (11D mode removed)

### Exploration
- **002**: ε-greedy (epsilon decays linearly)
- **003/004**: Maximum entropy (automatic entropy tuning via α parameter)

### Network Updates
- **002**: Single Q-network with target network
- **003/004**: Twin Q-networks + actor network + automatic alpha tuning

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

### "Import errors / Box2D missing"
- **Cause**: Box2D not installed
- **Fix**: `pip install 'gymnasium[box2d]'`

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
