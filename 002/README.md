# CarRacing-v3 DDQN Agent

Deep reinforcement learning implementation using Double Deep Q-Network (DDQN) to train an agent to play CarRacing-v3 from Gymnasium.

## Overview

This project implements DDQN for the CarRacing-v3 environment, which features a **continuous action space** (steering, gas, brake). The implementation discretizes the action space to make it compatible with DQN-based approaches.

### Key Features

- **Double DQN (DDQN)**: Reduces Q-value overestimation bias compared to standard DQN
- **Action Discretization**: Converts continuous actions to 9 discrete actions (3 steering × 3 gas/brake)
- **Experience Replay**: Stores and samples past experiences for stable learning
- **Frame Preprocessing**: Grayscale conversion, native 96×96 resolution, normalization, and frame stacking
- **Early Termination**: Stationary car detection for 3x training speedup
- **Reward Shaping**: Optional reward clipping to prevent catastrophic penalties
- **Checkpointing**: Save and resume training at any point
- **Visualization**: Watch trained agents play and compare with random baseline

## Environment

**CarRacing-v3** (Gymnasium)
- **Task**: Control a racing car to complete laps on randomly generated tracks
- **Observation**: Top-down RGB view (96×96×3)
- **Action Space**: Continuous `[steering, gas, brake]`
  - steering: [-1.0, 1.0] (left to right)
  - gas: [0.0, 1.0]
  - brake: [0.0, 1.0]
- **Reward**: +1000/N per track tile visited, -0.1 per frame, negative for off-track

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
# Basic training (2000 episodes)
python train.py

# Custom training
python train.py --episodes 1000 --learning-starts 10000

# Resume from checkpoint
python train.py --resume checkpoints/final_model.pt
```

### Watch Random Agent (Baseline)

```bash
python watch_random_agent.py --episodes 3
```

### Watch Trained Agent

```bash
python watch_agent.py --checkpoint checkpoints/final_model.pt --episodes 5
```

### Inspect Checkpoint

```bash
python inspect_checkpoint.py checkpoints/final_model.pt
```

## How It Works

### 1. Action Discretization

The continuous action space `[steering, gas, brake]` is discretized into 9 actions:

| Action | Steering | Gas/Brake | Description |
|--------|----------|-----------|-------------|
| 0 | LEFT (-0.8) | BRAKE (0.8) | Turn left while braking |
| 1 | STRAIGHT (0.0) | BRAKE (0.8) | Brake without turning |
| 2 | RIGHT (+0.8) | BRAKE (0.8) | Turn right while braking |
| 3 | LEFT (-0.8) | COAST (0.0) | Turn left, no gas/brake |
| 4 | STRAIGHT (0.0) | COAST (0.0) | Coast straight |
| 5 | RIGHT (+0.8) | COAST (0.0) | Turn right, no gas/brake |
| 6 | LEFT (-0.8) | GAS (0.8) | Turn left while accelerating |
| 7 | STRAIGHT (0.0) | GAS (0.8) | Accelerate straight |
| 8 | RIGHT (+0.8) | GAS (0.8) | Turn right while accelerating |

You can customize the discretization granularity with `--steering-bins` and `--gas-brake-bins`.

### 2. Frame Preprocessing

Raw frames undergo several transformations:

1. **RGB → Grayscale**: Reduces channels from 3 to 1 (preserves track boundaries)
2. **Native Resolution**: Uses CarRacing's native 96×96 resolution (no resize needed)
3. **Normalize**: [0, 255] → [0, 1] (better for neural networks)
4. **Frame Stacking**: Stack 4 consecutive frames to capture motion/velocity

Final shape: **(4, 96, 96)** - 4 stacked 96×96 grayscale frames

**Why native resolution?**
- Faster preprocessing (no resize operation)
- Better image quality (no information loss)
- Only 30% more pixels than 84×84 with minimal computational overhead

### 3. Double DQN Algorithm

DDQN improves upon DQN by decoupling action selection and evaluation:

**Standard DQN (overestimates Q-values):**
```
Q_target = r + γ * max_a' Q_target(s', a')
```

**Double DQN (reduces overestimation):**
```
Q_target = r + γ * Q_target(s', argmax_a' Q_policy(s', a'))
              └─ Use policy network to SELECT action
              └─ Use target network to EVALUATE action
```

### 4. Network Architecture

```
Input: (4, 96, 96) stacked frames (native CarRacing resolution)
  ↓
Conv1: 32 filters, 8×8 kernel, stride 4 → 23×23
  ↓ ReLU
Conv2: 64 filters, 4×4 kernel, stride 2 → 10×10
  ↓ ReLU
Conv3: 64 filters, 3×3 kernel, stride 1 → 8×8
  ↓ ReLU → Flatten (4096 features)
FC1: 512 neurons
  ↓ ReLU
FC2: 9 neurons (Q-values for each action)
```

**Note**: The network architecture is optimized for 96×96 input (4096 conv output features vs 3136 for 84×84).

## Training Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--episodes` | 2000 | Number of episodes to train |
| `--learning-starts` | 10000 | Steps before training begins |
| `--lr` | 0.00025 | Learning rate |
| `--gamma` | 0.99 | Discount factor |
| `--epsilon-decay` | 1000000 | Steps to decay epsilon |
| `--buffer-size` | 100000 | Replay buffer capacity |
| `--batch-size` | 32 | Training batch size |
| `--target-update-freq` | 10000 | Steps between target network updates |
| `--steering-bins` | 3 | Number of discrete steering values |
| `--gas-brake-bins` | 3 | Number of discrete gas/brake values |

## Training Timeline

CarRacing is more challenging than Atari games like Breakout. Expect longer training times:

| Steps | Epsilon | Expected Behavior |
|-------|---------|-------------------|
| 50k-100k | 0.90-0.95 | Mostly random exploration |
| 200k-500k | 0.61-0.80 | Learning basic control |
| 500k-1M | 0.37-0.61 | Learning to stay on track |
| 1M-2M | 0.01-0.37 | Improving racing strategy |
| 2M+ | 0.01 | Strong performance |

**Important**: Epsilon decays based on **steps**, not episodes. CarRacing episodes can be 1000+ frames, so patience is required!

## File Structure

```
.
├── CLAUDE.md                  # Context for Claude Code
├── README.md                  # This file
├── requirements.txt           # Python dependencies
├── .gitignore                # Git ignore rules
│
├── preprocessing.py           # Frame preprocessing & action discretization
├── ddqn_agent.py             # DDQN agent implementation
├── train.py                  # Main training script
│
├── watch_agent.py            # Visualize trained agent
├── watch_random_agent.py     # Visualize random agent (baseline)
├── inspect_checkpoint.py     # Inspect saved models
├── test_setup.py             # Verify installation
│
├── checkpoints/              # Saved model checkpoints
└── logs/                     # Training logs and plots
```

## Key Differences from Discrete Action Spaces

This project handles CarRacing's **continuous action space**, unlike typical DQN implementations for discrete-action games (e.g., Atari Breakout).

### Action Space
- **Discrete (Breakout)**: 4 discrete actions (NOOP, FIRE, LEFT, RIGHT)
- **Continuous (CarRacing)**: 3D continuous vector `[steering, gas, brake]`
  - **Solution**: Discretize into 9 discrete actions (or more with custom bins)

### State Representation
- **Breakout**: Grayscale works well (high contrast paddles and ball), resized to 84×84
- **CarRacing**: Grayscale preserves track boundaries (grass vs. track), native 96×96 resolution
  - No resize operation needed (faster preprocessing, better quality)

### Reward Structure
- **Breakout**: Sparse rewards (brick destroyed = +1)
- **CarRacing**: Dense rewards (continuous progress tracking)
  - Includes time penalties (-0.1 per frame)
  - Large negative rewards for going off-track (can be clipped)

## Recent Improvements (2025-11-03)

### Native 96×96 Resolution
- **Changed from**: 84×84 resized frames
- **Changed to**: 96×96 native CarRacing resolution
- **Benefits**: Faster preprocessing (no resize), better image quality, minimal overhead

### Unified Environment
- **Changed from**: Separate training/evaluation environments
- **Changed to**: Single environment for both training and evaluation
- **Benefits**: Faster evaluation (30-60s vs 2-5min), consistent behavior, simpler code
- **User Experience**: Real-time progress output during evaluation

## Tips for Better Performance

1. **Train Longer**: CarRacing requires 1M-2M+ steps for good performance
2. **Adjust Discretization**: Experiment with finer steering bins (5 instead of 3)
3. **Reward Shaping**: Tune negative reward clipping to balance exploration
4. **Check Epsilon**: Use `inspect_checkpoint.py` to verify epsilon is decaying
5. **Resume Training**: Don't start from scratch if epsilon is still high
6. **GPU Acceleration**: Training on Apple Silicon (MPS) or CUDA is much faster
7. **Monitor Evaluation**: Progress output now shows each evaluation episode in real-time

## Troubleshooting

### "Agent looks random after training"
- **Cause**: Epsilon is still high (agent was mostly exploring)
- **Fix**: Check epsilon with `inspect_checkpoint.py`, resume training for more episodes

### "Agent goes off track immediately"
- **Cause**: Insufficient training or poor hyperparameters
- **Fix**: Train for at least 1M steps, consider finer action discretization

### "Training is very slow"
- **Cause**: Running on CPU instead of GPU
- **Fix**: Verify MPS/CUDA is available with `test_setup.py`

### "Import errors / Box2D missing"
- **Cause**: Box2D not installed
- **Fix**: `pip install 'gymnasium[box2d]'`

## References

- **DDQN Paper**: [Deep Reinforcement Learning with Double Q-learning](https://arxiv.org/abs/1509.06461) (van Hasselt et al., 2015)
- **DQN Paper**: [Human-level control through deep reinforcement learning](https://www.nature.com/articles/nature14236) (Mnih et al., 2015)
- **Gymnasium Docs**: [CarRacing-v3](https://gymnasium.farama.org/environments/box2d/car_racing/)

## License

This is an educational project for learning deep reinforcement learning.

## Acknowledgments

Built with:
- [Gymnasium](https://gymnasium.farama.org/) - Reinforcement learning environments
- [PyTorch](https://pytorch.org/) - Deep learning framework
- [OpenCV](https://opencv.org/) - Image processing and visualization
