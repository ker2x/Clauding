# CarRacing-v3 SAC Agent (Project 006 - Streamlined)

Soft Actor-Critic (SAC) implementation for CarRacing-v3 with **continuous action space**, **custom 2D physics engine**, and **parallel selection training**.

## Overview

This project implements state-of-the-art reinforcement learning for racing using:
- **Parallel Selection Training**: N independent agents with evolutionary selection (RECOMMENDED)
- **Soft Actor-Critic (SAC)**: Maximum entropy RL with automatic tuning
- **Custom 2D Physics**: Clean, interpretable simulation with Magic Formula tires
- **Continuous Actions**: Native `[steering, acceleration]` without discretization
- **Vector Mode**: Fast 71D state representation (car + track + lookahead)
- **Streamlined Codebase**: Focused on vector mode for optimal performance

## Quick Start

### 1. Setup

```bash
# Activate shared virtual environment
source ../.venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Verify installation
python test_setup.py
```

### 2. Train an Agent (RECOMMENDED)

```bash
# Parallel selection training with 8 agents (fastest, best results)
python train_selection_parallel.py --num-agents 8 --episodes 2000

# With elite preservation (maintains diversity)
python train_selection_parallel.py --num-agents 8 --elite-count 2

# Quick test with fewer agents
python train_selection_parallel.py --num-agents 4 --episodes 500

# Resume from latest generation
python train_selection_parallel.py --resume checkpoints_selection_parallel/latest_generation.pt
```

**Why parallel selection?**
- True parallel execution (8× CPU utilization)
- Evolutionary pressure (best agents survive)
- Optional elite preservation (maintains diversity)
- Faster convergence through selection
- Sample efficient (8× data collection)
- Automatic checkpoints every tournament

### 3. Watch Your Agent

```bash
# Watch trained agent
python watch_agent.py --checkpoint checkpoints_selection_parallel/best_model.pt --episodes 5

# Watch random baseline
python watch_random_agent.py --episodes 3

# Play as human (try it yourself!)
python play_human.py
```

**Human controls:**
- Steering: A/D or Arrow Keys
- Gas: W or Up
- Brake: S or Down
- Reset: R

## Training Methods

### Parallel Selection Training (PRIMARY)

```bash
python train_selection_parallel.py --num-agents 8 --episodes 2000
```

**How it works:**
1. N agents train simultaneously on separate CPU cores
2. Every M episodes: synchronize and evaluate all agents
3. Select best performer, clone to all positions
4. Restart parallel training with winner

**Benefits:**
- ~8× wall-clock speedup with 8 agents
- Evolutionary selection pressure
- Optional elite preservation (--elite-count 2+) maintains diversity
- Automatic checkpoint saving every tournament
- No manual hyperparameter tuning needed

**Checkpoints saved:**
- `generation_N.pt`: Winner from each tournament
- `latest_generation.pt`: Most recent (for easy resume)
- `best_model.pt`: Best reward ever (only updated on improvement)

### Alternative Training Method

**Standard single-agent:**
```bash
python train.py --episodes 2000
```

See `TRAINING_COMPARISON.md` for detailed comparison.

## Environment

**CarRacing-v3** with custom physics:
- **Observation**: 71D vector (car state + track geometry + lookahead waypoints)
- **Action Space**: Continuous `[steering, acceleration]`
  - steering ∈ [-1, 1] (left to right)
  - acceleration ∈ [-1, 1] (brake to gas)
- **Rewards**:
  - +100 per checkpoint (15 total)
  - +1000 for completing lap
  - +0.1 per m/s forward velocity
  - -2.0 per frame (time pressure)
  - -1.0 per wheel off-track (when >2 wheels off)

## How It Works

### Soft Actor-Critic (SAC)

SAC is an off-policy, maximum entropy RL algorithm that:

1. **Maximizes reward AND entropy** (encourages exploration)
2. **Uses twin Q-networks** (reduces overestimation)
3. **Automatically tunes exploration** (learns alpha parameter)

**Components:**
- **Actor**: Learns stochastic policy (Gaussian distribution)
- **Critics**: Two Q-networks estimate action values
- **Entropy Coefficient**: Automatically adjusted for optimal exploration

### State Representation

**Vector Mode (71D):**
- Car state (11D): position, velocity, angle, wheel contacts, progress
- Track segment (5D): distance to center, angle, curvature
- Lookahead waypoints (40D): 20 future waypoints in car coordinates
- Vertical forces (4D): normal forces on each tire
- Additional features (11D): opponent info and extended state data

**Why vector mode?**
- Fast, efficient training
- No rendering overhead
- Full track geometry information
- Proven to work well for racing tasks

### Network Architecture

**Actor (Policy):**
```
71D state → FC(256)×3 → LeakyReLU
          → mean (2D) + log_std (2D)
          → Sample action from Gaussian
          → Squash with tanh to [-1, 1]
```

**Critic (Q-function):**
```
71D state + 2D action → FC(512)×4 → LeakyReLU
                      → Q-value (scalar)
```

Uses LayerNorm for stability, LeakyReLU to prevent dead neurons.

## Training Parameters

All default values are defined in `constants.py` for centralized configuration.

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--num-agents` | 4 | Parallel agents (selection training) |
| `--selection-frequency` | 50 | Episodes between tournaments |
| `--eval-episodes` | 10 | Episodes per tournament evaluation |
| `--elite-count` | 2 | Top N agents preserved (1=winner-takes-all, 2+=elite) |
| `--episodes` | 200 | Total training episodes per agent |
| `--learning-starts` | 5000 | Random steps before learning |
| `--lr-actor` | 1e-4 | Actor learning rate |
| `--lr-critic` | 1e-4 | Critic learning rate |
| `--lr-alpha` | 1e-3 | Alpha (entropy) learning rate |
| `--gamma` | 0.99 | Discount factor |
| `--tau` | 0.005 | Target network update rate |
| `--buffer-size` | 200000 | Replay buffer capacity |
| `--batch-size` | 512 | Training batch size |

**Note:** Intermediate evaluations during training use 5 episodes, while final evaluations use 10 episodes.

## Training Timeline

**With 8 parallel agents (RECOMMENDED):**

| Phase | Episodes/Agent | Total Samples | Expected Behavior |
|-------|----------------|---------------|-------------------|
| Exploration | 1-50 | ~400k | Random exploration, learning basics |
| Learning | 50-200 | ~1.6M | Consistent track following |
| Improvement | 200-500 | ~4M | Good racing lines, lap completion |
| Mastery | 500+ | 8M+ | Optimal performance, 700+ rewards |

**Selection pressure accelerates learning:** Best agents survive and improve each generation.

## Key Metrics

**Episode Metrics:**
- `reward`: Target 500+ for good performance
- `episode_steps`: Target 500+ steps (full laps)
- `best_agent`: Winner of each selection tournament

**SAC Metrics:**
- `actor_loss`: Can be negative (normal!)
- `critic_loss`: Should decrease and stabilize
- `alpha`: Decreases from ~0.8 to ~0.01-0.2
- `mean_q1/q2`: Should correlate with actual rewards

**Healthy training:**
- Rewards trending upward
- Alpha decreasing smoothly
- Critic losses stabilizing
- Episode length increasing
- Selection winners improving each generation

## File Structure

```
006/
├── env/
│   ├── car_racing.py       # Custom CarRacing environment
│   └── car_dynamics.py     # 2D physics with Magic Formula
│
├── sac_agent.py            # SAC implementation (vector mode only)
├── preprocessing.py         # Environment factory function
│
├── train_selection_parallel.py  # PRIMARY training method
├── train.py                     # Single-agent training
│
├── watch_agent.py          # Visualize trained agent
├── watch_random_agent.py   # Baseline comparison
├── play_human.py           # Human playable mode
├── test_setup.py           # Setup verification
│
├── checkpoints_selection_parallel/  # Saved models
├── logs_selection_parallel/         # Training logs
│
└── [Documentation files]
```

## Troubleshooting

### Agent doesn't learn
- **Check alpha:** Should decrease to 0.01-0.2 (not stay at 0.8)
- **Verify learning started:** Look for message at `learning_starts` steps
- **Train longer:** Needs 1M+ steps for good performance
- **Check selection:** Winners should improve each generation

### Training unstable
- **Reduce learning rates:** Try `--lr-actor 1e-4 --lr-critic 1e-4`
- **Reduce tau:** Try `--tau 0.002`
- **Increase batch size:** Try `--batch-size 512`

### Out of memory
- **Reduce buffer size:** Try `--buffer-size 500000`
- **Reduce agents:** Try `--num-agents 4`

### Slow training
- **Check CPU usage:** Should be ~800% with 8 agents
- **Reduce agents if limited cores:** 4 agents needs 4+ cores
- **Monitor progress:** Check `logs_selection_parallel/` directory

### Tournament gets stuck
- **Timeout protection:** Evaluation limited to 2500 steps per episode
- **Check diagnostics:** Should see "Sent EVALUATE to agent N" messages
- **Missing agents:** Assigned -inf reward after 2-minute timeout

## Reward Tuning

Edit reward constants in `env/car_racing.py:64-71`:

**Make agent faster (more aggressive):**
```python
STEP_PENALTY = 3.0  # Increase time pressure
```

**Make agent safer (less aggressive):**
```python
OFFTRACK_PENALTY = 0.5  # Reduce off-track penalty
OFFTRACK_THRESHOLD = 3  # Allow 3 wheels off track
```

**Increase progress incentive:**
```python
PROGRESS_REWARD_SCALE = 6000.0
```

## Documentation

- `CLAUDE.md`: Technical guide for Claude Code
- `SAC_EXPLAINED.md`: Deep dive into SAC algorithm
- `TRAINING_COMPARISON.md`: Comparison of training methods
- `PERFORMANCE_ANALYSIS.md`: Performance optimization analysis

## References

- **SAC Paper**: [Soft Actor-Critic (Haarnoja et al., 2018)](https://arxiv.org/abs/1801.01290)
- **Automatic Tuning**: [SAC Algorithms and Applications (Haarnoja et al., 2019)](https://arxiv.org/abs/1812.05905)
- **Environment**: [Gymnasium CarRacing-v3](https://gymnasium.farama.org/environments/box2d/car_racing/)

## Project 006 Cleanup

This version has been streamlined to focus on vector mode training:
- **Removed**: Visual mode (96x96 image observations - never used)
- **Removed**: Discrete action space (continuous-only now)
- **Removed**: Domain randomization (always disabled)
- **Removed**: Fuel tracking (unused penalty)
- **Removed**: Non-parallel selection training variants
- **Removed**: Multi-car training scripts
- **Removed**: VectorEnv training scripts
- **Kept**: Vector mode (71D state) for optimal performance
- **Kept**: Parallel selection training and standard training
- **Added**: Centralized configuration via `constants.py`
- **Result**: Cleaner, more maintainable codebase focused on what works best

The environment uses vector mode for both training and agent observations.
Rendering (600x400 RGB) is only used for human visualization via watch scripts.

## License

This is an educational project for learning deep reinforcement learning.

## Acknowledgments

Built with:
- [Gymnasium](https://gymnasium.farama.org/) - RL environments
- [PyTorch](https://pytorch.org/) - Deep learning framework
- [OpenCV](https://opencv.org/) - Visualization

---

*Project 006 - 2025 - Streamlined Vector Mode*
