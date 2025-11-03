# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is an educational Deep Q-Network (DQN) implementation for training reinforcement learning agents to play Atari Breakout. The codebase emphasizes clarity and learning, with extensive documentation explaining RL concepts.

## Environment Setup

**IMPORTANT**: This subproject (`001/`) uses a shared virtual environment located in the parent directory.

### Virtual Environment
- **Location**: `../venv/` (parent directory)
- **Activation**: Always activate before running any Python commands
  ```bash
  source ../.venv/bin/activate
  ```
- All Python package installations should be done within this activated virtual environment
- The virtual environment contains all dependencies from `requirements.txt`

### First-Time Setup
If packages are not installed in the parent virtual environment:
```bash
# Activate the parent virtual environment
source ../.venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

## Core Architecture

### DQN Agent (`dqn_agent.py`)
The agent uses two neural networks (policy and target) to learn Q-values:
- **Policy Network**: Actively trained, used for action selection
- **Target Network**: Frozen copy updated every 10k steps, provides stable learning targets
- **Replay Buffer**: Stores experiences (state, action, reward, next_state, done) for training
- **Network Architecture**: 3 convolutional layers → 2 fully connected layers → Q-values for each action

Key concept: The Bellman equation `Q(s,a) = r + γ * max Q(s',a')` drives learning through MSE loss between current and target Q-values.

### Preprocessing (`preprocessing.py`)
Atari frames undergo standard preprocessing plus game-specific wrappers:
1. **FireResetWrapper**: Auto-fires at game start and after losing lives (Breakout requires FIRE to launch ball)
2. **NoopFireLeftRightActions**: Simplifies action space from 4→3 actions (removes FIRE, keeps NOOP/RIGHT/LEFT)
3. RGB (210×160×3) → Grayscale (210×160)
4. Resize to 84×84 (reduces computation ~84%)
5. Stack 4 frames → (4, 84, 84) to capture motion/velocity
6. Optional reward clipping to {-1, 0, +1}

**Action Space**: The agent learns with 3 actions (NOOP, RIGHT, LEFT). FIRE is handled automatically by the wrapper.

### Training Loop (`train.py`)
1. Collect experiences by playing with epsilon-greedy policy
2. Store in replay buffer
3. Sample random batches and train policy network
4. Periodically update target network
5. Evaluate and checkpoint every N episodes

## Common Commands

**Note**: All commands assume the parent virtual environment is activated: `source ../.venv/bin/activate`

### Setup and Testing
```bash
# Activate virtual environment (if not already active)
source ../.venv/bin/activate

# Install dependencies (if needed)
pip install -r requirements.txt

# Verify installation
python test_setup.py

# Visualize preprocessing pipeline
python visualize_preprocessing.py
```

### Training
```bash
# Basic training (default: 10k episodes)
python train.py

# Custom training duration
python train.py --episodes 2000 --learning-starts 10000

# Resume from checkpoint (preserves epsilon)
python train.py --resume checkpoints/final_model.pt --episodes 1000

# Resume with reset epsilon (forces more exploration)
python train.py --resume checkpoints/final_model.pt --reset-epsilon --episodes 1000

# Fast epsilon decay training
python train_fast_decay.py
```

### Evaluation
```bash
# Watch trained agent play
python watch_agent.py --checkpoint checkpoints/final_model.pt

# Watch specific checkpoint for 3 episodes
python watch_agent.py --checkpoint checkpoints/checkpoint_ep1000.pt --episodes 3

# Watch random agent (test game mechanics)
python watch_random_agent.py --episodes 3

# Inspect checkpoint details (steps, epsilon, recommendations)
python inspect_checkpoint.py checkpoints/final_model.pt
```

## Critical Training Insights

### Epsilon-Greedy Exploration
- **Epsilon decays based on STEPS, not episodes**
- Default decay: 1.0 → 0.01 over 1M steps
- If epsilon is still high (>0.5), the agent needs much more training
- Training behavior: With ε=0.8, agent is 80% random, 20% learned policy
- Evaluation behavior: Always uses ε=0 (pure learned policy)

### Training Timeline Expectations
| Steps | Epsilon | Expected Behavior |
|-------|---------|-------------------|
| 50k-100k | 0.90-0.95 | Mostly random |
| 500k | 0.61 | Learning to position paddle |
| 1M | 0.37 | Good gameplay emerging |
| 2M+ | 0.01 | Strong performance |

**Key point**: Short training (150-350 episodes = ~250k steps) results in ε≈0.75-0.80. The agent is still mostly random during training.

### Device Support
- Auto-detects CUDA (NVIDIA), MPS (Apple Silicon), or CPU
- Training on M3 Mac uses MPS for GPU acceleration
- On CPU: ~3-5x slower than GPU

### Checkpointing
- `checkpoints/checkpoint_ep*.pt`: Saved every 500 episodes
- `checkpoints/final_model.pt`: Final model after training
- `logs/training_progress.png`: Training curves (rewards, loss, epsilon)

Checkpoints store:
- Policy network weights
- Target network weights
- Optimizer state
- `steps_done` (for epsilon calculation)

## Key Hyperparameters

| Parameter | Default | Purpose |
|-----------|---------|---------|
| learning_rate | 0.00025 | Gradient descent step size |
| gamma | 0.99 | Discount factor (future reward importance) |
| epsilon_start | 1.0 | Initial exploration rate |
| epsilon_end | 0.01 | Final exploration rate |
| epsilon_decay | 1M steps | Steps to decay epsilon |
| buffer_size | 100k | Replay buffer capacity |
| batch_size | 32 | Training batch size |
| learning_starts | 50k | Steps before training begins |
| target_update_freq | 10k | Steps between target network updates |

## Common Issues

### "Agent looks lazy/random after training"
**Root cause**: Epsilon is still high (agent was mostly random during training)

**Solution**: Check epsilon with `inspect_checkpoint.py`, then resume training for 1000+ more episodes

### "Agent learned a safe but boring strategy"
**Root cause**: Local optimum - common in RL

**Solutions**:
1. Train longer for more exploration
2. Resume with `--reset-epsilon` to force exploration
3. Try `train_fast_decay.py` for faster exploitation

### "Should I resume or start fresh?"
**Resume if**:
- Epsilon > 0.3 (still exploring)
- Loss is decreasing
- Want to save training time

**Start fresh if**:
- Epsilon < 0.1 AND performance is bad (possibly stuck)
- Testing new hyperparameters

## File Structure

```
.
├── dqn_agent.py                   # Core DQN implementation
├── preprocessing.py               # Frame preprocessing and env wrappers
├── train.py                       # Main training script
├── train_fast_decay.py            # Alternative training with faster epsilon decay
├── watch_agent.py                 # Visualize trained agent
├── watch_random_agent.py          # Test game mechanics with random actions
├── test_setup.py                  # Verify installation
├── inspect_checkpoint.py          # Inspect saved models
├── visualize_preprocessing.py     # Visualize preprocessing pipeline
├── requirements.txt               # Python dependencies
├── checkpoints/                   # Saved model checkpoints
└── logs/                          # Training logs and plots
```

## Implementation Details

### Replay Buffer

The replay buffer is a crucial component of DQN that stores past experiences and enables the agent to learn from them multiple times.

**What it is:**
- A fixed-size memory structure (default: 100,000 experiences) implemented as `collections.deque`
- Each experience is a tuple: `(state, action, reward, next_state, done)`
- When full, oldest experiences are automatically removed (FIFO - First In, First Out)
- During training, random batches are sampled from this buffer

**Why it's important:**
1. **Breaks temporal correlation**: Consecutive game frames are highly correlated (paddle barely moves). Training on consecutive experiences causes the network to overfit to recent patterns. Random sampling decorrelates the data.
2. **Data efficiency**: Each experience can be used for training multiple times, extracting more learning from each interaction with the environment.
3. **Stabilizes training**: Smooths out the distribution of experiences, reducing variance in gradient updates.

### Learning Starts and Buffer Filling Strategy

The `--learning-starts` parameter (default: 50,000 steps) determines when the agent begins training the neural network.

**What happens during this phase:**
- Agent plays the game with random actions (epsilon = 1.0)
- Experiences are collected and stored in the replay buffer
- NO gradient updates occur - networks remain at their random initialization
- This "warm-up" phase populates the buffer with diverse experiences

**Why wait before training:**
- **Avoid overfitting to initial experiences**: If training starts immediately, the first few experiences dominate learning. The agent might learn to replicate specific random behaviors rather than generalizable patterns.
- **Ensure diverse samples**: With 50k steps and buffer size of 100k, the buffer is 50% full. This provides sufficient variety for stable gradient estimates.
- **Better gradient estimates**: Larger sample pools reduce variance in randomly sampled batches.

**Tradeoffs of different `learning_starts` values:**

| Strategy | learning_starts | Buffer Fill % | Pros | cons |
|----------|----------------|---------------|------|------|
| **Very Early** | 1k-5k | 1-5% | • Faster time to first learning<br>• Quicker feedback loop | • High risk of overfitting to initial random experiences<br>• Unstable early training<br>• May learn bad habits that are hard to unlearn |
| **Early** | 10k-25k | 10-25% | • Starts learning relatively quickly<br>• Some diversity in buffer | • Moderate overfitting risk<br>• Gradient estimates still somewhat noisy |
| **Balanced** | 50k | 50% | • **Default and recommended**<br>• Good diversity of experiences<br>• Stable gradient estimates<br>• Balanced tradeoff | • Requires patience before seeing learning<br>• More computation spent collecting data |
| **Conservative** | 80k-100k | 80-100% | • Maximum buffer diversity<br>• Very stable initial training<br>• Best gradient estimates | • Long wait before learning starts<br>• Diminishing returns beyond 50%<br>• More wall-clock time |
| **Buffer Overflow** | 150k+ | 100% (old data replaced) | • Ensures only "recent" experiences | • Wasteful: early experiences discarded before being used<br>• No benefit over 100k<br>• Delays learning unnecessarily |

**Practical recommendations:**
- **Default (50k)**: Best for most cases. Balances stability with training time.
- **Reduce to 10k-25k if**: You're experimenting and want faster iteration cycles. Accept that early training may be noisier.
- **Increase to 80k if**: You're doing a final serious training run and want maximum stability. Only worthwhile if training for 1M+ steps total.
- **Never exceed buffer_size (100k)**: You'll just discard experiences before using them.

**Example command:**
```bash
# Quick experimentation (starts learning at 10k steps)
python train.py --learning-starts 10000 --episodes 2000

# Stable long training (starts learning at 80k steps)
python train.py --learning-starts 80000 --episodes 10000
```

**Monitoring tip:** After `learning_starts` steps, you'll see loss values appear in the training output, indicating that training has begun.

### Loss Computation
```python
# Current Q-value from policy network
current_Q = policy_net(state)[action]

# Target Q-value using Bellman equation
target_Q = reward + gamma * max(target_net(next_state))

# MSE loss
loss = (current_Q - target_Q)²
```

### Gradient Clipping
Clips gradients to max norm of 10 to prevent exploding gradients during training.

### Game-Specific Wrappers

#### FireResetWrapper
Atari Breakout requires pressing FIRE to launch the ball at game start and after losing each life. Without this, the game stays frozen. This wrapper:
- Automatically presses FIRE on episode reset
- Tracks lives and auto-presses FIRE whenever a life is lost
- Ensures continuous gameplay without manual intervention

#### NoopFireLeftRightActions
The original Breakout action space has 4 actions: NOOP, FIRE, RIGHT, LEFT. Since FireResetWrapper handles FIRE automatically, the agent doesn't need to learn when to press it. This wrapper:
- Simplifies action space from 4→3 actions (removes FIRE)
- Maps agent actions: 0→NOOP, 1→RIGHT, 2→LEFT
- Reduces learning complexity (agent focuses on paddle movement only)
- Makes training more efficient by removing an unnecessary action

**Why this matters**: Without these wrappers, agents often learn suboptimal policies (e.g., constantly pressing FIRE even when unnecessary) or fail to progress through lives.

### Rendering on macOS
The codebase uses OpenCV (`cv2.imshow`) instead of pygame for rendering because pygame's `render_mode='human'` can freeze on macOS. The scripts use `render_mode='rgb_array'` and manually display frames with OpenCV, providing reliable cross-platform visualization.

## Development Considerations

- All scripts use `argparse` for CLI configuration
- Checkpoints are PyTorch state dicts (`.pt` files)
- Progress plots use matplotlib and save to `logs/`
- The codebase extensively uses docstrings and inline comments for educational purposes
- Frame preprocessing follows the original DQN Nature paper specification
