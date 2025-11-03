# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a Deep Q-Network (DQN) / Double DQN (DDQN) implementation for training reinforcement learning agents to play CarRacing-v3 from Gymnasium. Unlike the sibling project (`001/` - Breakout with discrete actions), this project tackles a **continuous action space** environment.

### CarRacing-v3 Environment
- **Task**: Control a racing car to complete laps on a randomly generated track
- **Observation**: RGB images (96×96×3 by default) showing top-down view of track
- **Action Space**: Continuous 3D vector `[steering, gas, brake]`
  - `steering`: [-1.0, 1.0] (left to right)
  - `gas`: [0.0, 1.0] (throttle)
  - `brake`: [0.0, 1.0] (brake pedal)
- **Reward**: +1000/N for visiting each track tile (N = total tiles), -0.1 for each frame, negative for going off-track
- **Episode End**: All tiles visited OR 1000 frames without progress

### Continuous Action Space Challenge
DQN was originally designed for **discrete** actions (e.g., button presses). CarRacing requires **continuous** control. Common approaches:
1. **Action Discretization**: Quantize continuous actions into discrete bins (e.g., 5 steering levels × 3 gas levels = 15 actions)
2. **Hybrid Approaches**: Combine DQN with actor-critic for continuous control
3. **Native Continuous Methods**: DDPG, TD3, SAC (alternatives to DQN)

This project uses **action discretization** (9 discrete actions: 3 steering × 3 gas/brake).

## Implementation Status

**Status**: ✅ **COMPLETE AND TESTED**

All components are implemented and working:
- DDQN agent with experience replay
- Action discretization (continuous → 9 discrete actions)
- Frame preprocessing pipeline (grayscale, native 96×96 resolution, normalize, stack)
- **No reward shaping** - agent receives raw environment rewards (including -100 off-track penalty)
- Stationary car early termination (3x training speedup)
- Training loop with checkpointing and evaluation
- Unified environment (same config for training and evaluation)
- Visualization tools (watch agent, watch random agent)
- Apple Silicon GPU (MPS) acceleration working

**Verified on**: 2025-11-03
- Test training: 25 episodes, 25,000 steps completed successfully
- GPU (MPS) utilized properly
- Steps counting correctly
- Epsilon decay working (1.0 → 0.9752)
- Loss computation working (0.1323, 0.1085)
- Target network updates at 10k, 20k steps
- Native 96×96 resolution working (no resize operation needed)
- DQN network architecture updated for 4096 conv output (vs 3136 for 84×84)

## Critical Implementation Note

**IMPORTANT BUG FIX APPLIED**: The initial implementation had a circular dependency bug where `steps_done` was only incremented inside `train_step()`, but `train_step()` was only called when `steps_done >= learning_starts`. This caused training to never start.

**Fix Applied** (in `train.py` lines 253-263):
```python
# Increment step counter BEFORE checking learning_starts
agent.steps_done += 1

# Train agent (only after learning_starts steps)
if agent.steps_done >= args.learning_starts:
    loss = agent.train_step()
    if loss is not None:
        episode_loss.append(loss)

# Update epsilon based on steps
agent.update_epsilon()
```

This fix ensures:
1. `steps_done` increments every environment step
2. Training starts after `learning_starts` steps as intended
3. Epsilon decays properly based on steps
4. GPU is utilized (not stuck in CPU loop)

If you see `Steps: 0` in training output and no loss values, this fix may not be applied.

## Recent Updates (2025-11-03)

### Native 96×96 Resolution
**Changed from**: 84×84 resized frames
**Changed to**: 96×96 native CarRacing resolution

**Rationale**:
- Eliminates resize operation (faster preprocessing)
- Preserves full image quality (no information loss)
- Only 30% more pixels (9,216 vs 7,056 per frame)
- Minimal computational overhead with significant quality gain

**Implementation Changes**:
- `preprocessing.py`: Native 96×96 resolution, no `frame_size` parameter needed
- `ddqn_agent.py`: Updated DQN network `conv_output_size = 64 * 8 * 8 = 4096` (was 3136)
- `train.py`: Removed invalid `frame_size` parameter from `make_carracing_env()` call

**Bug Fixes**:
1. The DQN network had hardcoded conv output size for 84×84 input. This caused shape mismatch errors with 96×96 input. Fixed by updating the conv output calculation.
2. `train.py` was passing invalid `frame_size=(96, 96)` parameter to `make_carracing_env()`. The function doesn't accept this parameter since it uses native 96×96 resolution. Fixed by removing the parameter.

### Stationary Car Termination as Core Feature
**Changed from**: Preprocessing wrapper (`StationaryCarTerminator`)
**Changed to**: Built-in CarRacing environment feature (2025-11-03)

**Rationale**:
- Stationary car termination is a fundamental environment behavior, not preprocessing
- Cleaner architecture: core behavior belongs in the environment itself
- Easier to maintain and test
- More discoverable for users of the environment

**Implementation Changes**:
- **[2025-11-03 LATEST]** Moved stationary car termination logic from `StationaryCarTerminator` wrapper to `CarRacing` class
- Added `terminate_stationary`, `stationary_patience`, `stationary_min_steps` parameters to `CarRacing.__init__()`
- Removed `StationaryCarTerminator` wrapper from `preprocessing.py`
- Updated `make_carracing_env()` to pass parameters to `CarRacing` constructor
- Added comprehensive test suite: `test_stationary_termination.py`

**Testing**:
- All tests pass (stationary termination, normal movement, disable feature)
- Training script verified to work correctly with new implementation
- Preprocessing test updated and passes

### Unified Environment Configuration
**Changed from**: Separate training and evaluation environments
**Changed to**: Single unified environment for both training and evaluation

**Rationale**:
- Early termination (`terminate_stationary`) is part of environment design, not a training shortcut
- Evaluation should match the actual environment the agent will operate in
- Faster evaluation (30-60 seconds vs 2-5 minutes for final evaluation)
- Simpler, more maintainable code

**Implementation Changes**:
- `train.py`: Removed separate `eval_env` creation
- `train.py`: Updated all `evaluate_agent()` calls to use unified `env`
- `train.py`: Added progress output to `evaluate_agent()` function (shows each episode)

**User Experience**:
- Evaluation now shows real-time progress: `Eval episode 1/10: reward = 123.45`
- No more silent pauses during evaluation
- Consistent behavior across training, evaluation, and watching

## Reward Structure and Environment Behavior

**VERIFIED ON**: 2025-11-03 via `test_straight_full_speed.py`

### Reward System
CarRacing-v3 has the following reward structure:
- **Time penalty**: -0.1 per frame (encourages speed)
- **New tile visited**: +3 to +7 per tile (depends on tile value)
- **Off-track penalty**: **-100.00** (catastrophic, triggers immediate termination)

### Off-Track Detection
- Going off-track (onto grass) triggers an immediate **-100** reward and episode termination
- Test: Going STRAIGHT+GAS hits first curve at ~step 250 → -100 penalty → terminated
- **Design Decision**: The agent receives the full -100 penalty (no reward clipping)
  - This harsh penalty teaches the agent that going off-track is catastrophic
  - Reward shaping would weaken this signal and make the agent think off-track is "okay"
  - The natural reward structure is sufficient for learning

### Stationary Car Early Termination

**IMPLEMENTED**: Core environment feature (2025-11-03)

**Problem**: Early in training, agents often learn to brake/coast and sit still to avoid off-track penalties. This wastes ~80% of episode time (900+ frames) on a stationary car.

**Solution**: Stationary car termination is now a **core feature** of the CarRacing environment itself:
- Monitors positive rewards (new tiles visited)
- If no progress for 100 frames (configurable), episode terminates
- Minimum 50 steps before early termination can trigger
- Implemented directly in `env/car_racing.py`, not as a preprocessing wrapper

**Performance Impact**:
- Episodes with early termination: ~300 steps/episode (vs 1000 before) → **3x speedup**
- Episodes complete faster, allowing more training iterations per hour

**Implementation**:
- Built into the `CarRacing` class in `env/car_racing.py`
- Parameters: `terminate_stationary=True`, `stationary_patience=100`, `stationary_min_steps=50`
- **UNIFIED ENVIRONMENT**: Both training and evaluation use the same environment configuration
- Early termination is part of the environment design, not a training shortcut
- All episodes (training, evaluation, watching) use consistent environment behavior
- Configurable via `CarRacing(terminate_stationary=True, stationary_patience=100, stationary_min_steps=50)`
- Can be disabled by setting `terminate_stationary=False`
- Progress output during evaluation shows each episode's reward in real-time

**Testing**:
- Comprehensive test suite in `test_stationary_termination.py`
- Verifies termination when stationary, no termination during movement, and ability to disable

## Environment Setup

**IMPORTANT**: This subproject (`002/`) uses a shared virtual environment located in the parent directory.

### Virtual Environment
- **Location**: `../.venv/` (parent directory, shared with `001/`)
- **Activation**: Always activate before running any Python commands
  ```bash
  source ../.venv/bin/activate
  ```
- All Python package installations should be done within this activated virtual environment
- The virtual environment should contain gymnasium, torch, numpy, opencv-python, matplotlib

### First-Time Setup
If packages are not installed in the parent virtual environment:
```bash
# Activate the parent virtual environment
source ../.venv/bin/activate

# Install dependencies (will be created as requirements.txt)
pip install gymnasium[box2d] torch torchvision numpy opencv-python matplotlib
```

**Note**: CarRacing-v3 requires the `box2d` extra for Gymnasium. This installs Box2D physics engine dependencies.

## Expected Architecture

Based on the sibling project (`001/`), this codebase will likely include:

### Core Components
- **DQN/DDQN Agent**: Neural network(s) for Q-value estimation
  - **Double DQN**: Uses two networks to reduce overestimation bias
    - Action selection: `a* = argmax Q_policy(s', a')`
    - Value estimation: `Q_target(s', a*)`
    - Reduces optimistic bias from standard DQN's `max Q_target(s', a')`
  - **Network Architecture**: Optimized for 96×96 native input (4096 conv output)
- **Replay Buffer**: Experience storage for training stability
- **Action Discretization**: Maps continuous action space to discrete actions
- **Preprocessing**: Frame processing (grayscale conversion, native 96×96 resolution, normalize, frame stacking)

### Training Components
- **Training Script**: Main training loop with checkpointing
- **Evaluation Script**: Watch trained agent play
- **Visualization Tools**: Inspect preprocessing, training curves
- **Test Scripts**: Verify environment setup

## Common Commands

**Note**: All commands assume the parent virtual environment is activated: `source ../.venv/bin/activate`

### Setup and Testing
```bash
# Activate virtual environment (if not already active)
source ../.venv/bin/activate

# Install dependencies (including Box2D for CarRacing)
pip install -r requirements.txt
# Or: pip install 'gymnasium[box2d]' torch torchvision numpy opencv-python matplotlib

# Verify installation (IMPORTANT: run this first!)
python test_setup.py
```

### Training
```bash
# Quick test training (25 episodes, ~10 minutes)
python train.py --episodes 25 --learning-starts 500

# Short training (200 episodes, ~1 hour)
python train.py --episodes 200 --learning-starts 2000

# Full training (2000 episodes, several hours)
python train.py --episodes 2000 --learning-starts 10000

# Resume from checkpoint
python train.py --resume checkpoints/final_model.pt --episodes 1000

# Resume with reset epsilon (more exploration)
python train.py --resume checkpoints/final_model.pt --reset-epsilon --episodes 1000
```

### Evaluation
```bash
# Watch random agent (baseline, should get ~-50 to -100 reward)
python watch_random_agent.py --episodes 3

# Watch trained agent (should get +500 to +900 if well-trained)
python watch_agent.py --checkpoint checkpoints/final_model.pt --episodes 5

# Inspect checkpoint (check steps, epsilon, get recommendations)
python inspect_checkpoint.py checkpoints/final_model.pt
```

## Key Differences from Discrete Action DQN (001/)

### Action Space Handling
- **Discrete (Breakout)**: Direct output layer with N neurons (one per action)
- **Continuous (CarRacing)**: Must discretize or use hybrid approach
  - Example discretization: 5×3 grid = 15 actions
  - Network outputs 15 Q-values, each representing a (steering, gas/brake) combination

### Reward Structure
- **Breakout**: Simple sparse rewards (brick destroyed = +1, ball lost = 0)
- **CarRacing**: Dense rewards (continuous progress tracking, time penalties)
  - Requires different hyperparameter tuning (gamma, reward scaling)
  - Uses raw environment rewards without shaping

### State Preprocessing
- **Breakout**: Grayscale conversion, resize to 84×84
- **CarRacing**: Grayscale conversion, native 96×96 resolution (no resize)
  - Uses CarRacing's native resolution for better quality and faster preprocessing
  - Track boundaries are visually distinct even in grayscale
  - DQN network architecture adjusted for 96×96 input (4096 features vs 3136)

### Exploration Strategy
- **Discrete**: Epsilon-greedy straightforward (random discrete action)
- **Continuous**: Discretized epsilon-greedy OR noise injection
  - Must ensure discretized actions cover action space effectively
  - Exploration needs to discover "accelerate forward" strategy

## Critical Training Insights

### Epsilon-Greedy with Discretized Actions
- Epsilon still decays based on STEPS, not episodes
- Random action selection picks uniformly from discretized action set
- Ensure action discretization includes "forward acceleration" (critical for progress)

### CarRacing-Specific Challenges
1. **Sparse Exploration**: Agent must learn to stay on track before learning to race
   - The harsh -100 off-track penalty provides strong learning signal
   - May need longer training than Breakout
2. **Action Coordination**: Steering + gas must be coordinated
   - Poor discretization can lead to jerky, unstable control
   - May need finer discretization for steering than gas/brake
3. **Long Episodes**: Episodes can be 1000+ frames (vs ~200-300 for Breakout)
   - Slower training iteration
   - Need efficient preprocessing

### Device Support
- Auto-detects CUDA (NVIDIA), MPS (Apple Silicon), or CPU
- CarRacing uses native 96×96 resolution (30% more pixels than 84×84)
- GPU acceleration highly recommended for faster training
- MPS (Apple Silicon) verified working with 96×96 input

## Implementation Strategy

### Phase 1: Environment Setup
1. Create environment wrappers for CarRacing-v3
2. Define action discretization scheme
3. Implement preprocessing pipeline (grayscale, native 96×96 resolution, frame stacking)
4. Test with random agent to verify setup

### Phase 2: Core DQN/DDQN
1. Implement DQN agent with discretized action space
2. Adapt network architecture for CarRacing input size
3. Implement replay buffer and training loop
4. Add checkpointing and logging

### Phase 3: Training & Tuning
1. Initial training runs to establish baseline
2. Hyperparameter tuning (learning rate, epsilon decay, etc.)
3. Experiment with DDQN vs DQN
4. Monitor learning with raw environment rewards

### Phase 4: Advanced Improvements
1. Optimize action discretization based on results
2. Consider Dueling DQN or Prioritized Experience Replay
3. Advanced preprocessing experiments (already using native resolution for optimal quality)

## Development Considerations

- Follow similar structure to `001/` for consistency
- Use argparse for CLI configuration
- Checkpoints should store policy net, target net, optimizer state, steps_done
- Extensive documentation for educational purposes
- Use OpenCV for rendering (pygame can freeze on macOS)
- Frame stacking (4 frames) to capture velocity/motion

## Resources

- **Gymnasium CarRacing-v3 Docs**: https://gymnasium.farama.org/environments/box2d/car_racing/
- **DDQN Paper**: "Deep Reinforcement Learning with Double Q-learning" (van Hasselt et al., 2015)
- **DQN Nature Paper**: "Human-level control through deep reinforcement learning" (Mnih et al., 2015)

## Quick Reference

### Most Important Commands
```bash
# 1. Always activate venv first
source ../.venv/bin/activate

# 2. Verify setup (do this first!)
python test_setup.py

# 3. Watch random baseline
python watch_random_agent.py --episodes 3

# 4. Start training
python train.py --episodes 2000 --learning-starts 10000

# 5. Check training progress
python inspect_checkpoint.py checkpoints/final_model.pt

# 6. Watch trained agent
python watch_agent.py --checkpoint checkpoints/final_model.pt
```

### File Locations
- **Checkpoints**: `checkpoints/` (final_model.pt, best_model.pt, checkpoint_ep*.pt)
- **Training plots**: `logs/training_progress.png`
- **Code files**: All Python files in project root

### Expected Performance
- **Random agent**: -50 to -100 reward
- **Well-trained agent**: +500 to +900 reward
- **Training time**: 2000 episodes = several hours on Apple Silicon GPU
- **Minimum training**: 1M+ steps (epsilon < 0.5) for meaningful performance

### Action Discretization
9 discrete actions = 3 steering × 3 gas/brake:
- Steering: LEFT (-0.8), STRAIGHT (0.0), RIGHT (+0.8)
- Gas/Brake: BRAKE (0.8), COAST (0.0), GAS (0.8)
- Example: Action 7 = STRAIGHT + GAS

### Key Hyperparameters (defaults)
- `learning_rate`: 0.00025
- `gamma`: 0.99 (discount factor)
- `epsilon_decay`: 1M steps (1.0 → 0.01)
- `buffer_size`: 100k experiences
- `batch_size`: 32
- `target_update_freq`: 10k steps
- `learning_starts`: 10k steps (when training begins)
