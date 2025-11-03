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

**Status**: ✅ **COMPLETE AND OPTIMIZED**

All components are implemented and working:
- DDQN agent with experience replay
- Action discretization (continuous → 9 discrete actions)
- **Triple state modes**: Snapshot (RECOMMENDED), Vector (limited), and Visual (for watching)
- Frame preprocessing pipeline (grayscale, native 96×96 resolution, normalize, stack)
- **No reward shaping** - agent receives raw environment rewards (including -100 off-track penalty)
- Stationary car early termination (3x training speedup)
- Training loop with checkpointing and evaluation
- **Comprehensive file-based logging (CSV + human-readable logs)**
- Unified environment (same config for training and evaluation)
- Visualization tools (watch agent, watch random agent)
- Apple Silicon GPU (MPS) acceleration working
- **Snapshot state mode (3-5x faster training with track geometry and lookahead)**
- **Device selection support (auto, CPU, CUDA, MPS)**

**Latest Verification**: 2025-11-04
- Snapshot mode: 150-200 steps/sec (3-5x faster than visual, with proper track info)
- Vector mode: 313 steps/sec (fastest but too limited for learning)
- Visual mode: 57 steps/sec (full rendering, for watching only)
- Snapshot mode provides 36D state with track geometry and 10 lookahead waypoints
- Agent learns proper racing behavior in snapshot mode
- File-based logging: CSV metrics + training.log working correctly
- See STATE_MODES.md for comprehensive comparison

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

## Recent Updates

### Snapshot State Mode (2025-11-04)
**Major Performance Improvement**: Snapshot mode provides fast training with track geometry and lookahead!

**Problem**: Visual mode was slow (pygame rendering every step). Vector mode was fast but lacked track information for proper learning.

**Solution**: Snapshot mode - best of both worlds!
- **Snapshot Mode** (RECOMMENDED): 36-dimensional state with track geometry and lookahead
  - Car state: position, velocity, angle, wheel contacts, progress (11D)
  - Track segment info: distance to center, angle difference, curvature, segment progress (5D)
  - Lookahead waypoints: 10 upcoming waypoints in car-relative coordinates (20D)
  - No rendering required (3-5x faster than visual)
  - MLP network (256→256→128 vs CNN)
  - Agent learns proper racing behavior
  - **Default for training**

- **Vector Mode** (Not Recommended): 11-dimensional basic state (no track info)
  - Fastest (6x vs visual) but agent can't learn to drive well
  - Missing critical track geometry and lookahead information
  - Only useful for basic functionality testing

- **Visual Mode** (For Watching): 96×96 RGB images with full rendering
  - Full visualization for watching agents
  - CNN-based network
  - Used automatically by watch scripts
  - Too slow for training

**Performance Results**:
- Snapshot mode: 150-200 steps/second (3-5x faster than visual, proper learning)
- Vector mode: 313 steps/second (fastest but can't learn properly)
- Visual mode: 57 steps/second (baseline)
- 1M steps: 1.5-2 hours (snapshot) vs 0.9 hours (vector, but poor learning) vs 4.9 hours (visual)
- Memory: Low (36 values for snapshot vs 11 for vector vs 36,864 for visual)

**Implementation**:
- `env/car_racing.py`: Added `_create_snapshot_state()` with track geometry calculations
- `ddqn_agent.py`: Added `SnapshotDQN` network class (larger MLP for 36D state)
- `preprocessing.py`: Conditional preprocessing based on mode
- `train.py`: Added `--state-mode` argument (default: snapshot)
- `watch_*.py`: Hardcoded to visual mode for visualization
- `benchmark_state_modes.py`: Comprehensive comparison tool
- `STATE_MODES.md`: Complete documentation of all three modes

**Usage**:
```bash
# Training (RECOMMENDED, uses snapshot mode by default)
python train.py --episodes 2000

# Explicitly specify snapshot mode
python train.py --episodes 2000 --state-mode snapshot

# Watching (automatically uses visual mode)
python watch_agent.py --checkpoint checkpoints/final_model.pt

# Benchmark comparison
python benchmark_state_modes.py --episodes 50
```

**Key Advantage**: Snapshot mode is 3-5x faster than visual mode AND provides the track geometry/lookahead information needed for the agent to learn proper racing behavior. Vector mode is faster but too limited for actual learning.

**See**: `STATE_MODES.md` for comprehensive comparison and technical details.

### Native 96×96 Resolution (2025-11-03)
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

### Comprehensive File-Based Logging (2025-11-03)
**Added**: Complete logging infrastructure for monitoring training progress

**Problem**: Training could only be monitored through console output and plots (PNG images). This made it difficult to:
- Monitor training progress programmatically
- Parse metrics for custom analysis
- Review training history after completion
- Debug issues during long training runs

**Solution**: Comprehensive file-based logging with multiple formats

**Files Created**:
1. **`logs/training_metrics.csv`** - Per-episode training metrics
   - Columns: episode, total_steps, episode_steps, reward, avg_loss, epsilon, buffer_size, elapsed_time_sec, avg_reward_100, timestamp
   - Written every episode for complete history
   - Easy to parse with pandas/scripts

2. **`logs/evaluation_metrics.csv`** - Evaluation results
   - Columns: episode, total_steps, eval_mean_reward, eval_std_reward, eval_rewards, is_best, elapsed_time_sec, timestamp
   - Tracks evaluation performance over time
   - Records all individual episode rewards

3. **`logs/training.log`** - Human-readable timestamped log
   - Training session start with full configuration
   - Learning start milestone (when reaching `learning_starts`)
   - Episode progress (every 10 episodes)
   - All evaluation episodes with individual rewards
   - Best model saves and checkpoint saves
   - Training session summary

4. **`logs/system_info.txt`** - Complete configuration snapshot
   - Device, state mode, hyperparameters
   - Environment settings
   - Training parameters
   - For reproducibility

**Implementation**:
- `train.py`: Added `setup_logging()` function
- Modified `evaluate_agent()` to accept optional `log_handle` parameter
- CSV writing after every episode
- Log file writing with timestamps
- All logs use line buffering for real-time updates

**Usage**:
```bash
# Monitor training in real-time
tail -f logs/training.log

# Read CSV metrics
cat logs/training_metrics.csv
cat logs/evaluation_metrics.csv

# Check configuration
cat logs/system_info.txt

# Parse with Python
import pandas as pd
df = pd.read_csv('logs/training_metrics.csv')
print(df[['episode', 'reward', 'epsilon']].tail())
```

**Benefits**:
- Real-time monitoring without plots
- Complete training history
- Easy programmatic access to metrics
- Debugging support with timestamps
- Reproducibility with system info

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
- Additional loophole test in `test_stationary_loophole.py` (see below)

**Loophole Fix (2025-11-03)**:
- **Problem Discovered**: Agents learned to exploit the stationary termination by wiggling the steering wheel while remaining stationary. Since actions changed, the old logic (only checking new tile visits) didn't catch this. The agent could wiggle forever, avoiding both stationary termination and off-track penalties.
- **Fix Applied**: Now checks **both** new tile visits **and** car velocity. Car must either:
  1. Visit a new tile (step_reward > 0), OR
  2. Be moving with meaningful velocity (speed > 0.5 m/s)
- **Implementation**: Updated `env/car_racing.py` lines 627-651 to calculate speed from `car.hull.linearVelocity` and check `is_making_progress = (step_reward > 0) or (speed > 0.5)`
- **Result**: Agents can no longer exploit the wiggle loophole. Episodes with wiggling terminate correctly after ~100 frames.
- **Watch Scripts Fixed**: Both `watch_agent.py` and `watch_random_agent.py` now have `terminate_stationary=True` (was previously disabled for "full episodes", which allowed infinite wiggling)
- **Verification**:
  - `test_stationary_loophole.py`: Confirms wiggling terminates at ~101 steps, moving forward doesn't trigger false positives
  - `test_agent_quick.py`: Trained agent's wiggle strategy now gets caught and terminated properly
  - Manual testing: `watch_agent.py` now terminates wiggling agents instead of running forever

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
# Quick test training (25 episodes, ~2-3 minutes with snapshot mode)
python train.py --episodes 25 --learning-starts 500

# Short training (200 episodes, ~15-20 minutes with snapshot mode)
python train.py --episodes 200 --learning-starts 2000

# Full training (2000 episodes, ~1.5-2 hours with snapshot mode)
python train.py --episodes 2000 --learning-starts 10000

# Use vector mode (faster but agent can't learn well, not recommended)
python train.py --episodes 200 --state-mode vector

# Use visual mode for training (slowest, for debugging visuals)
python train.py --episodes 200 --state-mode visual

# Resume from checkpoint
python train.py --resume checkpoints/final_model.pt --episodes 1000

# Resume with reset epsilon (more exploration)
python train.py --resume checkpoints/final_model.pt --reset-epsilon --episodes 1000

# Force CPU mode (useful if CPU is faster than MPS)
python train.py --episodes 200 --device cpu
```

**Note**: Training uses snapshot mode by default (3-5x faster than visual, with track geometry). Watch scripts automatically use visual mode.

### Monitoring Training Progress
```bash
# Watch training log in real-time (during training)
tail -f logs/training.log

# View all training metrics
cat logs/training_metrics.csv

# View evaluation results
cat logs/evaluation_metrics.csv

# Check system configuration
cat logs/system_info.txt

# Parse metrics with Python
python -c "import pandas as pd; df = pd.read_csv('logs/training_metrics.csv'); print(df.tail(20))"

# Get latest training stats
python -c "import pandas as pd; df = pd.read_csv('logs/training_metrics.csv'); print(f\"Latest episode: {df.iloc[-1]['episode']}, Steps: {df.iloc[-1]['total_steps']}, Reward: {df.iloc[-1]['reward']:.2f}, Epsilon: {df.iloc[-1]['epsilon']:.4f}\")"
```

### Evaluation and Benchmarking
```bash
# Watch random agent (baseline, should get ~-50 to -100 reward)
python watch_random_agent.py --episodes 3

# Watch trained agent (should get +500 to +900 if well-trained)
python watch_agent.py --checkpoint checkpoints/final_model.pt --episodes 5

# Inspect checkpoint (check steps, epsilon, get recommendations)
python inspect_checkpoint.py checkpoints/final_model.pt

# Benchmark snapshot vs vector vs visual modes (~5 minutes)
python benchmark_state_modes.py --episodes 50
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

# 5. Monitor training progress (while training is running)
tail -f logs/training.log

# 6. Check training metrics
cat logs/training_metrics.csv | tail -20

# 7. Inspect checkpoint
python inspect_checkpoint.py checkpoints/final_model.pt

# 8. Watch trained agent
python watch_agent.py --checkpoint checkpoints/final_model.pt
```

### File Locations
- **Checkpoints**: `checkpoints/` (final_model.pt, best_model.pt, checkpoint_ep*.pt)
- **Training logs**: `logs/` (training_metrics.csv, evaluation_metrics.csv, training.log, system_info.txt)
- **Training plots**: `logs/training_progress.png`, `logs/benchmark_comparison.png`
- **Code files**: All Python files in project root

### Expected Performance
- **Random agent**: -50 to -100 reward
- **Well-trained agent**: +500 to +900 reward
- **Training time**: 2000 episodes ≈ 1.5-2 hours with snapshot mode on Apple Silicon GPU
- **Minimum training**: 1M+ steps (epsilon < 0.5) for meaningful performance
- **State modes**: Snapshot (RECOMMENDED, 150-200 steps/sec), Vector (313 steps/sec but limited), Visual (57 steps/sec)

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
