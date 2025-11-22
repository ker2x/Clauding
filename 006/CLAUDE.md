# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a **Soft Actor-Critic (SAC)** reinforcement learning agent for CarRacing-v3 using **continuous action space** and a **custom 2D physics engine**. The project implements maximum entropy RL with automatic entropy tuning, twin Q-networks, and uses **vector-based state representation** (default 53D base, configurable with frame stacking) for training.

### Key Features

- **Parallel Selection Training**: Primary training method using N independent agents with evolutionary selection
- **Custom 2D Physics**: Clean, interpretable physics simulation with Magic Formula tires
- **Physical Suspension System**: Per-wheel spring-damper with kinematic load transfer (default enabled)
- **Soft Actor-Critic**: State-of-the-art continuous control algorithm
- **Vector Mode**: Configurable state representation (default 53D base: car state + track geometry + lookahead waypoints + steering state)
- **Configurable Observation**: Adjustable waypoint count, spacing, and frame stacking
- **Frame Stacking**: Optional temporal information via multi-frame observations
- **Clean Architecture**: Simplified codebase focused on vector mode for optimal performance

### Suspension System (NEW - 2025-01-13)

The physics engine now includes a **realistic suspension system** using independent per-wheel spring-dampers with kinematic load transfer:

**Key Features:**
- Spring rate: 45,000 N/m (balanced street/track)
- Damping ratio: 1.09 (slightly overdamped, very stable)
- Natural frequency: 8.19 Hz (responsive)
- Load transfer: Visible weight shifts during cornering/braking
- Stability: Fixed feedback loops, bounded travel (±80mm max compression)

**Physics Fixes Applied:**
- ✅ Fixed positive feedback loop (switched to feedforward control)
- ✅ Fixed brake force (reduced from 780 → 50 rad/s² to prevent wheel locking)
- ✅ Fixed double-counting of weight forces
- ✅ Added combined bias caps to prevent suspension travel limit violations

**For Details:** See `SUSPENSION_FIX_SUMMARY.md`

**Testing:**
1. Record telemetry: `python play_human_gui.py --log-telemetry`
2. Text analysis: `python analyze_telemetry.py telemetry_*.csv`
3. Interactive visualization: `python telemetry_viewer.py telemetry_*.csv`

### Steering System (UPDATED - 2025-01-20)

The steering system now features **realistic rate limiting** and **steering state observation**:

**Key Changes:**
- **Steering rate**: Reduced from 3.0 → 1.5 rad/s (more realistic response)
- **Lock-to-lock time**: Increased from ~0.27s → ~0.53s (prevents instant steering changes)
- **Observation space**: Expanded from 71D → 73D
  - Added current steering angle (normalized by MAX_STEER_ANGLE)
  - Added steering rate/velocity (normalized by STEER_RATE)

**Benefits:**
- More realistic vehicle dynamics (steering wheel has inertia)
- Agent can observe current steering position for better control
- Agent can observe steering rate for smoother action planning
- **Breaking change**: Old 71D checkpoints incompatible with new 73D state space

**Configuration:**
- `config/physics_config.py`: STEER_RATE = 1.5 rad/s
- `env/car_racing.py`: Dynamic observation space
- `env/car_dynamics.py`: Steering rate tracking

### Observation Space (UPDATED - 2025-01-20)

The observation space is now **fully configurable** via `config/physics_config.py:ObservationParams`:

**Parameters:**
- **NUM_LOOKAHEAD**: Number of waypoints included in observation (default: 10)
- **WAYPOINT_STRIDE**: Spacing between waypoints (default: 2)
  - STRIDE=1: Consecutive waypoints
  - STRIDE=2: Every 2nd waypoint (2× lookahead horizon, same obs dimension) - DEFAULT
  - STRIDE=3: Every 3rd waypoint (3× lookahead horizon, same obs dimension)
- **FRAME_STACK**: Number of consecutive frames to stack (default: 1 = disabled)

**Base Observation Dimension:** `33 + (NUM_LOOKAHEAD × 2)`
- Fixed components (33D): car state, track info, speed, accelerations, slip angles/ratios, forces, steering
- Variable component: NUM_LOOKAHEAD waypoint (x, y) coordinates

**Examples:**
- `NUM_LOOKAHEAD=20, STRIDE=1`: 73D base, ~70m horizon
- `NUM_LOOKAHEAD=20, STRIDE=2`: 73D base, ~140m horizon (2× farther lookahead)
- `NUM_LOOKAHEAD=10, STRIDE=2`: 53D base, ~70m horizon (smaller network, same horizon) - DEFAULT

**Use Cases:**
- Increase STRIDE to see farther ahead without increasing network size
- Reduce NUM_LOOKAHEAD to decrease observation dimension for faster training
- Balance between lookahead horizon and network complexity

**Testing:** `python test_waypoint_config.py` to verify different configurations

**Important:** Changing observation parameters requires retraining (incompatible checkpoints)

### Frame Stacking (NEW - 2025-01-21)

Frame stacking concatenates observations from multiple consecutive timesteps, providing the agent with **temporal information** through finite differences.

**Configuration:** `config/physics_config.py:ObservationParams.FRAME_STACK`

**How it works:**
- `FRAME_STACK=1`: Disabled (default) - single frame observations
- `FRAME_STACK=2`: Stack 2 consecutive frames
- `FRAME_STACK=4`: Stack 4 consecutive frames (recommended for temporal dynamics)

**Final Observation Dimension:** `(33 + NUM_LOOKAHEAD × 2) × FRAME_STACK`
- Example: `NUM_LOOKAHEAD=10, STRIDE=2, FRAME_STACK=4` → 53 × 4 = 212D

**Benefits:**
- **Temporal derivatives**: Agent can compute rates of change via finite differences
- **Dynamics visibility**: See how slip angles, normal forces, and other quantities change over time
- **Predictive behavior**: Better anticipation of loss of traction, weight transfer, trajectory changes
- **No velocity ambiguity**: Can distinguish stationary vs. moving through same position

**What Frame Stacking Provides:**
With single frame, agent knows:
- Current slip angles (but not if they're increasing/decreasing)
- Current normal forces (but not rate of load transfer)
- Current distance to center (but not lateral velocity relative to track)

With 4-frame stacking, agent can compute:
- Slip angle rates → anticipate tire saturation before grip loss
- Load transfer rates → predict weight shift during maneuvers
- Lateral drift rate → correct trajectory deviations early
- Trends in all quantities → more predictive, smoother control

**Implementation Details:**
- Frames stored individually in replay buffer (memory efficient)
- Stacking happens during sampling (computational cost at training time)
- Episode boundaries respected (won't stack across episodes)
- Initial frames padded by repeating earliest available frame

**Performance Impact:**
- ✅ Memory: Minimal (stores single frames)
- ⚠️ Network size: Increases proportionally to FRAME_STACK
- ⚠️ Training time: Slightly slower due to larger networks
- ✅ Sample efficiency: Often improves convergence

**Important:** Changing FRAME_STACK requires retraining (incompatible checkpoint dimensions)

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

# Play as human with telemetry GUI
python play_human_gui.py
```

### Telemetry Analysis
```bash
# Record telemetry while driving
python play_human_gui.py --log-telemetry
python play_human_gui.py --log-telemetry --log-file my_session.csv --log-interval 5

# Text-based analysis (statistics and summaries)
python analyze_telemetry.py telemetry_20250113_123456.csv
python analyze_telemetry.py --all telemetry_*.csv          # All analyses
python analyze_telemetry.py --wheels --suspension file.csv # Specific sections

# Interactive professional telemetry viewer
python telemetry_viewer.py telemetry_20250113_123456.csv

# Telemetry viewer controls:
#   - Left/Right arrows : Navigate through time
#   - Space            : Play/Pause animation
#   - Click on plot    : Jump to that time
#   - Mouse wheel      : Zoom
#   - R                : Reset zoom
#   - Q                : Quit
```

**Telemetry Viewer Features:**
- Track map with speed-colored trajectory
- Real-time car position and orientation marker
- Synchronized time-series plots:
  - Speed (km/h)
  - Driver inputs (steering, acceleration/brake)
  - Wheel slip angles (all 4 wheels)
  - Wheel slip ratios (all 4 wheels)
  - Normal forces / tire loads (all 4 wheels)
- Interactive cursor with click-to-jump navigation
- Animation playback mode
- Professional motorsport color scheme

### Network Health Analysis
```bash
# Install weightwatcher (first time only)
pip install weightwatcher

# Analyze a trained checkpoint
python analyze_network_health.py --checkpoint checkpoints_selection_parallel/best_model.pt

# Generate detailed reports and visualizations
python analyze_network_health.py \
    --checkpoint checkpoints_selection_parallel/best_model.pt \
    --output-dir health_reports/

# Compare multiple checkpoints over training
python analyze_network_health.py \
    --checkpoint \
        checkpoints_selection_parallel/generation_100.pt \
        checkpoints_selection_parallel/generation_200.pt \
        checkpoints_selection_parallel/best_model.pt

# Analyze specific networks only
python analyze_network_health.py \
    --checkpoint checkpoints_selection_parallel/best_model.pt \
    --networks actor critic_1

# Quick test
bash test_network_health.sh
```

**Key Metrics Explained:**

- **Alpha (α)** - Power law exponent indicating generalization quality:
  - `α < 2.0`: Undertrained or random weights
  - `α ∈ [2.0, 4.0)`: **IDEAL** - Well-trained with good generalization
  - `α ∈ [4.0, 6.0)`: Borderline - May be overtraining
  - `α > 6.0`: Likely overfit - Poor generalization

- **Log Spectral Norm**: Layer conditioning (lower is better)
- **Stable Rank**: Effective dimensionality of weight matrices

**Output Files (with --output-dir):**
- `network_health_report.txt`: Detailed text report
- `{network}_details.csv`: Per-layer metrics
- `{network}_health_metrics.png`: Visualization plots

**Use Cases:**
- Diagnose why an agent isn't learning (α < 2.0 = undertrained)
- Detect overtraining (α > 4.0 = overfit)
- Compare checkpoints to find best generalization
- Monitor network health during training

**For detailed guide:** See `NETWORK_HEALTH_GUIDE.md`

## Architecture Overview

### State Representation

**Vector Mode (default 53D base, configurable):**
- Car state (11D): position, velocity, angle, wheel contacts, progress
- Track segment (5D): distance to center, angle, curvature, segment info
- Lookahead waypoints (NUM_LOOKAHEAD×2): future waypoints in car coordinates
  - Default: 10 waypoints × 2 = 20D
  - Configurable via `config/physics_config.py:ObservationParams`
- Speed (1D): velocity magnitude
- Accelerations (2D): longitudinal and lateral (body frame)
- Slip angles (4D): tire slip angles for all wheels
- Slip ratios (4D): tire slip ratios for all wheels
- Vertical forces (4D): normal forces on all wheels
- Steering state (2D): current steering angle and rate
- Fast training, no rendering required

**Base Observation Dimension:** 33 + (NUM_LOOKAHEAD × 2)
**Final Dimension (with frame stacking):** base_dim × FRAME_STACK

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
- Actor: [(33 + NUM_LOOKAHEAD×2) × FRAME_STACK]D → FC(256)×3 → 2D action (mean, log_std)
- Critic: [(33 + NUM_LOOKAHEAD×2) × FRAME_STACK]D + 2D action → FC(512)×4 → Q-value
- Default: 53D input with NUM_LOOKAHEAD=10, FRAME_STACK=1

Note: Network input dimension adjusts automatically based on observation and frame stacking configuration

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
├── play_human_gui.py       # Human playable with telemetry display
├── test_setup.py           # Verify installation
│
├── analyze_telemetry.py         # Text-based telemetry analysis
├── telemetry_viewer.py          # Interactive telemetry visualization
├── magic_formula_visualizer.py  # Tire model parameter tuning
├── analyze_network_health.py    # Network health analysis (WeightWatcher)
├── test_network_health.sh       # Quick test for network health tool
│
├── checkpoints_selection_parallel/  # Saved models (parallel selection)
├── logs_selection_parallel/         # Training logs
│
└── [Documentation files]
    ├── NETWORK_HEALTH_GUIDE.md  # Comprehensive guide for network analysis
```

## Key Hyperparameters

All default values are defined in `constants.py` for centralized configuration.

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--num-agents` | 4 | Number of parallel agents (selection training) |
| `--selection-frequency` | 50 | Episodes between selection tournaments |
| `--eval-episodes` | 10 | Episodes per tournament evaluation |
| `--elite-count` | 2 | Top N agents preserved (1=winner-takes-all) |
| `--learning-starts` | 5000 | Steps before training begins |
| `--lr-actor` | 1e-4 | Actor learning rate |
| `--lr-critic` | 1e-4 | Critic learning rate |
| `--lr-alpha` | 1e-3 | Alpha (entropy) learning rate |
| `--gamma` | 0.99 | Discount factor |
| `--tau` | 0.005 | Target network update rate |
| `--buffer-size` | 200000 | Replay buffer capacity |
| `--batch-size` | 512 | Training batch size |

**Evaluation Parameters (from constants.py):**
- `DEFAULT_INTERMEDIATE_EVAL_EPISODES = 5` - Periodic evaluations during training
- `DEFAULT_FINAL_EVAL_EPISODES = 10` - Final evaluation at end of training
- `DEFAULT_MAX_STEPS_PER_EPISODE = 2500` - Safety timeout for evaluation episodes

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
- `state_dim`: Observation dimension (base_dim × frame_stack)
- `action_dim`: 2
- Entropy tuning parameters (if enabled)

**Note:** Checkpoints are specific to observation configuration. Changing NUM_LOOKAHEAD, WAYPOINT_STRIDE, or FRAME_STACK requires retraining.

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
