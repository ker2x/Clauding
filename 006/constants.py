"""
Shared constants for SAC training on CarRacing-v3.

This module defines all shared constants used across the training codebase,
providing a single source of truth for hyperparameters and configuration values.
"""

from __future__ import annotations

# ===========================
# State and Action Space
# ===========================

STATE_DIM: int = 71  # Vector mode state dimension (car state + track geometry + lookahead)
ACTION_DIM: int = 2  # Continuous action space: [steering, gas & brake]

# ===========================
# Training Hyperparameters
# ===========================

# Learning rates (conservative values for initial learning to prevent collapse. you can easily multiply it by 10)
DEFAULT_LR_ACTOR: float = 1e-5
DEFAULT_LR_CRITIC: float = 1e-5
DEFAULT_LR_ALPHA: float = 1e-4

# SAC parameters
DEFAULT_GAMMA: float = 0.99      # Discount factor
DEFAULT_TAU: float = 0.005       # Soft target update rate
DEFAULT_ALPHA: float = 0.2       # Initial entropy coefficient (if not auto-tuning)

# Experience replay
DEFAULT_BUFFER_SIZE: int = 200000
DEFAULT_BATCH_SIZE: int = 128
DEFAULT_LEARNING_STARTS: int = 10000  # Steps of random exploration before learning

# Training schedule
DEFAULT_EPISODES: int = 200
DEFAULT_EVAL_FREQUENCY: int = 50      # Evaluate every N episodes
DEFAULT_CHECKPOINT_FREQUENCY: int = 100  # Save checkpoint every N episodes

# ===========================
# Evaluation Parameters
# ===========================

DEFAULT_EVAL_EPISODES: int = 10          # Episodes for standard evaluation
DEFAULT_INTERMEDIATE_EVAL_EPISODES: int = 5   # Episodes for periodic evaluation during training
DEFAULT_FINAL_EVAL_EPISODES: int = 10   # Episodes for final evaluation
DEFAULT_MAX_STEPS_PER_EPISODE: int = 2500  # Safety timeout for evaluation

# ===========================
# Selection Training Parameters
# ===========================

DEFAULT_NUM_AGENTS: int = 4             # Number of parallel agents
DEFAULT_SELECTION_FREQUENCY: int = 50   # Episodes between tournaments
DEFAULT_ELITE_COUNT: int = 2            # Top N agents preserved (1=winner-takes-all)

# ===========================
# Environment Parameters
# ===========================

# Early termination
DEFAULT_TERMINATE_STATIONARY: bool = True
DEFAULT_STATIONARY_PATIENCE: int = 150   # Will terminate episode if no progress for N steps
DEFAULT_STATIONARY_MIN_STEPS: int = 0   # Minimum steps required to terminate episode (it's useless at this stage since it's < patience)

# Episode constraints
DEFAULT_MAX_EPISODE_STEPS: int = 5000  # Max steps per episode (prevents infinite loops)
DEFAULT_MIN_EPISODE_STEPS: int = 200   # if episode end before this, apply penalty

# Reward shaping
DEFAULT_REWARD_SHAPING: bool = True
DEFAULT_SHORT_EPISODE_PENALTY: float = -100.0
MIN_EPISODE_REWARD: float = -500.0  # Cap worst-case total episode reward

# ===========================
# Reward Structure Configuration
# ===========================

# Progress and completion rewards
PROGRESS_REWARD_SCALE: float = 2000.0  # Reward scale for track progress (full lap = 2000 points)
LAP_COMPLETION_REWARD: float = 1000.0   # Large reward for completing a full lap (encourages finishing)

# Time and behavior penalties
# STEP_PENALTY == ONTRACK_REWARD for initial learning. then set step penalty to 0.6 or more to push the ai to drive fast
STEP_PENALTY: float = 0.6              # Penalty per frame (mild time pressure)
STATIONARY_PENALTY: float = 1.0        # Penalty per frame for being stationary (speed < threshold)
STATIONARY_SPEED_THRESHOLD: float = 0.5  # Speed threshold (m/s) below which car is considered stationary

# On-track rewards
ONTRACK_REWARD: float = 0.5            # Positive reward per frame for staying on track

# Off-track penalties
OFFTRACK_PENALTY: float = 1.0          # Penalty per wheel off track per frame
OFFTRACK_THRESHOLD: int = 0          # Number of wheels that can be off track before penalty applies (0 = any wheel off is penalized)
OFFTRACK_TERMINATION_PENALTY: float = 100.0  # Penalty when going completely off track

# Speed-based rewards
FORWARD_SPEED_REWARD_SCALE: float = 0.0  # 0.001 for initial learning then 0. (a small reward prevent the car from staying still during initial learning)

# ===========================
# Device Configuration
# ===========================

DEFAULT_DEVICE: str = 'cpu'  # 'auto', 'cpu', 'cuda', or 'mps'

# ===========================
# Logging and Checkpoints
# ===========================

DEFAULT_CHECKPOINT_DIR: str = 'checkpoints'
DEFAULT_LOG_DIR: str = 'logs'
DEFAULT_SELECTION_CHECKPOINT_DIR: str = 'checkpoints_selection_parallel'
DEFAULT_SELECTION_LOG_DIR: str = 'logs_selection_parallel'
