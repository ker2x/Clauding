"""
Shared constants for SAC training on CarRacing-v3.

This module defines all shared constants used across the training codebase,
providing a single source of truth for hyperparameters and configuration values.
"""

# ===========================
# State and Action Space
# ===========================

STATE_DIM = 71  # Vector mode state dimension (car state + track geometry + lookahead)
ACTION_DIM = 3  # Continuous action space: [steering, gas, brake]

# ===========================
# Training Hyperparameters
# ===========================

# Learning rates
DEFAULT_LR_ACTOR = 1e-4
DEFAULT_LR_CRITIC = 1e-4
DEFAULT_LR_ALPHA = 1e-4

# SAC parameters
DEFAULT_GAMMA = 0.99      # Discount factor
DEFAULT_TAU = 0.005       # Soft target update rate
DEFAULT_ALPHA = 0.2       # Initial entropy coefficient (if not auto-tuning)

# Experience replay
DEFAULT_BUFFER_SIZE = 200000
DEFAULT_BATCH_SIZE = 512
DEFAULT_LEARNING_STARTS = 5000  # Steps of random exploration before learning

# Training schedule
DEFAULT_EPISODES = 200
DEFAULT_EVAL_FREQUENCY = 50      # Evaluate every N episodes
DEFAULT_CHECKPOINT_FREQUENCY = 100  # Save checkpoint every N episodes

# ===========================
# Evaluation Parameters
# ===========================

DEFAULT_EVAL_EPISODES = 10          # Episodes for standard evaluation
DEFAULT_FINAL_EVAL_EPISODES = 10   # Episodes for final evaluation
DEFAULT_MAX_STEPS_PER_EPISODE = 2500  # Safety timeout for evaluation

# ===========================
# Selection Training Parameters
# ===========================

DEFAULT_NUM_AGENTS = 4             # Number of parallel agents
DEFAULT_SELECTION_FREQUENCY = 50   # Episodes between tournaments
DEFAULT_ELITE_COUNT = 2            # Top N agents preserved (1=winner-takes-all)

# ===========================
# Environment Parameters
# ===========================

# Early termination
DEFAULT_TERMINATE_STATIONARY = True
DEFAULT_STATIONARY_PATIENCE = 150
DEFAULT_STATIONARY_MIN_STEPS = 50

# Episode constraints
DEFAULT_MAX_EPISODE_STEPS = 5000  # Max steps per episode (prevents infinite loops)
DEFAULT_MIN_EPISODE_STEPS = 200   # Min steps before penalty applied

# Reward shaping
DEFAULT_REWARD_SHAPING = True
DEFAULT_SHORT_EPISODE_PENALTY = -50.0

# ===========================
# Device Configuration
# ===========================

DEFAULT_DEVICE = 'auto'  # 'auto', 'cpu', 'cuda', or 'mps'

# ===========================
# Logging and Checkpoints
# ===========================

DEFAULT_CHECKPOINT_DIR = 'checkpoints'
DEFAULT_LOG_DIR = 'logs'
DEFAULT_SELECTION_CHECKPOINT_DIR = 'checkpoints_selection_parallel'
DEFAULT_SELECTION_LOG_DIR = 'logs_selection_parallel'
