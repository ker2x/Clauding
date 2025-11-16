"""
Configuration module for CarRacing-v3 training.

This module provides centralized configuration for:
- Training hyperparameters (constants.py)
- Physics parameters (physics_config.py)
- Rendering settings (rendering_config.py)
"""

from .constants import *
from .physics_config import (
    PhysicsConfig,
    VehicleParams,
    TireParams,
    PacejkaParams,
    DrivetrainParams,
    AerodynamicsParams,
    SteeringParams,
    FrictionParams,
    get_physics_config,
    DEFAULT_PHYSICS_CONFIG,
)
from .rendering_config import (
    RenderingConfig,
    VideoConfig,
    CameraConfig,
    TrackVisualsConfig,
    FrictionDetectionConfig,
    StateNormalizationConfig,
    get_rendering_config,
    DEFAULT_RENDERING_CONFIG,
)

__all__ = [
    # From constants.py (imported via *)
    'STATE_DIM',
    'ACTION_DIM',
    'DEFAULT_LR_ACTOR',
    'DEFAULT_LR_CRITIC',
    'DEFAULT_LR_ALPHA',
    'DEFAULT_GAMMA',
    'DEFAULT_TAU',
    'DEFAULT_ALPHA',
    'DEFAULT_BUFFER_SIZE',
    'DEFAULT_BATCH_SIZE',
    'DEFAULT_LEARNING_STARTS',
    'DEFAULT_EPISODES',
    'DEFAULT_EVAL_FREQUENCY',
    'DEFAULT_CHECKPOINT_FREQUENCY',
    'DEFAULT_EVAL_EPISODES',
    'DEFAULT_INTERMEDIATE_EVAL_EPISODES',
    'DEFAULT_FINAL_EVAL_EPISODES',
    'DEFAULT_MAX_STEPS_PER_EPISODE',
    'DEFAULT_NUM_AGENTS',
    'DEFAULT_SELECTION_FREQUENCY',
    'DEFAULT_ELITE_COUNT',
    'DEFAULT_TERMINATE_STATIONARY',
    'DEFAULT_STATIONARY_PATIENCE',
    'DEFAULT_STATIONARY_MIN_STEPS',
    'DEFAULT_MAX_EPISODE_STEPS',
    'DEFAULT_MIN_EPISODE_STEPS',
    'DEFAULT_REWARD_SHAPING',
    'DEFAULT_SHORT_EPISODE_PENALTY',
    'PROGRESS_REWARD_SCALE',
    'LAP_COMPLETION_REWARD',
    'STEP_PENALTY',
    'STATIONARY_PENALTY',
    'STATIONARY_SPEED_THRESHOLD',
    'ONTRACK_REWARD',
    'OFFTRACK_PENALTY',
    'OFFTRACK_THRESHOLD',
    'OFFTRACK_TERMINATION_PENALTY',
    'FORWARD_SPEED_REWARD_SCALE',
    'DEFAULT_DEVICE',
    'DEFAULT_CHECKPOINT_DIR',
    'DEFAULT_LOG_DIR',
    'DEFAULT_SELECTION_CHECKPOINT_DIR',
    'DEFAULT_SELECTION_LOG_DIR',
    # Physics config
    'PhysicsConfig',
    'VehicleParams',
    'TireParams',
    'PacejkaParams',
    'DrivetrainParams',
    'AerodynamicsParams',
    'SteeringParams',
    'FrictionParams',
    'get_physics_config',
    'DEFAULT_PHYSICS_CONFIG',
    # Rendering config
    'RenderingConfig',
    'VideoConfig',
    'CameraConfig',
    'TrackVisualsConfig',
    'FrictionDetectionConfig',
    'StateNormalizationConfig',
    'get_rendering_config',
    'DEFAULT_RENDERING_CONFIG',
]
