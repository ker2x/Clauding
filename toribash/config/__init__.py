"""Toribash 2D Configuration Package.

This package contains all configuration dataclasses and constants used throughout
the game engine. Configuration is kept separate from game logic to ensure
testability and easy parameter tuning.

Modules:
    constants: Physics, game, and arena constants
    body_config: Ragdoll body part and joint definitions
    env_config: RL environment configuration

Example:
    >>> from config import EnvConfig, DEFAULT_BODY
    >>> config = EnvConfig(max_turns=30, opponent_type="random")
    >>> print(f"Using {DEFAULT_BODY.num_joints} joints")
"""

from .constants import (
    GRAVITY,
    DT,
    SPACE_ITERATIONS,
    STEPS_PER_TURN,
    MAX_TURNS,
    GROUND_Y,
    ARENA_WIDTH,
    ARENA_HEIGHT,
    SPAWN_OFFSET_X,
    CAT_GROUND,
    CAT_PLAYER_A,
    CAT_PLAYER_B,
    DAMAGE_IMPULSE_THRESHOLD,
    DISMEMBER_IMPULSE,
    DEFAULT_MOTOR_RATE,
    DEFAULT_MOTOR_MAX_FORCE,
    RELAX_MAX_FORCE,
)

from .body_config import (
    JointState,
    SegmentDef,
    JointDef,
    BodyConfig,
    DEFAULT_BODY,
)

from .env_config import EnvConfig

__all__ = [
    # Constants
    "GRAVITY",
    "DT",
    "SPACE_ITERATIONS",
    "STEPS_PER_TURN",
    "MAX_TURNS",
    "GROUND_Y",
    "ARENA_WIDTH",
    "ARENA_HEIGHT",
    "SPAWN_OFFSET_X",
    "CAT_GROUND",
    "CAT_PLAYER_A",
    "CAT_PLAYER_B",
    "DAMAGE_IMPULSE_THRESHOLD",
    "DISMEMBER_IMPULSE",
    "DEFAULT_MOTOR_RATE",
    "DEFAULT_MOTOR_MAX_FORCE",
    "RELAX_MAX_FORCE",
    # Body config
    "JointState",
    "SegmentDef",
    "JointDef",
    "BodyConfig",
    "DEFAULT_BODY",
    # Env config
    "EnvConfig",
]
