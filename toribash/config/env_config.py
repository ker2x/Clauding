"""Environment configuration dataclass."""

from dataclasses import dataclass, field
from .body_config import BodyConfig, DEFAULT_BODY
from .constants import (
    STEPS_PER_TURN, MAX_TURNS, DAMAGE_IMPULSE_THRESHOLD,
    DISMEMBER_IMPULSE, SPAWN_OFFSET_X,
)


@dataclass
class EnvConfig:
    body_config: BodyConfig = field(default_factory=lambda: DEFAULT_BODY)
    steps_per_turn: int = STEPS_PER_TURN
    max_turns: int = MAX_TURNS
    spawn_offset_x: float = SPAWN_OFFSET_X

    # Damage thresholds
    damage_impulse_threshold: float = DAMAGE_IMPULSE_THRESHOLD
    dismember_impulse: float = DISMEMBER_IMPULSE

    # Reward weights (for RL)
    reward_damage_dealt: float = 1.0
    reward_damage_taken: float = -0.5
    reward_ground_touch: float = -0.2
    reward_opponent_ground: float = 0.1
    reward_dismember: float = 5.0
    reward_ko: float = 10.0
    reward_win: float = 20.0

    # Opponent behavior (for single-agent RL)
    opponent_type: str = "hold"  # "hold", "random", "mirror"
