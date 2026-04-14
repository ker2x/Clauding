"""Damage calculation, ground-touch penalties, KO detection."""

from dataclasses import dataclass, field
from config.env_config import EnvConfig
from physics.collision import CollisionHandler
from physics.world import COLLISION_TYPE_A, COLLISION_TYPE_B


# Segments that are allowed to touch ground without penalty (Toribash rules)
EXEMPT_GROUND_SEGMENTS = {"foot_l", "foot_r", "hand_l", "hand_r"}

# Ground touch penalties per segment (Toribash rules - head is most critical)
# Negative = penalty, positive = bonus
GROUND_PENALTIES = {
    # Critical (head is most dangerous - instant KO risk)
    "head": -2.0,
    # Major body parts
    "chest": -1.0,
    "stomach": -0.8,
    # Limbs (less critical but still penalized)
    "upper_arm_l": -0.3,
    "upper_arm_r": -0.3,
    "lower_arm_l": -0.3,
    "lower_arm_r": -0.3,
    "upper_leg_l": -0.3,
    "upper_leg_r": -0.3,
    "lower_leg_l": -0.2,
    "lower_leg_r": -0.2,
    # Default for any unknown segment
    "default": -0.5,
}


@dataclass
class TurnResult:
    """Results from simulating one turn."""
    damage_a_to_b: float = 0.0  # damage player A dealt to B
    damage_b_to_a: float = 0.0  # damage player B dealt to A
    ground_segments_a: set[str] = field(default_factory=set)
    ground_segments_b: set[str] = field(default_factory=set)
    dismembered_a: list[str] = field(default_factory=list)  # joints dismembered on A this turn
    dismembered_b: list[str] = field(default_factory=list)  # joints dismembered on B this turn


def compute_turn_result(
    collision_handler: CollisionHandler,
    config: EnvConfig,
) -> TurnResult:
    """Analyze collision data from a turn and compute damage/contacts."""
    result = TurnResult()

    # Process fighter-fighter impulses with velocity-based attribution
    # The faster-moving body is the "striker" and deals more damage
    for impulse, seg_a, seg_b, vel_a, vel_b in collision_handler.turn_impulses:
        if impulse > config.damage_impulse_threshold:
            damage = (impulse - config.damage_impulse_threshold) / 1000.0
            total_vel = vel_a + vel_b
            if total_vel > 0:
                # A moving fast into B = A strikes B = damage_a_to_b
                result.damage_a_to_b += damage * (vel_a / total_vel)
                result.damage_b_to_a += damage * (vel_b / total_vel)
            else:
                # Both stationary (e.g. pressed together), split evenly
                result.damage_a_to_b += damage * 0.5
                result.damage_b_to_a += damage * 0.5

    # Ground contacts
    result.ground_segments_a = collision_handler.get_ground_segments(COLLISION_TYPE_A)
    result.ground_segments_b = collision_handler.get_ground_segments(COLLISION_TYPE_B)

    return result


def compute_reward(
    result: TurnResult,
    player: int,
    config: EnvConfig,
    ko: bool = False,
    won: bool = False,
) -> float:
    """Compute reward for a player from a turn result."""
    reward = 0.0

    if player == 0:
        damage_dealt = result.damage_a_to_b
        damage_taken = result.damage_b_to_a
        own_ground = result.ground_segments_a
        opp_ground = result.ground_segments_b
        own_dismembered = result.dismembered_a
        opp_dismembered = result.dismembered_b
    else:
        damage_dealt = result.damage_b_to_a
        damage_taken = result.damage_a_to_b
        own_ground = result.ground_segments_b
        opp_ground = result.ground_segments_a
        own_dismembered = result.dismembered_b
        opp_dismembered = result.dismembered_a

    # Damage
    reward += damage_dealt * config.reward_damage_dealt
    reward += damage_taken * config.reward_damage_taken

    # Ground contact penalty with segment-specific weights (Toribash rules)
    own_bad_ground = own_ground - EXEMPT_GROUND_SEGMENTS
    opp_bad_ground = opp_ground - EXEMPT_GROUND_SEGMENTS
    for seg in own_bad_ground:
        reward += GROUND_PENALTIES.get(seg, GROUND_PENALTIES["default"])
    for seg in opp_bad_ground:
        reward -= GROUND_PENALTIES.get(seg, GROUND_PENALTIES["default"])  # Opponent falling = positive

    # Dismemberment
    reward += len(opp_dismembered) * config.reward_dismember

    # KO / win bonuses
    if ko:
        reward += config.reward_ko
    if won:
        reward += config.reward_win

    return reward
