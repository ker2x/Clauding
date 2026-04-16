"""Damage calculation, ground-touch penalties, and reward computation.

This module implements the scoring rules for Toribash, including:
- Damage attribution from collision impulses (velocity-weighted)
- Ground-touch penalties based on body part (head is most critical)
- Ground-touch bonuses when opponent touches ground
- RL reward computation from turn results

Toribash Scoring Rules:
- Damage dealt increases your score
- Non-exempt body parts touching ground decreases your score
- Feet and hands are exempt (expected to touch ground)
- Head touching ground is the most severe penalty

Usage:
    >>> from game.scoring import compute_turn_result, compute_reward, TurnResult
    >>> result = compute_turn_result(collision_handler, config)
    >>> reward = compute_reward(result, player=0, config=config, won=False)
    >>> print(f"Reward for this turn: {reward:.2f}")
"""

from dataclasses import dataclass, field
from config.env_config import EnvConfig
from physics.collision import CollisionHandler
from physics.world import COLLISION_TYPE_A, COLLISION_TYPE_B


# Segments that are allowed to touch ground without penalty.
# Toribash rules: feet and hands are expected to be on the ground.
EXEMPT_GROUND_SEGMENTS: set[str] = {"foot_l", "foot_r", "hand_l", "hand_r"}

# Segments that trigger KO when touching the ground.
KO_GROUND_SEGMENTS: set[str] = {"head", "chest"}

# Ground touch penalties per segment (Toribash rules).
# Negative values are penalties, applied to the player's score.
# Head is most critical (instant KO risk), limbs are less severe.
GROUND_PENALTIES: dict[str, float] = {
    # Critical: head touching ground is very dangerous
    "head": -4.0,
    # Major body parts
    "chest": -2.0,
    "stomach": -1.6,
    # Limbs (less critical but still penalized)
    "upper_arm_l": -0.6,
    "upper_arm_r": -0.6,
    "lower_arm_l": -0.6,
    "lower_arm_r": -0.6,
    "upper_leg_l": -0.6,
    "upper_leg_r": -0.6,
    "lower_leg_l": -0.4,
    "lower_leg_r": -0.4,
    # Default for any unknown segment
    "default": -1.0,
}


@dataclass
class TurnResult:
    """Results from simulating one turn.
    
    This dataclass contains all the information about what happened during
    a single physics simulation turn. It is computed from collision data
    and used for both scoring and RL reward calculation.
    
    Attributes:
        damage_a_to_b: Damage player A dealt to player B this turn.
        damage_b_to_a: Damage player B dealt to player A this turn.
        ground_segments_a: Segments of player A currently touching ground.
        ground_segments_b: Segments of player B currently touching ground.
        dismembered_a: Joint names dismembered on player A this turn.
        dismembered_b: Joint names dismembered on player B this turn.
    """
    damage_a_to_b: float = 0.0  # damage player A dealt to B
    damage_b_to_a: float = 0.0  # damage player B dealt to A
    ground_segments_a: set[str] = field(default_factory=set)
    ground_segments_b: set[str] = field(default_factory=set)
    dismembered_a: list[str] = field(default_factory=list)
    dismembered_b: list[str] = field(default_factory=list)


def compute_turn_result(
    collision_handler: CollisionHandler,
    config: EnvConfig,
) -> TurnResult:
    """Analyze collision data from a turn and compute damage and contacts.
    
    This function processes all fighter-fighter collision impulses to determine
    damage attribution. The damage is distributed based on relative velocities:
    the faster-moving body is considered the "striker" and deals more damage.
    
    Damage Formula:
        For each impulse above threshold:
        - damage = (impulse - threshold) / 1000.0
        - A's share = damage × (vel_a / (vel_a + vel_b))
        - B's share = damage × (vel_b / (vel_a + vel_b))
    
    Args:
        collision_handler: Contains turn_impulses and ground_contacts.
        config: Environment config with damage thresholds.
    
    Returns:
        TurnResult with damage amounts and ground contact sets.
    """
    result = TurnResult()

    # Process fighter-fighter impulses with velocity-based attribution
    for impulse, seg_a, seg_b, vel_a, vel_b in collision_handler.turn_impulses:
        # Hands and feet don't deal or take hit damage
        if seg_a in EXEMPT_GROUND_SEGMENTS or seg_b in EXEMPT_GROUND_SEGMENTS:
            continue
        if impulse > config.damage_impulse_threshold:
            # Calculate damage above the threshold
            damage = (impulse - config.damage_impulse_threshold) / config.damage_divisor

            if seg_a == "head" or seg_b == "head":
                damage *= config.head_damage_multiplier

            total_vel = vel_a + vel_b
            
            if total_vel > 0:
                # Velocity-weighted damage attribution
                # Higher velocity body deals more damage (is the striker)
                result.damage_a_to_b += damage * (vel_a / total_vel)
                result.damage_b_to_a += damage * (vel_b / total_vel)
            else:
                # Both stationary (e.g., pressed together), split evenly
                result.damage_a_to_b += damage * 0.5
                result.damage_b_to_a += damage * 0.5

    # Record ground contacts for each player
    result.ground_segments_a = collision_handler.get_ground_segments(COLLISION_TYPE_A)
    result.ground_segments_b = collision_handler.get_ground_segments(COLLISION_TYPE_B)

    return result


def compute_reward(
    result: TurnResult,
    player: int,
    config: EnvConfig,
    ko: bool = False,
    ko_self: bool = False,
    won: bool = False,
    lost: bool = False,
) -> float:
    """Compute RL reward for a player from a turn result.

    Args:
        result: The turn result from simulate_turn().
        player: Which player to compute reward for (0 or 1).
        config: Environment config with reward weights.
        ko: Whether the opponent was knocked out this turn.
        ko_self: Whether this player was knocked out this turn.
        won: Whether this player won the match.
        lost: Whether this player lost the match.

    Returns:
        Scalar reward for this turn.
    """
    reward = 0.0

    # Extract relevant data based on which player we're rewarding
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

    # Damage rewards
    reward += damage_dealt * config.reward_damage_dealt
    reward += damage_taken * config.reward_damage_taken

    # Ground contact penalties (Toribash rules)
    # Only non-exempt segments count (feet/hands are fine)
    own_bad_ground = own_ground - EXEMPT_GROUND_SEGMENTS
    opp_bad_ground = opp_ground - EXEMPT_GROUND_SEGMENTS
    
    for seg in own_bad_ground:
        reward += GROUND_PENALTIES.get(seg, GROUND_PENALTIES["default"])
    
    for seg in opp_bad_ground:
        # Opponent falling is a positive reward for us
        reward -= GROUND_PENALTIES.get(seg, GROUND_PENALTIES["default"])

    # Dismemberment reward
    reward += len(opp_dismembered) * config.reward_dismember

    # KO bonuses/penalties
    if ko:
        reward += config.reward_ko
    if ko_self:
        reward += config.reward_ko_penalty

    # Win/loss bonuses
    if won:
        reward += config.reward_win
    if lost:
        reward += config.reward_loss

    return reward
