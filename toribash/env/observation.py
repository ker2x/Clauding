"""Build normalized observation vector from match state."""

import numpy as np
from config.body_config import DEFAULT_BODY, JointState
from config.constants import ARENA_WIDTH, ARENA_HEIGHT, GROUND_Y
from game.match import Match


def compute_obs_dim(num_joints: int, num_segments: int) -> int:
    """Calculate observation dimension."""
    per_ragdoll = (
        num_joints        # joint angles (normalized)
        + num_joints      # joint angular velocities
        + num_joints      # joint states (one-hot would be 4x, but we use normalized int)
        + num_segments * 2  # segment positions (x, y) relative to own torso
        + num_segments * 2  # segment velocities (vx, vy)
        + num_segments      # segment rotations
    )
    global_state = (
        2   # relative position between torsos (dx, dy)
        + 1  # turn progress (0 to 1)
        + 2  # scores [own, opponent]
    )
    return per_ragdoll * 2 + global_state


def build_observation(match: Match, player: int) -> np.ndarray:
    """Build observation vector from the perspective of `player` (0 or 1).

    The observation is ego-centric: own ragdoll data comes first,
    opponent data second. For player 1, positions are mirrored
    so the agent always "faces right".
    """
    if player == 0:
        own_rag = match.world.ragdoll_a
        opp_rag = match.world.ragdoll_b
        own_score = match.scores[0]
        opp_score = match.scores[1]
    else:
        own_rag = match.world.ragdoll_b
        opp_rag = match.world.ragdoll_a
        own_score = match.scores[1]
        opp_score = match.scores[0]

    obs = []

    # --- Own ragdoll ---
    obs.extend(_ragdoll_obs(own_rag, own_rag))

    # --- Opponent ragdoll ---
    obs.extend(_ragdoll_obs(opp_rag, own_rag))

    # --- Global state ---
    own_torso = own_rag.get_torso_position()
    opp_torso = opp_rag.get_torso_position()
    dx = (opp_torso.x - own_torso.x) / ARENA_WIDTH
    dy = (opp_torso.y - own_torso.y) / ARENA_HEIGHT
    obs.append(dx)
    obs.append(dy)

    # Turn progress
    turn_progress = match.turn / match.config.max_turns
    obs.append(turn_progress)

    # Scores (normalized loosely)
    obs.append(own_score / 100.0)
    obs.append(opp_score / 100.0)

    return np.array(obs, dtype=np.float32)


def _ragdoll_obs(ragdoll, ref_ragdoll) -> list[float]:
    """Build observation features for one ragdoll, relative to ref_ragdoll's torso."""
    obs = []
    config = ragdoll.body_config

    # Joint angles (normalized by limits)
    angles = ragdoll.get_joint_angles()
    for angle, jdef in zip(angles, config.joints):
        range_ = jdef.angle_max - jdef.angle_min
        if range_ > 0:
            normalized = 2.0 * (angle - jdef.angle_min) / range_ - 1.0
        else:
            normalized = 0.0
        obs.append(np.clip(normalized, -1.0, 1.0))

    # Joint angular velocities (normalized by motor rate)
    ang_vels = ragdoll.get_joint_angular_velocities()
    for vel, jdef in zip(ang_vels, config.joints):
        normalized = vel / (jdef.motor_rate * 2.0)
        obs.append(np.clip(normalized, -1.0, 1.0))

    # Joint states as normalized values (0, 0.33, 0.67, 1.0)
    for jdef in config.joints:
        state = ragdoll.joint_states.get(jdef.name, JointState.HOLD)
        obs.append(state / 3.0)

    # Segment positions relative to reference torso
    ref_pos = ref_ragdoll.get_torso_position()
    positions = ragdoll.get_segment_positions()
    for pos in positions:
        obs.append((pos.x - ref_pos.x) / ARENA_WIDTH)
        obs.append((pos.y - ref_pos.y) / ARENA_HEIGHT)

    # Segment velocities (normalized)
    velocities = ragdoll.get_segment_velocities()
    for vel in velocities:
        obs.append(vel.x / 500.0)  # ~500 cm/s max reasonable velocity
        obs.append(vel.y / 500.0)

    # Segment rotations (normalized to [-1, 1] via sin)
    seg_angles = ragdoll.get_segment_angles()
    for angle in seg_angles:
        obs.append(np.sin(angle))

    return obs
