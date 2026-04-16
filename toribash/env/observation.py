"""Build normalized observation vector from match state.

This module constructs the observation vector used by RL agents. The observation
is designed to be:
- Ego-centric: Always from the agent's perspective
- Normalized: All values scaled to roughly [-1, 1]
- Compact: 239-dimensional vector with essential game state

Observation Layout (per ragdoll, ×2):
    - Joint angles (14): normalized to [-1, 1] by angle limits
    - Joint angular velocities (14): normalized by 2× motor rate
    - Joint states (14): normalized 0.0, 0.33, 0.67, 1.0
    - Segment positions (30): x,y × 15 segments, relative to own torso
    - Segment velocities (30): vx,vy × 15 segments
    - Segment rotations (15): sin of absolute angle

Global State:
    - Relative torso dx, dy (2)
    - Turn progress (1)
    - Own/opponent scores (2)
    - Previous actions (14)

Usage:
    >>> from env.observation import build_observation, compute_obs_dim
    >>> obs_dim = compute_obs_dim(14, 15)
    >>> print(f"Observation dimension: {obs_dim}")
    >>> obs = build_observation(match, player=0, prev_actions=[...])
"""

import numpy as np
from config.body_config import DEFAULT_BODY, JointState
from config.constants import ARENA_WIDTH, ARENA_HEIGHT, GROUND_Y, SCORE_NORM, VELOCITY_NORM
from game.match import Match


def compute_obs_dim(num_joints: int, num_segments: int) -> int:
    """Calculate the total observation dimension.
    
    The observation is ego-centric: it contains data for both the agent's
    ragdoll and the opponent's ragdoll, plus global state.
    
    Args:
        num_joints: Number of joints in the body config (usually 14).
        num_segments: Number of segments in the body config (usually 15).
    
    Returns:
        Total dimension of the observation vector.
    
    Calculation:
        Per ragdoll:
            - joint angles: num_joints
            - joint angular velocities: num_joints
            - joint states: num_joints
            - segment positions: num_segments × 2 (x, y)
            - segment velocities: num_segments × 2 (vx, vy)
            - segment rotations: num_segments
        
        Global state:
            - relative torso position: 2 (dx, dy)
            - turn progress: 1
            - scores: 2 (own, opponent)
            - previous actions: num_joints
        
        Total = 2 × per_ragdoll + global_state
    """
    per_ragdoll = (
        num_joints        # joint angles (normalized)
        + num_joints      # joint angular velocities
        + num_joints      # joint states (normalized 0-1)
        + num_segments * 2  # segment positions (x, y)
        + num_segments * 2  # segment velocities (vx, vy)
        + num_segments      # segment rotations
        + num_segments      # self-collision flags (0/1)
        + num_segments      # cross-collision flags (0/1)
        + num_segments      # ground contact flags (0/1)
    )
    global_state = (
        2   # relative position between torsos (dx, dy)
        + 1  # turn progress (0 to 1)
        + 2  # scores [own, opponent]
        + num_joints  # previous actions (14 joint states)
    )
    return per_ragdoll * 2 + global_state


def build_observation(match: Match, player: int, prev_actions: list | None = None) -> np.ndarray:
    """Build observation vector from the perspective of `player` (0 or 1).
    
    The observation is ego-centric: own ragdoll data comes first,
    opponent data second. For player 1, positions are mirrored
    in physics (via facing=-1) so the agent always "faces right"
    from its own perspective.
    
    Args:
        match: The current match state.
        player: Which player to view from (0 or 1).
            - 0: View from player A's perspective (ragdoll_a)
            - 1: View from player B's perspective (ragdoll_b)
        prev_actions: List of previous turn's joint states (for temporal memory).
            Should be a list of JointState values, one per joint.
            If None, reads current joint states as fallback.
    
    Returns:
        numpy array of shape (obs_dim,) with float32 dtype.
        All values are normalized to roughly [-1, 1].
    
    Note:
        The observation includes the opponent's data with positions
        relative to the agent's torso, allowing the policy to learn
        spatial relationships regardless of which side it's playing.
    """
    # Determine which ragdoll is "own" and which is "opponent"
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
    ch = match.world.collision_handler

    # --- Own ragdoll state ---
    obs.extend(_ragdoll_obs(own_rag, own_rag, ch))

    # --- Opponent ragdoll state ---
    obs.extend(_ragdoll_obs(opp_rag, own_rag, ch))

    # --- Global state ---
    own_torso = own_rag.get_torso_position()
    opp_torso = opp_rag.get_torso_position()
    
    # Relative position between torsos (normalized by arena size)
    dx = (opp_torso.x - own_torso.x) / ARENA_WIDTH
    dy = (opp_torso.y - own_torso.y) / ARENA_HEIGHT
    obs.append(dx)
    obs.append(dy)

    # Turn progress (0 at start, 1 at max turns)
    turn_progress = match.turn / match.config.max_turns
    obs.append(turn_progress)

    # Scores (normalized loosely by 100)
    obs.append(own_score / SCORE_NORM)
    obs.append(opp_score / SCORE_NORM)

    # Previous actions (temporal memory, one per joint)
    if prev_actions is not None:
        for state in prev_actions:
            obs.append(float(state.value) / 3.0)  # Normalize to 0-1
    else:
        # Fallback: read current joint states (inaccurate but backward-compatible)
        for jdef in own_rag.body_config.joints:
            state = own_rag.joint_states.get(jdef.name, JointState.HOLD)
            obs.append(float(state.value) / 3.0)

    return np.array(obs, dtype=np.float32)


def _ragdoll_obs(ragdoll, ref_ragdoll, collision_handler) -> list[float]:
    """Build observation features for one ragdoll, relative to ref_ragdoll's torso.

    This helper function extracts state features for a single ragdoll:
    - Joint states (angles, angular velocities, motor states)
    - Segment states (positions, velocities, rotations)
    - Contact flags (self-collision, cross-collision, ground)

    All values are normalized to roughly [-1, 1] for stable neural network learning.

    Args:
        ragdoll: The Ragdoll instance to extract features from.
        ref_ragdoll: Reference ragdoll for computing relative positions.
        collision_handler: CollisionHandler for contact flag lookups.

    Returns:
        List of normalized float values for this ragdoll.
    """
    obs = []
    config = ragdoll.body_config

    # --- Joint angles (normalized by limits) ---
    # Maps angle range [angle_min, angle_max] to [-1, 1]
    angles = ragdoll.get_joint_angles()
    for angle, jdef in zip(angles, config.joints):
        angle_range = jdef.angle_max - jdef.angle_min
        if angle_range > 0:
            # Normalize: remap [min, max] to [-1, 1]
            normalized = 2.0 * (angle - jdef.angle_min) / angle_range - 1.0
        else:
            normalized = 0.0
        obs.append(np.clip(normalized, -1.0, 1.0))

    # --- Joint angular velocities (normalized by motor rate) ---
    # Divided by 2× motor_rate to expect values in [-0.5, 0.5] typical
    ang_vels = ragdoll.get_joint_angular_velocities()
    for vel, jdef in zip(ang_vels, config.joints):
        normalized = vel / (jdef.motor_rate * 2.0)
        obs.append(np.clip(normalized, -1.0, 1.0))

    # --- Joint states (encoded as normalized values) ---
    # 0, 1, 2, 3 → 0.0, 0.33, 0.67, 1.0
    for jdef in config.joints:
        state = ragdoll.joint_states.get(jdef.name, JointState.HOLD)
        obs.append(state / 3.0)

    # --- Segment positions (relative to reference torso) ---
    # This allows the network to learn opponent position relative to self
    ref_pos = ref_ragdoll.get_torso_position()
    positions = ragdoll.get_segment_positions()
    for pos in positions:
        obs.append((pos.x - ref_pos.x) / ARENA_WIDTH)
        obs.append((pos.y - ref_pos.y) / ARENA_HEIGHT)

    # --- Segment velocities (normalized by max expected velocity) ---
    # ~500 cm/s is a reasonable max velocity for ragdoll segments
    velocities = ragdoll.get_segment_velocities()
    for vel in velocities:
        obs.append(vel.x / VELOCITY_NORM)
        obs.append(vel.y / VELOCITY_NORM)

    # --- Segment rotations (via sin for angle normalization) ---
    # sin(angle) naturally maps angle to [-1, 1]
    seg_angles = ragdoll.get_segment_angles()
    for angle in seg_angles:
        obs.append(np.sin(angle))

    # --- Contact flags (binary 0/1 per segment) ---
    ct = ragdoll.collision_type
    self_contacts = collision_handler.get_self_contact_segments(ct)
    cross_contacts = collision_handler.get_cross_contact_segments(ct)
    ground_contacts = collision_handler.get_ground_segments(ct)
    for seg_def in config.segments:
        obs.append(1.0 if seg_def.name in self_contacts else 0.0)
    for seg_def in config.segments:
        obs.append(1.0 if seg_def.name in cross_contacts else 0.0)
    for seg_def in config.segments:
        obs.append(1.0 if seg_def.name in ground_contacts else 0.0)

    return obs
