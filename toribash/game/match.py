"""Match: orchestrates turn-based gameplay, scoring accumulation, and win conditions.

This module implements the Match class, which manages a complete Toribash match
between two fighters. It handles:
- Setting joint actions for each player
- Simulating physics turns
- Accumulating scores from damage and penalties
- Detecting dismemberment events
- Determining match winner

Match Flow:
    1. Create Match with config
    2. Loop:
       a. Set player 0 actions via set_actions(0, joint_states)
       b. Set player 1 actions via set_actions(1, joint_states)
       c. Call simulate_turn() to run physics and get results
       d. Check is_done() for match end
    3. Call get_winner() for final result

Usage:
    >>> from game.match import Match
    >>> from config.env_config import EnvConfig
    >>> from config.body_config import JointState
    >>> match = Match(EnvConfig(max_turns=10))
    >>> while not match.is_done():
    ...     match.set_actions(0, [JointState.CONTRACT] * 14)
    ...     match.set_actions(1, [JointState.HOLD] * 14)
    ...     result = match.simulate_turn()
    ...     print(f"Turn {match.turn}: Score A={match.scores[0]:.1f}")
    >>> winner = match.get_winner()
    >>> print(f"Winner: {'A' if winner == 0 else 'B'}")
"""

from config.body_config import JointState
from config.env_config import EnvConfig
from physics.world import PhysicsWorld, COLLISION_TYPE_A, COLLISION_TYPE_B
from .scoring import TurnResult, compute_turn_result, EXEMPT_GROUND_SEGMENTS, GROUND_PENALTIES, KO_GROUND_SEGMENTS


class Match:
    """A full Toribash match between two fighters.
    
    The Match class orchestrates the game loop, managing:
    - Physics world with two ragdolls
    - Turn counter and score tracking
    - Turn simulation and result computation
    - Dismemberment detection and application
    - Win condition checking
    
    Attributes:
        config: Environment configuration (turns, thresholds, etc.).
        world: PhysicsWorld with ragdolls and collision handling.
        turn: Current turn number (0-indexed).
        scores: [player_a_score, player_b_score] accumulated damage dealt.
        total_damage: [player_a_damage_taken, player_b_damage_taken].
        dismember_counts: [player_a_limbs_lost, player_b_limbs_lost].
        turn_results: List of TurnResult for each turn.
    
    Note:
        Player 0 = Player A = ragdoll_a (facing right)
        Player 1 = Player B = ragdoll_b (facing left)
    """
    
    def __init__(self, config: EnvConfig | None = None):
        """Initialize a new match.
        
        Args:
            config: Environment configuration (uses defaults if None).
        """
        self.config = config or EnvConfig()
        self.world = PhysicsWorld(self.config)
        self.turn = 0
        self.scores = [0.0, 0.0]  # accumulated damage dealt by each player
        self.total_damage = [0.0, 0.0]  # total damage taken by each player
        self.dismember_counts = [0, 0]  # limbs lost per player
        self.ko: int | None = None  # player index who was KO'd, None if no KO
        self.ko_reason: str | None = None  # "beheading", "ground_ko", or "mutual"
        self.turn_results: list[TurnResult] = []

    def set_actions(self, player: int, joint_states: list[JointState]) -> None:
        """Set joint states for a player before a turn.
        
        This should be called for both players before simulate_turn().
        The joint states remain in effect until changed.
        
        Args:
            player: Player index (0=A, 1=B).
            joint_states: List of JointState values, one per joint.
        
        Raises:
            ValueError: If player is not 0 or 1.
        """
        if player not in (0, 1):
            raise ValueError(f"Player must be 0 or 1, got {player}")
        
        ragdoll = self.world.ragdoll_a if player == 0 else self.world.ragdoll_b
        ragdoll.set_all_joint_states(joint_states)

    def simulate_turn(self) -> TurnResult:
        """Simulate one turn of physics and compute results.

        Runs the physics simulation for steps_per_turn frames, then
        processes scoring, dismemberment, and KO detection.

        Returns:
            TurnResult with damage and contact information for this turn.
        """
        # Run physics simulation (clears collision tracking internally)
        self.world.simulate_turn()
        return self.process_turn()

    def process_turn(self) -> TurnResult:
        """Process collision data after physics stepping.

        Call this after manually stepping physics (e.g. animated playback).
        Handles all post-physics logic:
        1. Computes collision results (damage, ground contacts)
        2. Checks for dismemberment (joints that exceeded impulse threshold)
        3. Updates scores with damage dealt and ground penalties
        4. Detects KO conditions
        5. Increments turn counter

        Returns:
            TurnResult with damage and contact information for this turn.
        """
        # Compute turn result from collision data
        result = compute_turn_result(
            self.world.collision_handler,
            self.config,
        )

        # Check for dismemberment events
        impulses_a, impulses_b = self._get_joint_impulses()

        for jname, imp_list in impulses_a.items():
            if max(imp_list) > self.config.dismember_impulse:
                if jname not in self.world.ragdoll_a.dismembered:
                    self.world.ragdoll_a.dismember_joint(jname)
                    result.dismembered_a.append(jname)
                    self.dismember_counts[0] += 1

        for jname, imp_list in impulses_b.items():
            if max(imp_list) > self.config.dismember_impulse:
                if jname not in self.world.ragdoll_b.dismembered:
                    self.world.ragdoll_b.dismember_joint(jname)
                    result.dismembered_b.append(jname)
                    self.dismember_counts[1] += 1

        # Update scores: damage dealt is points scored (Toribash rules)
        self.scores[0] += result.damage_a_to_b
        self.scores[1] += result.damage_b_to_a

        # Detached segments don't count for ground penalties
        detached_a = self._get_detached_segments(self.world.ragdoll_a)
        detached_b = self._get_detached_segments(self.world.ragdoll_b)

        # Apply ground-touch penalties per Toribash rules
        # Non-exempt and non-detached segments touching ground reduce score
        bad_a = result.ground_segments_a - EXEMPT_GROUND_SEGMENTS - detached_a
        bad_b = result.ground_segments_b - EXEMPT_GROUND_SEGMENTS - detached_b
        for seg in bad_a:
            self.scores[0] += GROUND_PENALTIES.get(seg, GROUND_PENALTIES["default"])
        for seg in bad_b:
            self.scores[1] += GROUND_PENALTIES.get(seg, GROUND_PENALTIES["default"])

        # Track total damage taken for each player
        self.total_damage[0] += result.damage_b_to_a
        self.total_damage[1] += result.damage_a_to_b

        # Check for KO: head or chest on ground, or head dismembered (instant KO)
        if self.ko is None:
            neck_a = "neck" in self.world.ragdoll_a.dismembered
            neck_b = "neck" in self.world.ragdoll_b.dismembered
            ground_a = bool((result.ground_segments_a - detached_a) & KO_GROUND_SEGMENTS)
            ground_b = bool((result.ground_segments_b - detached_b) & KO_GROUND_SEGMENTS)
            ko_a = neck_a or ground_a
            ko_b = neck_b or ground_b
            if ko_a and not ko_b:
                self.ko = 0
                self.ko_reason = "beheading" if neck_a else "ground_ko"
            elif ko_b and not ko_a:
                self.ko = 1
                self.ko_reason = "beheading" if neck_b else "ground_ko"
            elif ko_a and ko_b:
                self.ko_reason = "mutual"
                # Both down: higher score survives, tie = no KO
                if self.scores[0] > self.scores[1]:
                    self.ko = 1
                elif self.scores[1] > self.scores[0]:
                    self.ko = 0
            # KO'd player can't benefit from accumulated damage
            if self.ko is not None:
                if self.scores[self.ko] > 0:
                    self.scores[self.ko] = 0.0

        # Store result and increment turn
        self.turn_results.append(result)
        self.turn += 1
        return result

    def is_done(self) -> bool:
        """Check if the match is over (max turns or KO)."""
        return self.turn >= self.config.max_turns or self.ko is not None

    def get_winner(self) -> int | None:
        """Determine the match winner.

        KO'd player loses. Otherwise higher score wins, equal = draw.
        """
        if self.ko is not None:
            return 1 if self.ko == 0 else 0
        if self.scores[0] > self.scores[1]:
            return 0
        elif self.scores[1] > self.scores[0]:
            return 1
        return None

    def _get_detached_segments(self, ragdoll) -> set[str]:
        """Get all segments that are detached from a ragdoll due to dismemberment."""
        detached = set()
        mapping = self.config.body_config.joint_to_child_segments
        for jname in ragdoll.dismembered:
            detached |= mapping[jname]
        return detached

    def _get_joint_impulses(self) -> tuple[dict[str, list[float]], dict[str, list[float]]]:
        """Map joint names to impulse magnitudes, separated by player.

        For each collision impulse, attributes it to all joints adjacent
        to the involved segments. Only impulses where the opposing body
        has sufficient velocity are included — this filters out clipping
        artifacts where embedded segments generate high impulses despite
        neither body actually striking.

        Returns:
            (impulses_a, impulses_b) where each maps joint_name -> [impulse magnitudes]
        """
        seg_to_joints = self.config.body_config.segment_to_joints
        min_vel = self.config.dismember_min_velocity
        impulses_a: dict[str, list[float]] = {}
        impulses_b: dict[str, list[float]] = {}

        for impulse, seg_a, seg_b, vel_a, vel_b in self.world.collision_handler.turn_impulses:
            # A's joints take damage from B's strike — require B is moving
            if vel_b >= min_vel:
                for jname in seg_to_joints.get(seg_a, []):
                    impulses_a.setdefault(jname, []).append(impulse)
            # B's joints take damage from A's strike — require A is moving
            if vel_a >= min_vel:
                for jname in seg_to_joints.get(seg_b, []):
                    impulses_b.setdefault(jname, []).append(impulse)

        return impulses_a, impulses_b
