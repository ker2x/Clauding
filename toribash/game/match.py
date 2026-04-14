"""Match: orchestrates turn-based gameplay, scoring, win conditions."""

from config.body_config import JointState
from config.env_config import EnvConfig
from physics.world import PhysicsWorld, COLLISION_TYPE_A, COLLISION_TYPE_B
from .scoring import TurnResult, compute_turn_result


class Match:
    """A full Toribash match between two fighters."""

    def __init__(self, config: EnvConfig | None = None):
        self.config = config or EnvConfig()
        self.world = PhysicsWorld(self.config)
        self.turn = 0
        self.scores = [0.0, 0.0]  # accumulated damage dealt by each player
        self.total_damage = [0.0, 0.0]  # total damage taken by each player
        self.dismember_counts = [0, 0]  # limbs lost per player
        self.turn_results: list[TurnResult] = []

    def set_actions(self, player: int, joint_states: list[JointState]):
        """Set joint states for a player (0=A, 1=B)."""
        ragdoll = self.world.ragdoll_a if player == 0 else self.world.ragdoll_b
        ragdoll.set_all_joint_states(joint_states)

    def simulate_turn(self) -> TurnResult:
        """Simulate one turn of physics and compute results."""
        self.world.simulate_turn()

        result = compute_turn_result(
            self.world.collision_handler,
            self.config,
        )

        # Check for dismemberment
        for jname, impulses in self._get_joint_impulses().items():
            max_impulse = max(impulses) if impulses else 0
            if max_impulse > self.config.dismember_impulse:
                # Determine which ragdoll owns this joint
                if jname in self.world.ragdoll_a.joints and jname not in self.world.ragdoll_a.dismembered:
                    self.world.ragdoll_a.dismember_joint(jname)
                    result.dismembered_a.append(jname)
                    self.dismember_counts[0] += 1
                if jname in self.world.ragdoll_b.joints and jname not in self.world.ragdoll_b.dismembered:
                    self.world.ragdoll_b.dismember_joint(jname)
                    result.dismembered_b.append(jname)
                    self.dismember_counts[1] += 1

        # Update scores
        self.scores[0] += result.damage_a_to_b
        self.scores[1] += result.damage_b_to_a
        self.total_damage[0] += result.damage_b_to_a
        self.total_damage[1] += result.damage_a_to_b

        self.turn_results.append(result)
        self.turn += 1
        return result

    def is_done(self) -> bool:
        """Check if the match is over."""
        return self.turn >= self.config.max_turns

    def get_winner(self) -> int | None:
        """Return winner (0 or 1) or None for draw. Higher score wins."""
        if self.scores[0] > self.scores[1]:
            return 0
        elif self.scores[1] > self.scores[0]:
            return 1
        return None

    def _get_joint_impulses(self) -> dict[str, list[float]]:
        """Map joint names to impulses they received this turn (simplified)."""
        # For now, we don't have per-joint impulse tracking;
        # dismemberment will be handled by total impulse on adjacent segments
        return {}
