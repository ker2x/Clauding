"""Collision handling: damage tracking, impulse recording, dismemberment."""

import pymunk
from config.constants import CAT_PLAYER_A, CAT_PLAYER_B


class CollisionHandler:
    """Tracks collision impulses between fighters for damage/dismemberment."""

    def __init__(self, space: pymunk.Space, ct_a: int, ct_b: int):
        self.space = space
        # Accumulated impulses this turn: list of (impulse_mag, shape_a_name, shape_b_name)
        self.turn_impulses: list[tuple[float, str, str]] = []
        # Ground contacts this turn: set of (collision_type, segment_name)
        self.ground_contacts: set[tuple[int, str]] = set()

        # Fighter vs fighter collisions (pymunk 7.x API)
        space.on_collision(
            collision_type_a=ct_a,
            collision_type_b=ct_b,
            post_solve=self._on_fighter_collision,
        )

        # Fighter vs ground (ground collision_type = 0)
        # Use post_solve so contacts are tracked every step, not just on first touch
        for ct in (ct_a, ct_b):
            space.on_collision(
                collision_type_a=ct,
                collision_type_b=0,
                post_solve=self._on_ground_contact,
            )

    def _on_fighter_collision(self, arbiter: pymunk.Arbiter, space, data):
        """Record impulse from fighter-fighter collision."""
        impulse = arbiter.total_impulse.length
        if impulse > 0:
            shapes = arbiter.shapes
            name_a = getattr(shapes[0], 'segment_name', 'unknown')
            name_b = getattr(shapes[1], 'segment_name', 'unknown')
            self.turn_impulses.append((impulse, name_a, name_b))

    def _on_ground_contact(self, arbiter: pymunk.Arbiter, space, data):
        """Record ground contact."""
        shapes = arbiter.shapes
        for shape in shapes:
            if hasattr(shape, 'segment_name'):
                self.ground_contacts.add((shape.collision_type, shape.segment_name))

    def clear_turn(self):
        """Reset per-turn tracking."""
        self.turn_impulses.clear()
        self.ground_contacts.clear()

    def get_total_impulse_on(self, collision_type: int) -> float:
        """Get total impulse received by a specific player this turn."""
        total = 0.0
        for imp, name_a, name_b in self.turn_impulses:
            # Both shapes are involved; we attribute impulse to both sides
            total += imp
        return total

    def get_ground_segments(self, collision_type: int) -> set[str]:
        """Get set of segment names touching the ground for a player."""
        return {name for ct, name in self.ground_contacts if ct == collision_type}
