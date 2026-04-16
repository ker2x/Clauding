"""Collision handling for damage tracking, impulse recording, and dismemberment.

This module implements collision callbacks that track:
    1. Fighter-vs-fighter collisions: Records impulse magnitude and segment names
       for damage calculation. Also tracks body velocities for directional damage
       attribution (faster body is the "striker").
    2. Fighter-vs-ground collisions: Records which segments are touching ground
       for posture scoring.

The module uses pymunk 7.x collision handlers with post_solve callbacks,
which fire every physics step during active contact.

Usage:
    >>> handler = CollisionHandler(space, collision_type_a=1, collision_type_b=2)
    >>> # ... simulate physics ...
    >>> impulses = handler.turn_impulses  # List of collision data
    >>> ground_segments = handler.get_ground_segments(COLLISION_TYPE_A)
"""

import pymunk
from config.constants import CAT_PLAYER_A, CAT_PLAYER_B, IMPULSE_CAP


class CollisionHandler:
    """Tracks collision impulses and ground contacts for damage/dismemberment calculation.
    
    This class registers collision callbacks with the pymunk space and maintains
    per-turn tracking of:
    - Fighter-fighter collision impulses (with velocity data for attribution)
    - Fighter-ground contact segments
    
    Attributes:
        turn_impulses: List of (impulse, seg_a_name, seg_b_name, vel_a, vel_b).
            Cleared at start of each turn.
        ground_contacts: Set of (collision_type, segment_name) tuples.
            Cleared at start of each turn.
    
    Note:
        Uses pymunk 7.x API: space.on_collision() not add_collision_handler()
    """
    
    def __init__(self, space: pymunk.Space, ct_a: int, ct_b: int):
        """Initialize collision handler and register callbacks.
        
        Args:
            space: The pymunk Space to register callbacks with.
            ct_a: Collision type integer for player A segments.
            ct_b: Collision type integer for player B segments.
        """
        self.space = space
        self.ct_a = ct_a
        self.ct_b = ct_b

        # Accumulated impulses this turn: (impulse_mag, seg_a_name, seg_b_name, vel_a, vel_b)
        # vel_a and vel_b are body velocity magnitudes for directional damage attribution
        self.turn_impulses: list[tuple[float, str, str, float, float]] = []

        # Ground contacts this turn: set of (collision_type, segment_name)
        self.ground_contacts: set[tuple[int, str]] = set()

        # Contact flags per segment: which segments are touching what
        # self_contacts: segments colliding with own ragdoll (keyed by collision_type)
        # cross_contacts: segments colliding with opponent (keyed by collision_type)
        self.self_contacts: dict[int, set[str]] = {ct_a: set(), ct_b: set()}
        self.cross_contacts: dict[int, set[str]] = {ct_a: set(), ct_b: set()}

        # Register fighter vs fighter collision callback
        # post_solve fires every step while contact is active (not just at first touch)
        space.on_collision(
            collision_type_a=ct_a,
            collision_type_b=ct_b,
            post_solve=self._on_fighter_collision,
        )

        # Register self-collision callbacks for each player
        for ct in (ct_a, ct_b):
            space.on_collision(
                collision_type_a=ct,
                collision_type_b=ct,
                post_solve=self._on_self_collision,
            )

        # Register fighter vs ground collision callbacks for both players
        for ct in (ct_a, ct_b):
            space.on_collision(
                collision_type_a=ct,
                collision_type_b=0,  # Ground collision type
                post_solve=self._on_ground_contact,
            )

    def _on_fighter_collision(self, arbiter: pymunk.Arbiter, space, data) -> bool:
        """Callback for fighter-fighter collisions.
        
        Records the impulse magnitude and involved segment names for damage
        calculation. Uses velocity data to determine who is the "striker"
        (the faster-moving body deals more damage proportionally).
        
        Args:
            arbiter: Contains collision data including total_impulse and shapes.
            space: The pymunk Space (unused but required by callback signature).
            data: User data passed to callback (unused).
        
        Returns:
            True to process this collision, False to ignore.
        """
        impulse = arbiter.total_impulse.length
        if impulse > 0:
            # Cap impulse to filter clipping artifacts (embedded segments
            # produce escalating impulses as pymunk tries to separate them)
            impulse = min(impulse, IMPULSE_CAP)

            shapes = arbiter.shapes
            # Get segment names from shape attributes (set during ragdoll creation)
            name_a = getattr(shapes[0], 'segment_name', 'unknown')
            name_b = getattr(shapes[1], 'segment_name', 'unknown')

            # Record body velocities for directional damage attribution
            # The faster-moving body is considered the "striker"
            vel_a = shapes[0].body.velocity.length
            vel_b = shapes[1].body.velocity.length

            self.turn_impulses.append((impulse, name_a, name_b, vel_a, vel_b))

            # Track which segments are in cross-player contact
            self.cross_contacts[shapes[0].collision_type].add(name_a)
            self.cross_contacts[shapes[1].collision_type].add(name_b)

        return True

    def _on_self_collision(self, arbiter: pymunk.Arbiter, space, data) -> bool:
        """Callback for same-ragdoll self-collisions.

        Tracks which segments are colliding with their own body.
        Used for observation only — no damage or scoring.
        """
        shapes = arbiter.shapes
        ct = shapes[0].collision_type
        for shape in shapes:
            name = getattr(shape, 'segment_name', None)
            if name:
                self.self_contacts[ct].add(name)
        return True

    def _on_ground_contact(self, arbiter: pymunk.Arbiter, space, data) -> bool:
        """Callback for fighter-ground collisions.
        
        Records which segments are touching the ground for posture scoring.
        Ground contact with non-exempt segments (feet/hands are exempt) results
        in score penalties per Toribash rules.
        
        Args:
            arbiter: Contains collision shapes (one is the ground segment).
            space: The pymunk Space (unused but required).
            data: User data (unused).
        
        Returns:
            True to process this collision.
        """
        shapes = arbiter.shapes
        for shape in shapes:
            # Check if this shape has segment_name (it's a ragdoll segment)
            if hasattr(shape, 'segment_name'):
                # Record collision type + segment name
                self.ground_contacts.add((shape.collision_type, shape.segment_name))
        
        return True

    def clear_turn(self) -> None:
        """Reset per-turn tracking data.
        
        Called at the start of each turn's physics simulation to ensure
        impulses/contacts are counted only for the current turn.
        """
        self.turn_impulses.clear()
        self.ground_contacts.clear()
        for s in self.self_contacts.values():
            s.clear()
        for s in self.cross_contacts.values():
            s.clear()

    def get_ground_segments(self, collision_type: int) -> set[str]:
        """Get set of segment names touching the ground for a player.
        
        Args:
            collision_type: The collision type of the player to check.
        
        Returns:
            Set of segment names currently in ground contact for the player.
            This is used by scoring.py to apply ground-touch penalties.
        """
        return {name for ct, name in self.ground_contacts if ct == collision_type}

    def get_self_contact_segments(self, collision_type: int) -> set[str]:
        """Get segments colliding with own ragdoll this turn."""
        return self.self_contacts.get(collision_type, set())

    def get_cross_contact_segments(self, collision_type: int) -> set[str]:
        """Get segments colliding with opponent this turn."""
        return self.cross_contacts.get(collision_type, set())
