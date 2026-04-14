"""PhysicsWorld: pymunk Space wrapper with ground, ragdolls, and stepping."""

import pymunk
from config.body_config import BodyConfig, DEFAULT_BODY
from config.constants import (
    GRAVITY, DT, SPACE_ITERATIONS,
    GROUND_Y, ARENA_WIDTH, ARENA_HEIGHT,
    SPAWN_OFFSET_X,
)
from config.env_config import EnvConfig
from .ragdoll import Ragdoll
from .collision import CollisionHandler


COLLISION_TYPE_GROUND = 0
COLLISION_TYPE_A = 1
COLLISION_TYPE_B = 2


class PhysicsWorld:
    """Manages the pymunk simulation space with two ragdoll fighters."""

    def __init__(self, config: EnvConfig | None = None):
        self.config = config or EnvConfig()

        self.space = pymunk.Space()
        self.space.gravity = GRAVITY
        self.space.iterations = SPACE_ITERATIONS

        self._create_ground()

        center_x = ARENA_WIDTH / 2
        spawn_y = GROUND_Y  # feet at ground level

        self.ragdoll_a = Ragdoll(
            self.space,
            body_config=self.config.body_config,
            position=(center_x - self.config.spawn_offset_x, spawn_y),
            facing=1,
            collision_type=COLLISION_TYPE_A,
        )
        self.ragdoll_b = Ragdoll(
            self.space,
            body_config=self.config.body_config,
            position=(center_x + self.config.spawn_offset_x, spawn_y),
            facing=-1,
            collision_type=COLLISION_TYPE_B,
        )

        self.collision_handler = CollisionHandler(
            self.space, COLLISION_TYPE_A, COLLISION_TYPE_B
        )

    def _create_ground(self):
        """Create a static ground plane."""
        body = self.space.static_body
        # Wide ground segment
        ground = pymunk.Segment(body, (-100, GROUND_Y), (ARENA_WIDTH + 100, GROUND_Y), 5)
        ground.friction = 1.0
        ground.elasticity = 0.0
        ground.collision_type = COLLISION_TYPE_GROUND
        self.space.add(ground)

    def step(self, dt: float = DT):
        """Advance physics by one timestep."""
        self.space.step(dt)

    def simulate_turn(self, n_steps: int | None = None):
        """Run n_steps of physics simulation (one full turn)."""
        if n_steps is None:
            n_steps = self.config.steps_per_turn
        self.collision_handler.clear_turn()
        for _ in range(n_steps):
            self.step()
