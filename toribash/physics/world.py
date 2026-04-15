"""PhysicsWorld: pymunk Space wrapper with ground, ragdolls, and stepping.

This module implements the PhysicsWorld class, which manages the complete
physics simulation including:
- The pymunk Space with gravity and solver settings
- A static ground body
- Two ragdoll fighters (player A and player B)
- Collision handling for damage and ground contacts

Usage:
    >>> from physics.world import PhysicsWorld
    >>> from config.env_config import EnvConfig
    >>> world = PhysicsWorld(EnvConfig())
    >>> world.step()  # Advance by one physics timestep
    >>> world.simulate_turn()  # Run one full game turn (30 steps)
"""

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


# Collision type constants (local to physics module)
# These are used by CollisionHandler and Match to identify collision participants
COLLISION_TYPE_GROUND: int = 0
COLLISION_TYPE_A: int = 1
COLLISION_TYPE_B: int = 2


class PhysicsWorld:
    """Manages the pymunk simulation space with two ragdoll fighters.
    
    The PhysicsWorld encapsulates all physics simulation state and provides
    a clean interface for stepping the simulation. It creates:
    - A pymunk Space with gravity
    - A static ground body
    - Two ragdolls (A facing right, B facing left)
    - A collision handler for damage/contact tracking
    
    Attributes:
        config: The EnvConfig used to create this world.
        space: The pymunk Space containing all bodies and constraints.
        ragdoll_a: The first ragdoll (facing right, player 0).
        ragdoll_b: The second ragdoll (facing left, player 1).
        collision_handler: Handler for collision callbacks.
    """
    
    def __init__(self, config: EnvConfig | None = None):
        """Initialize the physics world with a new simulation.
        
        Args:
            config: Environment configuration (uses defaults if None).
                Includes body config, spawn positions, and steps per turn.
        """
        self.config = config or EnvConfig()

        # Create pymunk space with gravity and solver settings
        self.space = pymunk.Space()
        self.space.gravity = GRAVITY
        self.space.iterations = SPACE_ITERATIONS

        # Create ground plane
        self._create_ground()

        # Calculate spawn positions
        center_x = ARENA_WIDTH / 2
        spawn_y = GROUND_Y  # feet at ground level

        # Create both ragdolls
        # Player A spawns to the left of center, facing right (+1)
        self.ragdoll_a = Ragdoll(
            self.space,
            body_config=self.config.body_config,
            position=(center_x - self.config.spawn_offset_x, spawn_y),
            facing=1,
            collision_type=COLLISION_TYPE_A,
        )
        
        # Player B spawns to the right of center, facing left (-1)
        self.ragdoll_b = Ragdoll(
            self.space,
            body_config=self.config.body_config,
            position=(center_x + self.config.spawn_offset_x, spawn_y),
            facing=-1,
            collision_type=COLLISION_TYPE_B,
        )

        # Set up collision callbacks for damage and ground contact tracking
        self.collision_handler = CollisionHandler(
            self.space, COLLISION_TYPE_A, COLLISION_TYPE_B
        )

    def _create_ground(self) -> None:
        """Create a static ground plane at GROUND_Y.
        
        The ground is a thick static segment that extends beyond the arena
        bounds to prevent fighters from walking off the edge. It has:
        - High friction for standing
        - Zero elasticity (no bounce)
        - Collision type 0 (ground)
        """
        body = self.space.static_body
        # Wide segment extending beyond arena to catch edge cases
        ground = pymunk.Segment(
            body, 
            (-100, GROUND_Y),  # Left of arena
            (ARENA_WIDTH + 100, GROUND_Y),  # Right of arena
            5  # Thickness
        )
        ground.friction = 1.0
        ground.elasticity = 0.0
        ground.collision_type = COLLISION_TYPE_GROUND
        self.space.add(ground)

    def step(self, dt: float = DT) -> None:
        """Advance physics simulation by one timestep.
        
        This is the fundamental physics update. Call this 30 times to
        simulate one game turn.
        
        Args:
            dt: Timestep in seconds (default: 1/60).
        """
        self.space.step(dt)

    def simulate_turn(self, n_steps: int | None = None) -> None:
        """Run n_steps of physics simulation (one full turn).
        
        A "turn" in Toribash consists of multiple physics steps where
        players set joint states at the start and then watch the result.
        
        Args:
            n_steps: Number of physics steps (default: config.steps_per_turn).
                Typically 30 steps = 0.5 seconds of simulated time.
        
        Note:
            Clears collision tracking at the start of the turn.
        """
        if n_steps is None:
            n_steps = self.config.steps_per_turn
        
        # Clear collision tracking for the new turn
        self.collision_handler.clear_turn()
        
        # Run physics steps
        for _ in range(n_steps):
            self.step()
