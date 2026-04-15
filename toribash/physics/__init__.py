"""Toribash 2D Physics Package.

This package contains the pymunk-based physics simulation for the game.
It handles rigid body dynamics, joint constraints, and collision detection.

Modules:
    collision: Collision callbacks and impulse tracking
    ragdoll: Ragdoll creation and joint control
    world: Physics world management and stepping

Dependency Hierarchy:
    config ← physics ← game ← env
                   ↖ rendering ← scripts

Note:
    This layer has NO game logic or rendering - only raw physics.
    It depends only on the config module.

Example:
    >>> from physics.world import PhysicsWorld
    >>> from config.env_config import EnvConfig
    >>> world = PhysicsWorld(EnvConfig())
    >>> world.step()  # Advance physics by one timestep
"""

from .collision import CollisionHandler
from .ragdoll import Ragdoll
from .world import PhysicsWorld, COLLISION_TYPE_GROUND, COLLISION_TYPE_A, COLLISION_TYPE_B

__all__ = [
    "CollisionHandler",
    "Ragdoll",
    "PhysicsWorld",
    "COLLISION_TYPE_GROUND",
    "COLLISION_TYPE_A",
    "COLLISION_TYPE_B",
]
