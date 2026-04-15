"""Toribash 2D Rendering Package.

This package provides pygame-based visualization for the game.
It renders ragdolls, ground, UI elements, and supports human play.

Modules:
    pygame_renderer: Main rendering class

Dependency Hierarchy:
    config ← physics ← game ← env
                   ↖ rendering ← scripts

Note:
    This layer depends on physics and game for drawing,
    but NOT on env (no game logic in rendering).

Example:
    >>> from rendering.pygame_renderer import PygameRenderer
    >>> from game.match import Match
    >>> match = Match()
    >>> renderer = PygameRenderer(match, mode="human")
    >>> renderer.render(match)
"""

from .pygame_renderer import PygameRenderer

__all__ = [
    "PygameRenderer",
]
