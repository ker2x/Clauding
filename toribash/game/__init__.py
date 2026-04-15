"""Toribash 2D Game Package.

This package contains game logic and rules for the Toribash turn-based fighting game.
It handles scoring, turn management, and win conditions without any physics details.

Modules:
    match: Match orchestration, turn simulation, scoring accumulation
    scoring: Damage calculation, ground-touch penalties, reward computation

Dependency Hierarchy:
    config ← physics ← game ← env
                   ↖ rendering ← scripts

Note:
    This layer has NO physics internals or rendering - only game rules.
    It depends on physics (for collision data) and config.

Example:
    >>> from game.match import Match
    >>> from config.env_config import EnvConfig
    >>> from config.body_config import JointState
    >>> match = Match(EnvConfig())
    >>> match.set_actions(0, [JointState.CONTRACT] * 14)
    >>> match.set_actions(1, [JointState.HOLD] * 14)
    >>> result = match.simulate_turn()
    >>> print(f"Damage dealt: {result.damage_a_to_b}")
"""

from .scoring import (
    EXEMPT_GROUND_SEGMENTS,
    GROUND_PENALTIES,
    TurnResult,
    compute_turn_result,
    compute_reward,
)
from .match import Match

__all__ = [
    "EXEMPT_GROUND_SEGMENTS",
    "GROUND_PENALTIES",
    "TurnResult",
    "compute_turn_result",
    "compute_reward",
    "Match",
]
