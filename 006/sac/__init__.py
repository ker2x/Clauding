"""
Soft Actor-Critic (SAC) implementation for continuous control.

This module provides a complete SAC implementation with:
- VectorActor: Policy network (Gaussian) with LayerNorm
- VectorActorNoLN: Policy network without LayerNorm (for comparison)
- VectorCritic: Q-function network (twin critics) with LayerNorm
- VectorCriticNoLN: Q-function network without LayerNorm (for comparison)
- ReplayBuffer: Experience replay with optimized sampling
- SACAgent: Main SAC algorithm with automatic entropy tuning
"""

from .actor import VectorActor, VectorActorNoLN
from .critic import VectorCritic, VectorCriticNoLN
from .buffer import ReplayBuffer
from .agent import SACAgent

__all__ = [
    'VectorActor',
    'VectorActorNoLN',
    'VectorCritic',
    'VectorCriticNoLN',
    'ReplayBuffer',
    'SACAgent'
]
