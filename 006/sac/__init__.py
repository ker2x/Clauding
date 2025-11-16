"""
Soft Actor-Critic (SAC) implementation for continuous control.

This module provides a complete SAC implementation with:
- VectorActor: Policy network (Gaussian)
- VectorCritic: Q-function network (twin critics)
- ReplayBuffer: Experience replay with optimized sampling
- SACAgent: Main SAC algorithm with automatic entropy tuning
"""

from .actor import VectorActor
from .critic import VectorCritic
from .buffer import ReplayBuffer
from .agent import SACAgent

__all__ = ['VectorActor', 'VectorCritic', 'ReplayBuffer', 'SACAgent']
