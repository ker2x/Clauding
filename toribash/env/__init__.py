"""Toribash 2D Environment Package.

This package provides the gymnasium.Env interface for reinforcement learning training.
It bridges the game logic layer with RL frameworks like stable-baselines3.

Modules:
    toribash_env: Gymnasium environment implementation
    observation: Observation vector building and normalization

Dependency Hierarchy:
    config ← physics ← game ← env
                   ↖ rendering ← scripts

Note:
    This layer implements the RL interface (reset/step/render).
    It depends on game logic but not rendering.

Example:
    >>> import gymnasium as gym
    >>> from env.toribash_env import ToribashEnv
    >>> env = ToribashEnv()
    >>> obs, info = env.reset()
    >>> obs, reward, terminated, truncated, info = env.step([0, 0, 1, 1, 2, 2, 3, 3, 0, 0, 1, 1, 2, 2])
"""

from .toribash_env import ToribashEnv
from .observation import build_observation, compute_obs_dim

__all__ = [
    "ToribashEnv",
    "build_observation",
    "compute_obs_dim",
]
