"""
Preprocessing and environment wrappers for CarRacing-v3.

This module handles environment creation for vector mode only.
Vector mode provides a 71D state vector with track geometry and car state.

CarRacing-v3 uses a continuous action space: [steering, acceleration]
- steering: [-1.0, 1.0] (left to right)
- acceleration: [-1.0, 1.0] (brake to gas)

Continuous actions are used directly with SAC - no discretization.
"""

from __future__ import annotations

import gymnasium as gym

# Import local CarRacing environment
from env.car_racing import CarRacing
from config.domain_randomization import DomainRandomizationConfig


def make_carracing_env(
    terminate_stationary: bool = True,
    stationary_patience: int = 100,
    stationary_min_steps: int = 50,
    render_mode: str | None = None,
    reward_shaping: bool = True,
    min_episode_steps: int = 150,
    short_episode_penalty: float = -10.0,
    max_episode_steps: int = 1500,
    verbose: bool = False,
    domain_randomization_config: DomainRandomizationConfig | None = None,
) -> gym.Env:
    """
    Creates a CarRacing-v3 environment for SAC training in vector mode.

    Vector mode provides a 71D state vector containing:
    - Track geometry (lookahead sensors)
    - Car state (velocity, angular velocity, position)

    Args:
        terminate_stationary: Terminate early if car is stationary (default: True)
        stationary_patience: Frames without progress before termination (default: 100)
        stationary_min_steps: Minimum steps before early termination allowed (default: 50)
        render_mode: Rendering mode ('rgb_array', 'human', or None)
        reward_shaping: Apply reward shaping to discourage short episodes (default: True)
        min_episode_steps: Minimum episode length before penalty (default: 150)
        short_episode_penalty: Penalty for episodes shorter than min_episode_steps (default: -10.0)
        max_episode_steps: Maximum steps per episode (default: 1500, prevents infinite episodes)
        verbose: Enable verbose mode from environment for debugging (default: False)
        domain_randomization_config: Configuration for domain randomization (default: None, disabled)

    Returns:
        Environment ready for SAC training with continuous actions.
        All logic (timeout, reward shaping, state normalization) is built into CarRacing.
    """
    # Vector mode: All logic built into CarRacing itself - no wrappers needed!
    # This includes timeout, reward shaping, and state normalization
    env = CarRacing(
        render_mode=render_mode,
        verbose=verbose,
        terminate_stationary=terminate_stationary,
        stationary_patience=stationary_patience,
        stationary_min_steps=stationary_min_steps,
        state_mode="vector",
        max_episode_steps=max_episode_steps,
        reward_shaping=reward_shaping,
        min_episode_steps=min_episode_steps,
        short_episode_penalty=short_episode_penalty,
        domain_randomization_config=domain_randomization_config,
    )
    return env


if __name__ == "__main__":
    """Test preprocessing pipeline."""
    print("Creating CarRacing environment with vector mode...")

    env = make_carracing_env()
    print(f"Action space: {env.action_space}")
    print(f"Observation space: {env.observation_space}")
    print(f"Expected shape: (73,) for 73D vector state")

    obs, info = env.reset()
    print(f"Observation shape: {obs.shape}")
    print(f"Observation dtype: {obs.dtype}")
    print(f"Observation sample: {obs[:5]}...")

    # Test continuous action
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)
    print(f"Action: {action}")
    print(f"Reward: {reward:.3f}")
    env.close()

    print("\nPreprocessing test complete!")
