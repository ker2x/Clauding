"""
Preprocessing and environment wrappers for CarRacing-v3.

This module handles environment creation for vector mode only.
Vector mode provides a 71D state vector with track geometry and car state.

CarRacing-v3 has a continuous action space: [steering, gas, brake]
- steering: [-1.0, 1.0] (left to right)
- gas: [0.0, 1.0]
- brake: [0.0, 1.0]

This project uses continuous actions directly with SAC (no discretization).
"""

import gymnasium as gym

# Import local CarRacing environment
from env.car_racing import CarRacing


def make_carracing_env(
    terminate_stationary=True,
    stationary_patience=100,
    stationary_min_steps=50,
    render_mode=None,
    reward_shaping=True,
    min_episode_steps=150,
    short_episode_penalty=-10.0,
    max_episode_steps=1500,
    verbose=False,
    num_cars=1,
    suspension_config=None,
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
        num_cars: Number of cars racing simultaneously (default: 1)
        suspension_config: Suspension configuration dict (default: quarter-car sport mode)

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
        num_cars=num_cars,
        suspension_config=suspension_config,
    )
    return env


if __name__ == "__main__":
    """Test preprocessing pipeline."""
    print("Creating CarRacing environment with vector mode...")

    env = make_carracing_env()
    print(f"Action space: {env.action_space}")
    print(f"Observation space: {env.observation_space}")
    print(f"Expected shape: (71,) for 71D vector state")

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
