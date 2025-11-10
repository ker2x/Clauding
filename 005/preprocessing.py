"""
Preprocessing and environment wrappers for CarRacing-v3.

This module handles:
1. Frame preprocessing (grayscale conversion, normalization) for visual mode
2. Frame stacking (to capture motion/velocity) for visual mode
3. Reward shaping to encourage learning

CarRacing-v3 has a continuous action space: [steering, gas, brake]
- steering: [-1.0, 1.0] (left to right)
- gas: [0.0, 1.0]
- brake: [0.0, 1.0]

This project uses continuous actions directly with SAC (no discretization).
"""

import gymnasium as gym
import numpy as np
import cv2
from collections import deque

# Import local CarRacing environment
from env.car_racing import CarRacing


class GrayscaleWrapper(gym.ObservationWrapper):
    """
    Converts RGB observations to grayscale.

    CarRacing provides 96×96×3 RGB images. Converting to grayscale:
    1. Reduces computation by 3x
    2. Still preserves track boundaries (track vs grass)

    Note: CarRacing has distinct visual features even in grayscale
    (dark track, light grass, white lane markers).
    """

    def __init__(self, env):
        super().__init__(env)
        old_shape = env.observation_space.shape
        # New shape: (height, width) - single channel
        new_shape = (old_shape[0], old_shape[1])
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=new_shape, dtype=np.uint8
        )

    def observation(self, obs):
        """Convert RGB to grayscale using standard weights."""
        # Use OpenCV for efficient conversion (same as cv2.COLOR_RGB2GRAY)
        grayscale = cv2.cvtColor(obs, cv2.COLOR_RGB2GRAY)
        return grayscale


class NormalizeObservation(gym.ObservationWrapper):
    """
    Normalizes pixel values from [0, 255] to [0, 1].

    Neural networks train better with normalized inputs.
    """

    def __init__(self, env):
        super().__init__(env)
        old_shape = env.observation_space.shape
        self.observation_space = gym.spaces.Box(
            low=0.0, high=1.0, shape=old_shape, dtype=np.float32
        )

    def observation(self, obs):
        """Normalize to [0, 1] range."""
        return obs.astype(np.float32) / 255.0


class RewardShaper(gym.Wrapper):
    """
    Shapes rewards to discourage degenerate strategies.

    Specifically penalizes episodes that end too quickly (< min_steps),
    which prevents the agent from exploiting stationary termination as
    a "safe" strategy. This encourages the agent to actually drive and
    make progress.

    Penalty: -50 if episode ends in < min_steps (configurable)
    """

    def __init__(self, env, min_steps=150, short_episode_penalty=-50.0):
        """
        Args:
            env: Environment to wrap
            min_steps: Episodes shorter than this get penalized (default: 150)
            short_episode_penalty: Penalty for short episodes (default: -50.0)
        """
        super().__init__(env)
        self.min_steps = min_steps
        self.short_episode_penalty = short_episode_penalty
        self.steps_in_episode = 0

    def reset(self, **kwargs):
        """Reset episode step counter."""
        self.steps_in_episode = 0
        return self.env.reset(**kwargs)

    def step(self, action):
        """Apply reward shaping based on episode length."""
        obs, reward, terminated, truncated, info = self.env.step(action)
        self.steps_in_episode += 1

        # If episode is ending and it's too short, apply penalty
        if (terminated or truncated) and self.steps_in_episode < self.min_steps:
            shaped_reward = reward + self.short_episode_penalty
            info['reward_shaping'] = self.short_episode_penalty
            info['original_reward'] = reward
            return obs, shaped_reward, terminated, truncated, info

        return obs, reward, terminated, truncated, info


class FrameStack(gym.Wrapper):
    """
    Stacks the last N frames to capture motion and velocity.

    A single frame doesn't show:
    - Direction of car movement
    - Speed of car
    - Trajectory/momentum

    Stacking 4 frames allows the network to infer these temporal features.

    Output shape: (stack_size, height, width)
    Example: (4, 96, 96) for 4 stacked 96×96 frames
    """

    def __init__(self, env, stack_size=4):
        super().__init__(env)
        self.stack_size = stack_size
        self.frames = deque(maxlen=stack_size)

        # Update observation space
        old_shape = env.observation_space.shape
        new_shape = (stack_size,) + old_shape
        self.observation_space = gym.spaces.Box(
            low=0.0,
            high=1.0,
            shape=new_shape,
            dtype=np.float32
        )

    def reset(self, **kwargs):
        """Reset environment and initialize frame stack."""
        obs, info = self.env.reset(**kwargs)
        # Fill stack with initial frame
        for _ in range(self.stack_size):
            self.frames.append(obs)
        return self._get_observation(), info

    def step(self, action):
        """Take step and update frame stack."""
        obs, reward, terminated, truncated, info = self.env.step(action)
        self.frames.append(obs)
        return self._get_observation(), reward, terminated, truncated, info

    def _get_observation(self):
        """Stack frames along first dimension."""
        return np.stack(self.frames, axis=0)


def make_carracing_env(
    stack_size=4,
    terminate_stationary=True,
    stationary_patience=100,
    stationary_min_steps=50,
    render_mode=None,
    state_mode="vector",
    reward_shaping=True,
    min_episode_steps=150,
    short_episode_penalty=-50.0,
    max_episode_steps=1500,
    verbose=False
) -> gym.Env:
    """
    Creates a preprocessed CarRacing-v3 environment for SAC training.

    Args:
        stack_size: Number of frames to stack (default: 4) - only used in visual mode
        terminate_stationary: Terminate early if car is stationary (default: True)
        stationary_patience: Frames without progress before termination (default: 100)
        stationary_min_steps: Minimum steps before early termination allowed (default: 50)
        render_mode: Rendering mode ('rgb_array', 'human', or None)
        state_mode: 'vector' (67D track geometry - RECOMMENDED) or 'visual' (96x96 images - slow)
                    - vector: fastest, 67-value vector with track lookahead (most informative)
                              No preprocessing wrappers needed - all logic built-in
                    - visual: slowest, full rendering with frame stacking and wrappers
        reward_shaping: Apply reward shaping to discourage short episodes (default: True)
        min_episode_steps: Minimum episode length before penalty (default: 150)
        short_episode_penalty: Penalty for episodes shorter than min_episode_steps (default: -50.0)
        max_episode_steps: Maximum steps per episode (default: 1500, prevents infinite episodes)
        verbose: Enable verbose mode from environment for debugging (default: False)

    Returns:
        Environment ready for SAC training with continuous actions.
        Vector mode: CarRacing directly (no wrappers)
        Visual mode: Wrapped CarRacing with preprocessing
    """
    if state_mode == "vector":
        # Vector mode: All logic built into CarRacing itself - no wrappers needed!
        # This includes timeout, reward shaping, and state normalization
        env = CarRacing(
            render_mode=render_mode,
            verbose=verbose,
            continuous=True,
            terminate_stationary=terminate_stationary,
            stationary_patience=stationary_patience,
            stationary_min_steps=stationary_min_steps,
            state_mode=state_mode,
            max_episode_steps=max_episode_steps,
            reward_shaping=reward_shaping,
            min_episode_steps=min_episode_steps,
            short_episode_penalty=short_episode_penalty,
        )
        return env

    elif state_mode == "visual":
        # Visual mode: Use wrappers for backwards compatibility
        # Create base environment without built-in timeout/reward shaping
        # (let wrappers handle it for visual mode)
        env = CarRacing(
            render_mode=render_mode,
            verbose=verbose,
            continuous=True,
            terminate_stationary=terminate_stationary,
            stationary_patience=stationary_patience,
            stationary_min_steps=stationary_min_steps,
            state_mode=state_mode,
            max_episode_steps=None,  # Disable built-in timeout for visual mode
            reward_shaping=False,    # Disable built-in shaping for visual mode
        )

        # Reward shaping to discourage degenerate strategies
        if reward_shaping:
            env = RewardShaper(env, min_steps=min_episode_steps, short_episode_penalty=short_episode_penalty)

        # Frame preprocessing (native 96x96, no resize needed)
        env = GrayscaleWrapper(env)
        env = NormalizeObservation(env)
        # Frame stacking (must be last to stack preprocessed frames)
        env = FrameStack(env, stack_size=stack_size)

        # Add time limit to prevent infinite episodes
        if max_episode_steps is not None and max_episode_steps > 0:
            env = gym.wrappers.TimeLimit(env, max_episode_steps=max_episode_steps)

        return env

    else:
        raise ValueError(f"Invalid state_mode: {state_mode}. Must be 'vector' or 'visual'.")


if __name__ == "__main__":
    """Test preprocessing pipeline."""
    print("Creating CarRacing environment with preprocessing...")

    # Test vector mode (default)
    print("\n=== Testing Vector Mode ===")
    env_vector = make_carracing_env(state_mode='vector')
    print(f"Action space: {env_vector.action_space}")
    print(f"Observation space: {env_vector.observation_space}")
    print(f"Expected shape: (36,) for 36D vector state")

    obs, info = env_vector.reset()
    print(f"Observation shape: {obs.shape}")
    print(f"Observation dtype: {obs.dtype}")
    print(f"Observation sample: {obs[:5]}...")

    # Test continuous action
    action = env_vector.action_space.sample()
    obs, reward, terminated, truncated, info = env_vector.step(action)
    print(f"Action: {action}")
    print(f"Reward: {reward:.3f}")
    env_vector.close()

    # Test visual mode
    print("\n=== Testing Visual Mode ===")
    env_visual = make_carracing_env(state_mode='visual', render_mode='rgb_array')
    print(f"Observation space: {env_visual.observation_space}")
    print(f"Expected shape: (4, 96, 96) for 4 stacked 96×96 grayscale frames")

    obs, info = env_visual.reset()
    print(f"Observation shape: {obs.shape}")
    print(f"Observation dtype: {obs.dtype}")
    print(f"Observation range: [{obs.min():.3f}, {obs.max():.3f}]")

    env_visual.close()
    print("\nPreprocessing test complete!")
