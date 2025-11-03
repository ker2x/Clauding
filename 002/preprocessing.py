"""
Preprocessing and environment wrappers for CarRacing-v3.

This module handles:
1. Frame preprocessing (grayscale conversion, resizing, normalization)
2. Frame stacking (to capture motion/velocity)
3. Action discretization (continuous → discrete action space)

CarRacing-v3 has a continuous action space: [steering, gas, brake]
- steering: [-1.0, 1.0] (left to right)
- gas: [0.0, 1.0]
- brake: [0.0, 1.0]

We discretize this into a manageable set of actions for DQN.
"""

import gymnasium as gym
import numpy as np
import cv2
from collections import deque
from typing import Tuple, List

# Import local CarRacing environment
from env.car_racing import CarRacing


class ActionDiscretizer(gym.ActionWrapper):
    """
    Discretizes the continuous action space of CarRacing-v3.

    Default discretization creates 9 actions:
    - Steering: left (-0.8), straight (0.0), right (+0.8)
    - Gas/Brake: brake (0.0, 0.8), coast (0.0, 0.0), gas (0.5, 0.0)

    This can be customized for finer or coarser control.
    """

    def __init__(self, env, steering_bins=3, gas_brake_bins=3):
        """
        Args:
            env: The CarRacing environment
            steering_bins: Number of discrete steering values (default: 3)
            gas_brake_bins: Number of discrete gas/brake combinations (default: 3)
        """
        super().__init__(env)
        self.steering_bins = steering_bins
        self.gas_brake_bins = gas_brake_bins

        # Create discrete action space
        self.action_space = gym.spaces.Discrete(steering_bins * gas_brake_bins)

        # Define discrete actions
        self.discrete_actions = self._create_action_map()

    def _create_action_map(self) -> List[np.ndarray]:
        """
        Creates mapping from discrete action indices to continuous action vectors.

        Returns:
            List of action vectors [steering, gas, brake]
        """
        actions = []

        # Define steering values: from -1 (left) to +1 (right)
        if self.steering_bins == 3:
            steering_values = [-0.8, 0.0, 0.8]
        elif self.steering_bins == 5:
            steering_values = [-1.0, -0.5, 0.0, 0.5, 1.0]
        else:
            # Linearly space between -1 and 1
            steering_values = np.linspace(-1.0, 1.0, self.steering_bins)

        # Define gas/brake combinations
        if self.gas_brake_bins == 3:
            # brake, coast, gas
            gas_brake_values = [
                (0.0, 0.8),  # brake
                (0.0, 0.0),  # coast (no gas, no brake)
                (0.8, 0.0),  # gas
            ]
        elif self.gas_brake_bins == 2:
            # coast, gas (no braking - simpler)
            gas_brake_values = [
                (0.0, 0.0),  # coast
                (0.8, 0.0),  # gas
            ]
        else:
            # Default: coast and gas
            gas_brake_values = [(0.0, 0.0), (0.8, 0.0)]

        # Create all combinations of steering and gas/brake
        for gas, brake in gas_brake_values:
            for steering in steering_values:
                actions.append(np.array([steering, gas, brake], dtype=np.float32))

        return actions

    def action(self, action_idx: int) -> np.ndarray:
        """
        Converts discrete action index to continuous action vector.

        Args:
            action_idx: Discrete action index (0 to n_actions-1)

        Returns:
            Continuous action vector [steering, gas, brake]
        """
        return self.discrete_actions[action_idx]

    def get_action_meanings(self) -> List[str]:
        """Returns human-readable descriptions of each action."""
        meanings = []
        for action in self.discrete_actions:
            steering, gas, brake = action

            # Describe steering
            if steering < -0.3:
                steer_desc = "LEFT"
            elif steering > 0.3:
                steer_desc = "RIGHT"
            else:
                steer_desc = "STRAIGHT"

            # Describe gas/brake
            if brake > 0.1:
                pedal_desc = "BRAKE"
            elif gas > 0.1:
                pedal_desc = "GAS"
            else:
                pedal_desc = "COAST"

            meanings.append(f"{steer_desc}+{pedal_desc}")

        return meanings


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


class ResizeObservation(gym.ObservationWrapper):
    """
    Resizes observations to a target shape.

    Default CarRacing size: 96×96
    We resize to 84×84 to match DQN paper and reduce computation.
    """

    def __init__(self, env, shape=(84, 84)):
        super().__init__(env)
        self.shape = shape
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=shape, dtype=np.uint8
        )

    def observation(self, obs):
        """Resize using bilinear interpolation."""
        resized = cv2.resize(obs, self.shape, interpolation=cv2.INTER_AREA)
        return resized


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


class FrameStack(gym.Wrapper):
    """
    Stacks the last N frames to capture motion and velocity.

    A single frame doesn't show:
    - Direction of car movement
    - Speed of car
    - Trajectory/momentum

    Stacking 4 frames allows the network to infer these temporal features.

    Output shape: (stack_size, height, width)
    Example: (4, 84, 84) for 4 stacked 84×84 frames
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


class StationaryCarTerminator(gym.Wrapper):
    """
    Terminates episodes early if the car is stationary for too long.

    Problem: Agents can learn to brake and sit still to avoid negative rewards
    from going off-track. This wastes compute time and prevents exploration.

    Solution: Track car progress. If no positive rewards (no new tiles visited)
    for a certain number of frames, terminate the episode early.
    """

    def __init__(self, env, patience=100, min_steps=50):
        """
        Args:
            env: The environment to wrap
            patience: Number of frames without progress before terminating (default: 100)
            min_steps: Minimum steps before early termination is allowed (default: 50)
        """
        super().__init__(env)
        self.patience = patience
        self.min_steps = min_steps
        self.frames_since_progress = 0
        self.total_steps = 0

    def reset(self, **kwargs):
        """Reset counters on episode reset."""
        self.frames_since_progress = 0
        self.total_steps = 0
        return self.env.reset(**kwargs)

    def step(self, action):
        """Track progress and terminate if car is stationary."""
        obs, reward, terminated, truncated, info = self.env.step(action)
        self.total_steps += 1

        # Check if car made progress (positive reward = visited new tile)
        if reward > 0:
            self.frames_since_progress = 0
        else:
            self.frames_since_progress += 1

        # Terminate early if:
        # 1. We've taken enough steps (min_steps)
        # 2. No progress for 'patience' frames
        if self.total_steps >= self.min_steps and self.frames_since_progress >= self.patience:
            truncated = True
            info['stationary_termination'] = True

        return obs, reward, terminated, truncated, info


class RewardShaping(gym.RewardWrapper):
    """
    Optional reward shaping for CarRacing.

    Default CarRacing rewards:
    - +1000/N for each track tile visited (N = total tiles)
    - -0.1 for each frame
    - Large negative for going off track

    This wrapper can:
    1. Clip extreme negative rewards (prevent catastrophic penalties)
    2. Scale rewards for better learning dynamics
    """

    def __init__(self, env, clip_reward=True, negative_penalty=-5.0):
        super().__init__(env)
        self.clip_reward = clip_reward
        self.negative_penalty = negative_penalty

    def reward(self, reward):
        """Shape the reward."""
        if self.clip_reward:
            # Clip large negative rewards (e.g., from going off track)
            # This prevents single bad decisions from dominating learning
            if reward < 0:
                reward = max(reward, self.negative_penalty)

        return reward


def make_carracing_env(
    stack_size=4,
    frame_size=(84, 84),
    discretize_actions=True,
    steering_bins=3,
    gas_brake_bins=3,
    shape_rewards=True,
    terminate_stationary=True,
    stationary_patience=100,
    render_mode=None
) -> gym.Env:
    """
    Creates a preprocessed CarRacing-v3 environment for DQN training.

    Args:
        stack_size: Number of frames to stack (default: 4)
        frame_size: Target frame size (default: 84×84)
        discretize_actions: Whether to discretize the action space (default: True)
        steering_bins: Number of discrete steering values (default: 3)
        gas_brake_bins: Number of discrete gas/brake combinations (default: 3)
        shape_rewards: Whether to apply reward shaping (default: True)
        terminate_stationary: Terminate early if car is stationary (default: True)
        stationary_patience: Frames without progress before termination (default: 100)
        render_mode: Rendering mode ('rgb_array', 'human', or None)

    Returns:
        Wrapped environment ready for DQN training
    """
    # Create base environment using local CarRacing class
    env = CarRacing(render_mode=render_mode, continuous=True)

    # Apply wrappers in order
    if discretize_actions:
        env = ActionDiscretizer(env, steering_bins=steering_bins, gas_brake_bins=gas_brake_bins)

    if terminate_stationary:
        env = StationaryCarTerminator(env, patience=stationary_patience)

    if shape_rewards:
        env = RewardShaping(env)

    # Frame preprocessing
    env = GrayscaleWrapper(env)
    env = ResizeObservation(env, shape=frame_size)
    env = NormalizeObservation(env)

    # Frame stacking (must be last to stack preprocessed frames)
    env = FrameStack(env, stack_size=stack_size)

    return env


if __name__ == "__main__":
    """Test preprocessing pipeline."""
    print("Creating CarRacing environment with preprocessing...")
    env = make_carracing_env(render_mode='rgb_array')

    print(f"Original CarRacing action space: [steering, gas, brake]")
    print(f"Discretized action space: {env.action_space}")
    print(f"Number of discrete actions: {env.action_space.n}")
    print(f"\nAction meanings:")
    for i, meaning in enumerate(env.unwrapped.get_action_meanings()):
        print(f"  {i}: {meaning}")

    print(f"\nObservation space: {env.observation_space}")
    print(f"Expected shape: (4, 84, 84) for 4 stacked 84×84 grayscale frames")

    # Test environment
    print("\nTesting environment reset...")
    obs, info = env.reset()
    print(f"Observation shape: {obs.shape}")
    print(f"Observation dtype: {obs.dtype}")
    print(f"Observation range: [{obs.min():.3f}, {obs.max():.3f}]")

    print("\nTesting random actions...")
    for i in range(5):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        print(f"Step {i+1}: action={action}, reward={reward:.3f}, done={terminated or truncated}")

    env.close()
    print("\nPreprocessing test complete!")
