"""
Atari Frame Preprocessing

Atari games output 210x160 RGB frames at 60 Hz. This is too large and high-frequency
for efficient learning. Standard preprocessing includes:
1. Grayscale conversion (color is not important for most games)
2. Downsampling to 84x84 (reduces computation)
3. Frame stacking (stacking 4 frames to capture motion/velocity)
"""

import numpy as np
import gymnasium as gym
from collections import deque
import cv2

# Register ALE environments (required for gymnasium 1.0+)
try:
    import ale_py
    gym.register_envs(ale_py)
except Exception:
    pass  # ALE environments might already be registered


class AtariPreprocessing(gym.Wrapper):
    """
    Preprocessing wrapper for Atari environments

    This wrapper applies standard Atari preprocessing:
    - Grayscale conversion
    - Resizing to 84x84
    - Frame stacking (default 4 frames)
    """

    def __init__(self, env, frame_stack=4, screen_size=84):
        super().__init__(env)
        self.frame_stack = frame_stack
        self.screen_size = screen_size

        # Buffer to store stacked frames
        self.frames = deque(maxlen=frame_stack)

        # Update observation space
        self.observation_space = gym.spaces.Box(
            low=0,
            high=255,
            shape=(frame_stack, screen_size, screen_size),
            dtype=np.uint8
        )

    def reset(self, **kwargs):
        """Reset environment and initialize frame stack"""
        obs, info = self.env.reset(**kwargs)
        frame = self._preprocess_frame(obs)

        # Initialize stack with the same frame repeated
        for _ in range(self.frame_stack):
            self.frames.append(frame)

        return self._get_observation(), info

    def step(self, action):
        """Take a step and update frame stack"""
        obs, reward, terminated, truncated, info = self.env.step(action)
        frame = self._preprocess_frame(obs)
        self.frames.append(frame)

        return self._get_observation(), reward, terminated, truncated, info

    def _preprocess_frame(self, frame):
        """
        Preprocess a single frame

        Args:
            frame: Raw frame from environment (210, 160, 3)
        Returns:
            Preprocessed frame (84, 84)
        """
        # Convert to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)

        # Resize to 84x84
        resized = cv2.resize(gray, (self.screen_size, self.screen_size),
                           interpolation=cv2.INTER_AREA)

        return resized

    def _get_observation(self):
        """Get current observation (stacked frames)"""
        # Stack frames along first dimension: (frame_stack, 84, 84)
        return np.array(self.frames, dtype=np.uint8)


class RewardClipping(gym.Wrapper):
    """
    Clip rewards to {-1, 0, +1}

    This is a common technique in DQN for Atari games. It makes learning more stable
    by ensuring all rewards have the same scale across different games.
    """

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        # Clip reward to -1, 0, or 1
        reward = np.sign(reward)
        return obs, reward, terminated, truncated, info


def make_atari_env(env_name='ALE/Breakout-v5', frame_stack=4, clip_rewards=True):
    """
    Create and wrap an Atari environment with standard preprocessing

    Args:
        env_name: Name of Atari environment
        frame_stack: Number of frames to stack
        clip_rewards: Whether to clip rewards to {-1, 0, 1}

    Returns:
        Preprocessed Gymnasium environment
    """
    # Create base environment
    env = gym.make(env_name, render_mode='rgb_array')

    # Apply preprocessing
    env = AtariPreprocessing(env, frame_stack=frame_stack)

    if clip_rewards:
        env = RewardClipping(env)

    return env
