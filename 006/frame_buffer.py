"""
Frame buffer for stacking observations during environment interaction.

This is used during rollouts to provide stacked observations to the agent,
complementing the replay buffer's frame stacking during training.
"""

from __future__ import annotations

from collections import deque
import numpy as np
import numpy.typing as npt


class FrameBuffer:
    """
    Buffer for stacking consecutive frames during environment interaction.

    Unlike the replay buffer which stacks frames during sampling for training,
    this buffer maintains a sliding window of recent observations for action selection.

    Args:
        frame_stack: Number of frames to stack
        state_shape: Shape of individual frames

    Usage:
        buffer = FrameBuffer(frame_stack=4, state_shape=53)
        buffer.reset(initial_state)  # Initialize at episode start
        stacked = buffer.get()  # Get current stacked observation (212D)
        buffer.append(next_state)  # Add new frame
    """

    def __init__(self, frame_stack: int, state_shape: int | tuple[int, ...]) -> None:
        self.frame_stack = frame_stack
        self.state_shape = state_shape if isinstance(state_shape, tuple) else (state_shape,)
        self.frames: deque[npt.NDArray[np.float32]] = deque(maxlen=frame_stack)

    def reset(self, initial_state: npt.NDArray[np.float32]) -> None:
        """
        Reset buffer with initial state at episode start.

        Fills buffer by repeating the initial state (same as replay buffer padding).

        Args:
            initial_state: First observation of the episode
        """
        self.frames.clear()
        for _ in range(self.frame_stack):
            self.frames.append(initial_state.copy())

    def append(self, state: npt.NDArray[np.float32]) -> None:
        """
        Add a new frame to the buffer.

        Automatically maintains the sliding window (oldest frame is dropped).

        Args:
            state: New observation to add
        """
        self.frames.append(state.copy())

    def get(self) -> npt.NDArray[np.float32]:
        """
        Get current stacked observation.

        Returns:
            Stacked frames as a flat array (frame_stack * state_dim,)
        """
        if not self.frames:
            raise RuntimeError("Frame buffer is empty. Call reset() first.")

        # Stack frames: oldest to newest
        stacked = np.concatenate(list(self.frames), axis=0)
        return stacked

    def __len__(self) -> int:
        """Return number of frames currently in buffer."""
        return len(self.frames)
