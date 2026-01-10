"""
Gymnasium environment for 10x10 International Draughts.
"""

import gymnasium as gym
from gymnasium import spaces
import numpy as np
from typing import Tuple, Optional, Dict, Any
from ..engine.game import CheckersGame
from ..engine.moves import Move


class CheckersEnv(gym.Env):
    """
    Gymnasium environment for International Draughts.

    Action space: Discrete(150) - one action for each possible move
    Observation space: Box(8, 10, 10) - neural network input format

    The environment handles two-player games by alternating turns.
    """

    metadata = {"render_modes": ["human", "ansi"], "render_fps": 1}

    def __init__(self, render_mode: Optional[str] = None):
        super().__init__()

        self.render_mode = render_mode

        # Action space: we encode up to 150 possible moves
        # Actual legal moves are masked during action selection
        self.action_space = spaces.Discrete(150)

        # Observation space: 8 planes of 10x10
        self.observation_space = spaces.Box(
            low=0.0,
            high=1.0,
            shape=(8, 10, 10),
            dtype=np.float32
        )

        # Game state
        self.game: Optional[CheckersGame] = None
        self._move_list: list[Move] = []

    def _get_obs(self) -> np.ndarray:
        """Get current observation (neural network input)."""
        return self.game.to_neural_input()

    def _get_info(self) -> Dict[str, Any]:
        """Get auxiliary information."""
        legal_moves = self.game.get_legal_moves()
        self._move_list = legal_moves

        # Create legal moves mask
        legal_actions_mask = np.zeros(150, dtype=np.bool_)
        for i in range(min(len(legal_moves), 150)):
            legal_actions_mask[i] = True

        return {
            "legal_moves": legal_moves,
            "legal_actions_mask": legal_actions_mask,
            "num_legal_moves": len(legal_moves),
            "current_player": self.game.current_player,
            "move_count": self.game.move_count,
        }

    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Reset the environment to initial state."""
        super().reset(seed=seed)

        self.game = CheckersGame()
        self._move_list = []

        observation = self._get_obs()
        info = self._get_info()

        return observation, info

    def step(
        self, action: int
    ) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """
        Execute one move in the environment.

        Args:
            action: Integer index into the legal moves list

        Returns:
            observation: Neural network input (8, 10, 10)
            reward: +1.0 for win, -1.0 for loss, 0.0 for ongoing/draw
            terminated: True if game is over
            truncated: Always False (no time limits)
            info: Auxiliary information
        """
        if self.game is None:
            raise RuntimeError("Environment not initialized. Call reset() first.")

        # Get legal moves
        legal_moves = self.game.get_legal_moves()
        if not legal_moves:
            # No legal moves - current player loses
            observation = self._get_obs()
            info = self._get_info()
            return observation, -1.0, True, False, info

        # Validate action
        if action < 0 or action >= len(legal_moves):
            raise ValueError(
                f"Invalid action {action}. Must be in range [0, {len(legal_moves)})"
            )

        # Apply the move
        move = legal_moves[action]
        self.game.make_move(move)

        # Check if game is over
        terminated = self.game.is_terminal()
        reward = 0.0

        if terminated:
            # Game ended - get result from previous player's perspective
            # (since we just switched perspective)
            result = self.game.get_result()
            # Flip result because it's from new current player's perspective
            reward = -result

        observation = self._get_obs()
        info = self._get_info()

        return observation, reward, terminated, False, info

    def render(self):
        """Render the current game state."""
        if self.game is None:
            return

        if self.render_mode == "human" or self.render_mode == "ansi":
            return self.game.render()

    def close(self):
        """Clean up resources."""
        pass

    def action_to_move(self, action: int) -> Optional[Move]:
        """Convert action index to Move object."""
        if 0 <= action < len(self._move_list):
            return self._move_list[action]
        return None

    def move_to_action(self, move: Move) -> int:
        """Convert Move object to action index."""
        try:
            return self._move_list.index(move)
        except ValueError:
            return -1


def play_random_episode(env: CheckersEnv, max_steps: int = 200) -> Tuple[int, int]:
    """
    Play a random episode for testing.

    Returns:
        (winner, num_steps)
    """
    obs, info = env.reset()
    steps = 0

    while steps < max_steps:
        # Sample random legal action
        num_legal = info["num_legal_moves"]
        if num_legal == 0:
            break

        action = env.action_space.sample() % num_legal
        obs, reward, terminated, truncated, info = env.step(action)

        steps += 1

        if terminated or truncated:
            break

    winner = env.game.get_winner()
    return winner, steps


# Example usage
if __name__ == "__main__":
    env = CheckersEnv(render_mode="human")

    print("Playing random episode...")
    obs, info = env.reset()
    print(env.render())
    print(f"Legal moves: {info['num_legal_moves']}")

    done = False
    step_count = 0
    max_steps = 10

    while not done and step_count < max_steps:
        # Random action from legal moves
        num_legal = info["num_legal_moves"]
        action = env.action_space.sample() % num_legal

        obs, reward, terminated, truncated, info = env.step(action)

        step_count += 1
        done = terminated or truncated

        print(f"\nStep {step_count}:")
        print(env.render())
        print(f"Reward: {reward}, Done: {done}")
        print(f"Legal moves: {info['num_legal_moves']}")

    print(f"\nEpisode finished in {step_count} steps")
    print(f"Winner: {env.game.get_winner()}")

    env.close()
