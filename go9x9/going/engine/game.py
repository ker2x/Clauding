"""
Game state management for 9x9 Go.

No perspective switching — board stays in absolute coordinates.
Neural input uses "my stones" / "opponent stones" planes (AlphaGo Zero style).
"""

import numpy as np
from typing import List, Optional
from collections import deque

try:
    from .board import (
        BOARD_SIZE, EMPTY, BLACK, WHITE, opponent,
        get_legal_moves, apply_move, is_suicide, NEIGHBORS, find_group
    )
    from .zobrist import compute_hash, update_hash, toggle_side
    from .scoring import score_game
    from .action_encoder import (
        NUM_ACTIONS, PASS_ACTION, NUM_INTERSECTIONS,
        action_to_pos, pos_to_action
    )
except ImportError:
    from board import (
        BOARD_SIZE, EMPTY, BLACK, WHITE, opponent,
        get_legal_moves, apply_move, is_suicide, NEIGHBORS, find_group
    )
    from zobrist import compute_hash, update_hash, toggle_side
    from scoring import score_game
    from action_encoder import (
        NUM_ACTIONS, PASS_ACTION, NUM_INTERSECTIONS,
        action_to_pos, pos_to_action
    )

# Neural network input planes
NUM_HISTORY = 2  # 2 timesteps sufficient for ko context on 9x9
NUM_PLANES = NUM_HISTORY * 2  # 4 planes: my stones + opp stones × 2 history


class GoGame:
    """
    9x9 Go game state.

    Board stays in absolute coordinates. Neural input encodes
    "current player's stones" and "opponent's stones" relative to
    who is to move, following AlphaGo Zero.
    """

    def __init__(self):
        """Initialize empty board with black to play."""
        self.board = np.zeros(81, dtype=np.int8)
        self.current_player = BLACK
        self.ko_point = None  # Position illegal due to simple ko
        self.move_count = 0
        self.max_moves = 200
        self.komi = 7.5
        self.consecutive_passes = 0

        # Superko: set of (board_hash, player) seen
        self.zobrist_hash = compute_hash(self.board, self.current_player)
        self.hash_history = set()
        self.hash_history.add(self.zobrist_hash)

        # Board history for neural input (store board snapshots)
        # Each entry is a flat copy of the board at that point
        self.board_history = deque(maxlen=NUM_HISTORY)
        self.board_history.append(self.board.copy())

        # Captures count
        self.black_captures = 0
        self.white_captures = 0

    def clone(self) -> 'GoGame':
        """Create a deep copy of the game state."""
        game = GoGame.__new__(GoGame)
        game.board = self.board.copy()
        game.current_player = self.current_player
        game.ko_point = self.ko_point
        game.move_count = self.move_count
        game.max_moves = self.max_moves
        game.komi = self.komi
        game.consecutive_passes = self.consecutive_passes
        game.zobrist_hash = self.zobrist_hash
        game.hash_history = self.hash_history.copy()
        game.board_history = deque(
            [b.copy() for b in self.board_history],
            maxlen=NUM_HISTORY
        )
        game.black_captures = self.black_captures
        game.white_captures = self.white_captures
        return game

    def get_legal_actions(self) -> List[int]:
        """
        Get legal action indices for current player.

        Returns:
            List of valid action indices (0-81), always includes PASS (81).
        """
        legal_positions = get_legal_moves(
            self.board, self.current_player, self.ko_point
        )

        # Filter out superko violations
        actions = []
        for pos in legal_positions:
            if not self._violates_superko(pos):
                actions.append(pos)

        # Pass is always legal
        actions.append(PASS_ACTION)
        return actions

    def _violates_superko(self, pos: int) -> bool:
        """Check if placing a stone at pos would violate positional superko."""
        # Simulate the move to compute resulting hash
        test_board = self.board.copy()
        test_board[pos] = self.current_player

        # Find captures
        opp = opponent(self.current_player)
        h = self.zobrist_hash
        # Place stone
        h = update_hash(h, pos, EMPTY, self.current_player)

        # Check and remove captured groups
        for nb in NEIGHBORS[pos]:
            if test_board[nb] == opp:
                group, liberties = find_group(test_board, nb)
                if len(liberties) == 0:
                    for p in group:
                        h = update_hash(h, p, opp, EMPTY)
                        test_board[p] = EMPTY

        # Toggle side
        h = toggle_side(h)

        return h in self.hash_history

    def make_action(self, action: int) -> bool:
        """
        Apply action to game state.

        Args:
            action: Action index (0-81)

        Returns:
            True if action was legal and applied
        """
        if action == PASS_ACTION:
            return self._make_pass()

        pos = action
        if pos < 0 or pos >= NUM_INTERSECTIONS:
            return False

        # Check legality
        if self.board[pos] != EMPTY:
            return False
        if pos == self.ko_point:
            return False
        if is_suicide(self.board, pos, self.current_player):
            return False

        # Apply move
        color = self.current_player
        self.board, num_captured, new_ko = apply_move(self.board, pos, color)

        # Update captures
        if color == BLACK:
            self.black_captures += num_captured
        else:
            self.white_captures += num_captured

        self.ko_point = new_ko
        self.consecutive_passes = 0

        # Update Zobrist hash (recompute for correctness)
        self.current_player = opponent(color)
        self.zobrist_hash = compute_hash(self.board, self.current_player)
        self.hash_history.add(self.zobrist_hash)

        # Update history
        self.board_history.append(self.board.copy())
        self.move_count += 1

        return True

    def _make_pass(self) -> bool:
        """Apply a pass move."""
        self.ko_point = None
        self.consecutive_passes += 1
        self.current_player = opponent(self.current_player)

        # Update hash for new player
        self.zobrist_hash = compute_hash(self.board, self.current_player)

        # History keeps same board (pass doesn't change board)
        self.board_history.append(self.board.copy())
        self.move_count += 1

        return True

    def is_terminal(self) -> bool:
        """Check if game has ended."""
        # Two consecutive passes
        if self.consecutive_passes >= 2:
            return True
        # Move limit
        if self.move_count >= self.max_moves:
            return True
        return False

    def get_result(self) -> float:
        """
        Get game result from current player's perspective.

        Returns:
            1.0 if current player won
            -1.0 if current player lost
            0.0 for draw (shouldn't happen with 0.5 komi)
        """
        if not self.is_terminal():
            return 0.0

        _, _, result_str = score_game(self.board, self.komi)

        if result_str == "0":
            return 0.0

        # Parse result
        winner_color = BLACK if result_str.startswith("B") else WHITE

        if winner_color == self.current_player:
            return 1.0
        else:
            return -1.0

    def get_score_string(self) -> str:
        """Get the score result string (e.g. 'B+2.5')."""
        _, _, result_str = score_game(self.board, self.komi)
        return result_str

    def to_neural_input(self) -> np.ndarray:
        """
        Convert game state to neural network input (4 planes × 9 × 9).

        Planes 0-1:   Current player's stones over last 2 timesteps
        Planes 2-3:   Opponent's stones over last 2 timesteps

        Returns:
            numpy array of shape (4, 9, 9)
        """
        planes = np.zeros((NUM_PLANES, BOARD_SIZE, BOARD_SIZE), dtype=np.float32)

        my_color = self.current_player
        opp_color = opponent(my_color)

        # Fill history planes (most recent first)
        history_list = list(self.board_history)
        for t in range(NUM_HISTORY):
            if t < len(history_list):
                hist_board = history_list[-(t + 1)]
            else:
                # No history available, leave as zeros
                continue

            my_plane = (hist_board == my_color).reshape(BOARD_SIZE, BOARD_SIZE).astype(np.float32)
            opp_plane = (hist_board == opp_color).reshape(BOARD_SIZE, BOARD_SIZE).astype(np.float32)

            planes[t] = my_plane
            planes[NUM_HISTORY + t] = opp_plane

        return planes

    def render(self) -> str:
        """Render board as ASCII string."""
        lines = []
        lines.append(f"\n  A B C D E F G H J")
        for row in range(BOARD_SIZE - 1, -1, -1):
            line = f"{row + 1} "
            for col in range(BOARD_SIZE):
                pos = row * BOARD_SIZE + col
                stone = self.board[pos]
                if stone == BLACK:
                    line += "X "
                elif stone == WHITE:
                    line += "O "
                else:
                    line += ". "
            line += f"{row + 1}"
            lines.append(line)
        lines.append(f"  A B C D E F G H J")
        lines.append(f"")
        player_str = "Black(X)" if self.current_player == BLACK else "White(O)"
        lines.append(f"{player_str} to move | Move {self.move_count} | "
                     f"Captures B:{self.black_captures} W:{self.white_captures}")
        return "\n".join(lines)
