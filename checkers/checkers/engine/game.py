"""
Main game engine for 10x10 International Draughts.

Handles game state, move application, and game termination detection.
"""

import numpy as np
from typing import List, Optional, Tuple
from copy import deepcopy
from .bitboard import *
from .moves import Move, get_legal_moves


class CheckersGame:
    """
    10x10 International Draughts game state.

    State representation:
    - Bitboards for player_men, player_kings, opponent_men, opponent_kings
    - Current player perspective (always plays from "player" side)
    - Move history for draw detection
    """

    def __init__(self):
        """Initialize a new game at starting position."""
        # Player 1 starts at bottom (squares 0-19)
        self.player_men = INITIAL_PLAYER1
        self.player_kings = 0
        self.opponent_men = INITIAL_PLAYER2
        self.opponent_kings = 0

        # Game state
        self.current_player = 1  # 1 or 2
        self.move_count = 0
        self.moves_since_capture = 0

        # History for draw detection (store position hashes)
        self.position_history: List[int] = []
        self._update_position_hash()

    def clone(self) -> 'CheckersGame':
        """Create a deep copy of the game state."""
        new_game = CheckersGame()
        new_game.player_men = self.player_men
        new_game.player_kings = self.player_kings
        new_game.opponent_men = self.opponent_men
        new_game.opponent_kings = self.opponent_kings
        new_game.current_player = self.current_player
        new_game.move_count = self.move_count
        new_game.moves_since_capture = self.moves_since_capture
        new_game.position_history = self.position_history.copy()
        return new_game

    def _update_position_hash(self):
        """Compute and store hash of current position."""
        # Simple hash based on bitboard values
        position_hash = hash((
            self.player_men,
            self.player_kings,
            self.opponent_men,
            self.opponent_kings,
            self.current_player
        ))
        self.position_history.append(position_hash)

    def _count_position_repetitions(self) -> int:
        """Count how many times current position has occurred."""
        if not self.position_history:
            return 0
        current_hash = self.position_history[-1]
        return self.position_history.count(current_hash)

    def get_legal_moves(self) -> List[Move]:
        """Get all legal moves for current player."""
        return get_legal_moves(
            self.player_men,
            self.player_kings,
            self.opponent_men,
            self.opponent_kings
        )

    def make_move(self, move: Move) -> None:
        """
        Apply a move to the game state.
        Switches perspective after move (opponent becomes player).
        """
        # Remove piece from source
        self.player_men = clear_bit(self.player_men, move.from_square)
        self.player_kings = clear_bit(self.player_kings, move.from_square)

        # Remove captured pieces
        for captured_square in move.captured_squares:
            self.opponent_men = clear_bit(self.opponent_men, captured_square)
            self.opponent_kings = clear_bit(self.opponent_kings, captured_square)

        # Place piece at destination
        if move.promotes_to_king:
            self.player_kings = set_bit(self.player_kings, move.to_square)
        else:
            # Check if it was already a king
            was_king = get_bit(self.player_kings, move.from_square)
            if was_king:
                self.player_kings = set_bit(self.player_kings, move.to_square)
            else:
                self.player_men = set_bit(self.player_men, move.to_square)

        # Update game state
        self.move_count += 1

        if move.captured_squares:
            self.moves_since_capture = 0
        else:
            self.moves_since_capture += 1

        # Switch perspective (swap player and opponent, then flip board)
        self.player_men, self.opponent_men = self.opponent_men, self.player_men
        self.player_kings, self.opponent_kings = self.opponent_kings, self.player_kings

        # Flip all bitboards so current player is always at bottom
        self.player_men = flip_bitboard(self.player_men)
        self.player_kings = flip_bitboard(self.player_kings)
        self.opponent_men = flip_bitboard(self.opponent_men)
        self.opponent_kings = flip_bitboard(self.opponent_kings)

        self.current_player = 3 - self.current_player  # Switch between 1 and 2

        # Update position history
        self._update_position_hash()

    def is_terminal(self) -> bool:
        """Check if game is over."""
        # No legal moves (loss)
        if not self.get_legal_moves():
            return True

        # Draw by repetition (three-fold)
        if self._count_position_repetitions() >= 3:
            return True

        # Draw by 50-move rule (no captures for 50 moves)
        if self.moves_since_capture >= 50:
            return True

        return False

    def get_result(self) -> float:
        """
        Get game result from current player's perspective.
        Returns: 1.0 (win), -1.0 (loss), 0.0 (draw)
        """
        if not self.is_terminal():
            return 0.0  # Game not over

        # Check for draw conditions first
        if self._count_position_repetitions() >= 3:
            return 0.0  # Draw by repetition

        if self.moves_since_capture >= 50:
            return 0.0  # Draw by 50-move rule

        # No legal moves = current player lost
        if not self.get_legal_moves():
            return -1.0

        return 0.0  # Default to draw

    def get_winner(self) -> Optional[int]:
        """
        Get the winner of the game (1 or 2), or None for draw/ongoing.
        """
        if not self.is_terminal():
            return None

        result = self.get_result()
        if result == 1.0:
            return self.current_player
        elif result == -1.0:
            return 3 - self.current_player  # Opponent won
        else:
            return None  # Draw

    def get_piece_counts(self) -> Tuple[int, int, int, int]:
        """Return (player_men, player_kings, opponent_men, opponent_kings) counts."""
        return (
            count_bits(self.player_men),
            count_bits(self.player_kings),
            count_bits(self.opponent_men),
            count_bits(self.opponent_kings)
        )

    def to_array(self) -> np.ndarray:
        """Convert game state to 10x10 numpy array for visualization."""
        return bitboards_to_array(
            self.player_men,
            self.player_kings,
            self.opponent_men,
            self.opponent_kings
        )

    def to_neural_input(self) -> np.ndarray:
        """
        Convert game state to neural network input format.
        Returns: (8, 10, 10) array with the following planes:
        - Plane 0: Current player men
        - Plane 1: Current player kings
        - Plane 2: Opponent men
        - Plane 3: Opponent kings
        - Plane 4: Valid move destination squares
        - Plane 5: Two-fold repetition indicator
        - Plane 6: Three-fold repetition indicator (draw)
        - Plane 7: Constant bias plane (all 1.0)
        """
        planes = np.zeros((8, BOARD_SIZE, BOARD_SIZE), dtype=np.float32)

        # Planes 0-3: Piece positions
        for square in range(NUM_SQUARES):
            row, col = square_to_row_col(square)

            if get_bit(self.player_men, square):
                planes[0, row, col] = 1.0
            if get_bit(self.player_kings, square):
                planes[1, row, col] = 1.0
            if get_bit(self.opponent_men, square):
                planes[2, row, col] = 1.0
            if get_bit(self.opponent_kings, square):
                planes[3, row, col] = 1.0

        # Plane 4: Valid move destinations
        legal_moves = self.get_legal_moves()
        for move in legal_moves:
            row, col = square_to_row_col(move.to_square)
            planes[4, row, col] = 1.0

        # Planes 5-6: Repetition indicators
        repetitions = self._count_position_repetitions()
        if repetitions >= 2:
            planes[5, :, :] = 1.0
        if repetitions >= 3:
            planes[6, :, :] = 1.0

        # Plane 7: Constant bias
        planes[7, :, :] = 1.0

        return planes

    def render(self) -> str:
        """Render the board as ASCII art."""
        board = self.to_array()
        lines = ["\n  0 1 2 3 4 5 6 7 8 9"]

        piece_chars = {
            0: ".",   # Empty
            1: "w",   # White man (current player)
            2: "W",   # White king
            -1: "b",  # Black man (opponent)
            -2: "B",  # Black king
        }

        for row in range(BOARD_SIZE):
            line = f"{row} "
            for col in range(BOARD_SIZE):
                if (row + col) % 2 == 0:
                    line += "  "  # Light square
                else:
                    value = board[row, col]
                    line += piece_chars[value] + " "
            lines.append(line)

        lines.append(f"\nPlayer {self.current_player} to move")
        lines.append(f"Pieces - Player: {count_bits(self.player_men | self.player_kings)}, "
                    f"Opponent: {count_bits(self.opponent_men | self.opponent_kings)}")

        return "\n".join(lines)

    def __str__(self) -> str:
        return self.render()


def play_random_game(max_moves: int = 200) -> Tuple[Optional[int], int]:
    """
    Play a random game for testing.
    Returns: (winner, num_moves)
    """
    import random

    game = CheckersGame()
    moves_played = 0

    while not game.is_terminal() and moves_played < max_moves:
        legal_moves = game.get_legal_moves()
        if not legal_moves:
            break

        move = random.choice(legal_moves)
        game.make_move(move)
        moves_played += 1

    winner = game.get_winner()
    return winner, moves_played
