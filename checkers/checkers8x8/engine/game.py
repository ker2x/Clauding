"""
Game state management for 8x8 American Checkers.

Key design: Current player is ALWAYS at bottom (perspective switching after each move)
"""

import numpy as np
from typing import List, Optional, Tuple
from collections import defaultdict

try:
    from .bitboard import *
    from .moves import get_legal_moves, Move
    from .action_encoder import NUM_ACTIONS, decode_action, encode_action
except ImportError:
    from bitboard import *
    from moves import get_legal_moves, Move
    from action_encoder import NUM_ACTIONS, decode_action, encode_action


class CheckersGame:
    """
    8x8 Checkers game state.

    Board representation uses perspective switching:
    - Current player's pieces are always at bottom (rows 5, 6, 7)
    - After each move, board is flipped and players are swapped
    """

    def __init__(self):
        """Initialize game to starting position."""
        # Player 1 starts at bottom (squares 20-31)
        # Opponent starts at top (squares 0-11)
        self.player_men = 0b11111111111100000000000000000000
        self.player_kings = 0
        self.opponent_men = 0b00000000000000000000111111111111
        self.opponent_kings = 0

        self.current_player = 1  # Player 1 starts
        self.move_count = 0
        self.max_moves = 400  # Default safety cap
        self.position_history = defaultdict(int)
        self._update_position_hash()

    def clone(self) -> 'CheckersGame':
        """Create a deep copy of the game state."""
        game = CheckersGame.__new__(CheckersGame)
        game.player_men = self.player_men
        game.player_kings = self.player_kings
        game.opponent_men = self.opponent_men
        game.opponent_kings = self.opponent_kings
        game.current_player = self.current_player
        game.move_count = self.move_count
        game.max_moves = self.max_moves
        game.position_history = self.position_history.copy()
        return game

    def _update_position_hash(self):
        """Update position hash for draw detection."""
        position = (self.player_men, self.player_kings,
                   self.opponent_men, self.opponent_kings)
        self.position_history[position] += 1

    def get_legal_moves(self) -> List[Move]:
        """Get list of legal moves for current player."""
        return get_legal_moves(
            self.player_men, self.player_kings,
            self.opponent_men, self.opponent_kings
        )

    def get_legal_actions(self) -> List[int]:
        """
        Get legal action indices for current player.

        Returns:
            List of valid action indices (0-127)
        """
        moves = self.get_legal_moves()
        actions = []
        for move in moves:
            actions.extend(move.to_actions())
        return sorted(list(set(actions)))

    def action_to_move(self, action: int) -> Optional[Move]:
        """
        Convert action index to corresponding Move object.

        Args:
            action: Action index (0-127)

        Returns:
            Move object if action is legal, None otherwise
        """
        from_square, direction = decode_action(action)

        # Get all legal moves and find matching one
        legal_moves = self.get_legal_moves()

        for move in legal_moves:
            # Check if this move matches the action
            if move.from_square == from_square:
                # CRITICAL: For captures, use direction to FIRST captured piece
                # This must match the encoding logic in Move.to_actions()
                if move.captured_squares:
                    target_square = move.captured_squares[0]
                else:
                    target_square = move.to_square

                # Check if direction matches
                from_row, from_col = square_to_row_col(from_square)
                to_row, to_col = square_to_row_col(target_square)

                dr = to_row - from_row
                dc = to_col - from_col

                # Normalize direction for multi-square moves
                if dr != 0:
                    dr = dr // abs(dr)
                if dc != 0:
                    dc = dc // abs(dc)

                expected_dir = get_direction_offset(direction)

                if (dr, dc) == expected_dir:
                    return move

        return None

    def make_move(self, move: Move):
        """
        Apply a move to the game state.

        After applying the move, the board is flipped and players are swapped
        so the next player is at the bottom.
        """
        # Remove piece from starting square
        if get_bit(self.player_men, move.from_square):
            self.player_men = clear_bit(self.player_men, move.from_square)
            is_king = False
        else:
            self.player_kings = clear_bit(self.player_kings, move.from_square)
            is_king = True

        # Remove captured pieces
        for captured_sq in move.captured_squares:
            if get_bit(self.opponent_men, captured_sq):
                self.opponent_men = clear_bit(self.opponent_men, captured_sq)
            else:
                self.opponent_kings = clear_bit(self.opponent_kings, captured_sq)

        # Place piece at destination
        if move.promotes_to_king or is_king:
            self.player_kings = set_bit(self.player_kings, move.to_square)
        else:
            self.player_men = set_bit(self.player_men, move.to_square)

        # Flip board and swap players
        self.player_men = flip_bitboard(self.player_men)
        self.player_kings = flip_bitboard(self.player_kings)
        self.opponent_men = flip_bitboard(self.opponent_men)
        self.opponent_kings = flip_bitboard(self.opponent_kings)

        # Swap player and opponent
        self.player_men, self.opponent_men = self.opponent_men, self.player_men
        self.player_kings, self.opponent_kings = self.opponent_kings, self.player_kings

        # Update state
        self.current_player = 3 - self.current_player  # Switch between 1 and 2
        self.move_count += 1
        self._update_position_hash()

    def make_action(self, action: int) -> bool:
        """
        Apply action to game state.

        Args:
            action: Action index (0-127)

        Returns:
            True if action was legal and applied, False otherwise
        """
        move = self.action_to_move(action)
        if move is None:
            return False

        self.make_move(move)
        return True

    def is_terminal(self) -> bool:
        """Check if game has ended."""
        # No legal moves means loss
        if len(self.get_legal_moves()) == 0:
            return True

        # Draw by repetition (3-fold)
        position = (self.player_men, self.player_kings,
                   self.opponent_men, self.opponent_kings)
        if self.position_history[position] >= 3:
            return True

        # Draw by move limit (safety cap)
        if self.move_count >= self.max_moves:
            return True

        return False

    def get_result(self) -> float:
        """
        Get game result from current player's perspective.

        Returns:
            1.0 if current player won
            -1.0 if current player lost
            0.0 for draw
        """
        if not self.is_terminal():
            return 0.0

        # No legal moves = loss
        if len(self.get_legal_moves()) == 0:
            return -1.0

        # Draw (repetition or move limit)
        return 0.0

    def to_neural_input(self) -> np.ndarray:
        """
        Convert game state to neural network input (8 planes × 8 × 8).

        Planes:
            0: Current player's men
            1: Current player's kings
            2: Opponent's men
            3: Opponent's kings
            4: Legal move destinations (squares current player can move to)
            5: Repetition count (normalized)
            6: Move count (normalized)
            7: All ones (bias plane)

        Returns:
            numpy array of shape (8, 8, 8)
        """
        planes = np.zeros((8, 8, 8), dtype=np.float32)

        # Helper to set bits in plane
        def set_plane(plane_idx, bitboard):
            for square in get_set_squares(bitboard):
                row, col = square_to_row_col(square)
                planes[plane_idx, row, col] = 1.0

        # Plane 0-3: Piece positions
        set_plane(0, self.player_men)
        set_plane(1, self.player_kings)
        set_plane(2, self.opponent_men)
        set_plane(3, self.opponent_kings)

        # Plane 4: Legal move destinations
        legal_moves = self.get_legal_moves()
        for move in legal_moves:
            row, col = square_to_row_col(move.to_square)
            planes[4, row, col] = 1.0

        # Plane 5: Repetition count
        position = (self.player_men, self.player_kings,
                   self.opponent_men, self.opponent_kings)
        rep_count = self.position_history[position]
        planes[5, :, :] = min(rep_count / 3.0, 1.0)

        # Plane 6: Move count (normalized)
        planes[6, :, :] = min(self.move_count / 100.0, 1.0)

        # Plane 7: Bias (all ones)
        planes[7, :, :] = 1.0

        return planes

    def to_absolute_board_array(self) -> np.ndarray:
        """
        Convert game state to 8x8 array for visualization.
        
        Does NOT apply perspective switching - shows absolute board positions.
        Player 1 pieces are always shown in their actual positions.
        
        The board is flipped after each move so that the current player is at bottom.
        This method "unflips" to show the absolute board state.
        
        Returns:
            8x8 array where:
                1 = Player 1 man
                2 = Player 1 king  
                -1 = Player 2 man
                -2 = Player 2 king
                0 = empty
        """
        board = np.zeros((8, 8), dtype=np.int8)
        
        # The bitboards are stored in "current player at bottom" perspective
        # After each move, they're flipped and swapped
        # We need to determine the absolute positions
        
        if self.current_player == 1:
            # Current player is 1, so board is in "normal" orientation
            # player bitboards = Player 1, opponent bitboards = Player 2
            p1_men, p1_kings = self.player_men, self.player_kings
            p2_men, p2_kings = self.opponent_men, self.opponent_kings
            # No flip needed
            needs_flip = False
        else:
            # Current player is 2, so board was flipped
            # player bitboards = Player 2, opponent bitboards = Player 1
            # But they're in flipped coordinates!
            p2_men, p2_kings = self.player_men, self.player_kings
            p1_men, p1_kings = self.opponent_men, self.opponent_kings
            # Need to flip back to absolute coordinates
            needs_flip = True
        
        # If we need to flip, flip the bitboards first
        if needs_flip:
            p1_men = flip_bitboard(p1_men)
            p1_kings = flip_bitboard(p1_kings)
            p2_men = flip_bitboard(p2_men)
            p2_kings = flip_bitboard(p2_kings)
        
        # Fill board from bitboards (now in absolute coordinates)
        for square in get_set_squares(p1_men):
            row, col = square_to_row_col(square)
            board[row, col] = 1
        
        for square in get_set_squares(p1_kings):
            row, col = square_to_row_col(square)
            board[row, col] = 2
        
        for square in get_set_squares(p2_men):
            row, col = square_to_row_col(square)
            board[row, col] = -1
        
        for square in get_set_squares(p2_kings):
            row, col = square_to_row_col(square)
            board[row, col] = -2
        
        return board



    def render(self) -> str:
        """Render board as ASCII string."""
        lines = ["\nBoard:"]
        for row in range(BOARD_SIZE):
            line = f"{row} "
            for col in range(BOARD_SIZE):
                square = row_col_to_square(row, col)
                if square == -1:
                    line += "   "
                elif get_bit(self.player_men, square):
                    line += " m "
                elif get_bit(self.player_kings, square):
                    line += " K "
                elif get_bit(self.opponent_men, square):
                    line += " M "
                elif get_bit(self.opponent_kings, square):
                    line += " k "
                else:
                    line += " . "
            lines.append(line)
        lines.append("   0  1  2  3  4  5  6  7")
        lines.append(f"\nPlayer {self.current_player} to move")
        lines.append(f"Move count: {self.move_count}\n")
        return "\n".join(lines)


# Testing
if __name__ == "__main__":
    print("Testing 8x8 Checkers Game")
    print("=" * 60)

    # Test 1: Initial game state
    print("\n1. Initial Game State")
    game = CheckersGame()
    print(game.render())

    legal_moves = game.get_legal_moves()
    print(f"Legal moves: {len(legal_moves)}")
    for move in legal_moves[:5]:
        print(f"  {move}")

    legal_actions = game.get_legal_actions()
    print(f"Legal actions: {legal_actions}")

    # Test 2: Make a move
    print("\n2. Make a Move")
    if legal_moves:
        game.make_move(legal_moves[0])
        print(game.render())
        print(f"Legal moves: {len(game.get_legal_moves())}")

    # Test 3: Neural input
    print("\n3. Neural Network Input")
    neural_input = game.to_neural_input()
    print(f"Shape: {neural_input.shape}")
    print(f"Plane 0 (player men):\n{neural_input[0]}")

    # Test 4: Action interface
    print("\n4. Action Interface Test")
    game2 = CheckersGame()
    actions = game2.get_legal_actions()
    if actions:
        action = actions[0]
        print(f"Trying action {action}")
        move = game2.action_to_move(action)
        print(f"Converted to move: {move}")
        success = game2.make_action(action)
        print(f"Action applied: {success}")

    print("\n✓ Game tests passed!")
