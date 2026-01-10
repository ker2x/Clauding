"""
Bitboard utilities for 10x10 International Draughts.

The 10x10 board has 50 playable squares (dark squares only).
We use 4 uint64 bitboards to represent the game state:
- player_men: current player's regular pieces
- player_kings: current player's kings
- opponent_men: opponent's regular pieces
- opponent_kings: opponent's kings

Square numbering (0-49):
    0  1  2  3  4
  5  6  7  8  9
   10 11 12 13 14
  15 16 17 18 19
   20 21 22 23 24
  25 26 27 28 29
   30 31 32 33 34
  35 36 37 38 39
   40 41 42 43 44
  45 46 47 48 49
"""

import numpy as np
from typing import List, Tuple

# Constants
BOARD_SIZE = 10
NUM_SQUARES = 50

# Starting positions for 10x10 International Draughts
# Player 1 starts at bottom (rows 6-9, squares 30-49)
# Player 2 starts at top (rows 0-3, squares 0-19)
INITIAL_PLAYER1 = 0x3FFFFC0000000  # Squares 30-49 (bottom 20 pieces)
INITIAL_PLAYER2 = 0x00000000FFFFF  # Squares 0-19 (top 20 pieces)

# Row masks for each row (0-9)
ROW_MASKS = [
    0x000000000001F,  # Row 0: squares 0-4
    0x00000000003E0,  # Row 1: squares 5-9
    0x000000007C00,   # Row 2: squares 10-14
    0x0000000F8000,   # Row 3: squares 15-19
    0x0000001F00000,  # Row 4: squares 20-24
    0x000003E000000,  # Row 5: squares 25-29
    0x00007C0000000,  # Row 6: squares 30-34
    0x0000F80000000,  # Row 7: squares 35-39
    0x0001F000000000, # Row 8: squares 40-44
    0x003E0000000000, # Row 9: squares 45-49
]


def square_to_row_col(square: int) -> Tuple[int, int]:
    """Convert square index (0-49) to board (row, col) coordinates."""
    row = square // 5
    col = (square % 5) * 2 + (1 if row % 2 == 0 else 0)
    return row, col


def row_col_to_square(row: int, col: int) -> int:
    """Convert board (row, col) to square index (0-49). Returns -1 if invalid."""
    if not (0 <= row < BOARD_SIZE and 0 <= col < BOARD_SIZE):
        return -1
    if (row + col) % 2 == 0:  # Light square, not playable
        return -1
    square = row * 5 + (col // 2)
    return square if 0 <= square < NUM_SQUARES else -1


def set_bit(bitboard: int, square: int) -> int:
    """Set bit at given square."""
    return bitboard | (1 << square)


def clear_bit(bitboard: int, square: int) -> int:
    """Clear bit at given square."""
    return bitboard & ~(1 << square)


def get_bit(bitboard: int, square: int) -> bool:
    """Check if bit is set at given square."""
    return bool(bitboard & (1 << square))


def count_bits(bitboard: int) -> int:
    """Count number of set bits in bitboard."""
    return bin(bitboard).count('1')


def get_set_squares(bitboard: int) -> List[int]:
    """Return list of square indices where bits are set."""
    squares = []
    for square in range(NUM_SQUARES):
        if get_bit(bitboard, square):
            squares.append(square)
    return squares


def print_bitboard(bitboard: int, label: str = ""):
    """Print bitboard as 10x10 grid for debugging."""
    if label:
        print(f"\n{label}:")

    for row in range(BOARD_SIZE):
        line = ""
        for col in range(BOARD_SIZE):
            if (row + col) % 2 == 0:
                line += " . "  # Light square (not playable)
            else:
                square = row_col_to_square(row, col)
                if get_bit(bitboard, square):
                    line += " X "
                else:
                    line += " - "
        print(line)


def flip_bitboard(bitboard: int) -> int:
    """Flip bitboard vertically (for switching perspectives)."""
    flipped = 0
    for square in range(NUM_SQUARES):
        if get_bit(bitboard, square):
            row = square // 5
            col_offset = square % 5
            flipped_row = BOARD_SIZE - 1 - row
            flipped_square = flipped_row * 5 + col_offset
            flipped = set_bit(flipped, flipped_square)
    return flipped


def get_adjacent_squares(square: int) -> List[int]:
    """Get adjacent diagonal squares (for regular moves)."""
    row, col = square_to_row_col(square)
    adjacent = []

    # Four diagonal directions
    for dr, dc in [(-1, -1), (-1, 1), (1, -1), (1, 1)]:
        new_row, new_col = row + dr, col + dc
        new_square = row_col_to_square(new_row, new_col)
        if new_square != -1:
            adjacent.append(new_square)

    return adjacent


def get_jump_target(square: int, direction: Tuple[int, int]) -> Tuple[int, int]:
    """
    Get the landing square and jumped square for a jump in given direction.
    Returns (jumped_square, landing_square) or (-1, -1) if invalid.
    """
    row, col = square_to_row_col(square)
    dr, dc = direction

    # Jumped square (one step)
    jumped_row, jumped_col = row + dr, col + dc
    jumped_square = row_col_to_square(jumped_row, jumped_col)
    if jumped_square == -1:
        return -1, -1

    # Landing square (two steps)
    landing_row, landing_col = row + 2*dr, col + 2*dc
    landing_square = row_col_to_square(landing_row, landing_col)
    if landing_square == -1:
        return -1, -1

    return jumped_square, landing_square


def bitboards_to_array(player_men: int, player_kings: int,
                       opponent_men: int, opponent_kings: int) -> np.ndarray:
    """Convert bitboards to 10x10 numpy array for visualization."""
    board = np.zeros((BOARD_SIZE, BOARD_SIZE), dtype=np.int8)

    for square in range(NUM_SQUARES):
        row, col = square_to_row_col(square)
        if get_bit(player_men, square):
            board[row, col] = 1  # Current player man
        elif get_bit(player_kings, square):
            board[row, col] = 2  # Current player king
        elif get_bit(opponent_men, square):
            board[row, col] = -1  # Opponent man
        elif get_bit(opponent_kings, square):
            board[row, col] = -2  # Opponent king

    return board


def array_to_bitboards(board: np.ndarray) -> Tuple[int, int, int, int]:
    """Convert 10x10 numpy array to bitboards."""
    player_men = 0
    player_kings = 0
    opponent_men = 0
    opponent_kings = 0

    for row in range(BOARD_SIZE):
        for col in range(BOARD_SIZE):
            if (row + col) % 2 == 1:  # Only dark squares
                square = row_col_to_square(row, col)
                value = board[row, col]
                if value == 1:
                    player_men = set_bit(player_men, square)
                elif value == 2:
                    player_kings = set_bit(player_kings, square)
                elif value == -1:
                    opponent_men = set_bit(opponent_men, square)
                elif value == -2:
                    opponent_kings = set_bit(opponent_kings, square)

    return player_men, player_kings, opponent_men, opponent_kings
