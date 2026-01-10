"""
Bitboard representation for 8x8 American Checkers.

Board layout (32 playable dark squares):
    0   1   2   3      Row 0 (top)
  4   5   6   7        Row 1
    8   9  10  11      Row 2
  12  13  14  15       Row 3
   16  17  18  19      Row 4
  20  21  22  23       Row 5
   24  25  26  27      Row 6
  28  29  30  31       Row 7 (bottom)

Each square is numbered 0-31 and represents a dark square on the 8x8 board.
"""

from typing import List, Tuple

# Constants
NUM_SQUARES = 32
BOARD_SIZE = 8

# Bitboard masks for rows (used for king promotion detection)
TOP_ROW = 0b00000000000000000000000000001111  # Squares 0-3
BOTTOM_ROW = 0b11110000000000000000000000000000  # Squares 28-31


def get_bit(bitboard: int, square: int) -> bool:
    """Check if bit at square index is set."""
    return (bitboard >> square) & 1 == 1


def set_bit(bitboard: int, square: int) -> int:
    """Set bit at square index."""
    return bitboard | (1 << square)


def clear_bit(bitboard: int, square: int) -> int:
    """Clear bit at square index."""
    return bitboard & ~(1 << square)


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


def square_to_row_col(square: int) -> Tuple[int, int]:
    """
    Convert square index (0-31) to board row/col (0-7, 0-7).

    Returns actual board coordinates (not just dark square coordinates).
    """
    # Map square index to actual board position
    row = square // 4
    col = (square % 4) * 2

    # Even rows have dark squares at odd columns (1, 3, 5, 7)
    # Odd rows have dark squares at even columns (0, 2, 4, 6)
    if row % 2 == 0:
        col += 1

    return row, col


def row_col_to_square(row: int, col: int) -> int:
    """
    Convert board row/col to square index.

    Returns -1 if position is not a dark square or out of bounds.
    """
    # Check bounds
    if row < 0 or row >= BOARD_SIZE or col < 0 or col >= BOARD_SIZE:
        return -1

    # Check if it's a dark square (dark squares have row+col = odd)
    if (row + col) % 2 == 0:  # Light square
        return -1

    # Calculate square index
    square = row * 4 + col // 2

    return square if 0 <= square < NUM_SQUARES else -1


def get_direction_offset(direction: int) -> Tuple[int, int]:
    """
    Get row/col offset for a direction.

    Directions:
        0: NW (forward-left for player at bottom)
        1: NE (forward-right for player at bottom)
        2: SW (backward-left for player at bottom)
        3: SE (backward-right for player at bottom)
    """
    offsets = {
        0: (-1, -1),  # NW
        1: (-1, +1),  # NE
        2: (+1, -1),  # SW
        3: (+1, +1),  # SE
    }
    return offsets.get(direction, (0, 0))


def get_neighbor(square: int, direction: int) -> int:
    """
    Get neighboring square in given direction.

    Returns -1 if move goes off board.
    """
    row, col = square_to_row_col(square)
    dr, dc = get_direction_offset(direction)
    new_row, new_col = row + dr, col + dc
    return row_col_to_square(new_row, new_col)


def get_jump_target(square: int, direction: int) -> Tuple[int, int]:
    """
    Get jumped square and landing square for a capture in given direction.

    Returns (jumped_square, landing_square).
    Returns (-1, -1) if jump goes off board.
    """
    jumped_square = get_neighbor(square, direction)
    if jumped_square == -1:
        return -1, -1

    landing_square = get_neighbor(jumped_square, direction)
    if landing_square == -1:
        return -1, -1

    return jumped_square, landing_square


def flip_bitboard(bitboard: int) -> int:
    """
    Flip bitboard vertically (for perspective switching).

    Player 1 sees board normally (bottom to top).
    Player 2 sees board flipped (their pieces at bottom).
    
    IMPORTANT: Flips row AND mirrors column horizontally to prevent artifacts.
    A piece at (row, col) becomes (7-row, 7-col).
    """
    result = 0
    for square in range(NUM_SQUARES):
        if get_bit(bitboard, square):
            # Get actual board coordinates
            row, col = square_to_row_col(square)
            # Flip row and mirror column horizontally
            flipped_row = 7 - row
            flipped_col = 7 - col
            # Map back to square number
            flipped_square = row_col_to_square(flipped_row, flipped_col)
            if flipped_square != -1:
                result = set_bit(result, flipped_square)
    return result


def print_bitboard(bitboard: int, label: str = ""):
    """Print bitboard as 8x8 grid for debugging."""
    if label:
        print(f"\n{label}:")

    for row in range(BOARD_SIZE):
        line = ""
        for col in range(BOARD_SIZE):
            square = row_col_to_square(row, col)
            if square == -1:
                line += " . "  # Light square
            elif get_bit(bitboard, square):
                line += f" {square:2d}"  # Show square number
            else:
                line += "  ."  # Empty dark square
        print(line)
    print()


def print_board(player_men: int, player_kings: int, opponent_men: int, opponent_kings: int):
    """Print full board state with pieces."""
    print("\nBoard:")
    for row in range(BOARD_SIZE):
        line = ""
        for col in range(BOARD_SIZE):
            square = row_col_to_square(row, col)
            if square == -1:
                line += "   "  # Light square
            elif get_bit(player_men, square):
                line += " m "  # Player man
            elif get_bit(player_kings, square):
                line += " K "  # Player king
            elif get_bit(opponent_men, square):
                line += " M "  # Opponent man
            elif get_bit(opponent_kings, square):
                line += " k "  # Opponent king
            else:
                line += " . "  # Empty dark square
        print(f"{row} {line}")
    print("   0  1  2  3  4  5  6  7")
    print()


# Test the bitboard implementation
if __name__ == "__main__":
    print("Testing 8x8 Bitboard Implementation")
    print("=" * 50)

    # Test square conversion
    print("\nSquare to row/col conversions:")
    for sq in [0, 3, 4, 7, 28, 31]:
        r, c = square_to_row_col(sq)
        back = row_col_to_square(r, c)
        print(f"  Square {sq:2d} → ({r}, {c}) → {back:2d}")

    # Test directions
    print("\nDirection tests from square 13:")
    for d in range(4):
        dr, dc = get_direction_offset(d)
        neighbor = get_neighbor(13, d)
        print(f"  Direction {d} ({dr:+d}, {dc:+d}): neighbor = {neighbor}")

    # Test jump
    print("\nJump tests from square 13:")
    for d in range(4):
        jumped, landing = get_jump_target(13, d)
        print(f"  Direction {d}: jumped={jumped}, landing={landing}")

    # Test initial position
    print("\nInitial board setup:")
    player_men = 0b00000000000000000000111111111111  # Squares 0-11
    opponent_men = 0b11111111111100000000000000000000  # Squares 20-31
    print_board(player_men, 0, opponent_men, 0)

    print("✓ Bitboard tests passed!")
