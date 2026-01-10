"""
Move generation for 10x10 International Draughts.

International Draughts rules:
- Men move forward diagonally, one square
- Kings move diagonally any distance
- Captures are mandatory (forced)
- Multiple captures in sequence (chains)
- Long-range king captures
- Maximum capture rule (choose sequence with most captures)
"""

from typing import List, Tuple, Set
from dataclasses import dataclass
from .bitboard import *


@dataclass(frozen=True)
class Move:
    """Represents a move in checkers."""
    from_square: int
    to_square: int
    captured_squares: Tuple[int, ...] = ()  # Squares of captured pieces
    promotes_to_king: bool = False

    def __hash__(self):
        return hash((self.from_square, self.to_square, self.captured_squares))

    def __eq__(self, other):
        if not isinstance(other, Move):
            return False
        return (self.from_square == other.from_square and
                self.to_square == other.to_square and
                self.captured_squares == other.captured_squares)

    def __str__(self):
        if self.captured_squares:
            return f"{self.from_square}x{self.to_square} (captures: {self.captured_squares})"
        return f"{self.from_square}-{self.to_square}"


def generate_man_moves(square: int, all_pieces: int) -> List[Move]:
    """Generate forward moves for a regular piece (man)."""
    moves = []
    row, col = square_to_row_col(square)

    # Men move forward only (decreasing row numbers)
    for dc in [-1, 1]:
        new_row, new_col = row - 1, col + dc
        new_square = row_col_to_square(new_row, new_col)

        if new_square != -1 and not get_bit(all_pieces, new_square):
            # Check if this move promotes to king (reaches row 0)
            promotes = (new_row == 0)
            moves.append(Move(square, new_square, (), promotes))

    return moves


def generate_king_moves(square: int, all_pieces: int) -> List[Move]:
    """Generate moves for a king (flying king - can move any distance)."""
    moves = []
    row, col = square_to_row_col(square)

    # Four diagonal directions
    for dr, dc in [(-1, -1), (-1, 1), (1, -1), (1, 1)]:
        # Try all distances in this direction
        distance = 1
        while True:
            new_row, new_col = row + dr * distance, col + dc * distance
            new_square = row_col_to_square(new_row, new_col)

            if new_square == -1:  # Off board
                break

            if get_bit(all_pieces, new_square):  # Blocked by piece
                break

            moves.append(Move(square, new_square, (), False))
            distance += 1

    return moves


def generate_man_captures(square: int, player_pieces: int, opponent_pieces: int,
                          all_pieces: int, captured_so_far: Set[int],
                          current_square: int) -> List[Move]:
    """
    Generate capture moves for a man from current position.
    Uses recursion to handle capture chains.
    """
    captures = []
    row, col = square_to_row_col(current_square)

    # Men can capture in all four diagonal directions (forward and backward when capturing)
    for dr, dc in [(-1, -1), (-1, 1), (1, -1), (1, 1)]:
        jumped_square, landing_square = get_jump_target(current_square, (dr, dc))

        if jumped_square == -1 or landing_square == -1:
            continue

        # Must jump over an opponent piece that hasn't been captured yet
        if (get_bit(opponent_pieces, jumped_square) and
            jumped_square not in captured_so_far and
            not get_bit(all_pieces, landing_square)):

            # Check if this landing promotes to king
            landing_row, _ = square_to_row_col(landing_square)
            promotes = (landing_row == 0)

            new_captured = captured_so_far | {jumped_square}

            if promotes:
                # If promoted, can't continue capturing as a man
                captures.append(Move(square, landing_square,
                                   tuple(sorted(new_captured)), True))
            else:
                # Try to continue capturing (recursive)
                further_captures = generate_man_captures(
                    square, player_pieces, opponent_pieces,
                    all_pieces, new_captured, landing_square
                )

                if further_captures:
                    # Found longer captures
                    captures.extend(further_captures)
                else:
                    # This is a terminal capture
                    captures.append(Move(square, landing_square,
                                       tuple(sorted(new_captured)), False))

    return captures


def generate_king_captures(square: int, player_pieces: int, opponent_pieces: int,
                           all_pieces: int, captured_so_far: Set[int],
                           current_square: int) -> List[Move]:
    """
    Generate capture moves for a king (flying king).
    Kings can jump over pieces at any distance and land anywhere beyond.
    """
    captures = []
    row, col = square_to_row_col(current_square)

    # Four diagonal directions
    for dr, dc in [(-1, -1), (-1, 1), (1, -1), (1, 1)]:
        # Look for an opponent piece in this direction
        distance = 1
        jumped_square = -1

        while True:
            check_row, check_col = row + dr * distance, col + dc * distance
            check_square = row_col_to_square(check_row, check_col)

            if check_square == -1:  # Off board
                break

            if get_bit(player_pieces, check_square):  # Blocked by own piece
                break

            if get_bit(opponent_pieces, check_square):
                if check_square not in captured_so_far:
                    jumped_square = check_square
                break  # Can't have multiple pieces in one direction

            distance += 1

        # If found an opponent piece, try all landing squares beyond it
        if jumped_square != -1:
            land_distance = distance + 1
            while True:
                land_row, land_col = row + dr * land_distance, col + dc * land_distance
                landing_square = row_col_to_square(land_row, land_col)

                if landing_square == -1:  # Off board
                    break

                if get_bit(all_pieces, landing_square):  # Blocked
                    break

                new_captured = captured_so_far | {jumped_square}

                # Try to continue capturing from landing square
                further_captures = generate_king_captures(
                    square, player_pieces, opponent_pieces,
                    all_pieces, new_captured, landing_square
                )

                if further_captures:
                    captures.extend(further_captures)
                else:
                    # Terminal capture
                    captures.append(Move(square, landing_square,
                                       tuple(sorted(new_captured)), False))

                land_distance += 1

    return captures


def get_all_captures(player_men: int, player_kings: int,
                     opponent_men: int, opponent_kings: int) -> List[Move]:
    """
    Get all possible captures for current player.
    Returns all captures with maximum number of pieces captured.
    """
    player_pieces = player_men | player_kings
    opponent_pieces = opponent_men | opponent_kings
    all_pieces = player_pieces | opponent_pieces

    all_captures = []

    # Generate captures for each man
    for square in get_set_squares(player_men):
        captures = generate_man_captures(square, player_pieces, opponent_pieces,
                                        all_pieces, set(), square)
        all_captures.extend(captures)

    # Generate captures for each king
    for square in get_set_squares(player_kings):
        captures = generate_king_captures(square, player_pieces, opponent_pieces,
                                         all_pieces, set(), square)
        all_captures.extend(captures)

    if not all_captures:
        return []

    # Maximum capture rule: only return captures with maximum pieces taken
    max_captures = max(len(move.captured_squares) for move in all_captures)
    return [move for move in all_captures if len(move.captured_squares) == max_captures]


def get_all_quiet_moves(player_men: int, player_kings: int,
                        opponent_men: int, opponent_kings: int) -> List[Move]:
    """Get all non-capturing moves for current player."""
    player_pieces = player_men | player_kings
    opponent_pieces = opponent_men | opponent_kings
    all_pieces = player_pieces | opponent_pieces

    all_moves = []

    # Generate moves for each man
    for square in get_set_squares(player_men):
        moves = generate_man_moves(square, all_pieces)
        all_moves.extend(moves)

    # Generate moves for each king
    for square in get_set_squares(player_kings):
        moves = generate_king_moves(square, all_pieces)
        all_moves.extend(moves)

    return all_moves


def get_legal_moves(player_men: int, player_kings: int,
                    opponent_men: int, opponent_kings: int) -> List[Move]:
    """
    Get all legal moves for current player.
    Captures are mandatory - if any captures exist, only return captures.
    """
    # Check for captures first (mandatory)
    captures = get_all_captures(player_men, player_kings, opponent_men, opponent_kings)
    if captures:
        return captures

    # No captures available, return quiet moves
    return get_all_quiet_moves(player_men, player_kings, opponent_men, opponent_kings)
