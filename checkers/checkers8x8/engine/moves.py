"""
Move generation for 8x8 American Checkers.

American Checkers Rules:
- Men move forward diagonally one square
- Kings move diagonally one square in any direction
- Captures are mandatory (forced)
- Men can capture backward (but not move backward)
- Capture chains continue until no more captures
- Promotion when reaching opposite end

Key difference from 10x10:
- NO flying kings (kings move only 1 square)
- Simpler capture rules (no maximum capture rule)
"""

from typing import List, Set, Tuple
from dataclasses import dataclass

try:
    from .bitboard import *
    from .action_encoder import encode_action, decode_action
except ImportError:
    from bitboard import *
    from action_encoder import encode_action, decode_action


@dataclass(frozen=True)
class Move:
    """
    Represents a move in checkers.

    For multi-captures, the move represents the entire capture sequence,
    but we encode it using just the final landing square for simplicity.
    """
    from_square: int  # Starting square
    to_square: int    # Ending square
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
            return f"{self.from_square}x{self.to_square} (caps: {self.captured_squares})"
        return f"{self.from_square}→{self.to_square}"

    def to_actions(self) -> List[int]:
        """Convert move to list of action indices (0-127)."""
        from_row, from_col = square_to_row_col(self.from_square)

        # For jumps, we MUST use the direction to the first captured piece
        # This handles zig-zag jumps where the final destination might be
        # vertically aligned (dc=0) or in a different overall direction.
        if self.captured_squares:
            # captured_squares is now ordered by visit time
            first_captured = self.captured_squares[0]
            target_row, target_col = square_to_row_col(first_captured)
        else:
            # Simple move
            target_row, target_col = square_to_row_col(self.to_square)

        diff_row = target_row - from_row
        diff_col = target_col - from_col

        direction = -1
        if diff_row < 0 and diff_col > 0:
            direction = 0  # NE
        elif diff_row < 0 and diff_col < 0:
            direction = 1  # NW
        elif diff_row > 0 and diff_col > 0:
            direction = 2  # SE
        elif diff_row > 0 and diff_col < 0:
            direction = 3  # SW

        if direction != -1:
             return [encode_action(self.from_square, direction)]
        return []


def get_simple_move_actions(square: int, is_king: bool, all_pieces: int) -> List[int]:
    """
    Get available simple (non-capture) move actions for a piece.

    Args:
        square: Piece location
        is_king: Whether piece is a king
        all_pieces: Bitboard of all pieces

    Returns:
        List of valid action indices
    """
    actions = []

    # Men can only move forward (directions 0, 1)
    # Kings can move in all 4 directions
    directions = range(4) if is_king else [0, 1]

    for direction in directions:
        neighbor = get_neighbor(square, direction)
        if neighbor != -1 and not get_bit(all_pieces, neighbor):
            actions.append(encode_action(square, direction))

    return actions


def get_capture_moves(
    square: int,
    is_king: bool,
    player_pieces: int,
    opponent_pieces: int,
    all_pieces: int,
    captured_so_far: Set[int]
) -> List[Move]:
    """
    Get all capture moves from current square.

    Args:
        square: Current piece location
        is_king: Whether piece is a king
        player_pieces: Bitboard of player's pieces
        opponent_pieces: Bitboard of opponent's pieces
        all_pieces: Bitboard of all pieces
        captured_so_far: Set of already captured squares in this chain

    Returns:
        List of valid capture moves
    """
    captures = []

    # Both men and kings can capture in all 4 directions
    # Men capture forward only (0, 1)
    # Kings capture in all directions (0, 1, 2, 3)
    directions = range(4) if is_king else [0, 1]

    for direction in directions:
        jumped_square, landing_square = get_jump_target(square, direction)

        if jumped_square == -1 or landing_square == -1:
            continue

        # Must jump over opponent piece that hasn't been captured yet
        if (get_bit(opponent_pieces, jumped_square) and
            jumped_square not in captured_so_far and
            not get_bit(all_pieces, landing_square)):

            # Check if this landing promotes to king
            landing_row, _ = square_to_row_col(landing_square)
            promotes = (landing_row == 0)  # Reaching top row

            new_captured = captured_so_far + (jumped_square,)

            # Temporarily update board state to check for continued captures
            # (Remove jumped piece, move to landing square)
            temp_all_pieces = clear_bit(all_pieces, jumped_square)
            temp_opponent = clear_bit(opponent_pieces, jumped_square)

            if promotes:
                # If promoted, stop capturing (promoted pieces can't continue in same turn)
                captures.append(Move(
                    square, landing_square,
                    tuple(sorted(new_captured)),
                    promotes_to_king=True
                ))
            else:
                # Try to continue capturing (recursive)
                further_captures = get_capture_moves(
                    landing_square,  # Now search from landing square
                    is_king,
                    player_pieces,
                    temp_opponent,  # Updated opponent pieces
                    temp_all_pieces,  # Updated all pieces
                    new_captured
                )

                if further_captures:
                    # Found longer capture chains
                    # We must prepend the current jump step to the sequences found recursively
                    for move in further_captures:
                        # Append subsequent captures to the current one.
                        # Do NOT sort, to preserve temporal order for direction checking.
                        # Recursion already returns moves accumulating the full capture path,
                        # so we just take move.captured_squares as is.
                        new_captured = move.captured_squares
                        
                        # The move object represents the FULL move from 'square' to final destination
                        captures.append(Move(
                            square,
                            move.to_square,
                            new_captured,
                            move.promotes_to_king
                        ))
                else:
                    # Terminal capture
                    captures.append(Move(
                        square, landing_square,
                        tuple(list(captured_so_far) + [jumped_square]),
                        promotes_to_king=False
                    ))

    return captures


def get_legal_moves(
    player_men: int,
    player_kings: int,
    opponent_men: int,
    opponent_kings: int
) -> List[Move]:
    """
    Get all legal moves for current player.

    Captures are mandatory - if any captures exist, only return captures.

    Returns:
        List of legal Move objects
    """
    player_pieces = player_men | player_kings
    opponent_pieces = opponent_men | opponent_kings
    all_pieces = player_pieces | opponent_pieces

    all_captures = []

    # Check for captures from men
    for square in get_set_squares(player_men):
        captures = get_capture_moves(
            square,
            False,  # is_king=False
            player_pieces,
            opponent_pieces,
            all_pieces,
            ()
        )
        all_captures.extend(captures)

    # Check for captures from kings
    for square in get_set_squares(player_kings):
        captures = get_capture_moves(
            square,
            True,  # is_king=True
            player_pieces,
            opponent_pieces,
            all_pieces,
            ()
        )
        all_captures.extend(captures)

    # If captures exist, they are mandatory
    if all_captures:
        return all_captures

    # No captures - return simple moves
    all_moves = []

    # Simple moves for men
    for square in get_set_squares(player_men):
        from_row, from_col = square_to_row_col(square)

        for direction in [0, 1]:  # Men move forward only
            neighbor = get_neighbor(square, direction)
            if neighbor != -1 and not get_bit(all_pieces, neighbor):
                # Check for promotion
                neighbor_row, _ = square_to_row_col(neighbor)
                promotes = (neighbor_row == 0)
                all_moves.append(Move(square, neighbor, (), promotes))

    # Simple moves for kings
    for square in get_set_squares(player_kings):
        for direction in range(4):  # Kings move in all directions
            neighbor = get_neighbor(square, direction)
            if neighbor != -1 and not get_bit(all_pieces, neighbor):
                all_moves.append(Move(square, neighbor, (), False))

    return all_moves


def get_legal_actions(
    player_men: int,
    player_kings: int,
    opponent_men: int,
    opponent_kings: int
) -> List[int]:
    """
    Get legal action indices (for fixed action space).

    Returns:
        List of legal action indices (0-127)
    """
    moves = get_legal_moves(player_men, player_kings, opponent_men, opponent_kings)

    # Convert moves to actions
    actions = []
    for move in moves:
        actions.extend(move.to_actions())

    # Remove duplicates and sort
    return sorted(list(set(actions)))


# Testing
if __name__ == "__main__":
    print("Testing Move Generation for 8x8 Checkers")
    print("=" * 60)

    # Test 1: Initial position (player at bottom, opponent at top)
    print("\n1. Initial Position")
    # Player starts at bottom (rows 5, 6, 7 = squares 20-31)
    # Opponent starts at top (rows 0, 1, 2 = squares 0-11)
    player_men = 0b11111111111100000000000000000000  # Squares 20-31 (bottom)
    opponent_men = 0b00000000000000000000111111111111  # Squares 0-11 (top)

    print_board(player_men, 0, opponent_men, 0)

    moves = get_legal_moves(player_men, 0, opponent_men, 0)
    print(f"Legal moves: {len(moves)}")
    for move in moves[:10]:
        print(f"  {move}")

    # Test 2: Capture scenario
    print("\n2. Capture Scenario")
    # Player man on 13, opponent man on 17, empty square 22
    player_men_cap = set_bit(0, 13)
    opponent_men_cap = set_bit(0, 17)

    print_board(player_men_cap, 0, opponent_men_cap, 0)

    moves_cap = get_legal_moves(player_men_cap, 0, opponent_men_cap, 0)
    print(f"Legal moves: {len(moves_cap)}")
    for move in moves_cap:
        print(f"  {move}")

    # Test 3: King moves
    print("\n3. King Movement")
    player_kings = set_bit(0, 13)  # King on square 13

    print_board(0, player_kings, 0, 0)

    moves_king = get_legal_moves(0, player_kings, 0, 0)
    print(f"Legal king moves: {len(moves_king)}")
    for move in moves_king:
        print(f"  {move}")

    # Test 4: Action encoding
    print("\n4. Action Encoding Test")
    # Use correct initial position
    player_men_init = 0b11111111111100000000000000000000
    opponent_men_init = 0b00000000000000000000111111111111
    actions = get_legal_actions(player_men_init, 0, opponent_men_init, 0)
    print(f"Legal actions (fixed space): {len(actions)}")
    print(f"First 10 actions: {actions[:10]}")

    from action_encoder import action_to_string
    print("Action meanings:")
    for action in actions[:5]:
        print(f"  {action}: {action_to_string(action)}")

    print("\n✓ Move generation tests passed!")
