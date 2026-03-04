"""
Zobrist hashing for Go board positions.

Used for superko detection (positional superko).
"""

import numpy as np

# Fixed seed for reproducible hash tables
_rng = np.random.RandomState(seed=314159)

# Hash tables: [color][position] where color 0=black, 1=white
# 2 colors × 81 positions
ZOBRIST_TABLE = _rng.randint(0, 2**63, size=(2, 81), dtype=np.uint64)

# Hash for side to move (XOR when it's white's turn)
ZOBRIST_BLACK_TO_MOVE = np.uint64(_rng.randint(0, 2**63))


def compute_hash(board: np.ndarray, current_player: int) -> int:
    """
    Compute Zobrist hash for a board position.

    Args:
        board: Flat array of 81 ints (0=empty, 1=black, 2=white)
        current_player: 1=black, 2=white

    Returns:
        64-bit hash value
    """
    h = np.uint64(0)
    for pos in range(81):
        stone = board[pos]
        if stone == 1:
            h ^= ZOBRIST_TABLE[0, pos]
        elif stone == 2:
            h ^= ZOBRIST_TABLE[1, pos]

    if current_player == 2:
        h ^= ZOBRIST_BLACK_TO_MOVE

    return int(h)


def update_hash(h: int, pos: int, old_color: int, new_color: int) -> int:
    """
    Incrementally update Zobrist hash after a stone change.

    Args:
        h: Current hash
        pos: Board position (0-80)
        old_color: Previous color at position (0=empty, 1=black, 2=white)
        new_color: New color at position

    Returns:
        Updated hash
    """
    h = np.uint64(h)
    if old_color == 1:
        h ^= ZOBRIST_TABLE[0, pos]
    elif old_color == 2:
        h ^= ZOBRIST_TABLE[1, pos]

    if new_color == 1:
        h ^= ZOBRIST_TABLE[0, pos]
    elif new_color == 2:
        h ^= ZOBRIST_TABLE[1, pos]

    return int(h)


def toggle_side(h: int) -> int:
    """Toggle the side-to-move component of the hash."""
    return int(np.uint64(h) ^ ZOBRIST_BLACK_TO_MOVE)
