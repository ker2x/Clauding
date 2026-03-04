"""
9x9 Go board with group tracking, liberties, and captures.

Board is a flat numpy array of 81 ints:
  0 = empty, 1 = black, 2 = white
"""

import numpy as np
from collections import deque
from typing import List, Set, Tuple, Optional

BOARD_SIZE = 9
EMPTY = 0
BLACK = 1
WHITE = 2


def opponent(color: int) -> int:
    """Return the opponent's color."""
    return 3 - color


def pos_to_rc(pos: int) -> Tuple[int, int]:
    """Convert flat position to (row, col)."""
    return pos // BOARD_SIZE, pos % BOARD_SIZE


def rc_to_pos(row: int, col: int) -> int:
    """Convert (row, col) to flat position."""
    return row * BOARD_SIZE + col


# Precompute neighbor lists for each position
NEIGHBORS = []
for _p in range(81):
    _r, _c = pos_to_rc(_p)
    _nb = []
    if _r > 0: _nb.append(_p - BOARD_SIZE)
    if _r < BOARD_SIZE - 1: _nb.append(_p + BOARD_SIZE)
    if _c > 0: _nb.append(_p - 1)
    if _c < BOARD_SIZE - 1: _nb.append(_p + 1)
    NEIGHBORS.append(tuple(_nb))


def find_group(board: np.ndarray, pos: int) -> Tuple[Set[int], Set[int]]:
    """
    Find the group of connected stones containing pos, and its liberties.

    Args:
        board: Flat board array
        pos: Starting position (must contain a stone)

    Returns:
        (group, liberties): Sets of positions
    """
    color = board[pos]
    assert color in (BLACK, WHITE), f"No stone at position {pos}"

    group = set()
    liberties = set()
    queue = deque([pos])
    group.add(pos)

    while queue:
        p = queue.popleft()
        for nb in NEIGHBORS[p]:
            if nb in group:
                continue
            if board[nb] == EMPTY:
                liberties.add(nb)
            elif board[nb] == color:
                group.add(nb)
                queue.append(nb)

    return group, liberties


def count_liberties(board: np.ndarray, pos: int) -> int:
    """Count liberties of the group containing pos."""
    _, liberties = find_group(board, pos)
    return len(liberties)


def remove_group(board: np.ndarray, group: Set[int]) -> np.ndarray:
    """Remove a group of stones from the board. Returns modified board."""
    for pos in group:
        board[pos] = EMPTY
    return board


def get_captures(board: np.ndarray, pos: int, color: int) -> List[Set[int]]:
    """
    Find opponent groups that would be captured by placing color at pos.

    Args:
        board: Current board (pos should already have the stone placed)
        pos: Position where stone was just placed
        color: Color of the placed stone

    Returns:
        List of groups (sets of positions) that are captured
    """
    opp = opponent(color)
    captured_groups = []
    checked = set()

    for nb in NEIGHBORS[pos]:
        if board[nb] == opp and nb not in checked:
            group, liberties = find_group(board, nb)
            checked.update(group)
            if len(liberties) == 0:
                captured_groups.append(group)

    return captured_groups


def is_suicide(board: np.ndarray, pos: int, color: int) -> bool:
    """
    Check if placing a stone at pos would be suicide.

    A move is suicide if:
    1. The placed stone has no liberties after placement
    2. It doesn't capture any opponent stones
    3. It doesn't connect to a friendly group that still has other liberties

    Args:
        board: Board BEFORE placing the stone
        pos: Position to check
        color: Color of the stone

    Returns:
        True if the move would be suicide
    """
    # Temporarily place the stone
    board[pos] = color

    # Check if we capture any opponent groups
    captures = get_captures(board, pos, color)
    if captures:
        board[pos] = EMPTY
        return False

    # Check if our group has any liberties
    _, liberties = find_group(board, pos)

    board[pos] = EMPTY

    return len(liberties) == 0


def get_legal_moves(board: np.ndarray, color: int, ko_point: Optional[int] = None) -> List[int]:
    """
    Get all legal moves for a color.

    A move is legal if:
    1. The intersection is empty
    2. It's not suicide
    3. It's not a ko recapture

    Args:
        board: Current board state
        color: Color to move (1=black, 2=white)
        ko_point: Position that is illegal due to ko rule (or None)

    Returns:
        List of legal position indices (0-80). Does NOT include pass.
    """
    legal = []

    for pos in range(81):
        if board[pos] != EMPTY:
            continue
        if pos == ko_point:
            continue
        if is_suicide(board, pos, color):
            continue
        legal.append(pos)

    return legal


def apply_move(board: np.ndarray, pos: int, color: int) -> Tuple[np.ndarray, int, Optional[int]]:
    """
    Apply a move to the board.

    Args:
        board: Current board (will be modified in place)
        pos: Position to place stone (0-80)
        color: Color of stone (1=black, 2=white)

    Returns:
        (board, num_captured, ko_point):
        - board: Modified board
        - num_captured: Number of stones captured
        - ko_point: Position that becomes illegal for opponent's next move (ko),
                     or None if no ko
    """
    board[pos] = color

    # Find and remove captured groups
    captured_groups = get_captures(board, pos, color)
    total_captured = 0
    captured_positions = set()

    for group in captured_groups:
        captured_positions.update(group)
        total_captured += len(group)
        remove_group(board, group)

    # Determine ko point
    ko_point = None
    if total_captured == 1 and len(captured_positions) == 1:
        # Potential ko: single stone captured
        # Check if the capturing stone has exactly 1 liberty (the captured position)
        _, liberties = find_group(board, pos)
        if len(liberties) == 1:
            ko_point = captured_positions.pop()
        else:
            captured_positions.pop()

    return board, total_captured, ko_point
