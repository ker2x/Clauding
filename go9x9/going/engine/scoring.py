"""
Chinese area scoring for Go.

Chinese scoring: score = stones_on_board + empty_territory - komi (for white)
Territory = empty points surrounded entirely by one color.
"""

import numpy as np
from collections import deque

BOARD_SIZE = 9
EMPTY = 0
BLACK = 1
WHITE = 2


def get_neighbors(pos: int) -> list:
    """Get adjacent positions for a board position."""
    row, col = pos // BOARD_SIZE, pos % BOARD_SIZE
    neighbors = []
    if row > 0:
        neighbors.append(pos - BOARD_SIZE)
    if row < BOARD_SIZE - 1:
        neighbors.append(pos + BOARD_SIZE)
    if col > 0:
        neighbors.append(pos - 1)
    if col < BOARD_SIZE - 1:
        neighbors.append(pos + 1)
    return neighbors


def compute_territory(board: np.ndarray) -> tuple:
    """
    Compute territory for both players using flood fill.

    An empty region is territory for a color if ALL adjacent stones
    around that region are that color.

    Args:
        board: Flat array of 81 ints (0=empty, 1=black, 2=white)

    Returns:
        (black_territory, white_territory): Counts of empty points
        that are territory for each player
    """
    visited = np.zeros(81, dtype=bool)
    black_territory = 0
    white_territory = 0

    for start in range(81):
        if visited[start] or board[start] != EMPTY:
            continue

        # BFS to find connected empty region
        region = []
        border_colors = set()
        queue = deque([start])
        visited[start] = True

        while queue:
            pos = queue.popleft()
            region.append(pos)

            for nb in get_neighbors(pos):
                if visited[nb]:
                    continue
                if board[nb] == EMPTY:
                    visited[nb] = True
                    queue.append(nb)
                else:
                    border_colors.add(board[nb])

        # Region is territory if bordered by exactly one color
        if border_colors == {BLACK}:
            black_territory += len(region)
        elif border_colors == {WHITE}:
            white_territory += len(region)

    return black_territory, white_territory


def score_game(board: np.ndarray, komi: float = 7.5) -> tuple:
    """
    Score a finished game using Chinese area scoring.

    Score = stones + territory
    Black wins if black_score > white_score + komi

    Args:
        board: Flat array of 81 ints (0=empty, 1=black, 2=white)
        komi: Compensation for white (default 7.5)

    Returns:
        (black_score, white_score_with_komi, result_str)
        result_str is like "B+2.5" or "W+0.5"
    """
    black_stones = int(np.sum(board == BLACK))
    white_stones = int(np.sum(board == WHITE))

    black_territory, white_territory = compute_territory(board)

    black_score = black_stones + black_territory
    white_score = white_stones + white_territory + komi

    diff = black_score - white_score
    if diff > 0:
        result_str = f"B+{diff:.1f}"
    elif diff < 0:
        result_str = f"W+{-diff:.1f}"
    else:
        result_str = "0"  # Jigo (tie) — extremely rare with 0.5 komi

    return black_score, white_score, result_str
