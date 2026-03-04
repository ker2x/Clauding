"""
Action encoding for 9x9 Go.

Action space: 82 actions total
  - Actions 0-80: Place stone at intersection (row * 9 + col)
  - Action 81: Pass

GTP coordinate conversion:
  - Columns: A-J (skipping I) → 0-8
  - Rows: 1-9 → 0-8 (GTP row 1 = our row 0 = bottom)
"""

BOARD_SIZE = 9
NUM_INTERSECTIONS = BOARD_SIZE * BOARD_SIZE  # 81
PASS_ACTION = NUM_INTERSECTIONS  # 81
NUM_ACTIONS = NUM_INTERSECTIONS + 1  # 82

# GTP column letters (I is skipped in Go)
GTP_COLUMNS = "ABCDEFGHJ"


def action_to_pos(action: int) -> tuple:
    """
    Convert action index to (row, col) or None for pass.

    Args:
        action: Action index (0-81)

    Returns:
        (row, col) tuple or None if pass
    """
    if action == PASS_ACTION:
        return None
    row = action // BOARD_SIZE
    col = action % BOARD_SIZE
    return (row, col)


def pos_to_action(row: int, col: int) -> int:
    """Convert (row, col) to action index."""
    return row * BOARD_SIZE + col


def action_to_gtp(action: int) -> str:
    """
    Convert action index to GTP move string.

    Args:
        action: Action index (0-81)

    Returns:
        GTP move string like "D4" or "pass"
    """
    if action == PASS_ACTION:
        return "pass"
    row = action // BOARD_SIZE
    col = action % BOARD_SIZE
    # GTP: column letter + row number (1-indexed, bottom=1)
    return f"{GTP_COLUMNS[col]}{row + 1}"


def gtp_to_action(move_str: str) -> int:
    """
    Convert GTP move string to action index.

    Args:
        move_str: GTP move string like "D4" or "pass"

    Returns:
        Action index (0-81)
    """
    move_str = move_str.strip().lower()
    if move_str == "pass":
        return PASS_ACTION

    col_char = move_str[0].upper()
    row_num = int(move_str[1:])

    col = GTP_COLUMNS.index(col_char)
    row = row_num - 1  # GTP is 1-indexed

    return pos_to_action(row, col)
