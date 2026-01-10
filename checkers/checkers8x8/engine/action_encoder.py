"""
Fixed Action Space Encoding for 8x8 Checkers.

This is the KEY innovation that solves the dynamic action space problem.

Action Encoding: from_square × 4 + direction

Directions:
    0: NW (forward-left, row-1 col-1)
    1: NE (forward-right, row-1 col+1)
    2: SW (backward-left, row+1 col-1)
    3: SE (backward-right, row+1 col+1)

Total action space: 32 squares × 4 directions = 128 actions

Example:
    Action 0  = Move from square 0, direction NW
    Action 1  = Move from square 0, direction NE
    Action 53 = Move from square 13, direction NE (13*4 + 1)

This encoding works for:
    - Simple moves (1 square diagonal)
    - Captures (jump over opponent piece in that direction)
    - Multi-captures (multiple jumps, each encoded separately... or handled in move generation)

The network learns:
    "Action 53 is good" = "Moving from square 13 toward NE is strong"
    NOT just "the 5th legal move is good"
"""

from typing import Tuple, Optional

try:
    from .bitboard import NUM_SQUARES, get_direction_offset
except ImportError:
    # For standalone testing
    from bitboard import NUM_SQUARES, get_direction_offset

# Action space size
NUM_ACTIONS = 128  # 32 squares × 4 directions


def encode_action(from_square: int, direction: int) -> int:
    """
    Encode (from_square, direction) as a single action index.

    Args:
        from_square: Source square (0-31)
        direction: Direction (0-3)

    Returns:
        Action index (0-127)
    """
    assert 0 <= from_square < NUM_SQUARES, f"Invalid from_square: {from_square}"
    assert 0 <= direction < 4, f"Invalid direction: {direction}"

    return from_square * 4 + direction


def decode_action(action: int) -> Tuple[int, int]:
    """
    Decode action index to (from_square, direction).

    Args:
        action: Action index (0-127)

    Returns:
        (from_square, direction) tuple
    """
    assert 0 <= action < NUM_ACTIONS, f"Invalid action: {action}"

    from_square = action // 4
    direction = action % 4

    return from_square, direction


def get_direction_name(direction: int) -> str:
    """Get human-readable direction name."""
    names = {0: "NW", 1: "NE", 2: "SW", 3: "SE"}
    return names.get(direction, "??")


def action_to_string(action: int) -> str:
    """Convert action to human-readable string."""
    from_square, direction = decode_action(action)
    dir_name = get_direction_name(direction)
    return f"sq{from_square:2d}→{dir_name}"


# Test the encoder
if __name__ == "__main__":
    print("Testing Fixed Action Space Encoder")
    print("=" * 50)

    # Test encoding/decoding
    print("\nEncoding tests:")
    test_cases = [
        (0, 0),    # Square 0, NW
        (0, 1),    # Square 0, NE
        (13, 1),   # Square 13, NE (should be action 53)
        (31, 3),   # Square 31, SE (should be action 127)
    ]

    for from_sq, direction in test_cases:
        action = encode_action(from_sq, direction)
        decoded_sq, decoded_dir = decode_action(action)

        print(f"  Square {from_sq:2d}, dir {direction} → "
              f"action {action:3d} → "
              f"square {decoded_sq:2d}, dir {decoded_dir} | "
              f"{action_to_string(action)}")

        assert from_sq == decoded_sq and direction == decoded_dir, "Encoding mismatch!"

    # Test all actions
    print(f"\nTesting all {NUM_ACTIONS} actions...")
    for action in range(NUM_ACTIONS):
        from_square, direction = decode_action(action)
        reconstructed = encode_action(from_square, direction)
        assert action == reconstructed, f"Mismatch at action {action}"

    print(f"✓ All {NUM_ACTIONS} actions encode/decode correctly!")

    # Show some example actions
    print("\nExample actions:")
    for action in [0, 1, 52, 53, 126, 127]:
        print(f"  Action {action:3d}: {action_to_string(action)}")

    print("\n✓ Action encoder tests passed!")
