"""
Unit tests for the checkers game engine.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from checkers.engine.game import CheckersGame, play_random_game
from checkers.engine.bitboard import *
from checkers.engine.moves import Move


def test_initial_position():
    """Test that initial position is set up correctly."""
    game = CheckersGame()

    # Check piece counts
    player_men, player_kings, opp_men, opp_kings = game.get_piece_counts()
    assert player_men == 20, f"Expected 20 player men, got {player_men}"
    assert player_kings == 0, f"Expected 0 player kings, got {player_kings}"
    assert opp_men == 20, f"Expected 20 opponent men, got {opp_men}"
    assert opp_kings == 0, f"Expected 0 opponent kings, got {opp_kings}"

    # Check that there are legal moves
    legal_moves = game.get_legal_moves()
    assert len(legal_moves) > 0, "No legal moves at start"
    print(f"Initial position has {len(legal_moves)} legal moves")


def test_move_generation():
    """Test that moves are generated correctly."""
    game = CheckersGame()
    legal_moves = game.get_legal_moves()

    # All initial moves should be forward moves (no captures)
    for move in legal_moves:
        assert len(move.captured_squares) == 0, "Initial position should have no captures"
        print(f"Move: {move}")

    # Test that we can make a move
    if legal_moves:
        first_move = legal_moves[0]
        print(f"\nMaking move: {first_move}")
        game.make_move(first_move)
        print(game.render())


def test_game_termination():
    """Test game termination detection."""
    game = CheckersGame()

    # Initially, game should not be terminal
    assert not game.is_terminal(), "Game should not be terminal at start"

    # Play until terminal
    move_count = 0
    max_moves = 300

    while not game.is_terminal() and move_count < max_moves:
        legal_moves = game.get_legal_moves()
        if not legal_moves:
            break
        game.make_move(legal_moves[0])  # Take first move
        move_count += 1

    print(f"\nGame ended after {move_count} moves")
    print(f"Terminal: {game.is_terminal()}")
    print(f"Result: {game.get_result()}")
    print(f"Winner: {game.get_winner()}")


def test_clone():
    """Test game state cloning."""
    game = CheckersGame()
    legal_moves = game.get_legal_moves()

    if legal_moves:
        # Clone before move
        clone = game.clone()

        # Make move in original
        game.make_move(legal_moves[0])

        # Clone should be unchanged
        assert clone.move_count == 0
        assert clone.current_player == 1
        clone_pieces = clone.get_piece_counts()
        assert clone_pieces[0] == 20  # Still has 20 men

        print("Clone test passed")


def test_neural_input():
    """Test neural network input generation."""
    game = CheckersGame()
    neural_input = game.to_neural_input()

    # Check shape
    assert neural_input.shape == (8, 10, 10), f"Wrong shape: {neural_input.shape}"

    # Check that planes 0-3 have pieces
    assert neural_input[0].sum() == 20, "Plane 0 should have 20 pieces"
    assert neural_input[2].sum() == 20, "Plane 2 should have 20 pieces"

    # Check that plane 7 is all ones
    assert neural_input[7].sum() == 100, "Plane 7 should be all ones"

    print("Neural input test passed")
    print(f"Input shape: {neural_input.shape}")
    print(f"Plane 0 sum (player men): {neural_input[0].sum()}")
    print(f"Plane 2 sum (opponent men): {neural_input[2].sum()}")


def test_random_game():
    """Test playing a complete random game."""
    print("\nPlaying random game...")
    winner, moves = play_random_game(max_moves=200)
    print(f"Game finished in {moves} moves, winner: {winner}")


def run_all_tests():
    """Run all tests."""
    print("=" * 60)
    print("Testing Checkers Game Engine")
    print("=" * 60)

    tests = [
        ("Initial Position", test_initial_position),
        ("Move Generation", test_move_generation),
        ("Game Termination", test_game_termination),
        ("Clone", test_clone),
        ("Neural Input", test_neural_input),
        ("Random Game", test_random_game),
    ]

    passed = 0
    failed = 0

    for name, test_func in tests:
        try:
            print(f"\n--- Test: {name} ---")
            test_func()
            print(f"✓ {name} passed")
            passed += 1
        except Exception as e:
            print(f"✗ {name} failed: {e}")
            import traceback
            traceback.print_exc()
            failed += 1

    print("\n" + "=" * 60)
    print(f"Results: {passed} passed, {failed} failed")
    print("=" * 60)


if __name__ == "__main__":
    run_all_tests()
