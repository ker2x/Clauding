"""Quick test to check if both players can move."""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from checkers.engine.game import CheckersGame

game = CheckersGame()

print("Initial position:")
print(game.render())
print(f"Player {game.current_player} has {len(game.get_legal_moves())} legal moves")

# Make a move
moves = game.get_legal_moves()
if moves:
    print(f"\nMaking move: {moves[0]}")
    game.make_move(moves[0])

print("\nAfter first move:")
print(game.render())
print(f"Player {game.current_player} has {len(game.get_legal_moves())} legal moves")
print(f"Game terminal: {game.is_terminal()}")

# Try second move
moves2 = game.get_legal_moves()
if moves2:
    print(f"\nMaking second move: {moves2[0]}")
    game.make_move(moves2[0])

    print("\nAfter second move:")
    print(game.render())
    print(f"Player {game.current_player} has {len(game.get_legal_moves())} legal moves")
