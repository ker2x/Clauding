#!/usr/bin/env python3
"""
Debug script to catch and analyze early draws.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import checkers_cpp
import random

def analyze_early_draw():
    """Try to reproduce an early draw and analyze it."""
    print("=" * 60)
    print("Trying to Reproduce Early Draw")
    print("=" * 60)

    # Try multiple games to catch an early draw
    for game_num in range(100):
        game = checkers_cpp.Game()

        for move_num in range(20):
            if game.is_terminal():
                # Found a terminal state!
                if move_num < 10 and game.player_kings == 0 and game.opponent_kings == 0:
                    print(f"\n🚨 EARLY DRAW DETECTED at move {move_num}!")
                    print(f"   Game #{game_num + 1}")
                    print(f"   No kings present: player_kings={game.player_kings}, opponent_kings={game.opponent_kings}")

                    # Analyze position history
                    history = game.position_history
                    print(f"\n   Position History (length={len(history)}):")
                    for i, pos in enumerate(history):
                        print(f"     [{i}] {pos}")

                    # Count duplicates
                    pos_counts = {}
                    for pos in history:
                        if pos in pos_counts:
                            pos_counts[pos] += 1
                        else:
                            pos_counts[pos] = 1

                    print(f"\n   Duplicate Positions:")
                    found_dup = False
                    for pos, count in pos_counts.items():
                        if count >= 3:
                            print(f"     {pos}: appears {count} times ← DRAW TRIGGER")
                            found_dup = True
                        elif count > 1:
                            print(f"     {pos}: appears {count} times")
                            found_dup = True

                    if not found_dup:
                        print(f"     No duplicates found! This shouldn't cause a draw!")

                    # Check current position
                    current_pos = (game.player_men, game.player_kings, game.opponent_men, game.opponent_kings)
                    print(f"\n   Current Position: {current_pos}")
                    print(f"   Current in history: {pos_counts.get(current_pos, 0)} times")

                    return True

            # Make a random move
            actions = game.get_legal_actions()
            if not actions:
                break

            # Choose randomly to create unpredictable patterns
            action = random.choice(actions)
            game.make_action(action)

    print("\n   Could not reproduce early draw in 100 games")
    return False

def main():
    print("\n" + "=" * 60)
    print("Early Draw Debug Script")
    print("=" * 60)

    random.seed(42)  # For reproducibility
    success = analyze_early_draw()

    if not success:
        print("\n⚠️  No early draw detected. Try running training to see the bug.")
        return 0

    return 1

if __name__ == "__main__":
    sys.exit(main())
