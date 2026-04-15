#!/usr/bin/env python3
"""Watch random agents fight each other.

This script runs an automated demo where two random agents continuously
fight each other. Useful for:
- Testing physics stability
- Visual debugging
- Watching chaos

Controls:
    SPACE: Toggle auto-play on/off
    R: Reset match
    Q/ESC: Quit

Usage:
    ../.venv/bin/python scripts/watch_random.py
"""

import sys
sys.path.insert(0, sys.path[0] + '/..')

import random
import pygame
from config.body_config import JointState, DEFAULT_BODY
from config.env_config import EnvConfig
from game.match import Match
from rendering.pygame_renderer import PygameRenderer


def main():
    """Run the random agent demo loop."""
    # Create match with default config
    config = EnvConfig(max_turns=20, steps_per_turn=30)
    match = Match(config)
    renderer = PygameRenderer(match, mode="human")

    # Get number of joints for action generation
    n_joints = DEFAULT_BODY.num_joints
    
    # Game state
    running = True
    auto_play = True
    turn_timer = 0
    sim_phase = False
    sim_step = 0

    # Initialize pygame before event loop
    pygame.init()

    # Main game loop
    while running:
        # Process pygame events
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_q or event.key == pygame.K_ESCAPE:
                    running = False
                elif event.key == pygame.K_r:
                    # Reset match
                    match = Match(config)
                    sim_phase = False
                    print("Reset!")
                elif event.key == pygame.K_SPACE:
                    # Toggle auto-play
                    auto_play = not auto_play
                    print(f"Auto-play: {'ON' if auto_play else 'OFF'}")

        # Auto-play logic
        if auto_play and not match.is_done():
            if not sim_phase:
                # Set random actions for both players
                for player in range(2):
                    states = [JointState(random.randint(0, 3)) for _ in range(n_joints)]
                    match.set_actions(player, states)
                sim_phase = True
                sim_step = 0
                match.world.collision_handler.clear_turn()

            if sim_phase:
                # Animate simulation step-by-step
                match.world.step()
                sim_step += 1

                if sim_step >= config.steps_per_turn:
                    sim_phase = False
                    # Compute and apply turn results
                    from game.scoring import compute_turn_result, EXEMPT_GROUND_SEGMENTS, GROUND_PENALTIES
                    result = compute_turn_result(match.world.collision_handler, match.config)
                    match.scores[0] += result.damage_a_to_b
                    match.scores[1] += result.damage_b_to_a
                    bad_a = result.ground_segments_a - EXEMPT_GROUND_SEGMENTS
                    bad_b = result.ground_segments_b - EXEMPT_GROUND_SEGMENTS
                    for seg in bad_a:
                        match.scores[0] += GROUND_PENALTIES.get(seg, GROUND_PENALTIES["default"])
                    for seg in bad_b:
                        match.scores[1] += GROUND_PENALTIES.get(seg, GROUND_PENALTIES["default"])
                    match.turn_results.append(result)
                    match.turn += 1

                    print(f"Turn {match.turn}/{config.max_turns}: "
                          f"A={match.scores[0]:.1f} B={match.scores[1]:.1f}")

        # Auto-reset after match ends
        if match.is_done() and auto_play:
            winner = match.get_winner()
            print(f"\nMatch over! Winner: {'A' if winner == 0 else 'B' if winner == 1 else 'Draw'}")
            print(f"Final scores: A={match.scores[0]:.1f}, B={match.scores[1]:.1f}")
            
            # Pause before auto-reset
            pygame.time.wait(1500)
            match = Match(config)
            sim_phase = False

        # Render frame
        renderer.render(match)
        renderer.clock.tick(60)

    # Clean up
    renderer.close()


if __name__ == "__main__":
    main()
