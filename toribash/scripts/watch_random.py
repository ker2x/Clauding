#!/usr/bin/env python3
"""Watch random agents fight each other."""

import sys
sys.path.insert(0, sys.path[0] + '/..')

import random
import pygame
from config.body_config import JointState, DEFAULT_BODY
from config.env_config import EnvConfig
from game.match import Match
from rendering.pygame_renderer import PygameRenderer


def main():
    config = EnvConfig(max_turns=20, steps_per_turn=30)
    match = Match(config)
    renderer = PygameRenderer(match, mode="human")

    n_joints = DEFAULT_BODY.num_joints
    running = True
    auto_play = True
    turn_timer = 0
    SIM_PHASE = False
    sim_step = 0

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_q or event.key == pygame.K_ESCAPE:
                    running = False
                elif event.key == pygame.K_r:
                    match = Match(config)
                    SIM_PHASE = False
                    print("Reset!")
                elif event.key == pygame.K_SPACE:
                    auto_play = not auto_play
                    print(f"Auto-play: {'ON' if auto_play else 'OFF'}")

        if auto_play and not match.is_done():
            if not SIM_PHASE:
                # Set random actions for both players
                for player in range(2):
                    states = [JointState(random.randint(0, 3)) for _ in range(n_joints)]
                    match.set_actions(player, states)
                SIM_PHASE = True
                sim_step = 0
                match.world.collision_handler.clear_turn()

            if SIM_PHASE:
                # Animate simulation
                match.world.step()
                sim_step += 1

                if sim_step >= config.steps_per_turn:
                    SIM_PHASE = False
                    from game.scoring import compute_turn_result
                    result = compute_turn_result(match.world.collision_handler, match.config)
                    match.scores[0] += result.damage_a_to_b
                    match.scores[1] += result.damage_b_to_a
                    match.turn_results.append(result)
                    match.turn += 1

                    print(f"Turn {match.turn}/{config.max_turns}: "
                          f"A={match.scores[0]:.1f} B={match.scores[1]:.1f}")

        if match.is_done() and auto_play:
            winner = match.get_winner()
            print(f"\nMatch over! Winner: {'A' if winner == 0 else 'B' if winner == 1 else 'Draw'}")
            print(f"Final scores: A={match.scores[0]:.1f}, B={match.scores[1]:.1f}")
            # Auto-reset after a pause
            pygame.time.wait(1500)
            match = Match(config)
            SIM_PHASE = False

        renderer.render(match)
        renderer.clock.tick(60)

    renderer.close()


if __name__ == "__main__":
    main()
