#!/usr/bin/env python3
"""Human vs human (or human vs AI) Toribash 2D gameplay."""

import sys
sys.path.insert(0, sys.path[0] + '/..')

import pygame
from config.body_config import JointState, DEFAULT_BODY
from config.env_config import EnvConfig
from game.match import Match
from rendering.pygame_renderer import PygameRenderer, VIEWPORT_HEIGHT, SCREEN_WIDTH


def main():
    config = EnvConfig(max_turns=20, steps_per_turn=30)
    match = Match(config)

    renderer = PygameRenderer(match, mode="human")
    renderer.render(match)

    # Which player is being edited (0=A, 1=B)
    active_player = 0
    running = True
    simulating = False
    sim_steps_remaining = 0

    # For smooth simulation playback
    PLAYBACK_STEPS = config.steps_per_turn
    playback_step = 0

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_q or event.key == pygame.K_ESCAPE:
                    running = False

                elif event.key == pygame.K_SPACE and not simulating:
                    if match.is_done():
                        print(f"Match over! Scores: A={match.scores[0]:.1f}, B={match.scores[1]:.1f}")
                        winner = match.get_winner()
                        print(f"Winner: {'A' if winner == 0 else 'B' if winner == 1 else 'Draw'}")
                    else:
                        # Start animated simulation
                        simulating = True
                        playback_step = 0
                        match.world.collision_handler.clear_turn()

                elif event.key == pygame.K_r:
                    match = Match(config)
                    active_player = 0
                    print("Match reset!")

                elif event.key == pygame.K_TAB:
                    active_player = 1 - active_player
                    print(f"Editing Player {'A' if active_player == 0 else 'B'}")

                # Quick joint state keys (when hovering would be complex, use number keys for all)
                elif event.key == pygame.K_1:
                    _set_all_joints(match, active_player, JointState.CONTRACT)
                elif event.key == pygame.K_2:
                    _set_all_joints(match, active_player, JointState.EXTEND)
                elif event.key == pygame.K_3:
                    _set_all_joints(match, active_player, JointState.HOLD)
                elif event.key == pygame.K_4:
                    _set_all_joints(match, active_player, JointState.RELAX)

            elif event.type == pygame.MOUSEBUTTONDOWN and not simulating:
                mx, my = event.pos
                # Check if clicking on a joint in the panel
                for player_idx in range(2):
                    joint_name = renderer.get_joint_at_pos(event.pos, player_idx)
                    if joint_name:
                        ragdoll = match.world.ragdoll_a if player_idx == 0 else match.world.ragdoll_b
                        current = ragdoll.joint_states.get(joint_name, JointState.HOLD)
                        if event.button == 1:  # left click: cycle forward
                            new_state = JointState((current + 1) % 4)
                        else:  # right click: cycle backward
                            new_state = JointState((current - 1) % 4)
                        ragdoll.set_joint_state(joint_name, new_state)
                        break

        # Animated simulation
        if simulating:
            match.world.step()
            playback_step += 1

            if playback_step >= PLAYBACK_STEPS:
                simulating = False
                # Finalize turn scoring
                from game.scoring import compute_turn_result
                result = compute_turn_result(match.world.collision_handler, match.config)
                match.scores[0] += result.damage_a_to_b
                match.scores[1] += result.damage_b_to_a
                match.total_damage[0] += result.damage_b_to_a
                match.total_damage[1] += result.damage_a_to_b
                match.turn_results.append(result)
                match.turn += 1

                print(f"Turn {match.turn}: A={match.scores[0]:.1f} B={match.scores[1]:.1f} "
                      f"| Ground A: {result.ground_segments_a} | Ground B: {result.ground_segments_b}")

        renderer.render(match)
        renderer.clock.tick(60)

    renderer.close()


def _set_all_joints(match: Match, player: int, state: JointState):
    """Set all joints for a player to a given state."""
    ragdoll = match.world.ragdoll_a if player == 0 else match.world.ragdoll_b
    ragdoll.set_all_joint_states([state] * DEFAULT_BODY.num_joints)
    print(f"Player {'A' if player == 0 else 'B'}: all joints -> {state.name}")


if __name__ == "__main__":
    main()
