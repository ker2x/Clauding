#!/usr/bin/env python3
"""Human vs Human (or human vs AI) Toribash 2D gameplay.

This script provides an interactive game loop where humans can:
- Set joint states by clicking in the bottom panel
- Use keyboard shortcuts (1-4) for quick state changes
- Watch animated turn simulation at 60 FPS
- TAB between players to set their joint states

Controls:
    Mouse click: Cycle joint state (left=forward, right=backward)
    1: All joints CONTRACT
    2: All joints EXTEND
    3: All joints HOLD
    4: All joints RELAX
    TAB: Switch active player (A or B)
    SPACE: Simulate turn (animated at 60 FPS)
    R: Reset match
    Q/ESC: Quit

Usage:
    ../.venv/bin/python scripts/play_human.py
"""

import sys
sys.path.insert(0, sys.path[0] + '/..')

import pygame
from config.body_config import JointState, DEFAULT_BODY
from config.env_config import EnvConfig
from game.match import Match
from rendering.pygame_renderer import PygameRenderer, VIEWPORT_HEIGHT, SCREEN_WIDTH


def main():
    """Run the human vs human game loop."""
    # Create match with default config
    config = EnvConfig(max_turns=20, steps_per_turn=30)
    match = Match(config)

    # Create renderer
    renderer = PygameRenderer(match, mode="human")
    renderer.render(match)

    # Game state
    active_player = 0  # 0 = Player A, 1 = Player B
    running = True
    simulating = False
    sim_steps_remaining = 0

    # For smooth simulation playback
    PLAYBACK_STEPS = config.steps_per_turn
    playback_step = 0

    # Main game loop
    while running:
        # Process pygame events
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_q or event.key == pygame.K_ESCAPE:
                    running = False

                # Start turn simulation
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

                # Reset match
                elif event.key == pygame.K_r:
                    match = Match(config)
                    active_player = 0
                    print("Match reset!")

                # Switch active player
                elif event.key == pygame.K_TAB:
                    active_player = 1 - active_player
                    print(f"Editing Player {'A' if active_player == 0 else 'B'}")

                # Quick joint state keys (set all joints at once)
                elif event.key == pygame.K_1:
                    _set_all_joints(match, active_player, JointState.CONTRACT)
                elif event.key == pygame.K_2:
                    _set_all_joints(match, active_player, JointState.EXTEND)
                elif event.key == pygame.K_3:
                    _set_all_joints(match, active_player, JointState.HOLD)
                elif event.key == pygame.K_4:
                    _set_all_joints(match, active_player, JointState.RELAX)

            # Mouse click for individual joint control
            elif event.type == pygame.MOUSEBUTTONDOWN and not simulating:
                # Check if clicking on a joint in the panel
                for player_idx in range(2):
                    joint_name = renderer.get_joint_at_pos(event.pos, player_idx)
                    if joint_name:
                        ragdoll = match.world.ragdoll_a if player_idx == 0 else match.world.ragdoll_b
                        current = ragdoll.joint_states.get(joint_name, JointState.HOLD)
                        
                        if event.button == 1:  # Left click: cycle forward
                            new_state = JointState((current + 1) % 4)
                        else:  # Right click: cycle backward
                            new_state = JointState((current - 1) % 4)
                        
                        ragdoll.set_joint_state(joint_name, new_state)
                        break

        # Animated simulation (step-by-step physics)
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

        # Render frame
        renderer.render(match)
        renderer.clock.tick(60)

    # Clean up
    renderer.close()


def _set_all_joints(match: Match, player: int, state: JointState):
    """Set all joints for a player to a given state.
    
    Args:
        match: Current match instance.
        player: Player index (0 or 1).
        state: JointState to set all joints to.
    """
    ragdoll = match.world.ragdoll_a if player == 0 else match.world.ragdoll_b
    ragdoll.set_all_joint_states([state] * DEFAULT_BODY.num_joints)
    print(f"Player {'A' if player == 0 else 'B'}: all joints -> {state.name}")


if __name__ == "__main__":
    main()
