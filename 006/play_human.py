"""
Human-controlled playable CarRacing-v3 with arcade-style keyboard controls.

This script uses Pygame to properly handle continuous key presses (key-down)
and key releases, providing a much more intuitive "arcade" driving experience.

Usage:
    python play_human.py

Controls:
    - Steering:   A/D, Left/Right
    - Gas:        W, Z, Up Arrow
    - Brake:      S, Down Arrow
    - Reset:      R
    - Quit:       ESC or Q
"""

import argparse
import numpy as np
import time
import pygame
import sys

from preprocessing import make_carracing_env
from utils.display import format_action, get_car_speed


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description='Play CarRacing-v3 as a human')
    parser.add_argument('--episodes', type=int, default=5,
                        help='Number of episodes to play (default: 5)')
    parser.add_argument('--fps', type=int, default=50,
                        help='Display FPS (default: 50)')
    parser.add_argument('--no-render', action='store_true',
                        help='Disable rendering (just compute rewards)')
    return parser.parse_args()


def render_info(screen, font, episode, step, reward, total_reward, action, speed_kmh=0.0):
    """Render text overlay onto the pygame screen."""
    info_area_height = 130
    w, h = screen.get_size()

    # Clear the top info bar
    screen.fill((0, 0, 0), (0, 0, w, info_area_height))

    def draw_text(text, y, color=(255, 255, 255)):
        text_surf = font.render(text, True, color)
        screen.blit(text_surf, (10, y))

    def draw_text_right(text, y, color=(255, 255, 255)):
        text_surf = font.render(text, True, color)
        screen.blit(text_surf, (w - text_surf.get_width() - 10, y))

    draw_text(f"HUMAN PLAYER (You're in control!)", 10, (0, 255, 100))
    draw_text(f"Episode: {episode}", 30)
    draw_text(f"Step: {step}", 50)

    action_str = format_action(action)
    draw_text(f"Action: {action_str}", 70, (0, 255, 0))
    draw_text(f"Controls: Arrows/ZQSD (W/S) | R: reset | Q/ESC: quit", 90, (100, 100, 255))

    draw_text_right(f"Reward: {reward:+.2f}", 30)
    draw_text_right(f"Total: {total_reward:+.2f}", 50)
    draw_text_right(f"Speed: {speed_kmh:.1f} km/h", 70, (255, 255, 100))


def play_human(args):
    """Play episodes as a human using arcade-style controls."""

    # Initialize Pygame
    pygame.init()
    pygame.font.init()
    font = pygame.font.Font(None, 24)
    screen = None

    # Create environment
    render_mode = None if args.no_render else 'rgb_array'
    env = make_carracing_env(
        terminate_stationary=False,
        stationary_patience=100,
        render_mode=render_mode
    )

    action_dim = env.action_space.shape[0]
    state_shape = env.observation_space.shape

    print("=" * 60)
    print(f"CarRacing-v3 - HUMAN PLAYER (Arcade Controls)")
    print("=" * 60)
    print(f"Episodes: {args.episodes}")
    print(f"State shape: {state_shape}")
    print("=" * 60)
    print("\nKEYBOARD CONTROLS:")
    print("  - Steering:   A / D or Left / Right Arrow")
    print("  - Gas:        W / Z or Up Arrow")
    print("  - Brake:      S or Down Arrow")
    print("  - Reset:      R")
    print("  - Quit:       Q or ESC")
    print("\nNote: Releasing keys will return to neutral (coast).")
    print("=" * 60)

    episode_rewards = []
    target_frame_time = 1.0 / args.fps

    try:
        for episode in range(args.episodes):
            state, _ = env.reset()
            total_reward = 0
            step = 0
            done = False

            # Action: [steering, acceleration] (2D action space)
            action = np.array([0.0, 0.0], dtype=np.float32)

            # Smooth steering state
            current_steering = 0.0

            print(f"\n{'-' * 60}")
            print(f"Episode {episode + 1}/{args.episodes}")
            print(f"{'-' * 60}")
            print("Ready! Use keyboard to control the car")

            while not done:
                frame_start = time.time()

                # --- Pygame Event Handling (Quit/Reset) ---
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        print("\nQuitting...")
                        env.close()
                        pygame.quit()
                        sys.exit()
                    if event.type == pygame.KEYDOWN:
                        if event.key == pygame.K_r:
                            print("Resetting episode...")
                            done = True  # Break inner loop to reset
                        if event.key == pygame.K_x or event.key == pygame.K_ESCAPE:
                            print("\nQuitting...")
                            env.close()
                            pygame.quit()
                            sys.exit()

                # --- Arcade-Style Input (Key-Down State) ---
                keys = pygame.key.get_pressed()

                # 1. Acceleration (combines gas and brake into single dimension)
                # Note: ZQSD (French AZERTY) or WASD (US QWERTY)
                gas = 1.0 if keys[pygame.K_w] or keys[pygame.K_z] or keys[pygame.K_UP] else 0.0
                brake = 1.0 if keys[pygame.K_s] or keys[pygame.K_DOWN] else 0.0

                # Convert to single acceleration dimension: [-1, +1]
                if brake > 0:
                    acceleration = -brake  # Negative = brake
                else:
                    acceleration = gas     # Positive = gas

                # 2. Steering (Smoothed)
                # Note: Q/D (AZERTY) or A/D (QWERTY)
                target_steering = 0.0
                if keys[pygame.K_a] or keys[pygame.K_q] or keys[pygame.K_LEFT]:
                    target_steering = -1.0
                if keys[pygame.K_d] or keys[pygame.K_RIGHT]:
                    target_steering = 1.0  # Use 'if' not 'elif' to handle both keys

                # Smoothly interpolate to the target steering
                # This creates a "return to center" effect when keys are released
                STEER_SPEED = 0.1 # How fast to turn the wheel (10% per frame)
                current_steering = (1.0 - STEER_SPEED) * current_steering + STEER_SPEED * target_steering

                # Assemble the final action: [steering, acceleration]
                action = np.array([current_steering, acceleration], dtype=np.float32)

                # --- Step Environment ---
                next_state, reward, terminated, truncated, _ = env.step(action)
                if done: # If we reset, 'terminated' might be false
                    break

                done = terminated or truncated
                total_reward += reward
                step += 1

                # --- Get car speed ---
                speed_kmh = get_car_speed(env)

                # --- Render (if enabled) ---
                if not args.no_render:
                    rgb_frame = env.render()
                    frame_h, frame_w = rgb_frame.shape[:2]
                    info_area_height = 130 # Space for text

                    # Create screen on first frame
                    if screen is None:
                        screen = pygame.display.set_mode((frame_w, frame_h + info_area_height))
                        pygame.display.set_caption("CarRacing-v3 - Human Player")

                    # Convert env frame (H, W, 3) to pygame surface (W, H)
                    # We must transpose the array for pygame
                    frame_pygame = np.transpose(rgb_frame, (1, 0, 2))
                    surf = pygame.surfarray.make_surface(frame_pygame)

                    # Draw game frame and info
                    screen.blit(surf, (0, info_area_height))
                    render_info(screen, font, episode + 1, step, reward, total_reward, action, speed_kmh)

                    pygame.display.flip()

                    # FPS control
                    elapsed = time.time() - frame_start
                    if elapsed < target_frame_time:
                        time.sleep(target_frame_time - elapsed)

                state = next_state

            # Episode summary
            episode_rewards.append(total_reward)
            print(f"Episode {episode + 1} finished:")
            print(f"  Steps: {step}")
            print(f"  Total reward: {total_reward:.2f}")
            print(f"  Average reward so far: {np.mean(episode_rewards):.2f}")

    except KeyboardInterrupt:
        print("\nInterrupted by user")
    finally:
        env.close()
        pygame.quit()

    # Final statistics
    print("\n" + "=" * 60)
    print("GAME SUMMARY")
    print("=" * 60)
    print(f"Episodes completed: {len(episode_rewards)}")
    if episode_rewards:
        print(f"Average reward: {np.mean(episode_rewards):.2f}")
        print(f"Std dev: {np.std(episode_rewards):.2f}")
        print(f"Min reward: {np.min(episode_rewards):.2f}")
        print(f"Max reward: {np.max(episode_rewards):.2f}")
    print("=" * 60)


if __name__ == "__main__":
    args = parse_args()
    play_human(args)
