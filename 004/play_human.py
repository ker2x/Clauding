"""
Human-controlled playable CarRacing-v3.

This script lets you drive the car directly using keyboard controls.
Useful for understanding the environment and getting a feel for the game difficulty.

Usage:
    # Play a game
    python play_human.py

    # Play multiple episodes
    python play_human.py --episodes 5

    # Faster playback
    python play_human.py --fps 60

    # Skip rendering (just compute rewards, for testing)
    python play_human.py --no-render
"""

import argparse
import cv2
import numpy as np
import time

from preprocessing import make_carracing_env


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description='Play CarRacing-v3 as a human')

    parser.add_argument('--episodes', type=int, default=1,
                        help='Number of episodes to play (default: 1)')
    parser.add_argument('--fps', type=int, default=30,
                        help='Display FPS (default: 30)')
    parser.add_argument('--no-render', action='store_true',
                        help='Disable rendering (just compute rewards)')

    return parser.parse_args()


def format_action(action):
    """
    Format continuous action for display.

    Args:
        action: Continuous action [steering, gas, brake]

    Returns:
        Human-readable action description
    """
    steering, gas, brake = action

    # Describe steering
    if steering < -0.3:
        steer_desc = f"LEFT({steering:.2f})"
    elif steering > 0.3:
        steer_desc = f"RIGHT({steering:.2f})"
    else:
        steer_desc = f"STRAIGHT({steering:.2f})"

    # Describe gas/brake
    if brake > 0.1:
        pedal_desc = f"BRAKE({brake:.2f})"
    elif gas > 0.1:
        pedal_desc = f"GAS({gas:.2f})"
    else:
        pedal_desc = "COAST"

    return f"{steer_desc} + {pedal_desc}"


def render_frame(frame, episode, step, reward, total_reward, action, controls_hint=None):
    """
    Render frame with overlay information.

    Args:
        frame: RGB frame from environment
        episode: Current episode number
        step: Current step number
        reward: Current step reward
        total_reward: Cumulative episode reward
        action: Continuous action [steering, gas, brake]
        controls_hint: Optional controls hint text

    Returns:
        Frame with overlay text
    """
    # Convert to BGR for OpenCV
    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

    # Add black bar at top for text
    frame = cv2.copyMakeBorder(frame, 130, 0, 0, 0, cv2.BORDER_CONSTANT, value=(0, 0, 0))

    # Add text overlay
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.5
    thickness = 1
    color = (255, 255, 255)

    cv2.putText(frame, f"HUMAN PLAYER (You're in control!)", (10, 20), font, font_scale, (0, 255, 100), thickness)
    cv2.putText(frame, f"Episode: {episode}", (10, 40), font, font_scale, color, thickness)
    cv2.putText(frame, f"Step: {step}", (10, 60), font, font_scale, color, thickness)

    # Display continuous action values
    action_str = format_action(action)
    cv2.putText(frame, f"Action: {action_str}", (10, 80), font, font_scale, (0, 255, 0), thickness)

    # Controls hint
    if controls_hint:
        cv2.putText(frame, controls_hint, (10, 100), font, font_scale, (100, 100, 255), thickness)

    cv2.putText(frame, f"Reward: {reward:+.2f}", (350, 40), font, font_scale, color, thickness)
    cv2.putText(frame, f"Total: {total_reward:+.2f}", (350, 60), font, font_scale, color, thickness)

    return frame


def update_action_from_key(current_action, key):
    """
    Update action based on a single key press.

    Args:
        current_action: Current action [steering, gas, brake]
        key: Key code from cv2.waitKey()

    Returns:
        Updated action [steering, gas, brake]

    Note: Action space is:
          steering: [-1, 1]
          gas: [0, 1]
          brake: [0, 1]
    """
    steering, gas, brake = current_action

    # Steering: A/D or arrow keys
    if key == ord('a') or key == ord('A'):
        steering = -1.0
    elif key == ord('d') or key == ord('D'):
        steering = 1.0
    elif key == 81:  # Left arrow (special OpenCV code)
        steering = -1.0
    elif key == 83:  # Right arrow (special OpenCV code)
        steering = 1.0

    # Gas and brake (in [0, 1] range)
    elif key == ord('w') or key == ord('W'):
        gas = 1.0   # Full acceleration
        brake = 0.0  # No braking
    elif key == ord('s') or key == ord('S'):
        brake = 1.0   # Full braking
        gas = 0.0  # No acceleration

    action = np.array([steering, gas, brake], dtype=np.float32)
    return action


def play_human(args):
    """Play episodes as a human."""
    # Create environment with rendering
    render_mode = None if args.no_render else 'rgb_array'
    env = make_carracing_env(
        stack_size=4,
        terminate_stationary=True,
        stationary_patience=100,
        render_mode=render_mode,
        state_mode='visual'  # Always visual for human play
    )

    action_dim = env.action_space.shape[0]
    state_shape = env.observation_space.shape

    print("=" * 60)
    print(f"CarRacing-v3 - HUMAN PLAYER")
    print("=" * 60)
    print(f"Episodes: {args.episodes}")
    print(f"State shape: {state_shape}")
    print(f"Action space: Continuous")
    print(f"  - Steering: [{env.action_space.low[0]:.1f}, {env.action_space.high[0]:.1f}]")
    print(f"  - Gas:      [{env.action_space.low[1]:.1f}, {env.action_space.high[1]:.1f}]")
    print(f"  - Brake:    [{env.action_space.low[2]:.1f}, {env.action_space.high[2]:.1f}]")
    print("=" * 60)

    if not args.no_render:
        print("\nKEYBOARD CONTROLS:")
        print("  Steering:")
        print("    - Arrow Left / A: Steer left")
        print("    - Arrow Right / D: Steer right")
        print("  Speed Control:")
        print("    - W / Arrow Up: Accelerate (gas)")
        print("    - S / Arrow Down: Brake")
        print("  Episode Control:")
        print("    - SPACE: Reset action to neutral (coast)")
        print("    - R: Reset current episode")
        print("    - Q / ESC: Quit")
        print("\nNote: You can combine controls (e.g., steer left + gas at same time)")
        print("=" * 60)

    # Statistics
    episode_rewards = []
    target_frame_time = 1.0 / args.fps

    try:
        for episode in range(args.episodes):
            state, _ = env.reset()
            total_reward = 0
            step = 0
            done = False

            # Current action state (persists between frames)
            # Action space: steering [-1, 1], gas [0, 1], brake [0, 1]
            action = np.array([0.0, 0.0, 0.0], dtype=np.float32)
            key_timeout = 0  # Frames since last key press

            print(f"\n{'-' * 60}")
            print(f"Episode {episode + 1}/{args.episodes}")
            print(f"{'-' * 60}")
            print("Ready! Use WASD/Arrows to control the car")

            while not done:
                frame_start = time.time()

                # Decay action after timeout (return to neutral if no input)
                # Action persists for ~10 frames, then gradually decays
                key_timeout += 1
                if key_timeout > 10:
                    # Smooth decay of action
                    action = action * 0.95

                # Take step with current action
                next_state, reward, terminated, truncated, _ = env.step(action)
                done = terminated or truncated

                total_reward += reward
                step += 1

                # Render if enabled
                if not args.no_render:
                    # Get RGB frame for display
                    rgb_frame = env.render()

                    # Add overlay
                    controls_hint = "SPACE: coast | R: reset | Q: quit"
                    display_frame = render_frame(
                        rgb_frame, episode + 1, step, reward, total_reward,
                        action, controls_hint
                    )

                    # Display
                    cv2.imshow('CarRacing-v3 - Human Player', display_frame)

                    # Handle keyboard input (non-blocking)
                    key = cv2.waitKey(1) & 0xFF

                    if key != 255:  # 255 means no key pressed
                        key_timeout = 0  # Reset timeout on any input

                        if key == 27 or key == ord('x') or key == ord('X'):  # ESC or x
                            print("\nQuitting...")
                            env.close()
                            cv2.destroyAllWindows()
                            return

                        elif key == ord(' '):  # SPACE - reset to neutral
                            action = np.array([0.0, 0.0, 0.0], dtype=np.float32)
                            print("Reset to neutral (coasting)")

                        elif key == ord('r') or key == ord('R'):  # R - reset episode
                            print("Resetting episode...")
                            break

                        # Control keys
                        else:
                            action = update_action_from_key(action, key)

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
        if not args.no_render:
            cv2.destroyAllWindows()

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
    print("\nTips for better performance:")
    print("- Smooth steering is better than sharp turns")
    print("- Avoid going off-road (grass)")
    print("- Keep consistent speed rather than braking hard")
    print("- A trained SAC agent can achieve 500+ reward!")
    print("=" * 60)


if __name__ == "__main__":
    args = parse_args()
    play_human(args)
