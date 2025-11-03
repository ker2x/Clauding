"""
Watch a random agent play CarRacing-v3.

This script is useful for:
1. Testing environment setup and rendering
2. Understanding baseline performance (random policy)
3. Visualizing the discretized action space
4. Debugging environment issues

The random agent selects actions uniformly at random from the discretized action space.

Usage:
    # Watch random agent
    python watch_random_agent.py

    # Watch for specific number of episodes
    python watch_random_agent.py --episodes 3

    # Use finer action discretization
    python watch_random_agent.py --steering-bins 5 --gas-brake-bins 3
"""

import argparse
import cv2
import numpy as np
import time

from preprocessing import make_carracing_env


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description='Watch random agent play CarRacing-v3')

    parser.add_argument('--episodes', type=int, default=3,
                        help='Number of episodes to watch (default: 3)')
    parser.add_argument('--fps', type=int, default=30,
                        help='Display FPS (default: 30)')
    parser.add_argument('--steering-bins', type=int, default=3,
                        help='Number of discrete steering values (default: 3)')
    parser.add_argument('--gas-brake-bins', type=int, default=3,
                        help='Number of discrete gas/brake values (default: 3)')
    parser.add_argument('--no-render', action='store_true',
                        help='Disable rendering (just compute rewards)')

    return parser.parse_args()


def render_frame(frame, episode, step, reward, total_reward, action_meaning):
    """
    Render frame with overlay information.

    Args:
        frame: RGB frame from environment
        episode: Current episode number
        step: Current step number
        reward: Current step reward
        total_reward: Cumulative episode reward
        action_meaning: Human-readable action description

    Returns:
        Frame with overlay text
    """
    # Convert to BGR for OpenCV
    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

    # Resize for better visibility
    display_size = (600, 600)
    frame = cv2.resize(frame, display_size, interpolation=cv2.INTER_NEAREST)

    # Add black bar at top for text
    frame = cv2.copyMakeBorder(frame, 80, 0, 0, 0, cv2.BORDER_CONSTANT, value=(0, 0, 0))

    # Add text overlay
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.5
    thickness = 1
    color = (255, 255, 255)

    cv2.putText(frame, f"RANDOM AGENT", (10, 20), font, font_scale, (0, 255, 255), thickness)
    cv2.putText(frame, f"Episode: {episode}", (10, 40), font, font_scale, color, thickness)
    cv2.putText(frame, f"Step: {step}", (10, 60), font, font_scale, color, thickness)

    cv2.putText(frame, f"Action: {action_meaning}", (300, 20), font, font_scale, color, thickness)
    cv2.putText(frame, f"Reward: {reward:+.2f}", (300, 40), font, font_scale, color, thickness)
    cv2.putText(frame, f"Total: {total_reward:+.2f}", (300, 60), font, font_scale, color, thickness)

    return frame


def watch_random_agent(args):
    """Watch random agent play episodes."""
    # Create environment with rendering
    render_mode = None if args.no_render else 'rgb_array'
    env = make_carracing_env(
        stack_size=4,
        frame_size=(84, 84),
        discretize_actions=True,
        steering_bins=args.steering_bins,
        gas_brake_bins=args.gas_brake_bins,
        shape_rewards=True,
        terminate_stationary=False,  # Full episodes for watching
        render_mode=render_mode
    )

    n_actions = env.action_space.n
    state_shape = env.observation_space.shape

    print("=" * 60)
    print(f"Random Agent on CarRacing-v3")
    print("=" * 60)
    print(f"Episodes: {args.episodes}")
    print(f"State shape: {state_shape}")
    print(f"Number of actions: {n_actions}")
    print(f"Steering bins: {args.steering_bins}")
    print(f"Gas/Brake bins: {args.gas_brake_bins}")
    print("=" * 60)

    # Get action meanings
    from preprocessing import ActionDiscretizer
    temp_env = env
    while hasattr(temp_env, 'env'):
        if isinstance(temp_env, ActionDiscretizer):
            action_meanings = temp_env.get_action_meanings()
            break
        temp_env = temp_env.env
    else:
        action_meanings = [f"Action {i}" for i in range(n_actions)]

    print("\nDiscrete actions:")
    for i, meaning in enumerate(action_meanings):
        print(f"  {i}: {meaning}")

    if not args.no_render:
        print("\nControls:")
        print("  ESC or Q: Quit")
        print("  SPACE: Pause/Resume")
        print("  R: Reset episode")

    # Statistics
    episode_rewards = []

    # Target frame time for FPS control
    frame_time = 1.0 / args.fps

    try:
        for episode in range(args.episodes):
            state, _ = env.reset()
            total_reward = 0
            step = 0
            done = False
            paused = False

            print(f"\n{'-' * 60}")
            print(f"Episode {episode + 1}/{args.episodes}")
            print(f"{'-' * 60}")

            while not done:
                frame_start = time.time()

                # Select random action
                action = env.action_space.sample()
                action_meaning = action_meanings[action]

                # Take step
                next_state, reward, terminated, truncated, _ = env.step(action)
                done = terminated or truncated

                total_reward += reward
                step += 1

                # Render if enabled
                if not args.no_render:
                    # Get RGB frame for display
                    rgb_frame = env.render()

                    # Add overlay
                    display_frame = render_frame(
                        rgb_frame, episode + 1, step, reward, total_reward,
                        action_meaning
                    )

                    # Display
                    cv2.imshow('Random Agent - CarRacing-v3', display_frame)

                    # Handle keyboard input
                    key = cv2.waitKey(1) & 0xFF

                    if key == 27 or key == ord('q'):  # ESC or Q
                        print("\nQuitting...")
                        env.close()
                        cv2.destroyAllWindows()
                        return

                    elif key == ord(' '):  # SPACE
                        paused = not paused
                        if paused:
                            print("Paused (press SPACE to resume)")
                        else:
                            print("Resumed")

                    elif key == ord('r'):  # R
                        print("Resetting episode...")
                        break

                    # Handle pause
                    while paused:
                        key = cv2.waitKey(100) & 0xFF
                        if key == ord(' '):
                            paused = False
                            print("Resumed")
                        elif key == 27 or key == ord('q'):
                            env.close()
                            cv2.destroyAllWindows()
                            return

                    # FPS control
                    elapsed = time.time() - frame_start
                    if elapsed < frame_time:
                        time.sleep(frame_time - elapsed)

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
    print("Summary")
    print("=" * 60)
    print(f"Episodes completed: {len(episode_rewards)}")
    print(f"Average reward: {np.mean(episode_rewards):.2f}")
    print(f"Std dev: {np.std(episode_rewards):.2f}")
    print(f"Min reward: {np.min(episode_rewards):.2f}")
    print(f"Max reward: {np.max(episode_rewards):.2f}")
    print("=" * 60)
    print("\nNote: This is baseline performance with random actions.")
    print("A trained agent should significantly outperform this!")


if __name__ == "__main__":
    args = parse_args()
    watch_random_agent(args)
