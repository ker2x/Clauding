"""
Watch a trained SAC agent play CarRacing-v3.

This script loads a trained agent from a checkpoint and visualizes its performance.
Uses OpenCV for rendering (more reliable than pygame on macOS).

Usage:
    # Watch agent from checkpoint
    python watch_agent.py --checkpoint checkpoints/best_model.pt

    # Watch for specific number of episodes
    python watch_agent.py --checkpoint checkpoints/best_model.pt --episodes 3

    # Watch without display (just print rewards)
    python watch_agent.py --checkpoint checkpoints/best_model.pt --no-render
"""

import argparse
import cv2
import numpy as np
import time
import torch

from preprocessing import make_carracing_env
from sac_agent import SACAgent


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description='Watch trained SAC agent play CarRacing-v3')

    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to agent checkpoint')
    parser.add_argument('--episodes', type=int, default=5,
                        help='Number of episodes to watch (default: 5)')
    parser.add_argument('--no-render', action='store_true',
                        help='Disable rendering (just compute rewards)')
    parser.add_argument('--fps', type=int, default=30,
                        help='Display FPS (default: 30)')

    return parser.parse_args()


def format_action(action):
    """
    Format continuous action for display.

    Args:
        action: Continuous action [steering, acceleration] (2D) or [steering, gas, brake] (3D, old)

    Returns:
        Human-readable action description
    """
    if len(action) == 2:
        # New 2D action space: [steering, acceleration]
        steering, accel = action

        # Describe steering
        if steering < -0.3:
            steer_desc = f"LEFT({steering:.2f})"
        elif steering > 0.3:
            steer_desc = f"RIGHT({steering:.2f})"
        else:
            steer_desc = f"STRAIGHT({steering:.2f})"

        # Describe acceleration
        if accel > 0.1:
            pedal_desc = f"GAS({accel:.2f})"
        elif accel < -0.1:
            pedal_desc = f"BRAKE({-accel:.2f})"
        else:
            pedal_desc = "COAST"
    else:
        # Old 3D action space: [steering, gas, brake]
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


def render_frame(frame, episode, step, reward, total_reward, action, alpha):
    """
    Render frame with overlay information.

    Args:
        frame: RGB frame from environment
        episode: Current episode number
        step: Current step number
        reward: Current step reward
        total_reward: Cumulative episode reward
        action: Continuous action [steering, gas, brake]
        alpha: Entropy coefficient

    Returns:
        Frame with overlay text
    """
    # Convert to BGR for OpenCV
    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

    # Add black bar at top for text
    frame = cv2.copyMakeBorder(frame, 100, 0, 0, 0, cv2.BORDER_CONSTANT, value=(0, 0, 0))

    # Add text overlay
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.5
    thickness = 1
    color = (255, 255, 255)

    cv2.putText(frame, f"SAC AGENT (Deterministic Policy)", (10, 20), font, font_scale, (0, 255, 255), thickness)
    cv2.putText(frame, f"Episode: {episode}", (10, 40), font, font_scale, color, thickness)
    cv2.putText(frame, f"Step: {step}", (10, 60), font, font_scale, color, thickness)

    # Display continuous action values
    action_str = format_action(action)
    cv2.putText(frame, f"Action: {action_str}", (10, 80), font, font_scale, (0, 255, 0), thickness)

    cv2.putText(frame, f"Reward: {reward:+.2f}", (350, 40), font, font_scale, color, thickness)
    cv2.putText(frame, f"Total: {total_reward:+.2f}", (350, 60), font, font_scale, color, thickness)
    cv2.putText(frame, f"Alpha: {alpha:.4f}", (350, 80), font, font_scale, color, thickness)

    return frame


def watch_agent(args):
    """Watch agent play episodes."""
    # First, detect state mode from checkpoint
    checkpoint = torch.load(args.checkpoint, map_location='cpu')

    # Try to detect state_mode from checkpoint
    if 'state_mode' in checkpoint:
        state_mode = checkpoint['state_mode']
        print(f"Detected state mode from checkpoint: {state_mode}")
    else:
        # Auto-detect from network architecture
        actor_state = checkpoint['actor']
        if 'conv1.weight' in actor_state:
            state_mode = 'visual'
            print("Auto-detected state mode: visual (has conv layers)")
        else:
            state_mode = 'vector'
            print("Auto-detected state mode: vector (no conv layers)")

    # Use same state mode as training (agent architecture must match checkpoint)
    # But use rendering for visualization
    render_mode = None if args.no_render else 'rgb_array'
    env = make_carracing_env(
        stack_size=4,
        terminate_stationary=True,
        stationary_patience=100,
        render_mode=render_mode,
        state_mode=state_mode  # Use detected state mode from checkpoint
    )

    action_dim = env.action_space.shape[0]
    state_shape = env.observation_space.shape

    print("=" * 60)
    print(f"Watching SAC Agent on CarRacing-v3")
    print("=" * 60)
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Episodes: {args.episodes}")
    print(f"State mode: {state_mode}")
    print(f"State shape: {state_shape}")
    print(f"Action space: Continuous ({action_dim}D)")
    if action_dim == 2:
        print(f"  - Steering:      [{env.action_space.low[0]:.1f}, {env.action_space.high[0]:.1f}]")
        print(f"  - Acceleration:  [{env.action_space.low[1]:.1f}, {env.action_space.high[1]:.1f}]")
    else:  # 3D (old checkpoints)
        print(f"  - Steering: [{env.action_space.low[0]:.1f}, {env.action_space.high[0]:.1f}]")
        print(f"  - Gas:      [{env.action_space.low[1]:.1f}, {env.action_space.high[1]:.1f}]")
        print(f"  - Brake:    [{env.action_space.low[2]:.1f}, {env.action_space.high[2]:.1f}]")
    print("=" * 60)

    # Create agent with same state mode as training
    agent = SACAgent(
        state_shape=state_shape,
        action_dim=action_dim,
        state_mode=state_mode  # Use detected state mode from checkpoint
    )
    agent.load(args.checkpoint)

    alpha_value = agent.alpha.item() if isinstance(agent.alpha, torch.Tensor) else agent.alpha

    print(f"\nAgent loaded successfully!")
    print(f"Agent alpha (entropy coefficient): {alpha_value:.4f}")
    print(f"Using deterministic policy (mean action) for evaluation")

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

                # Select action (deterministic, use mean)
                action = agent.select_action(state, evaluate=True)

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
                        action, alpha_value
                    )

                    # Display
                    cv2.imshow('SAC Agent - CarRacing-v3', display_frame)

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


if __name__ == "__main__":
    args = parse_args()
    watch_agent(args)
