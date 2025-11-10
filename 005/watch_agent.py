"""
Watch a trained SAC agent play CarRacing-v3.

This script loads a trained agent from a checkpoint and visualizes its performance.
Uses OpenCV for rendering (more reliable than pygame on macOS).

For vector mode agents, this displays a custom visualization showing:
- Top-down view of the 20 track waypoints the model sees (car-relative coordinates)
- Car state (position, velocity, wheel contacts, etc.)
- Track information (curvature, distance to center, etc.)
- Tire dynamics (slip angles and slip ratios for each wheel)

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
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyArrow, Rectangle
from matplotlib.collections import LineCollection

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


def get_car_speed(env):
    """Extract car speed from the environment and convert to km/h."""
    speed_kmh = 0.0

    if hasattr(env, 'unwrapped') and hasattr(env.unwrapped, 'car'):
        car = env.unwrapped.car
        if car is not None and hasattr(car, 'vx') and hasattr(car, 'vy'):
            # Calculate speed magnitude from velocity components (m/s)
            speed_ms = np.sqrt(car.vx**2 + car.vy**2)
            # Convert m/s to km/h
            speed_kmh = speed_ms * 3.6

    return speed_kmh


def visualize_vector_state(state_vector, episode, step, reward, total_reward, action, alpha, speed_kmh=0.0):
    """
    Visualize the 67D vector state to show what the model sees.

    Vector state structure (67D):
    - Car state (11): x, y, vx, vy, angle, angular_vel, wheel_contacts[4], track_progress
    - Track segment info (5): dist_to_center, angle_diff, curvature, dist_along, seg_len
    - Waypoints (40): 20 waypoints × (x, y) in car-relative coordinates
    - Speed (1): magnitude of velocity
    - Accelerations (2): longitudinal, lateral
    - Slip angles (4): for each wheel [FL, FR, RL, RR]
    - Slip ratios (4): for each wheel [FL, FR, RL, RR]

    Args:
        state_vector: 67D numpy array
        episode: Episode number
        step: Step number
        reward: Current reward
        total_reward: Cumulative reward
        action: Action taken
        alpha: SAC alpha parameter
        speed_kmh: Car speed in km/h

    Returns:
        RGB image (numpy array) showing the visualization
    """
    # Parse state vector
    car_x = state_vector[0]
    car_y = state_vector[1]
    vx = state_vector[2]
    vy = state_vector[3]
    angle = state_vector[4]
    angular_vel = state_vector[5]
    wheel_contacts = state_vector[6:10]
    track_progress = state_vector[10]

    dist_to_center = state_vector[11]
    angle_diff = state_vector[12]
    curvature = state_vector[13]
    dist_along = state_vector[14]
    seg_len = state_vector[15]

    # Extract waypoints (20 waypoints × 2 coordinates = 40 values)
    waypoints_flat = state_vector[16:56]
    waypoints = waypoints_flat.reshape(20, 2)

    speed = state_vector[56]
    ax = state_vector[57]
    ay = state_vector[58]

    slip_angles = state_vector[59:63]
    slip_ratios = state_vector[63:67]

    # Create figure with subplots
    fig = plt.figure(figsize=(14, 8))

    # Main plot: Top-down view with waypoints
    ax_main = plt.subplot(2, 2, (1, 3))
    ax_main.set_aspect('equal')
    ax_main.set_xlim(-0.3, 0.3)
    ax_main.set_ylim(-0.1, 0.5)
    ax_main.set_facecolor('#1a1a1a')
    ax_main.grid(True, alpha=0.2, color='white')
    ax_main.set_xlabel('X (car-relative, normalized)', color='white')
    ax_main.set_ylabel('Y (car-relative, normalized)', color='white')
    ax_main.tick_params(colors='white')
    ax_main.set_title('Model\'s View: Track Lookahead (20 Waypoints)',
                      color='white', fontweight='bold', fontsize=12)

    # Draw car at origin (since waypoints are in car-relative coordinates)
    car_length = 0.02
    car_width = 0.01
    car_rect = Rectangle((-car_width/2, -car_length/2), car_width, car_length,
                         facecolor='cyan', edgecolor='white', linewidth=2, zorder=10)
    ax_main.add_patch(car_rect)

    # Draw forward direction arrow
    arrow = FancyArrow(0, 0, 0, 0.04, width=0.01,
                      head_width=0.02, head_length=0.01,
                      facecolor='yellow', edgecolor='white', linewidth=1, zorder=11)
    ax_main.add_patch(arrow)

    # Draw waypoints
    if len(waypoints) > 0:
        # Connect waypoints with lines
        segments = []
        for i in range(len(waypoints) - 1):
            segments.append([waypoints[i], waypoints[i+1]])

        lc = LineCollection(segments, colors='lime', linewidths=2, alpha=0.7, zorder=5)
        ax_main.add_collection(lc)

        # Draw waypoint markers with gradient color (closer = brighter)
        colors = plt.cm.hot(np.linspace(0.3, 1.0, len(waypoints)))
        for i, (wx, wy) in enumerate(waypoints):
            ax_main.plot(wx, wy, 'o', color=colors[i], markersize=6,
                        markeredgecolor='white', markeredgewidth=0.5, zorder=6)

        # Add distance annotations for first few waypoints
        for i in [0, 4, 9, 14, 19]:
            if i < len(waypoints):
                wx, wy = waypoints[i]
                dist = np.sqrt(wx**2 + wy**2)
                ax_main.annotate(f'{i+1}', (wx, wy),
                               textcoords='offset points', xytext=(0, 5),
                               fontsize=7, color='white', ha='center',
                               bbox=dict(boxstyle='round,pad=0.3', facecolor='black', alpha=0.7))

    # Car state info (top right)
    ax_info = plt.subplot(2, 2, 2)
    ax_info.axis('off')
    ax_info.set_facecolor('#1a1a1a')

    info_text = f"""
    EPISODE: {episode}    STEP: {step}
    REWARD: {reward:+.2f}    TOTAL: {total_reward:+.1f}

    === CAR STATE ===
    Position: ({car_x:.3f}, {car_y:.3f})
    Velocity: vx={vx:.2f}, vy={vy:.2f}
    Speed: {speed:.2f} m/s ({speed_kmh:.1f} km/h)
    Angle: {angle:.3f} rad
    Angular vel: {angular_vel:.3f} rad/s
    Track progress: {track_progress*100:.1f}%

    === TRACK INFO ===
    Dist to center: {dist_to_center:.3f}
    Angle diff: {angle_diff:.3f} rad
    Curvature: {curvature:.3f}

    === ACCELERATIONS ===
    Longitudinal: {ax:.2f} m/s²
    Lateral: {ay:.2f} m/s²

    === WHEEL CONTACTS ===
    FL: {'✓' if wheel_contacts[0] > 0.5 else '✗'}    FR: {'✓' if wheel_contacts[1] > 0.5 else '✗'}
    RL: {'✓' if wheel_contacts[2] > 0.5 else '✗'}    RR: {'✓' if wheel_contacts[3] > 0.5 else '✗'}
    """

    ax_info.text(0.05, 0.95, info_text,
                transform=ax_info.transAxes,
                fontfamily='monospace', fontsize=9,
                verticalalignment='top', color='white')

    # Slip angles and ratios (bottom right)
    ax_slip = plt.subplot(2, 2, 4)
    ax_slip.axis('off')
    ax_slip.set_facecolor('#1a1a1a')

    slip_text = f"""
    === SLIP ANGLES (rad) ===
    FL: {slip_angles[0]:+.3f}    FR: {slip_angles[1]:+.3f}
    RL: {slip_angles[2]:+.3f}    RR: {slip_angles[3]:+.3f}

    === SLIP RATIOS ===
    FL: {slip_ratios[0]:+.3f}    FR: {slip_ratios[1]:+.3f}
    RL: {slip_ratios[2]:+.3f}    RR: {slip_ratios[3]:+.3f}

    === ACTION ===
    {format_action(action)}

    === SAC PARAMETERS ===
    Alpha (entropy): {alpha:.4f}
    """

    ax_slip.text(0.05, 0.95, slip_text,
                transform=ax_slip.transAxes,
                fontfamily='monospace', fontsize=9,
                verticalalignment='top', color='white')

    # Convert matplotlib figure to numpy array
    fig.tight_layout()
    fig.canvas.draw()

    # Convert to numpy array (RGB) - using buffer_rgba() for compatibility
    buf = fig.canvas.buffer_rgba()
    img = np.asarray(buf)

    plt.close(fig)

    # Convert RGBA to BGR for OpenCV (drop alpha channel)
    img_bgr = cv2.cvtColor(img, cv2.COLOR_RGBA2BGR)

    return img_bgr


def render_frame(frame, episode, step, reward, total_reward, action, alpha, speed_kmh=0.0):
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
        speed_kmh: Car speed in km/h

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
    cv2.putText(frame, f"Speed: {speed_kmh:.1f} km/h", (350, 80), font, font_scale, (0, 255, 255), thickness)

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
                    # Get car speed
                    speed_kmh = get_car_speed(env)

                    if state_mode == 'vector':
                        # Use vector state visualization
                        display_frame = visualize_vector_state(
                            state, episode + 1, step, reward, total_reward,
                            action, alpha_value, speed_kmh
                        )
                        window_title = 'SAC Agent - Vector State Visualization'
                    else:
                        # Get RGB frame for display (visual mode)
                        rgb_frame = env.render()

                        # Add overlay
                        display_frame = render_frame(
                            rgb_frame, episode + 1, step, reward, total_reward,
                            action, alpha_value, speed_kmh
                        )
                        window_title = 'SAC Agent - CarRacing-v3'

                    # Display
                    cv2.imshow(window_title, display_frame)

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
