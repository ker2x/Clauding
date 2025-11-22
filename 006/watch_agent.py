"""
Watch a trained SAC agent play CarRacing-v3.

This script loads a trained agent from a checkpoint and visualizes its performance.
Uses OpenCV for rendering (more reliable than pygame on macOS).

For vector mode agents, this displays a custom visualization showing:
- Top-down view of the track waypoints the model sees (car-relative coordinates)
- Car state (position, velocity, wheel contacts, etc.)
- Track information (curvature, distance to center, etc.)
- Tire dynamics (slip angles and slip ratios for each wheel)

Usage:
    # Watch agent with deterministic policy (default)
    python watch_agent.py --checkpoint checkpoints/best_model.pt

    # Watch with stochastic policy (sample from distribution)
    python watch_agent.py --checkpoint checkpoints/best_model.pt --stochastic

    # Watch with stochastic policy and custom temperature
    # Higher temperature = more exploration/randomness
    python watch_agent.py --checkpoint checkpoints/best_model.pt --stochastic --temperature 1.5

    # Lower temperature = less exploration (more deterministic)
    python watch_agent.py --checkpoint checkpoints/best_model.pt --stochastic --temperature 0.5

    # Watch with only game view
    python watch_agent.py --checkpoint checkpoints/best_model.pt --view game

    # Watch with only vector state visualization
    python watch_agent.py --checkpoint checkpoints/best_model.pt --view vector

    # Watch for specific number of episodes
    python watch_agent.py --checkpoint checkpoints/best_model.pt --episodes 3

    # Watch without display (just print rewards)
    python watch_agent.py --checkpoint checkpoints/best_model.pt --no-render
"""

from __future__ import annotations

import argparse
from typing import Any
import numpy.typing as npt
import cv2
import numpy as np
import time
import torch
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyArrow, Rectangle
from matplotlib.collections import LineCollection
import gymnasium as gym

from preprocessing import make_carracing_env
from sac import SACAgent
from utils.display import format_action, get_car_speed
from config.physics_config import ObservationParams, get_base_observation_dim

# Load observation configuration
_OBS_PARAMS = ObservationParams()


def select_action_with_temperature(
    agent: SACAgent,
    state: npt.NDArray[np.float32],
    temperature: float = 1.0,
    stochastic: bool = False
) -> npt.NDArray[np.float32]:
    """
    Select action with optional temperature-scaled stochastic sampling.

    Args:
        agent: SACAgent instance
        state: Current state
        temperature: Temperature parameter for stochastic sampling (default: 1.0)
                    - temperature = 1.0: Normal stochastic sampling (policy as trained)
                    - temperature > 1.0: More exploration (wider distribution)
                    - temperature < 1.0: Less exploration (sharper distribution)
                    - temperature → 0: Approaches deterministic (mean)
        stochastic: If False, use deterministic policy (ignores temperature)

    Returns:
        action: Action to take
    """
    if not stochastic:
        # Deterministic: use mean action
        return agent.select_action(state, evaluate=True)

    # Stochastic with temperature scaling
    state_tensor = torch.FloatTensor(state).unsqueeze(0).to(agent.device)

    with torch.no_grad():
        mean, log_std = agent.actor(state_tensor)

        # Apply temperature scaling to standard deviation
        # temperature > 1 → wider distribution (more exploration)
        # temperature < 1 → narrower distribution (less exploration)
        # temperature → 0 → deterministic (std → 0)
        std = log_std.exp() * temperature

        # Sample from temperature-scaled distribution
        normal = torch.distributions.Normal(mean, std)
        z = normal.sample()

        # Apply action bounds (tanh for both steering and acceleration)
        action = torch.tanh(z)

    return action.cpu().numpy()[0]


def parse_args() -> argparse.Namespace:
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
    parser.add_argument('--view', type=str, default='game', choices=['game', 'vector', 'both'],
                        help='Which view to show for vector mode: game, vector, or both (default: both)')

    # Stochastic inference options
    parser.add_argument('--stochastic', action='store_true',
                        help='Use stochastic policy (sample from distribution) instead of deterministic (mean)')
    parser.add_argument('--temperature', type=float, default=1.0,
                        help='Temperature for stochastic sampling (default: 1.0). '
                             'Higher = more exploration, lower = more deterministic. '
                             'Only used with --stochastic flag.')

    return parser.parse_args()


def visualize_vector_state(
    state_vector: npt.NDArray[np.float32],
    episode: int,
    step: int,
    reward: float,
    total_reward: float,
    action: npt.NDArray[np.float32],
    alpha: float,
    speed_kmh: float = 0.0
) -> npt.NDArray[np.uint8]:
    """
    Visualize the vector state to show what the model sees.

    Vector state structure (default 73D, configurable via config/physics_config.py):
    - Car state (11): x, y, vx, vy, angle, angular_vel, wheel_contacts[4], track_progress
    - Track segment info (5): dist_to_center, angle_diff, curvature, dist_along, seg_len
    - Waypoints (NUM_LOOKAHEAD × 2): waypoints in car-relative coordinates
    - Speed (1): magnitude of velocity
    - Accelerations (2): longitudinal, lateral
    - Slip angles (4): for each wheel [FL, FR, RL, RR]
    - Slip ratios (4): for each wheel [FL, FR, RL, RR]

    Dimension: 33 + (NUM_LOOKAHEAD × 2)

    Args:
        state_vector: Vector state numpy array (dimension based on config)
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

    # Extract waypoints dynamically based on configuration
    # NUM_LOOKAHEAD waypoints × 2 coordinates
    num_waypoints = _OBS_PARAMS.NUM_LOOKAHEAD
    waypoint_start_idx = 16
    waypoint_end_idx = waypoint_start_idx + (num_waypoints * 2)

    waypoints_flat = state_vector[waypoint_start_idx:waypoint_end_idx]
    waypoints_raw = waypoints_flat.reshape(num_waypoints, 2)

    # Transform waypoints for correct visualization orientation
    # Environment stores as (longitudinal, lateral) in car frame where:
    #   - First coord (rel_x) = forward/backward (longitudinal)
    #   - Second coord (rel_y) = left (positive) / right (negative) (lateral)
    # For plotting, we want:
    #   - X-axis (horizontal) = right (positive) / left (negative)
    #   - Y-axis (vertical) = forward (positive)
    # So: plot_x = -rel_y (negate to flip left/right), plot_y = rel_x
    waypoints = np.column_stack([-waypoints_raw[:, 1], waypoints_raw[:, 0]])

    # Calculate dynamic indices for remaining components
    speed_idx = waypoint_end_idx
    speed = state_vector[speed_idx]
    ax = state_vector[speed_idx + 1]
    ay = state_vector[speed_idx + 2]

    slip_angles = state_vector[speed_idx + 3:speed_idx + 7]
    slip_ratios = state_vector[speed_idx + 7:speed_idx + 11]

    # Create figure with subplots
    fig = plt.figure(figsize=(14, 8))

    # Main plot: Top-down view with waypoints
    ax_main = plt.subplot(2, 2, (1, 3))
    ax_main.set_aspect('equal')
    ax_main.set_xlim(-0.3, 0.3)
    ax_main.set_ylim(-0.1, 0.5)
    ax_main.set_facecolor('#1a1a1a')
    ax_main.grid(True, alpha=0.2, color='white')
    ax_main.set_xlabel('Lateral (left ← | → right)', color='white')
    ax_main.set_ylabel('Longitudinal (forward ↑)', color='white')
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

    # Convert world-frame velocities to car-relative frame for clearer display
    # Model sees world velocities, but car-relative makes more intuitive sense
    angle_world = angle * 2 * np.pi  # denormalize
    cos_a = np.cos(angle_world)
    sin_a = np.sin(angle_world)
    # Transform to car frame: forward (x) and lateral (y)
    v_forward = vx * cos_a + vy * sin_a  # Longitudinal velocity
    v_lateral = -vx * sin_a + vy * cos_a  # Lateral velocity

    # angle_diff tells us how misaligned we are with the track
    # Positive = car pointing right of track, Negative = car pointing left of track
    angle_diff_deg = angle_diff * 360  # Convert to degrees for readability

    info_text = f"""
    EPISODE: {episode}    STEP: {step}
    REWARD: {reward:+.2f}    TOTAL: {total_reward:+.1f}

    === CAR STATE (World Frame) ===
    Position: ({car_x:.3f}, {car_y:.3f})
    World velocity: vx={vx:.2f}, vy={vy:.2f} m/s
    Speed: {speed:.2f} m/s ({speed_kmh:.1f} km/h)
    Track progress: {track_progress*100:.1f}%

    === CAR STATE (Car Frame) ===
    Forward vel: {v_forward:.2f} m/s
    Lateral vel: {v_lateral:.2f} m/s
    Angular vel: {angular_vel:.3f} rad/s

    === TRACK ALIGNMENT ===
    Angle diff: {angle_diff_deg:+.1f}° (car vs track)
    {'→ Car pointing RIGHT of track' if angle_diff > 0.02 else '← Car pointing LEFT of track' if angle_diff < -0.02 else '↑ Car aligned with track'}
    Dist to center: {dist_to_center:.3f}
    Curvature: {curvature:.3f}

    === ACCELERATIONS (Body Frame) ===
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


def render_frame(
    frame: npt.NDArray[np.uint8],
    episode: int,
    step: int,
    reward: float,
    total_reward: float,
    action: npt.NDArray[np.float32],
    alpha: float,
    speed_kmh: float = 0.0,
    stochastic: bool = False,
    temperature: float = 1.0
) -> npt.NDArray[np.uint8]:
    """
    Render frame with overlay information.

    Args:
        frame: RGB frame from environment
        episode: Current episode number
        step: Current step number
        reward: Current step reward
        total_reward: Cumulative episode reward
        action: Continuous action [steering, acceleration]
        alpha: Entropy coefficient
        speed_kmh: Car speed in km/h
        stochastic: Whether using stochastic policy
        temperature: Temperature parameter (only relevant if stochastic=True)

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

    # Policy mode indicator
    if stochastic:
        policy_text = f"SAC AGENT (Stochastic, T={temperature:.2f})"
        policy_color = (255, 165, 0)  # Orange for stochastic
    else:
        policy_text = f"SAC AGENT (Deterministic)"
        policy_color = (0, 255, 255)  # Cyan for deterministic

    cv2.putText(frame, policy_text, (10, 20), font, font_scale, policy_color, thickness)
    cv2.putText(frame, f"Episode: {episode}", (10, 40), font, font_scale, color, thickness)
    cv2.putText(frame, f"Step: {step}", (10, 60), font, font_scale, color, thickness)

    # Display continuous action values
    action_str = format_action(action)
    cv2.putText(frame, f"Action: {action_str}", (10, 80), font, font_scale, (0, 255, 0), thickness)

    cv2.putText(frame, f"Reward: {reward:+.2f}", (350, 40), font, font_scale, color, thickness)
    cv2.putText(frame, f"Total: {total_reward:+.2f}", (350, 60), font, font_scale, color, thickness)
    cv2.putText(frame, f"Speed: {speed_kmh:.1f} km/h", (350, 80), font, font_scale, (0, 255, 255), thickness)

    return frame


def watch_agent(args: argparse.Namespace) -> None:
    """Watch agent play episodes."""
    # Load checkpoint and verify it's vector mode
    checkpoint = torch.load(args.checkpoint, map_location='cpu')

    # Try to detect state_mode from checkpoint
    if 'state_mode' in checkpoint:
        state_mode = checkpoint['state_mode']
        print(f"Detected state mode from checkpoint: {state_mode}")
    else:
        # Auto-detect from network architecture
        actor_state = checkpoint['actor']
        if 'conv1.weight' in actor_state:
            raise ValueError("Visual mode checkpoint detected but not supported in 006/. Use vector mode checkpoints only.")
        else:
            state_mode = 'vector'
            print("Auto-detected state mode: vector")

    if state_mode != 'vector':
        raise ValueError(f"Only vector mode is supported in 006/, got '{state_mode}'")

    # Use rendering for visualization
    # Import training constants for consistency
    from config.constants import (
        DEFAULT_TERMINATE_STATIONARY,
        DEFAULT_STATIONARY_PATIENCE,
        DEFAULT_REWARD_SHAPING,
        DEFAULT_MIN_EPISODE_STEPS,
        DEFAULT_SHORT_EPISODE_PENALTY,
        DEFAULT_MAX_EPISODE_STEPS,
    )

    render_mode = None if args.no_render else 'rgb_array'
    env = make_carracing_env(
        terminate_stationary=DEFAULT_TERMINATE_STATIONARY,
        stationary_patience=DEFAULT_STATIONARY_PATIENCE,
        render_mode=render_mode,
        reward_shaping=DEFAULT_REWARD_SHAPING,
        min_episode_steps=DEFAULT_MIN_EPISODE_STEPS,
        short_episode_penalty=DEFAULT_SHORT_EPISODE_PENALTY,
        max_episode_steps=DEFAULT_MAX_EPISODE_STEPS,
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
    # Extract env state_dim from state_shape (for vector mode: (53,) -> 53)
    env_state_dim = state_shape[0] if len(state_shape) == 1 else state_shape

    # Check checkpoint state dimension
    checkpoint_temp = torch.load(args.checkpoint, map_location='cpu')
    checkpoint_state_dim = checkpoint_temp.get('state_dim', None)

    # Calculate dimensions
    base_obs_dim = get_base_observation_dim(_OBS_PARAMS.NUM_LOOKAHEAD)
    frame_stack = _OBS_PARAMS.FRAME_STACK

    # Environment handles frame stacking internally now
    if frame_stack > 1:
        expected_env_dim = base_obs_dim * frame_stack
        print(f"✓ Frame stacking: {frame_stack} frames (handled by environment)")
        print(f"  Base observation: {base_obs_dim}D")
        print(f"  Stacked observation: {expected_env_dim}D")
    else:
        expected_env_dim = base_obs_dim
        print(f"✓ No frame stacking (single frame: {base_obs_dim}D)")

    # Verify dimensions match between checkpoint and environment
    if checkpoint_state_dim and checkpoint_state_dim != env_state_dim:
        print(f"\n{'='*60}")
        print(f"⚠️  WARNING: Dimension Mismatch!")
        print(f"{'='*60}")
        print(f"Checkpoint dimension: {checkpoint_state_dim}D")
        print(f"Current environment: {env_state_dim}D")
        print(f"\nCurrent config (config/physics_config.py):")
        print(f"  NUM_LOOKAHEAD = {_OBS_PARAMS.NUM_LOOKAHEAD}")
        print(f"  WAYPOINT_STRIDE = {_OBS_PARAMS.WAYPOINT_STRIDE}")
        print(f"  FRAME_STACK = {_OBS_PARAMS.FRAME_STACK}")
        print(f"\nExpected: {base_obs_dim}D base × {frame_stack} frames = {expected_env_dim}D")
        print(f"{'='*60}")
        raise RuntimeError(f"Cannot load checkpoint with {checkpoint_state_dim}D using environment {env_state_dim}D")

    del checkpoint_temp  # Free memory

    agent = SACAgent(
        state_dim=checkpoint_state_dim if checkpoint_state_dim else env_state_dim,
        action_dim=action_dim
    )
    agent.load(args.checkpoint)

    alpha_value = agent.alpha.item() if isinstance(agent.alpha, torch.Tensor) else agent.alpha

    print(f"\nAgent loaded successfully!")
    print(f"Agent alpha (entropy coefficient): {alpha_value:.4f}")

    # Display policy mode
    if args.stochastic:
        print(f"Policy mode: STOCHASTIC (sampling from distribution)")
        print(f"Temperature: {args.temperature:.2f}")
        if args.temperature > 1.0:
            print(f"  → Higher temperature = more exploration")
        elif args.temperature < 1.0:
            print(f"  → Lower temperature = less exploration")
        else:
            print(f"  → Normal stochastic sampling (policy as trained)")
    else:
        print(f"Policy mode: DETERMINISTIC (mean action)")

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
            state, _ = env.reset()  # Environment returns stacked observation

            total_reward = 0
            step = 0
            done = False
            paused = False

            print(f"\n{'-' * 60}")
            print(f"Episode {episode + 1}/{args.episodes}")
            print(f"{'-' * 60}")

            while not done:
                frame_start = time.time()

                # Select action (state is already stacked by environment)
                action = select_action_with_temperature(
                    agent, state,
                    temperature=args.temperature,
                    stochastic=args.stochastic
                )

                # Take step
                next_state, reward, terminated, truncated, _ = env.step(action)
                done = terminated or truncated

                total_reward += reward
                step += 1

                # Render if enabled
                if not args.no_render:
                    # Get car speed
                    speed_kmh = get_car_speed(env)

                    # Vector mode: Show game render and/or vector visualization based on --view option
                    # Get game render if needed
                    if args.view in ['game', 'both']:
                        rgb_frame = env.render()
                        game_render = render_frame(
                            rgb_frame, episode + 1, step, reward, total_reward,
                            action, alpha_value, speed_kmh,
                            stochastic=args.stochastic, temperature=args.temperature
                        )
                        cv2.imshow('SAC Agent - Game View', game_render)

                    # Get vector state visualization if needed
                    if args.view in ['vector', 'both']:
                        vector_viz = visualize_vector_state(
                            state, episode + 1, step, reward, total_reward,
                            action, alpha_value, speed_kmh
                        )
                        cv2.imshow('SAC Agent - Vector State (Model View)', vector_viz)

                    # Handle keyboard input
                    key = cv2.waitKey(1) & 0xFF

                    # Process keyboard input
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
