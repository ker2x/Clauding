"""
Test script for vector state visualization.

This script creates a dummy vector state and tests the visualization function
without requiring a trained model.

Usage:
    python test_vector_visualization.py
"""

import numpy as np
import cv2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, FancyArrow
from matplotlib.collections import LineCollection


def format_action(action):
    """Format action for display."""
    if len(action) == 2:
        steering, accel = action
        if steering < -0.3:
            steer_desc = f"LEFT({steering:.2f})"
        elif steering > 0.3:
            steer_desc = f"RIGHT({steering:.2f})"
        else:
            steer_desc = f"STRAIGHT({steering:.2f})"

        if accel > 0.1:
            pedal_desc = f"GAS({accel:.2f})"
        elif accel < -0.1:
            pedal_desc = f"BRAKE({-accel:.2f})"
        else:
            pedal_desc = "COAST"
    else:
        steering, gas, brake = action
        if steering < -0.3:
            steer_desc = f"LEFT({steering:.2f})"
        elif steering > 0.3:
            steer_desc = f"RIGHT({steering:.2f})"
        else:
            steer_desc = f"STRAIGHT({steering:.2f})"

        if brake > 0.1:
            pedal_desc = f"BRAKE({brake:.2f})"
        elif gas > 0.1:
            pedal_desc = f"GAS({gas:.2f})"
        else:
            pedal_desc = "COAST"

    return f"{steer_desc} + {pedal_desc}"


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
    waypoints_raw = waypoints_flat.reshape(20, 2)

    # Swap X and Y for correct visualization orientation
    # Environment stores as (lateral, longitudinal) but we want (longitudinal, lateral) for plotting
    # So swap columns: waypoints[:, [1, 0]] means take column 1 first, then column 0
    waypoints = waypoints_raw[:, [1, 0]]

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


def create_dummy_vector_state():
    """Create a dummy 67D vector state for testing."""
    # Create a realistic-looking vector state
    state = np.zeros(67, dtype=np.float32)

    # Car state (11)
    state[0] = 0.5  # car_x
    state[1] = 0.5  # car_y
    state[2] = 10.0  # vx (velocity x)
    state[3] = 2.0   # vy (velocity y)
    state[4] = 0.1   # angle
    state[5] = 0.05  # angular_vel
    state[6:10] = [1.0, 1.0, 1.0, 1.0]  # all wheels on track
    state[10] = 0.25  # 25% track progress

    # Track segment info (5)
    state[11] = 0.02   # dist_to_center
    state[12] = -0.05  # angle_diff
    state[13] = 0.1    # curvature
    state[14] = 0.5    # dist_along
    state[15] = 1.0    # seg_len

    # Waypoints (40): Create a curved track ahead
    # Simulate a right turn
    for i in range(20):
        t = i / 20.0
        # Curve to the right
        x = 0.05 * np.sin(t * 0.5) + 0.02 * t  # Slight right curve
        y = 0.02 * (i + 1)  # Forward progress
        state[16 + i*2] = x
        state[16 + i*2 + 1] = y

    # Speed (1)
    state[56] = 10.2  # speed magnitude

    # Accelerations (2)
    state[57] = 0.5   # longitudinal acceleration
    state[58] = -0.2  # lateral acceleration (turning right)

    # Slip angles (4) - realistic values for turning
    state[59] = 0.1   # FL
    state[60] = 0.15  # FR (higher for right turn)
    state[61] = 0.05  # RL
    state[62] = 0.08  # RR

    # Slip ratios (4) - light acceleration
    state[63] = 0.02  # FL
    state[64] = 0.02  # FR
    state[65] = 0.05  # RL (rear wheels driving)
    state[66] = 0.05  # RR

    return state


def main():
    """Test the vector visualization."""
    print("=" * 60)
    print("Testing Vector State Visualization")
    print("=" * 60)

    # Create dummy state and action
    state = create_dummy_vector_state()
    action = np.array([0.2, 0.5])  # steering right, accelerating
    episode = 1
    step = 42
    reward = 5.0
    total_reward = 123.5
    alpha = 0.2
    speed_kmh = 36.7

    print("\n✓ Created dummy 67D vector state")
    print(f"  State shape: {state.shape}")
    print(f"  State dtype: {state.dtype}")
    print(f"  Action: {action}")

    # Generate visualization
    print("\n✓ Generating visualization...")
    img = visualize_vector_state(
        state, episode, step, reward, total_reward,
        action, alpha, speed_kmh
    )

    print(f"✓ Visualization created")
    print(f"  Image shape: {img.shape}")
    print(f"  Image dtype: {img.dtype}")

    # Save to file
    output_path = "test_vector_viz.png"
    cv2.imwrite(output_path, img)
    print(f"\n✓ Visualization saved to: {output_path}")

    print("\n" + "=" * 60)
    print("Test completed successfully!")
    print("=" * 60)
    print("\nThe visualization shows:")
    print("  • Car at origin (cyan rectangle with yellow forward arrow)")
    print("  • 20 waypoints showing track ahead (colored gradient)")
    print("  • Car state information (position, velocity, etc.)")
    print("  • Track information (curvature, distance to center)")
    print("  • Tire dynamics (slip angles and ratios)")


if __name__ == "__main__":
    main()
