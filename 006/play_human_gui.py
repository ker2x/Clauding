"""
Human-controlled CarRacing-v3 with Real-time Telemetry GUI.

This script provides an interactive GUI showing real-time vehicle dynamics
while playing the game. It includes:
- Real-time wheel slip display (slip angle and slip ratio for all 4 wheels)
- Normal force (tire load) visualization per wheel
- Suspension travel display per wheel
- Optional CSV telemetry logging for analysis
- Standard keyboard controls for gameplay

Usage:
    # Basic usage with GUI
    python play_human_gui.py

    # With telemetry logging (every frame)
    python play_human_gui.py --log-telemetry

    # Custom log file and interval (every 5 frames)
    python play_human_gui.py --log-telemetry --log-file my_session.csv --log-interval 5

Controls:
    - Steering:   Q/D, Left/Right (AZERTY: Q/D)
    - Gas:        Z, Up Arrow
    - Brake:      S, Down Arrow
    - Reset:      R
    - Quit:       ESC

CSV Format:
    The telemetry log includes: timestamp, episode, step, speed, steering,
    acceleration, rewards, car state (x, y, angle, velocities), and per-wheel
    data (slip angle, slip ratio, normal force, suspension travel).
"""

import argparse
import numpy as np
import time
import pygame
import sys
import csv
from datetime import datetime
from preprocessing import make_carracing_env


class TelemetryLogger:
    """Log vehicle telemetry data to CSV file."""

    def __init__(self, filename=None, log_interval=1):
        """
        Initialize telemetry logger.

        Args:
            filename: Output CSV filename (None = auto-generate)
            log_interval: Log every N frames (default: 1 = every frame)
        """
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"telemetry_{timestamp}.csv"

        self.filename = filename
        self.log_interval = log_interval
        self.frame_count = 0
        self.file = None
        self.writer = None

        # CSV columns
        self.fieldnames = [
            'timestamp', 'episode', 'step', 'speed_kmh',
            'steering', 'acceleration', 'reward', 'total_reward',
            'car_x', 'car_y', 'car_angle', 'car_vx', 'car_vy', 'car_yaw_rate',
            # Wheel data: FL, FR, RL, RR
            'fl_slip_angle', 'fl_slip_ratio', 'fl_normal_force', 'fl_suspension',
            'fr_slip_angle', 'fr_slip_ratio', 'fr_normal_force', 'fr_suspension',
            'rl_slip_angle', 'rl_slip_ratio', 'rl_normal_force', 'rl_suspension',
            'rr_slip_angle', 'rr_slip_ratio', 'rr_normal_force', 'rr_suspension',
        ]

        self._open_file()

    def _open_file(self):
        """Open CSV file and write header."""
        self.file = open(self.filename, 'w', newline='')
        self.writer = csv.DictWriter(self.file, fieldnames=self.fieldnames)
        self.writer.writeheader()
        self.file.flush()

    def log_frame(self, episode, step, speed_kmh, action, reward, total_reward,
                  wheel_data, car_state=None):
        """
        Log a single frame of telemetry data.

        Args:
            episode: Current episode number
            step: Current step number
            speed_kmh: Car speed in km/h
            action: [steering, acceleration]
            reward: Current frame reward
            total_reward: Cumulative episode reward
            wheel_data: Dict with wheel telemetry [FL, FR, RL, RR]
            car_state: Optional dict with car state (x, y, angle, vx, vy, yaw_rate)
        """
        self.frame_count += 1

        # Only log every N frames
        if self.frame_count % self.log_interval != 0:
            return

        # Default car state
        if car_state is None:
            car_state = {
                'x': 0.0, 'y': 0.0, 'angle': 0.0,
                'vx': 0.0, 'vy': 0.0, 'yaw_rate': 0.0
            }

        # Build row
        row = {
            'timestamp': datetime.now().isoformat(),
            'episode': episode,
            'step': step,
            'speed_kmh': f"{speed_kmh:.2f}",
            'steering': f"{action[0]:.4f}",
            'acceleration': f"{action[1]:.4f}",
            'reward': f"{reward:.4f}",
            'total_reward': f"{total_reward:.2f}",
            'car_x': f"{car_state['x']:.4f}",
            'car_y': f"{car_state['y']:.4f}",
            'car_angle': f"{car_state['angle']:.4f}",
            'car_vx': f"{car_state['vx']:.4f}",
            'car_vy': f"{car_state['vy']:.4f}",
            'car_yaw_rate': f"{car_state['yaw_rate']:.4f}",
        }

        # Add wheel data
        wheel_names = ['fl', 'fr', 'rl', 'rr']
        for i, name in enumerate(wheel_names):
            row[f'{name}_slip_angle'] = f"{wheel_data[i]['slip_angle']:.6f}"
            row[f'{name}_slip_ratio'] = f"{wheel_data[i]['slip_ratio']:.6f}"
            row[f'{name}_normal_force'] = f"{wheel_data[i]['normal_force']:.2f}"
            row[f'{name}_suspension'] = f"{wheel_data[i]['suspension_travel']:.6f}"

        # Write to file
        self.writer.writerow(row)
        self.file.flush()  # Ensure data is written immediately

    def close(self):
        """Close the log file."""
        if self.file:
            self.file.close()
            print(f"\n✓ Telemetry logged to: {self.filename}")


class TelemetryGUI:
    """GUI for displaying real-time vehicle telemetry."""

    def __init__(self, width=420, height=300):
        self.width = width
        self.height = height
        self.font = pygame.font.Font(None, 18)
        self.font_small = pygame.font.Font(None, 14)

        # Wheel telemetry data (updated each frame)
        self.wheel_data = {
            0: {'slip_angle': 0.0, 'slip_ratio': 0.0, 'normal_force': 0.0, 'suspension_travel': 0.0},  # FL
            1: {'slip_angle': 0.0, 'slip_ratio': 0.0, 'normal_force': 0.0, 'suspension_travel': 0.0},  # FR
            2: {'slip_angle': 0.0, 'slip_ratio': 0.0, 'normal_force': 0.0, 'suspension_travel': 0.0},  # RL
            3: {'slip_angle': 0.0, 'slip_ratio': 0.0, 'normal_force': 0.0, 'suspension_travel': 0.0},  # RR
        }

    def update_data(self, wheel_data):
        """Update wheel telemetry data from the environment."""
        if wheel_data:
            self.wheel_data = wheel_data

    def draw(self, surface, speed_kmh=0.0):
        """Draw telemetry panel."""
        # Background
        surface.fill((25, 25, 30))
        pygame.draw.rect(surface, (60, 60, 70), (0, 0, self.width, self.height), 2)

        # Title
        title = self.font.render("Vehicle Telemetry", True, (255, 200, 100))
        surface.blit(title, (10, 5))

        # Speed display (right-aligned)
        speed_text = self.font.render(f"{speed_kmh:.1f} km/h", True, (255, 255, 100))
        surface.blit(speed_text, (self.width - speed_text.get_width() - 10, 5))

        # Wheel labels and positions
        wheel_names = ['FL', 'FR', 'RL', 'RR']
        wheel_colors = [(100, 150, 255), (100, 255, 150), (255, 150, 100), (255, 100, 150)]

        y_start = 28
        row_height = 68  # Height per wheel row

        for i, (name, color) in enumerate(zip(wheel_names, wheel_colors)):
            y = y_start + i * row_height

            # Wheel label
            label = self.font_small.render(name, True, color)
            surface.blit(label, (10, y))

            # Get telemetry data
            slip_angle = self.wheel_data[i]['slip_angle'] * 180 / np.pi  # Convert to degrees
            slip_ratio = self.wheel_data[i]['slip_ratio']
            normal_force = self.wheel_data[i]['normal_force']
            suspension = self.wheel_data[i]['suspension_travel'] * 1000  # Convert to mm

            # === LINE 1: Slip Angle ===
            angle_text = self.font_small.render(f"SA:{slip_angle:+.1f}°", True, (200, 200, 200))
            surface.blit(angle_text, (40, y))

            # Slip angle bar (range: -25 to +25 degrees)
            bar_x = 115
            bar_width = 120
            bar_height = 8
            max_angle = 25.0

            pygame.draw.rect(surface, (40, 40, 45), (bar_x, y, bar_width, bar_height))

            center_x = bar_x + bar_width // 2
            fill_width = int((slip_angle / max_angle) * (bar_width // 2))
            fill_width = max(-bar_width // 2, min(bar_width // 2, fill_width))

            if fill_width > 0:
                pygame.draw.rect(surface, (100, 150, 255), (center_x, y, fill_width, bar_height))
            elif fill_width < 0:
                pygame.draw.rect(surface, (255, 150, 100), (center_x + fill_width, y, -fill_width, bar_height))

            pygame.draw.line(surface, (100, 100, 100), (center_x, y), (center_x, y + bar_height), 1)
            pygame.draw.rect(surface, (80, 80, 85), (bar_x, y, bar_width, bar_height), 1)

            # Slip ratio
            ratio_text = self.font_small.render(f"SR:{slip_ratio:+.2f}", True, (200, 200, 200))
            surface.blit(ratio_text, (bar_x + bar_width + 8, y))

            ratio_bar_x = bar_x + bar_width + 65
            ratio_bar_width = 80

            pygame.draw.rect(surface, (40, 40, 45), (ratio_bar_x, y, ratio_bar_width, bar_height))

            ratio_center_x = ratio_bar_x + ratio_bar_width // 2
            ratio_fill_width = int(slip_ratio * (ratio_bar_width // 2))
            ratio_fill_width = max(-ratio_bar_width // 2, min(ratio_bar_width // 2, ratio_fill_width))

            if ratio_fill_width > 0:
                pygame.draw.rect(surface, (100, 255, 100), (ratio_center_x, y, ratio_fill_width, bar_height))
            elif ratio_fill_width < 0:
                pygame.draw.rect(surface, (255, 100, 100), (ratio_center_x + ratio_fill_width, y, -ratio_fill_width, bar_height))

            pygame.draw.line(surface, (100, 100, 100), (ratio_center_x, y), (ratio_center_x, y + bar_height), 1)
            pygame.draw.rect(surface, (80, 80, 85), (ratio_bar_x, y, ratio_bar_width, bar_height), 1)

            # === LINE 2: Normal Force ===
            y_load = y + 15
            # Fix force scale - show in kN for readability
            load_kn = normal_force / 1000.0
            load_text = self.font_small.render(f"Load: {load_kn:.2f}kN", True, (255, 255, 150))
            surface.blit(load_text, (40, y_load))

            # Normal force bar (relative to nominal load)
            nominal_load = 2.6  # kN (2600 N for 1060kg car)
            load_bar_x = 115
            load_bar_width = 265
            load_bar_height = 8

            pygame.draw.rect(surface, (40, 40, 45), (load_bar_x, y_load, load_bar_width, load_bar_height))

            # Scale: 0 to 2x nominal (0-5.2kN)
            max_display_load = nominal_load * 2.0
            fill_ratio = min(load_kn / max_display_load, 1.0)
            fill_width = int(fill_ratio * load_bar_width)

            # Color based on load
            if load_kn < nominal_load * 0.7:
                bar_color = (100, 100, 255)  # Blue (light)
            elif load_kn < nominal_load * 1.2:
                bar_color = (100, 255, 100)  # Green (nominal)
            elif load_kn < nominal_load * 1.5:
                bar_color = (255, 255, 100)  # Yellow (heavy)
            else:
                bar_color = (255, 100, 100)  # Red (very heavy)

            pygame.draw.rect(surface, bar_color, (load_bar_x, y_load, fill_width, load_bar_height))

            # Nominal load marker
            nominal_x = load_bar_x + int((nominal_load / max_display_load) * load_bar_width)
            pygame.draw.line(surface, (150, 150, 150), (nominal_x, y_load), (nominal_x, y_load + load_bar_height), 2)

            pygame.draw.rect(surface, (80, 80, 85), (load_bar_x, y_load, load_bar_width, load_bar_height), 1)

            # === LINE 3: Suspension Travel ===
            y_susp = y + 30
            susp_text = self.font_small.render(f"Susp: {suspension:+.1f}mm", True, (200, 255, 200))
            surface.blit(susp_text, (40, y_susp))

            # Suspension travel bar (range: -120mm to +80mm)
            susp_bar_x = 115
            susp_bar_width = 265
            susp_bar_height = 8

            max_extension = -120.0  # mm (negative = droop)
            max_compression = 80.0  # mm (positive = bump)
            equilibrium = 58.0  # mm (static compression)

            pygame.draw.rect(surface, (40, 40, 45), (susp_bar_x, y_susp, susp_bar_width, susp_bar_height))

            # Calculate bar position (0 = max extension, 1 = max compression)
            susp_range = max_compression - max_extension
            susp_norm = (suspension - max_extension) / susp_range
            susp_fill_x = int(susp_norm * susp_bar_width)

            # Color: green near equilibrium, yellow/red at extremes
            deviation = abs(suspension - equilibrium)
            if deviation < 10:
                susp_color = (100, 255, 100)  # Green (normal)
            elif deviation < 30:
                susp_color = (255, 255, 100)  # Yellow (moderate)
            else:
                susp_color = (255, 100, 100)  # Red (extreme)

            pygame.draw.rect(surface, susp_color, (susp_bar_x, y_susp, susp_fill_x, susp_bar_height))

            # Equilibrium marker
            eq_norm = (equilibrium - max_extension) / susp_range
            eq_x = susp_bar_x + int(eq_norm * susp_bar_width)
            pygame.draw.line(surface, (200, 200, 200), (eq_x, y_susp), (eq_x, y_susp + susp_bar_height), 2)

            pygame.draw.rect(surface, (80, 80, 85), (susp_bar_x, y_susp, susp_bar_width, susp_bar_height), 1)


def format_action(action):
    """Format continuous action for display."""
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


def render_info(screen, font, episode, step, reward, total_reward, action, speed_kmh=0.0, info_y_offset=0):
    """Render text overlay onto the pygame screen."""
    info_area_height = 100
    w = screen.get_size()[0]

    # Clear the info bar
    screen.fill((0, 0, 0), (0, info_y_offset, w, info_area_height))

    def draw_text(text, y, color=(255, 255, 255)):
        text_surf = font.render(text, True, color)
        screen.blit(text_surf, (10, y))

    def draw_text_right(text, y, color=(255, 255, 255)):
        text_surf = font.render(text, True, color)
        screen.blit(text_surf, (w - text_surf.get_width() - 10, y))

    y_base = info_y_offset
    draw_text(f"Episode: {episode} | Step: {step}", y_base + 10)

    action_str = format_action(action)
    draw_text(f"Action: {action_str}", y_base + 30, (0, 255, 0))
    draw_text(f"Controls: ZQSD/Arrows | R: reset | ESC: quit", y_base + 50, (100, 100, 255))

    draw_text_right(f"Reward: {reward:+.2f}", y_base + 10)
    draw_text_right(f"Total: {total_reward:+.2f}", y_base + 30)
    draw_text_right(f"Speed: {speed_kmh:.1f} km/h", y_base + 50, (255, 255, 100))


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


def get_wheel_data(env):
    """Extract wheel telemetry data from the environment."""
    wheel_data = {
        0: {'slip_angle': 0.0, 'slip_ratio': 0.0, 'normal_force': 0.0, 'suspension_travel': 0.0},
        1: {'slip_angle': 0.0, 'slip_ratio': 0.0, 'normal_force': 0.0, 'suspension_travel': 0.0},
        2: {'slip_angle': 0.0, 'slip_ratio': 0.0, 'normal_force': 0.0, 'suspension_travel': 0.0},
        3: {'slip_angle': 0.0, 'slip_ratio': 0.0, 'normal_force': 0.0, 'suspension_travel': 0.0},
    }

    if hasattr(env, 'unwrapped') and hasattr(env.unwrapped, 'car'):
        car = env.unwrapped.car
        if car is not None:
            # Use stored tire forces from last step
            forces = car.last_tire_forces if hasattr(car, 'last_tire_forces') else None

            if forces:
                for i in range(4):
                    if i in forces:
                        wheel_data[i]['slip_angle'] = forces[i].get('slip_angle', 0.0)
                        wheel_data[i]['slip_ratio'] = forces[i].get('slip_ratio', 0.0)
                        wheel_data[i]['normal_force'] = forces[i].get('normal_force', 0.0)

            # Get suspension travel
            if hasattr(car, 'suspension_travel'):
                for i in range(4):
                    wheel_data[i]['suspension_travel'] = car.suspension_travel[i]

    return wheel_data


def get_car_state(env):
    """Extract detailed car state for logging."""
    car_state = {
        'x': 0.0, 'y': 0.0, 'angle': 0.0,
        'vx': 0.0, 'vy': 0.0, 'yaw_rate': 0.0
    }

    if hasattr(env, 'unwrapped') and hasattr(env.unwrapped, 'car'):
        car = env.unwrapped.car
        if car is not None:
            car_state['x'] = getattr(car, 'x', 0.0)
            car_state['y'] = getattr(car, 'y', 0.0)
            car_state['angle'] = getattr(car, 'angle', 0.0)
            car_state['vx'] = getattr(car, 'vx', 0.0)
            car_state['vy'] = getattr(car, 'vy', 0.0)
            car_state['yaw_rate'] = getattr(car, 'yaw_rate', 0.0)

    return car_state


def play_human_gui(args):
    """Play episodes with real-time telemetry GUI."""

    # Initialize Pygame
    pygame.init()
    pygame.font.init()
    font = pygame.font.Font(None, 24)

    # Create GUI
    gui = TelemetryGUI(width=420, height=300)

    # Create telemetry logger if requested
    logger = None
    if args.log_telemetry:
        logger = TelemetryLogger(
            filename=args.log_file,
            log_interval=args.log_interval
        )
        print(f"✓ Telemetry logging enabled: {logger.filename}")
        print(f"  Log interval: every {args.log_interval} frame(s)")

    # Create environment
    render_mode = None if args.no_render else 'rgb_array'
    env = make_carracing_env(
        terminate_stationary=False,
        stationary_patience=100,
        render_mode=render_mode
    )

    action_dim = env.action_space.shape[0]
    state_shape = env.observation_space.shape

    print("=" * 80)
    print(f"CarRacing-v3 - HUMAN PLAYER with Telemetry GUI")
    print("=" * 80)
    print(f"Episodes: {args.episodes}")
    print(f"State shape: {state_shape}")
    if logger:
        print(f"Telemetry logging: ENABLED")
        print(f"  File: {logger.filename}")
        print(f"  Interval: every {args.log_interval} frame(s)")
    else:
        print(f"Telemetry logging: DISABLED (use --log-telemetry to enable)")
    print("=" * 80)
    print("\nKEYBOARD CONTROLS (AZERTY):")
    print("  - Steering:   Q / D or Left / Right Arrow")
    print("  - Gas:        Z or Up Arrow")
    print("  - Brake:      S or Down Arrow")
    print("  - Reset:      R")
    print("  - Quit:       ESC")
    print("\nTELEMETRY DISPLAY:")
    print("  - Slip angle (SA) and slip ratio (SR) per wheel")
    print("  - Normal force (tire load) per wheel")
    print("  - Suspension travel per wheel")
    print("=" * 80)

    episode_rewards = []
    target_frame_time = 1.0 / args.fps
    screen = None

    try:
        for episode in range(args.episodes):
            state, _ = env.reset()

            total_reward = 0
            step = 0
            done = False

            action = np.array([0.0, 0.0], dtype=np.float32)
            current_steering = 0.0

            print(f"\n{'-' * 80}")
            print(f"Episode {episode + 1}/{args.episodes}")
            print(f"{'-' * 80}")
            print("Ready! Use keyboard to control the car")

            while not done:
                frame_start = time.time()

                # --- Pygame Event Handling ---
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        print("\nQuitting...")
                        env.close()
                        pygame.quit()
                        sys.exit()

                    if event.type == pygame.KEYDOWN:
                        if event.key == pygame.K_r:
                            print("Resetting episode...")
                            done = True
                        if event.key == pygame.K_ESCAPE:
                            print("\nQuitting...")
                            env.close()
                            pygame.quit()
                            sys.exit()

                # --- Keyboard Input ---
                keys = pygame.key.get_pressed()

                gas = 1.0 if keys[pygame.K_z] or keys[pygame.K_UP] else 0.0
                brake = 1.0 if keys[pygame.K_s] or keys[pygame.K_DOWN] else 0.0

                acceleration = -brake if brake > 0 else gas

                target_steering = 0.0
                if keys[pygame.K_q] or keys[pygame.K_LEFT]:
                    target_steering = -1.0
                if keys[pygame.K_d] or keys[pygame.K_RIGHT]:
                    target_steering = 1.0

                STEER_SPEED = 0.1
                current_steering = (1.0 - STEER_SPEED) * current_steering + STEER_SPEED * target_steering

                action = np.array([current_steering, acceleration], dtype=np.float32)

                # --- Step Environment ---
                next_state, reward, terminated, truncated, _ = env.step(action)
                if done:
                    break

                done = terminated or truncated
                total_reward += reward
                step += 1

                # --- Get wheel data and car speed ---
                wheel_data = get_wheel_data(env)
                gui.update_data(wheel_data)
                speed_kmh = get_car_speed(env)

                # --- Log telemetry if enabled ---
                if logger:
                    car_state = get_car_state(env)
                    logger.log_frame(
                        episode=episode + 1,
                        step=step,
                        speed_kmh=speed_kmh,
                        action=action,
                        reward=reward,
                        total_reward=total_reward,
                        wheel_data=wheel_data,
                        car_state=car_state
                    )

                # --- Render ---
                if not args.no_render:
                    rgb_frame = env.render()
                    frame_h, frame_w = rgb_frame.shape[:2]
                    info_area_height = 100

                    # Create screen on first frame
                    if screen is None:
                        # Layout: Game | Telemetry Panel
                        total_width = frame_w + gui.width
                        game_area_height = frame_h + info_area_height
                        total_height = max(game_area_height, gui.height)
                        screen = pygame.display.set_mode((total_width, total_height))
                        pygame.display.set_caption("CarRacing-v3 - Telemetry Display")

                    # Clear screen
                    screen.fill((0, 0, 0))

                    # Draw game frame
                    frame_pygame = np.transpose(rgb_frame, (1, 0, 2))
                    surf = pygame.surfarray.make_surface(frame_pygame)
                    screen.blit(surf, (0, info_area_height))

                    # Draw info overlay
                    render_info(screen, font, episode + 1, step, reward, total_reward, action, speed_kmh, info_y_offset=0)

                    # Draw telemetry GUI (right of game)
                    gui_surface = pygame.Surface((gui.width, gui.height))
                    gui.draw(gui_surface, speed_kmh=speed_kmh)
                    screen.blit(gui_surface, (frame_w, 0))

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
        if logger:
            logger.close()
        env.close()
        pygame.quit()

    # Final statistics
    print("\n" + "=" * 80)
    print("GAME SUMMARY")
    print("=" * 80)
    print(f"Episodes completed: {len(episode_rewards)}")
    if episode_rewards:
        print(f"Average reward: {np.mean(episode_rewards):.2f}")
        print(f"Std dev: {np.std(episode_rewards):.2f}")
        print(f"Min reward: {np.min(episode_rewards):.2f}")
        print(f"Max reward: {np.max(episode_rewards):.2f}")
    print("=" * 80)


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description='Play CarRacing-v3 with real-time telemetry display'
    )
    parser.add_argument('--episodes', type=int, default=5,
                        help='Number of episodes to play (default: 5)')
    parser.add_argument('--fps', type=int, default=50,
                        help='Display FPS (default: 50)')
    parser.add_argument('--no-render', action='store_true',
                        help='Disable rendering (just compute rewards)')

    # Telemetry logging options
    parser.add_argument('--log-telemetry', action='store_true',
                        help='Enable telemetry logging to CSV file')
    parser.add_argument('--log-file', type=str, default=None,
                        help='Output CSV filename (default: auto-generated with timestamp)')
    parser.add_argument('--log-interval', type=int, default=1,
                        help='Log every N frames (default: 1 = every frame)')

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    play_human_gui(args)
