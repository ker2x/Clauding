"""
Human-controlled CarRacing-v3 with Real-time Telemetry GUI.

This script provides an interactive GUI showing real-time vehicle dynamics
while playing the game. It includes:
- Real-time wheel slip display (slip angle and slip ratio for all 4 wheels)
- Normal force (tire load) visualization per wheel
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
    data (slip angle, slip ratio, normal force).
"""

import argparse
import numpy as np
import time
import pygame
import sys
import csv
from datetime import datetime
from preprocessing import make_carracing_env
from utils.display import format_action, get_car_speed


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
            'fl_slip_angle', 'fl_slip_ratio', 'fl_normal_force',
            'fr_slip_angle', 'fr_slip_ratio', 'fr_normal_force',
            'rl_slip_angle', 'rl_slip_ratio', 'rl_normal_force',
            'rr_slip_angle', 'rr_slip_ratio', 'rr_normal_force',
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

    def __init__(self, width=420, height=310):
        self.width = width
        self.height = height
        self.font = pygame.font.Font(None, 18)
        self.font_small = pygame.font.Font(None, 14)

        # Wheel telemetry data (updated each frame)
        self.wheel_data = {
            0: {'slip_angle': 0.0, 'slip_ratio': 0.0, 'normal_force': 0.0},  # FL
            1: {'slip_angle': 0.0, 'slip_ratio': 0.0, 'normal_force': 0.0},  # FR
            2: {'slip_angle': 0.0, 'slip_ratio': 0.0, 'normal_force': 0.0},  # RL
            3: {'slip_angle': 0.0, 'slip_ratio': 0.0, 'normal_force': 0.0},  # RR
        }

        # Powertrain telemetry data (updated each frame)
        self.powertrain_data = {
            'gear': 0,
            'torque': 0.0,
            'hp': 0.0,
        }

    def update_data(self, wheel_data, powertrain_data=None):
        """Update wheel and powertrain telemetry data from the environment."""
        if wheel_data:
            self.wheel_data = wheel_data
        if powertrain_data:
            self.powertrain_data = powertrain_data

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

        # === POWERTRAIN TELEMETRY ===
        powertrain_y = 28

        # Get powertrain data
        gear = self.powertrain_data.get('gear', 0)
        torque = self.powertrain_data.get('torque', 0.0)
        hp = self.powertrain_data.get('hp', 0.0)

        # Line 1: Gear and Torque
        # Gear display
        gear_label = self.font_small.render("Gear:", True, (200, 200, 200))
        surface.blit(gear_label, (10, powertrain_y))

        gear_text = self.font.render(f"{gear}", True, (100, 255, 255))
        surface.blit(gear_text, (50, powertrain_y - 2))

        # Torque display with bar
        torque_label_x = 90
        torque_text = self.font_small.render(f"Torque: {torque:.0f}Nm", True, (255, 200, 150))
        surface.blit(torque_text, (torque_label_x, powertrain_y))

        # Torque bar (0 to 210 Nm max)
        torque_bar_x = torque_label_x + 105
        torque_bar_width = 200
        torque_bar_height = 10
        max_torque = 210.0

        pygame.draw.rect(surface, (40, 40, 45), (torque_bar_x, powertrain_y, torque_bar_width, torque_bar_height))

        # Fill based on torque (support negative torque for engine braking)
        if torque >= 0:
            fill_ratio = min(abs(torque) / max_torque, 1.0)
            fill_width = int(fill_ratio * torque_bar_width)
            bar_color = (255, 150, 100)  # Orange for positive torque
            pygame.draw.rect(surface, bar_color, (torque_bar_x, powertrain_y, fill_width, torque_bar_height))
        else:
            # Negative torque (engine braking)
            fill_ratio = min(abs(torque) / 80.0, 1.0)  # Max 80 Nm engine braking
            fill_width = int(fill_ratio * torque_bar_width)
            bar_color = (100, 150, 255)  # Blue for engine braking
            pygame.draw.rect(surface, bar_color, (torque_bar_x, powertrain_y, fill_width, torque_bar_height))

        pygame.draw.rect(surface, (80, 80, 85), (torque_bar_x, powertrain_y, torque_bar_width, torque_bar_height), 1)

        # Line 2: HP display with bar
        hp_y = powertrain_y + 18
        hp_text = self.font_small.render(f"Power: {hp:.0f}hp", True, (150, 255, 150))
        surface.blit(hp_text, (90, hp_y))

        # HP bar (0 to 185 hp max)
        hp_bar_x = 90 + 85
        hp_bar_width = 220
        hp_bar_height = 10
        max_hp = 185.0

        pygame.draw.rect(surface, (40, 40, 45), (hp_bar_x, hp_y, hp_bar_width, hp_bar_height))

        fill_ratio = min(abs(hp) / max_hp, 1.0)
        fill_width = int(fill_ratio * hp_bar_width)

        # Color based on power output
        if hp < max_hp * 0.3:
            bar_color = (100, 200, 100)  # Green (low power)
        elif hp < max_hp * 0.7:
            bar_color = (255, 255, 100)  # Yellow (medium power)
        else:
            bar_color = (255, 150, 100)  # Orange (high power)

        pygame.draw.rect(surface, bar_color, (hp_bar_x, hp_y, fill_width, hp_bar_height))
        pygame.draw.rect(surface, (80, 80, 85), (hp_bar_x, hp_y, hp_bar_width, hp_bar_height), 1)

        # Separator line
        separator_y = powertrain_y + 40
        pygame.draw.line(surface, (60, 60, 70), (5, separator_y), (self.width - 5, separator_y), 1)

        # === WHEEL TELEMETRY ===
        # Wheel labels and positions
        wheel_names = ['FL', 'FR', 'RL', 'RR']
        wheel_colors = [(100, 150, 255), (100, 255, 150), (255, 150, 100), (255, 100, 150)]

        y_start = separator_y + 8
        row_height = 50  # Height per wheel row (2 lines: slip angle/ratio + normal force)

        for i, (name, color) in enumerate(zip(wheel_names, wheel_colors)):
            y = y_start + i * row_height

            # Wheel label
            label = self.font_small.render(name, True, color)
            surface.blit(label, (10, y))

            # Get telemetry data
            slip_angle = self.wheel_data[i]['slip_angle'] * 180 / np.pi  # Convert to degrees
            slip_ratio = self.wheel_data[i]['slip_ratio']
            normal_force = self.wheel_data[i]['normal_force']

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


def get_wheel_data(env):
    """Extract wheel telemetry data from the environment."""
    wheel_data = {
        0: {'slip_angle': 0.0, 'slip_ratio': 0.0, 'normal_force': 0.0},
        1: {'slip_angle': 0.0, 'slip_ratio': 0.0, 'normal_force': 0.0},
        2: {'slip_angle': 0.0, 'slip_ratio': 0.0, 'normal_force': 0.0},
        3: {'slip_angle': 0.0, 'slip_ratio': 0.0, 'normal_force': 0.0},
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


def get_powertrain_data(env, throttle=0.0):
    """Extract powertrain telemetry data from the environment."""
    powertrain_data = {
        'gear': 0,
        'torque': 0.0,
        'hp': 0.0,
    }

    if hasattr(env, 'unwrapped') and hasattr(env.unwrapped, 'car'):
        car = env.unwrapped.car
        if car is not None and hasattr(car, 'powertrain'):
            powertrain = car.powertrain
            powertrain_data['gear'] = powertrain.gearbox.current_gear
            powertrain_data['torque'] = powertrain.engine.get_torque(throttle)
            powertrain_data['hp'] = powertrain.engine.get_power_hp()

    return powertrain_data


def play_human_gui(args):
    """Play episodes with real-time telemetry GUI."""

    # Initialize Pygame
    pygame.init()
    pygame.font.init()
    font = pygame.font.Font(None, 24)

    # Create GUI
    gui = TelemetryGUI(width=420, height=310)

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
            current_gas = 0.0
            current_brake = 0.0

            print(f"\n{'-' * 80}")
            print(f"Episode {episode + 1}/{args.episodes}")
            print(f"{'-' * 80}")
            print("Ready! Use keyboard to control the car (smooth progressive controls)")

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

                # --- Keyboard Input with Progressive Smoothing ---
                keys = pygame.key.get_pressed()

                # Progressive gas/brake (ramps up/down smoothly)
                # Faster ramp up, slower ramp down for better control
                GAS_RAMP_UP = 0.08      # Speed when pressing gas (slower = more progressive)
                GAS_RAMP_DOWN = 0.15    # Speed when releasing gas (faster release)
                BRAKE_RAMP_UP = 0.12    # Speed when pressing brake
                BRAKE_RAMP_DOWN = 0.20  # Speed when releasing brake
                MAX_GAS = 0.85          # Maximum gas (< 1.0 for easier control)
                MAX_BRAKE = 0.90        # Maximum brake (< 1.0 for easier control)

                # Gas pedal (progressive)
                if keys[pygame.K_z] or keys[pygame.K_UP]:
                    target_gas = MAX_GAS
                    current_gas = current_gas + (target_gas - current_gas) * GAS_RAMP_UP
                else:
                    current_gas = current_gas * (1.0 - GAS_RAMP_DOWN)
                    if current_gas < 0.01:
                        current_gas = 0.0

                # Brake pedal (progressive)
                if keys[pygame.K_s] or keys[pygame.K_DOWN]:
                    target_brake = MAX_BRAKE
                    current_brake = current_brake + (target_brake - current_brake) * BRAKE_RAMP_UP
                else:
                    current_brake = current_brake * (1.0 - BRAKE_RAMP_DOWN)
                    if current_brake < 0.01:
                        current_brake = 0.0

                # Combine into acceleration (-1 = brake, +1 = gas)
                # Brake takes priority over gas
                if current_brake > 0.05:
                    acceleration = -current_brake
                else:
                    acceleration = current_gas

                # Steering (smooth with progressive return to center)
                # Reduced max steering for easier small adjustments
                MAX_STEER = 0.70    # Maximum steering angle (< 1.0 for gentler turns)
                STEER_SPEED = 0.05  # Very slow ramp for precise control

                target_steering = 0.0
                if keys[pygame.K_q] or keys[pygame.K_LEFT]:
                    target_steering = -MAX_STEER
                if keys[pygame.K_d] or keys[pygame.K_RIGHT]:
                    target_steering = MAX_STEER

                # Very smooth steering response for precise control
                current_steering = current_steering + (target_steering - current_steering) * STEER_SPEED

                action = np.array([current_steering, acceleration], dtype=np.float32)

                # --- Step Environment ---
                next_state, reward, terminated, truncated, _ = env.step(action)
                if done:
                    break

                done = terminated or truncated
                total_reward += reward
                step += 1

                # --- Get wheel data, powertrain data, and car speed ---
                wheel_data = get_wheel_data(env)
                # Extract throttle from action for accurate torque display
                throttle = max(0.0, acceleration) if acceleration > 0 else 0.0
                powertrain_data = get_powertrain_data(env, throttle)
                gui.update_data(wheel_data, powertrain_data)
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
