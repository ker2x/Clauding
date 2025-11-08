"""
Human-controlled CarRacing-v3 with Magic Formula Parameter GUI.

This script provides an interactive GUI for tuning Pacejka Magic Formula parameters
while playing the game. It includes:
- Real-time parameter adjustment sliders
- Live tire force curve visualization
- Real-time wheel slip display (slip angle and slip ratio for all 4 wheels)
- Standard keyboard controls for gameplay

Usage:
    python play_human_gui.py

Controls:
    - Steering:   Q/D, Left/Right (AZERTY: Q/D)
    - Gas:        Z, Up Arrow
    - Brake:      S, Down Arrow
    - Reset:      R
    - Quit:       ESC only

GUI Controls:
    - Sliders adjust Pacejka parameters in real-time
    - Graphs show lateral and longitudinal force curves
    - Wheel slip bars show real-time SA (slip angle) and SR (slip ratio) for FL, FR, RL, RR
"""

import argparse
import numpy as np
import time
import pygame
import sys
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg
from preprocessing import make_carracing_env


class PacejkaGUI:
    """GUI for controlling Pacejka Magic Formula parameters."""

    def __init__(self, width=400, height=800):
        self.width = width
        self.height = height
        self.font = pygame.font.Font(None, 20)
        self.font_small = pygame.font.Font(None, 16)

        # Default Pacejka parameters (from car_dynamics.py)
        self.params = {
            'B_lat': 10.0,
            'C_lat': 1.9,
            'D_lat': 1.1,
            'E_lat': 0.95,
            'B_lon': 9.0,
            'C_lon': 1.9,
            'D_lon': 1.4,
            'E_lon': 0.95,
        }

        # Parameter ranges for sliders
        self.param_ranges = {
            'B_lat': (5.0, 20.0),   # Stiffness factor
            'C_lat': (1.0, 2.5),    # Shape factor
            'D_lat': (0.8, 1.5),    # Peak friction
            'E_lat': (0.5, 1.5),    # Curvature factor
            'B_lon': (5.0, 20.0),
            'C_lon': (1.0, 2.5),
            'D_lon': (0.8, 1.5),
            'E_lon': (0.5, 1.5),
        }

        # Slider definitions (y position, parameter name, label)
        y_start = 50
        y_spacing = 60
        self.sliders = []
        param_names = ['B_lat', 'C_lat', 'D_lat', 'E_lat', 'B_lon', 'C_lon', 'D_lon', 'E_lon']
        labels = [
            'B_lat (Lateral Stiffness)',
            'C_lat (Lateral Shape)',
            'D_lat (Lateral Peak)',
            'E_lat (Lateral Curve)',
            'B_lon (Longitudinal Stiffness)',
            'C_lon (Longitudinal Shape)',
            'D_lon (Longitudinal Peak)',
            'E_lon (Longitudinal Curve)',
        ]

        for i, (param, label) in enumerate(zip(param_names, labels)):
            y = y_start + i * y_spacing
            slider = {
                'param': param,
                'label': label,
                'y': y,
                'x': 20,
                'width': self.width - 40,
                'height': 10,
                'dragging': False,
            }
            self.sliders.append(slider)

        # Create matplotlib figure for tire curves
        self.fig, (self.ax_lat, self.ax_lon) = plt.subplots(1, 2, figsize=(8, 3))
        self.fig.tight_layout(pad=2.0)
        self.canvas = FigureCanvasAgg(self.fig)

        # Wheel slip data (updated each frame)
        self.wheel_slip_data = {
            0: {'slip_angle': 0.0, 'slip_ratio': 0.0},  # FL
            1: {'slip_angle': 0.0, 'slip_ratio': 0.0},  # FR
            2: {'slip_angle': 0.0, 'slip_ratio': 0.0},  # RL
            3: {'slip_angle': 0.0, 'slip_ratio': 0.0},  # RR
        }

        self.update_graphs()

    def get_slider_value(self, slider):
        """Get normalized value (0-1) for a slider based on current parameter."""
        param_name = slider['param']
        value = self.params[param_name]
        min_val, max_val = self.param_ranges[param_name]
        return (value - min_val) / (max_val - min_val)

    def set_slider_value(self, slider, normalized_value):
        """Set parameter value from normalized slider position (0-1)."""
        param_name = slider['param']
        min_val, max_val = self.param_ranges[param_name]
        self.params[param_name] = min_val + normalized_value * (max_val - min_val)
        self.update_graphs()

    def handle_event(self, event, offset_x=0, offset_y=0):
        """Handle mouse events for sliders. Returns True if event was handled."""
        if event.type == pygame.MOUSEBUTTONDOWN:
            mouse_x, mouse_y = event.pos
            mouse_x -= offset_x
            mouse_y -= offset_y

            for slider in self.sliders:
                if (slider['x'] <= mouse_x <= slider['x'] + slider['width'] and
                    slider['y'] - 5 <= mouse_y <= slider['y'] + slider['height'] + 5):
                    slider['dragging'] = True
                    # Set value based on click position
                    normalized_x = (mouse_x - slider['x']) / slider['width']
                    normalized_x = max(0.0, min(1.0, normalized_x))
                    self.set_slider_value(slider, normalized_x)
                    return True

        elif event.type == pygame.MOUSEBUTTONUP:
            for slider in self.sliders:
                slider['dragging'] = False

        elif event.type == pygame.MOUSEMOTION:
            mouse_x, mouse_y = event.pos
            mouse_x -= offset_x
            mouse_y -= offset_y

            for slider in self.sliders:
                if slider['dragging']:
                    normalized_x = (mouse_x - slider['x']) / slider['width']
                    normalized_x = max(0.0, min(1.0, normalized_x))
                    self.set_slider_value(slider, normalized_x)
                    return True

        return False

    def update_graphs(self):
        """Update tire force curve graphs."""
        # Clear axes
        self.ax_lat.clear()
        self.ax_lon.clear()

        # Lateral force curve (vs slip angle in degrees)
        slip_angles = np.linspace(-20, 20, 200) * np.pi / 180  # -20 to 20 degrees
        lateral_forces = []

        normal_force = 2600  # N (approx weight per wheel)
        max_friction = 1.0

        for sa in slip_angles:
            sa_clip = np.clip(sa, -np.pi / 2, np.pi / 2)
            arg = self.params['B_lat'] * sa_clip
            F = (self.params['D_lat'] * normal_force * max_friction *
                 np.sin(self.params['C_lat'] * np.arctan(
                     arg - self.params['E_lat'] * (arg - np.arctan(arg)))))
            lateral_forces.append(F)

        self.ax_lat.plot(slip_angles * 180 / np.pi, lateral_forces, 'b-', linewidth=2)
        self.ax_lat.set_xlabel('Slip Angle (degrees)')
        self.ax_lat.set_ylabel('Lateral Force (N)')
        self.ax_lat.set_title('Lateral Grip Curve')
        self.ax_lat.grid(True, alpha=0.3)
        self.ax_lat.axhline(y=0, color='k', linestyle='-', linewidth=0.5)
        self.ax_lat.axvline(x=0, color='k', linestyle='-', linewidth=0.5)

        # Longitudinal force curve (vs slip ratio)
        slip_ratios = np.linspace(-1, 1, 200)
        longitudinal_forces = []

        for sr in slip_ratios:
            sr_clip = np.clip(sr, -1.0, 1.0)
            arg = self.params['B_lon'] * sr_clip
            F = (self.params['D_lon'] * normal_force * max_friction *
                 np.sin(self.params['C_lon'] * np.arctan(
                     arg - self.params['E_lon'] * (arg - np.arctan(arg)))))
            longitudinal_forces.append(F)

        self.ax_lon.plot(slip_ratios, longitudinal_forces, 'r-', linewidth=2)
        self.ax_lon.set_xlabel('Slip Ratio')
        self.ax_lon.set_ylabel('Longitudinal Force (N)')
        self.ax_lon.set_title('Longitudinal Grip Curve')
        self.ax_lon.grid(True, alpha=0.3)
        self.ax_lon.axhline(y=0, color='k', linestyle='-', linewidth=0.5)
        self.ax_lon.axvline(x=0, color='k', linestyle='-', linewidth=0.5)

        self.fig.tight_layout()
        self.canvas.draw()

    def get_graph_surface(self):
        """Get pygame surface of the current graphs."""
        # Convert matplotlib canvas to pygame surface
        canvas = self.canvas
        renderer = canvas.get_renderer()
        raw_data = renderer.buffer_rgba()
        size = canvas.get_width_height()
        return pygame.image.frombuffer(raw_data, size, "RGBA")

    def draw(self, surface):
        """Draw the GUI panel on the given surface."""
        # Background
        surface.fill((40, 40, 40))

        # Title
        title = self.font.render("Magic Formula Parameters", True, (255, 255, 255))
        surface.blit(title, (10, 10))

        # Draw sliders
        for slider in self.sliders:
            # Label
            label_surf = self.font_small.render(slider['label'], True, (200, 200, 200))
            surface.blit(label_surf, (slider['x'], slider['y'] - 20))

            # Value
            param_value = self.params[slider['param']]
            value_text = f"{param_value:.2f}"
            value_surf = self.font_small.render(value_text, True, (255, 255, 100))
            surface.blit(value_surf, (slider['x'] + slider['width'] - 50, slider['y'] - 20))

            # Slider track
            track_rect = pygame.Rect(slider['x'], slider['y'], slider['width'], slider['height'])
            pygame.draw.rect(surface, (80, 80, 80), track_rect)
            pygame.draw.rect(surface, (120, 120, 120), track_rect, 1)

            # Slider thumb
            normalized_pos = self.get_slider_value(slider)
            thumb_x = slider['x'] + int(normalized_pos * slider['width'])
            thumb_rect = pygame.Rect(thumb_x - 5, slider['y'] - 5, 10, slider['height'] + 10)
            color = (100, 200, 255) if slider['dragging'] else (150, 150, 255)
            pygame.draw.rect(surface, color, thumb_rect)
            pygame.draw.rect(surface, (200, 200, 200), thumb_rect, 2)

    def update_slip_data(self, slip_data):
        """Update wheel slip data from the environment."""
        if slip_data:
            self.wheel_slip_data = slip_data

    def draw_slip_panel(self, surface, y_offset=0):
        """Draw real-time wheel slip visualization."""
        panel_width = self.width
        panel_height = 200

        # Background
        pygame.draw.rect(surface, (25, 25, 30), (0, y_offset, panel_width, panel_height))
        pygame.draw.rect(surface, (60, 60, 70), (0, y_offset, panel_width, panel_height), 2)

        # Title
        title = self.font.render("Wheel Slip (Real-time)", True, (255, 200, 100))
        surface.blit(title, (10, y_offset + 5))

        # Wheel labels and positions
        wheel_names = ['FL', 'FR', 'RL', 'RR']
        wheel_colors = [(100, 150, 255), (100, 255, 150), (255, 150, 100), (255, 100, 150)]

        y_start = y_offset + 35
        row_height = 40

        for i, (name, color) in enumerate(zip(wheel_names, wheel_colors)):
            y = y_start + i * row_height

            # Wheel label
            label = self.font_small.render(name, True, color)
            surface.blit(label, (10, y))

            # Get slip data
            slip_angle = self.wheel_slip_data[i]['slip_angle'] * 180 / np.pi  # Convert to degrees
            slip_ratio = self.wheel_slip_data[i]['slip_ratio']

            # Slip angle bar (range: -25 to +25 degrees)
            angle_text = self.font_small.render(f"SA: {slip_angle:+.1f}°", True, (200, 200, 200))
            surface.blit(angle_text, (45, y))

            # Slip angle bar
            bar_x = 130
            bar_width = 100
            bar_height = 12
            max_angle = 25.0

            # Draw bar background
            pygame.draw.rect(surface, (40, 40, 45), (bar_x, y, bar_width, bar_height))

            # Draw bar fill (centered at middle)
            center_x = bar_x + bar_width // 2
            fill_width = int((slip_angle / max_angle) * (bar_width // 2))
            fill_width = max(-bar_width // 2, min(bar_width // 2, fill_width))

            if fill_width > 0:
                pygame.draw.rect(surface, (100, 150, 255), (center_x, y, fill_width, bar_height))
            elif fill_width < 0:
                pygame.draw.rect(surface, (255, 150, 100), (center_x + fill_width, y, -fill_width, bar_height))

            # Center line
            pygame.draw.line(surface, (100, 100, 100), (center_x, y), (center_x, y + bar_height), 1)
            pygame.draw.rect(surface, (80, 80, 85), (bar_x, y, bar_width, bar_height), 1)

            # Slip ratio bar (range: -1 to +1)
            ratio_text = self.font_small.render(f"SR: {slip_ratio:+.2f}", True, (200, 200, 200))
            surface.blit(ratio_text, (bar_x + bar_width + 10, y))

            # Slip ratio bar
            ratio_bar_x = bar_x + bar_width + 75
            ratio_bar_width = 80

            # Draw bar background
            pygame.draw.rect(surface, (40, 40, 45), (ratio_bar_x, y, ratio_bar_width, bar_height))

            # Draw bar fill (centered at middle)
            ratio_center_x = ratio_bar_x + ratio_bar_width // 2
            ratio_fill_width = int(slip_ratio * (ratio_bar_width // 2))
            ratio_fill_width = max(-ratio_bar_width // 2, min(ratio_bar_width // 2, ratio_fill_width))

            if ratio_fill_width > 0:
                pygame.draw.rect(surface, (100, 255, 100), (ratio_center_x, y, ratio_fill_width, bar_height))
            elif ratio_fill_width < 0:
                pygame.draw.rect(surface, (255, 100, 100), (ratio_center_x + ratio_fill_width, y, -ratio_fill_width, bar_height))

            # Center line
            pygame.draw.line(surface, (100, 100, 100), (ratio_center_x, y), (ratio_center_x, y + bar_height), 1)
            pygame.draw.rect(surface, (80, 80, 85), (ratio_bar_x, y, ratio_bar_width, bar_height), 1)


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


def render_info(screen, font, episode, step, reward, total_reward, action, info_y_offset=0):
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


def update_car_parameters(env, params):
    """Update the car's Pacejka parameters in the environment."""
    if hasattr(env, 'unwrapped') and hasattr(env.unwrapped, 'car'):
        car = env.unwrapped.car
        if car is not None and hasattr(car, 'tire'):
            car.tire.B_lat = params['B_lat']
            car.tire.C_lat = params['C_lat']
            car.tire.D_lat = params['D_lat']
            car.tire.E_lat = params['E_lat']
            car.tire.B_lon = params['B_lon']
            car.tire.C_lon = params['C_lon']
            car.tire.D_lon = params['D_lon']
            car.tire.E_lon = params['E_lon']
            return True
    return False


def get_wheel_slip_data(env):
    """Extract wheel slip data from the environment."""
    slip_data = {
        0: {'slip_angle': 0.0, 'slip_ratio': 0.0},
        1: {'slip_angle': 0.0, 'slip_ratio': 0.0},
        2: {'slip_angle': 0.0, 'slip_ratio': 0.0},
        3: {'slip_angle': 0.0, 'slip_ratio': 0.0},
    }

    if hasattr(env, 'unwrapped') and hasattr(env.unwrapped, 'car'):
        car = env.unwrapped.car
        if car is not None:
            # Compute tire forces to get slip data
            friction = car._get_surface_friction() if hasattr(car, '_get_surface_friction') else 1.0
            forces = car._compute_tire_forces(friction) if hasattr(car, '_compute_tire_forces') else None

            if forces:
                for i in range(4):
                    if i in forces:
                        slip_data[i]['slip_angle'] = forces[i].get('slip_angle', 0.0)
                        slip_data[i]['slip_ratio'] = forces[i].get('slip_ratio', 0.0)

    return slip_data


def play_human_gui(args):
    """Play episodes with Magic Formula parameter GUI."""

    # Initialize Pygame
    pygame.init()
    pygame.font.init()
    font = pygame.font.Font(None, 24)

    # Create GUI
    gui = PacejkaGUI(width=400, height=550)

    # Create environment
    render_mode = None if args.no_render else 'rgb_array'
    env = make_carracing_env(
        stack_size=4,
        terminate_stationary=False,
        stationary_patience=100,
        render_mode=render_mode,
        state_mode='visual'
    )

    action_dim = env.action_space.shape[0]
    state_shape = env.observation_space.shape

    print("=" * 80)
    print(f"CarRacing-v3 - HUMAN PLAYER with Magic Formula GUI")
    print("=" * 80)
    print(f"Episodes: {args.episodes}")
    print(f"State shape: {state_shape}")
    print("=" * 80)
    print("\nKEYBOARD CONTROLS (AZERTY):")
    print("  - Steering:   Q / D or Left / Right Arrow")
    print("  - Gas:        Z or Up Arrow")
    print("  - Brake:      S or Down Arrow")
    print("  - Reset:      R")
    print("  - Quit:       ESC only")
    print("\nGUI CONTROLS:")
    print("  - Use mouse to drag sliders and adjust Pacejka parameters")
    print("  - Graphs show real-time tire force curves")
    print("=" * 80)

    episode_rewards = []
    target_frame_time = 1.0 / args.fps
    screen = None

    try:
        for episode in range(args.episodes):
            state, _ = env.reset()

            # Update car parameters from GUI
            update_car_parameters(env, gui.params)

            total_reward = 0
            step = 0
            done = False

            action = np.array([0.0, 0.0], dtype=np.float32)
            current_steering = 0.0

            print(f"\n{'-' * 80}")
            print(f"Episode {episode + 1}/{args.episodes}")
            print(f"{'-' * 80}")
            print("Ready! Use keyboard to control, mouse to adjust parameters")

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

                    # Handle GUI events (pass offset for GUI panel location)
                    if not args.no_render and screen is not None:
                        frame_w = screen.get_size()[0] - gui.width
                        if gui.handle_event(event, offset_x=frame_w, offset_y=0):
                            # Parameters changed, update car
                            update_car_parameters(env, gui.params)

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

                # --- Get wheel slip data ---
                slip_data = get_wheel_slip_data(env)
                gui.update_slip_data(slip_data)

                # --- Render ---
                if not args.no_render:
                    rgb_frame = env.render()
                    frame_h, frame_w = rgb_frame.shape[:2]
                    info_area_height = 100

                    # Create screen on first frame
                    if screen is None:
                        total_width = frame_w + gui.width
                        total_height = max(frame_h + info_area_height, gui.height + 450)
                        screen = pygame.display.set_mode((total_width, total_height))
                        pygame.display.set_caption("CarRacing-v3 - Magic Formula GUI")

                    # Clear screen
                    screen.fill((0, 0, 0))

                    # Draw game frame
                    frame_pygame = np.transpose(rgb_frame, (1, 0, 2))
                    surf = pygame.surfarray.make_surface(frame_pygame)
                    screen.blit(surf, (0, info_area_height))

                    # Draw info overlay
                    render_info(screen, font, episode + 1, step, reward, total_reward, action, info_y_offset=0)

                    # Draw GUI panel (right side)
                    gui_surface = pygame.Surface((gui.width, gui.height))
                    gui.draw(gui_surface)
                    screen.blit(gui_surface, (frame_w, 0))

                    # Draw slip panel (right side, below parameter sliders)
                    slip_surface = pygame.Surface((gui.width, 200))
                    gui.draw_slip_panel(slip_surface, y_offset=0)
                    screen.blit(slip_surface, (frame_w, gui.height))

                    # Draw graphs (below slip panel)
                    graph_surface = gui.get_graph_surface()
                    screen.blit(graph_surface, (frame_w - 400, gui.height + 200))

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
        description='Play CarRacing-v3 with Magic Formula parameter GUI'
    )
    parser.add_argument('--episodes', type=int, default=5,
                        help='Number of episodes to play (default: 5)')
    parser.add_argument('--fps', type=int, default=50,
                        help='Display FPS (default: 50)')
    parser.add_argument('--no-render', action='store_true',
                        help='Disable rendering (just compute rewards)')
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    play_human_gui(args)
