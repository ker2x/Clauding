"""
Standalone Magic Formula Parameter Visualizer.

This script provides an interactive GUI for visualizing Pacejka Magic Formula
tire force curves without needing to run the full game environment.

Usage:
    python magic_formula_visualizer.py

Controls:
    - Use mouse to drag sliders
    - Watch tire force curves update in real-time
    - Press ESC to quit, R to reset
"""

import pygame
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg
import sys


class MagicFormulaVisualizer:
    """Interactive visualizer for Pacejka Magic Formula parameters."""

    def __init__(self):
        pygame.init()
        pygame.font.init()

        self.font = pygame.font.Font(None, 24)
        self.font_small = pygame.font.Font(None, 18)

        # Window size
        self.panel_width = 450
        self.graph_width = 1000
        self.height = 700
        self.screen = pygame.display.set_mode((self.panel_width + self.graph_width, self.height))
        pygame.display.set_caption("Magic Formula Visualizer")

        # Default Pacejka parameters
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

        # Parameter ranges
        self.param_ranges = {
            'B_lat': (3.0, 25.0),
            'C_lat': (1.0, 3.0),
            'D_lat': (0.5, 2.0),
            'E_lat': (0.3, 2.0),
            'B_lon': (3.0, 25.0),
            'C_lon': (1.0, 3.0),
            'D_lon': (0.5, 2.0),
            'E_lon': (0.3, 2.0),
        }

        # Parameter descriptions
        self.descriptions = {
            'B_lat': 'Stiffness - initial slope',
            'C_lat': 'Shape - curve peakiness',
            'D_lat': 'Peak - max grip multiplier',
            'E_lat': 'Curvature - falloff shape',
            'B_lon': 'Stiffness - initial slope',
            'C_lon': 'Shape - curve peakiness',
            'D_lon': 'Peak - max grip multiplier',
            'E_lon': 'Curvature - falloff shape',
        }

        # Create sliders
        self.sliders = []
        y_start = 80
        y_spacing = 70

        # Lateral parameters
        for i, param in enumerate(['B_lat', 'C_lat', 'D_lat', 'E_lat']):
            self.sliders.append({
                'param': param,
                'y': y_start + i * y_spacing,
                'x': 20,
                'width': self.panel_width - 40,
                'height': 12,
                'dragging': False,
            })

        # Longitudinal parameters
        y_start_lon = y_start + 4 * y_spacing + 60
        for i, param in enumerate(['B_lon', 'C_lon', 'D_lon', 'E_lon']):
            self.sliders.append({
                'param': param,
                'y': y_start_lon + i * y_spacing,
                'x': 20,
                'width': self.panel_width - 40,
                'height': 12,
                'dragging': False,
            })

        # Create matplotlib figure
        self.fig, (self.ax_lat, self.ax_lon) = plt.subplots(2, 1, figsize=(10, 7))
        self.fig.tight_layout(pad=3.0)
        self.canvas = FigureCanvasAgg(self.fig)

        self.update_graphs()
        self.running = True
        self.clock = pygame.time.Clock()

    def get_slider_value(self, slider):
        """Get normalized value (0-1) for a slider."""
        param_name = slider['param']
        value = self.params[param_name]
        min_val, max_val = self.param_ranges[param_name]
        return (value - min_val) / (max_val - min_val)

    def set_slider_value(self, slider, normalized_value):
        """Set parameter value from normalized slider position."""
        param_name = slider['param']
        min_val, max_val = self.param_ranges[param_name]
        self.params[param_name] = min_val + normalized_value * (max_val - min_val)
        self.update_graphs()

    def handle_events(self):
        """Handle pygame events."""
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False

            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    self.running = False
                elif event.key == pygame.K_r:
                    # Reset to defaults
                    self.params = {
                        'B_lat': 10.0, 'C_lat': 1.9, 'D_lat': 1.1, 'E_lat': 0.95,
                        'B_lon': 9.0, 'C_lon': 1.9, 'D_lon': 1.4, 'E_lon': 0.95,
                    }
                    self.update_graphs()

            elif event.type == pygame.MOUSEBUTTONDOWN:
                mouse_x, mouse_y = event.pos
                for slider in self.sliders:
                    if (slider['x'] <= mouse_x <= slider['x'] + slider['width'] and
                        slider['y'] - 5 <= mouse_y <= slider['y'] + slider['height'] + 5):
                        slider['dragging'] = True
                        normalized_x = (mouse_x - slider['x']) / slider['width']
                        normalized_x = max(0.0, min(1.0, normalized_x))
                        self.set_slider_value(slider, normalized_x)

            elif event.type == pygame.MOUSEBUTTONUP:
                for slider in self.sliders:
                    slider['dragging'] = False

            elif event.type == pygame.MOUSEMOTION:
                mouse_x, mouse_y = event.pos
                for slider in self.sliders:
                    if slider['dragging']:
                        normalized_x = (mouse_x - slider['x']) / slider['width']
                        normalized_x = max(0.0, min(1.0, normalized_x))
                        self.set_slider_value(slider, normalized_x)

    def update_graphs(self):
        """Update tire force curve graphs."""
        self.ax_lat.clear()
        self.ax_lon.clear()

        normal_force = 2600  # N (approx weight per wheel)
        max_friction = 1.0

        # Lateral force curve
        slip_angles = np.linspace(-25, 25, 300) * np.pi / 180
        lateral_forces = []

        for sa in slip_angles:
            sa_clip = min(np.pi / 2, max(-np.pi / 2, sa))
            arg = self.params['B_lat'] * sa_clip
            F = (self.params['D_lat'] * normal_force * max_friction *
                 np.sin(self.params['C_lat'] * np.arctan(
                     arg - self.params['E_lat'] * (arg - np.arctan(arg)))))
            lateral_forces.append(F)

        # Find peak
        max_force = max(lateral_forces)
        peak_idx = lateral_forces.index(max_force)
        peak_angle = slip_angles[peak_idx] * 180 / np.pi

        self.ax_lat.plot(slip_angles * 180 / np.pi, lateral_forces, 'b-', linewidth=2.5, label='Lateral Force')
        self.ax_lat.axhline(y=max_force, color='r', linestyle='--', alpha=0.5, label=f'Peak: {max_force:.0f}N')
        self.ax_lat.axvline(x=peak_angle, color='g', linestyle='--', alpha=0.5, label=f'Peak angle: {peak_angle:.1f}°')
        self.ax_lat.set_xlabel('Slip Angle (degrees)', fontsize=12)
        self.ax_lat.set_ylabel('Lateral Force (N)', fontsize=12)
        self.ax_lat.set_title('Lateral (Cornering) Grip Curve', fontsize=14, fontweight='bold')
        self.ax_lat.grid(True, alpha=0.3)
        self.ax_lat.legend(loc='upper right')
        self.ax_lat.axhline(y=0, color='k', linestyle='-', linewidth=0.5)
        self.ax_lat.axvline(x=0, color='k', linestyle='-', linewidth=0.5)

        # Add parameter text
        param_text = f"B={self.params['B_lat']:.2f}  C={self.params['C_lat']:.2f}  D={self.params['D_lat']:.2f}  E={self.params['E_lat']:.2f}"
        self.ax_lat.text(0.02, 0.98, param_text, transform=self.ax_lat.transAxes,
                        fontsize=10, verticalalignment='top',
                        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

        # Longitudinal force curve
        slip_ratios = np.linspace(-1, 1, 300)
        longitudinal_forces = []

        for sr in slip_ratios:
            sr_clip = min(1.0, max(-1.0, sr))
            arg = self.params['B_lon'] * sr_clip
            F = (self.params['D_lon'] * normal_force * max_friction *
                 np.sin(self.params['C_lon'] * np.arctan(
                     arg - self.params['E_lon'] * (arg - np.arctan(arg)))))
            longitudinal_forces.append(F)

        # Find peak
        max_force = max(longitudinal_forces)
        peak_idx = longitudinal_forces.index(max_force)
        peak_ratio = slip_ratios[peak_idx]

        self.ax_lon.plot(slip_ratios, longitudinal_forces, 'r-', linewidth=2.5, label='Longitudinal Force')
        self.ax_lon.axhline(y=max_force, color='b', linestyle='--', alpha=0.5, label=f'Peak: {max_force:.0f}N')
        self.ax_lon.axvline(x=peak_ratio, color='g', linestyle='--', alpha=0.5, label=f'Peak ratio: {peak_ratio:.2f}')
        self.ax_lon.set_xlabel('Slip Ratio', fontsize=12)
        self.ax_lon.set_ylabel('Longitudinal Force (N)', fontsize=12)
        self.ax_lon.set_title('Longitudinal (Acceleration/Braking) Grip Curve', fontsize=14, fontweight='bold')
        self.ax_lon.grid(True, alpha=0.3)
        self.ax_lon.legend(loc='upper right')
        self.ax_lon.axhline(y=0, color='k', linestyle='-', linewidth=0.5)
        self.ax_lon.axvline(x=0, color='k', linestyle='-', linewidth=0.5)

        # Add parameter text
        param_text = f"B={self.params['B_lon']:.2f}  C={self.params['C_lon']:.2f}  D={self.params['D_lon']:.2f}  E={self.params['E_lon']:.2f}"
        self.ax_lon.text(0.02, 0.98, param_text, transform=self.ax_lon.transAxes,
                        fontsize=10, verticalalignment='top',
                        bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5))

        self.fig.tight_layout()
        self.canvas.draw()

    def draw_panel(self):
        """Draw control panel."""
        # Background
        panel_rect = pygame.Rect(0, 0, self.panel_width, self.height)
        self.screen.fill((30, 30, 35), panel_rect)

        # Title
        title = self.font.render("Magic Formula Parameters", True, (255, 255, 255))
        self.screen.blit(title, (10, 10))

        # Instructions
        inst = self.font_small.render("Drag sliders to adjust | R: Reset | ESC: Quit", True, (150, 150, 150))
        self.screen.blit(inst, (10, 40))

        # Section headers
        lat_header = self.font.render("LATERAL (Cornering)", True, (100, 150, 255))
        self.screen.blit(lat_header, (20, 60))

        lon_header = self.font.render("LONGITUDINAL (Accel/Brake)", True, (255, 100, 100))
        self.screen.blit(lon_header, (20, 360))

        # Draw sliders
        for slider in self.sliders:
            param = slider['param']

            # Parameter name
            name_text = param.replace('_', ' ').upper()
            name_surf = self.font_small.render(name_text, True, (220, 220, 220))
            self.screen.blit(name_surf, (slider['x'], slider['y'] - 35))

            # Description
            desc_surf = self.font_small.render(self.descriptions[param], True, (150, 150, 150))
            self.screen.blit(desc_surf, (slider['x'], slider['y'] - 18))

            # Current value
            value = self.params[param]
            min_val, max_val = self.param_ranges[param]
            value_text = f"{value:.2f}"
            value_surf = self.font.render(value_text, True, (255, 255, 100))
            self.screen.blit(value_surf, (slider['x'] + slider['width'] - 60, slider['y'] - 35))

            # Slider track
            track_rect = pygame.Rect(slider['x'], slider['y'], slider['width'], slider['height'])
            pygame.draw.rect(self.screen, (60, 60, 65), track_rect)
            pygame.draw.rect(self.screen, (100, 100, 105), track_rect, 2)

            # Slider fill (shows current value)
            normalized_pos = self.get_slider_value(slider)
            fill_width = int(normalized_pos * slider['width'])
            if fill_width > 0:
                fill_rect = pygame.Rect(slider['x'], slider['y'], fill_width, slider['height'])
                color = (100, 150, 255) if 'lat' in param else (255, 100, 100)
                pygame.draw.rect(self.screen, color, fill_rect)

            # Slider thumb
            thumb_x = slider['x'] + int(normalized_pos * slider['width'])
            thumb_rect = pygame.Rect(thumb_x - 6, slider['y'] - 4, 12, slider['height'] + 8)
            thumb_color = (150, 200, 255) if slider['dragging'] else (200, 200, 200)
            pygame.draw.rect(self.screen, thumb_color, thumb_rect, border_radius=3)
            pygame.draw.rect(self.screen, (255, 255, 255), thumb_rect, 2, border_radius=3)

            # Min/max labels
            min_label = self.font_small.render(f"{min_val:.1f}", True, (100, 100, 100))
            max_label = self.font_small.render(f"{max_val:.1f}", True, (100, 100, 100))
            self.screen.blit(min_label, (slider['x'], slider['y'] + slider['height'] + 2))
            self.screen.blit(max_label, (slider['x'] + slider['width'] - 30, slider['y'] + slider['height'] + 2))

    def draw_graphs(self):
        """Draw matplotlib graphs."""
        # Convert matplotlib canvas to pygame surface
        raw_data = self.canvas.get_renderer().buffer_rgba()
        size = self.canvas.get_width_height()
        graph_surf = pygame.image.frombuffer(raw_data, size, "RGBA")

        # Draw on screen
        self.screen.blit(graph_surf, (self.panel_width, 0))

    def run(self):
        """Main loop."""
        while self.running:
            self.handle_events()

            # Draw
            self.screen.fill((20, 20, 25))
            self.draw_panel()
            self.draw_graphs()

            pygame.display.flip()
            self.clock.tick(60)

        pygame.quit()


if __name__ == "__main__":
    print("=" * 80)
    print("Magic Formula Parameter Visualizer")
    print("=" * 80)
    print("\nThis tool lets you experiment with Pacejka Magic Formula parameters")
    print("and see how they affect tire force curves in real-time.")
    print("\nControls:")
    print("  - Drag sliders with mouse to adjust parameters")
    print("  - Press R to reset to default values")
    print("  - Press ESC to quit")
    print("\nParameter Guide:")
    print("  B (Stiffness): Controls initial slope and responsiveness")
    print("  C (Shape): Controls curve shape and peakiness")
    print("  D (Peak): Peak grip multiplier")
    print("  E (Curvature): Controls falloff after peak")
    print("=" * 80)

    visualizer = MagicFormulaVisualizer()
    visualizer.run()
