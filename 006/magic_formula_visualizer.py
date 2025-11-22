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

from __future__ import annotations

import pygame
import numpy as np
import numpy.typing as npt
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg
import sys
from typing import Any
from config.physics_config import PhysicsConfig


class MagicFormulaVisualizer:
    """Interactive visualizer for Pacejka Magic Formula parameters."""

    def __init__(self) -> None:
        pygame.init()
        pygame.font.init()

        self.font: Any = pygame.font.Font(None, 24)
        self.font_small: Any = pygame.font.Font(None, 18)

        # Window size
        self.panel_width: int = 450
        self.graph_width: int = 1000
        self.height: int = 700
        self.screen: Any = pygame.display.set_mode((self.panel_width + self.graph_width, self.height))
        pygame.display.set_caption("Magic Formula Visualizer")

        # Load default Pacejka parameters from physics config
        config = PhysicsConfig()

        # Default Pacejka parameters (loaded from env/config)
        # FULLY UNIFIED - same parameters for lateral AND longitudinal
        self.params: dict[str, float] = {
            'B': config.pacejka.B,  # Stiffness (unified)
            'C': config.pacejka.C,  # Shape (unified)
            'D': config.pacejka.D,  # Peak friction (unified)
            'E': config.pacejka.E,  # Curvature (unified)
        }

        # Store defaults for reset functionality
        self.default_params = self.params.copy()

        # Parameter ranges
        self.param_ranges: dict[str, tuple[float, float]] = {
            'B': (3.0, 25.0),
            'C': (1.0, 3.0),
            'D': (0.5, 2.0),
            'E': (0.3, 2.0),
        }

        # Parameter descriptions
        self.descriptions: dict[str, str] = {
            'B': 'Stiffness - initial slope (unified)',
            'C': 'Shape - curve peakiness (unified)',
            'D': 'Peak friction - traction circle (unified)',
            'E': 'Curvature - falloff shape (unified)',
        }

        # Create sliders - FULLY UNIFIED (same for lateral and longitudinal)
        self.sliders: list[dict[str, Any]] = []
        y_start: int = 120
        y_spacing: int = 80

        # All 4 unified parameters
        for i, param in enumerate(['B', 'C', 'D', 'E']):
            self.sliders.append({
                'param': param,
                'y': y_start + i * y_spacing,
                'x': 20,
                'width': self.panel_width - 40,
                'height': 12,
                'dragging': False,
            })

        # Create matplotlib figure
        self.fig: Any
        self.ax_lat: Any
        self.ax_lon: Any
        self.fig, (self.ax_lat, self.ax_lon) = plt.subplots(2, 1, figsize=(10, 7))
        self.fig.tight_layout(pad=3.0)
        self.canvas: Any = FigureCanvasAgg(self.fig)

        self.update_graphs()
        self.running: bool = True
        self.clock: Any = pygame.time.Clock()

    def get_slider_value(self, slider: dict[str, Any]) -> float:
        """Get normalized value (0-1) for a slider."""
        param_name: str = slider['param']
        value: float = self.params[param_name]
        min_val: float
        max_val: float
        min_val, max_val = self.param_ranges[param_name]
        return (value - min_val) / (max_val - min_val)

    def set_slider_value(self, slider: dict[str, Any], normalized_value: float) -> None:
        """Set parameter value from normalized slider position."""
        param_name: str = slider['param']
        min_val: float
        max_val: float
        min_val, max_val = self.param_ranges[param_name]
        self.params[param_name] = min_val + normalized_value * (max_val - min_val)
        self.update_graphs()

    def handle_events(self) -> None:
        """Handle pygame events."""
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False

            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    self.running = False
                elif event.key == pygame.K_r:
                    # Reset to defaults (loaded from physics config)
                    self.params = self.default_params.copy()
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
        """Update tire force curve graphs - FULLY UNIFIED parameters."""
        self.ax_lat.clear()
        self.ax_lon.clear()

        # Multiple normal force values to show load sensitivity
        # 2000N = light load (inside wheels during cornering)
        # 2600N = nominal load (static weight per wheel)
        # 3200N = heavy load (outside wheels during cornering/braking)
        normal_forces = [2000, 2600, 3200]
        colors = ['lightblue', 'blue', 'darkblue']
        line_styles = ['--', '-', '-.']
        max_friction = 1.0

        # FULLY UNIFIED parameters (same for both curves)
        B = self.params['B']
        C = self.params['C']
        D = self.params['D']
        E = self.params['E']

        # Lateral force curves at different loads
        slip_angles = np.linspace(-25, 25, 300) * np.pi / 180

        for normal_force, color, line_style in zip(normal_forces, colors, line_styles):
            lateral_forces = []

            for sa in slip_angles:
                sa_clip = min(np.pi / 2, max(-np.pi / 2, sa))
                arg = B * sa_clip
                F = (D * normal_force * max_friction *
                     np.sin(C * np.arctan(arg - E * (arg - np.arctan(arg)))))
                lateral_forces.append(F)

            # Find peak
            max_force = max(lateral_forces)
            peak_idx = lateral_forces.index(max_force)
            peak_angle = slip_angles[peak_idx] * 180 / np.pi

            # Plot curve
            label = f'Fz={normal_force}N ({max_force/(normal_force):.2f}g peak)'
            linewidth = 2.5 if normal_force == 2600 else 2.0
            self.ax_lat.plot(slip_angles * 180 / np.pi, lateral_forces, color=color,
                           linestyle=line_style, linewidth=linewidth, label=label)

        self.ax_lat.set_xlabel('Slip Angle (degrees)', fontsize=12)
        self.ax_lat.set_ylabel('Lateral Force (N)', fontsize=12)
        self.ax_lat.set_title('Lateral (Cornering) - FULLY UNIFIED - Multiple Loads', fontsize=14, fontweight='bold')
        self.ax_lat.grid(True, alpha=0.3)
        self.ax_lat.legend(loc='upper right', fontsize=9)
        self.ax_lat.axhline(y=0, color='k', linestyle='-', linewidth=0.5)
        self.ax_lat.axvline(x=0, color='k', linestyle='-', linewidth=0.5)

        # Add parameter text
        param_text = f"UNIFIED: B={B:.2f}  C={C:.2f}  D={D:.2f}  E={E:.2f}"
        self.ax_lat.text(0.02, 0.98, param_text, transform=self.ax_lat.transAxes,
                        fontsize=10, verticalalignment='top',
                        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

        # Longitudinal force curves at different loads (SAME parameters as lateral)
        slip_ratios = np.linspace(-1, 1, 300)

        for normal_force, color, line_style in zip(normal_forces, colors, line_styles):
            longitudinal_forces = []

            for sr in slip_ratios:
                sr_clip = min(1.0, max(-1.0, sr))
                arg = B * sr_clip
                F = (D * normal_force * max_friction *
                     np.sin(C * np.arctan(arg - E * (arg - np.arctan(arg)))))
                longitudinal_forces.append(F)

            # Find peak
            max_force = max(longitudinal_forces)
            peak_idx = longitudinal_forces.index(max_force)
            peak_ratio = slip_ratios[peak_idx]

            # Plot curve
            label = f'Fz={normal_force}N ({max_force/(normal_force):.2f}g peak)'
            linewidth = 2.5 if normal_force == 2600 else 2.0
            self.ax_lon.plot(slip_ratios, longitudinal_forces, color=color,
                           linestyle=line_style, linewidth=linewidth, label=label)

        self.ax_lon.set_xlabel('Slip Ratio', fontsize=12)
        self.ax_lon.set_ylabel('Longitudinal Force (N)', fontsize=12)
        self.ax_lon.set_title('Longitudinal (Accel/Brake) - FULLY UNIFIED - Multiple Loads', fontsize=14, fontweight='bold')
        self.ax_lon.grid(True, alpha=0.3)
        self.ax_lon.legend(loc='upper right', fontsize=9)
        self.ax_lon.axhline(y=0, color='k', linestyle='-', linewidth=0.5)
        self.ax_lon.axvline(x=0, color='k', linestyle='-', linewidth=0.5)

        # Add parameter text
        param_text = f"UNIFIED: B={B:.2f}  C={C:.2f}  D={D:.2f}  E={E:.2f}"
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
        title = self.font.render("Magic Formula - UNIFIED", True, (255, 255, 255))
        self.screen.blit(title, (10, 10))

        # Instructions
        inst = self.font_small.render("Drag sliders to adjust | R: Reset | ESC: Quit", True, (150, 150, 150))
        self.screen.blit(inst, (10, 40))

        # Section header
        unified_header = self.font.render("UNIFIED PARAMETERS", True, (255, 200, 100))
        self.screen.blit(unified_header, (20, 70))

        # Subtitle
        subtitle = self.font_small.render("Same for Lateral & Longitudinal", True, (180, 180, 180))
        self.screen.blit(subtitle, (20, 95))

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
                # Color coding for unified parameters
                color_map = {
                    'B': (100, 200, 255),  # Light blue
                    'C': (150, 255, 150),  # Light green
                    'D': (255, 150, 50),   # Orange
                    'E': (255, 200, 100),  # Yellow
                }
                color = color_map.get(param, (200, 200, 200))
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
