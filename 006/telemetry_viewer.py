#!/usr/bin/env python3
"""
Professional Interactive Telemetry Viewer

A comprehensive visualization tool for analyzing racing telemetry data.
Features synchronized multi-plot displays, track map visualization, and
interactive data inspection.

Usage:
    python telemetry_viewer.py telemetry_20250113_123456.csv
    python telemetry_viewer.py --compare file1.csv file2.csv

Controls:
    - Click on any plot to inspect data at that time
    - Mouse wheel to zoom
    - Left/Right arrows: Navigate through time
    - Space: Play/Pause animation
    - R: Reset zoom
    - Q: Quit

Requirements:
    pip install matplotlib numpy
"""

from __future__ import annotations

import argparse
import csv
import sys
import math
from typing import Any
import numpy as np
import numpy.typing as npt
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.widgets import Slider, Button
from matplotlib.patches import Rectangle, Circle
import matplotlib.patches as mpatches


class TelemetryData:
    """Container for telemetry data."""

    def __init__(self, filename: str) -> None:
        self.filename: str = filename
        self.data: list[dict[str, Any]] = []
        self.timestamps: list[str] = []
        self.load_data()

    def load_data(self) -> None:
        """Load telemetry CSV file."""
        try:
            with open(self.filename, 'r') as f:
                reader = csv.DictReader(f)
                for i, row in enumerate(reader):
                    # Convert numeric fields
                    converted_row = {'index': i}
                    for key, value in row.items():
                        try:
                            if key == 'timestamp':
                                converted_row[key] = value
                            else:
                                converted_row[key] = float(value)
                        except (ValueError, KeyError):
                            converted_row[key] = value
                    self.data.append(converted_row)

            print(f"✓ Loaded: {self.filename}")
            print(f"  Frames: {len(self.data)}")
            if self.data:
                episodes = set(row['episode'] for row in self.data)
                print(f"  Episodes: {sorted(episodes)}")
        except FileNotFoundError:
            print(f"✗ File not found: {self.filename}")
            sys.exit(1)
        except Exception as e:
            print(f"✗ Error loading file: {e}")
            sys.exit(1)

    def get_column(self, column_name: str) -> npt.NDArray[np.float64]:
        """Extract a column as numpy array."""
        return np.array([row.get(column_name, 0) for row in self.data])

    def get_value_at_index(self, idx: int) -> dict[str, Any] | None:
        """Get data row at specific index."""
        if 0 <= idx < len(self.data):
            return self.data[idx]
        return None


class TelemetryViewer:
    """Interactive telemetry visualization application."""

    def __init__(self, telemetry_files: list[str]) -> None:
        self.telemetries: list[TelemetryData] = [TelemetryData(f) for f in telemetry_files]
        self.current_file: int = 0
        self.current_data: TelemetryData = self.telemetries[self.current_file]

        # Animation state
        self.current_index: int = 0
        self.playing: bool = False
        self.animation_timer: Any = None

        # Color scheme - professional motorsport colors
        self.colors: dict[str, str] = {
            'speed': '#00D9FF',      # Cyan
            'steering': '#FFD700',   # Gold
            'accel': '#00FF00',      # Green
            'brake': '#FF4444',      # Red
            'slip_angle': '#FF6B00', # Orange
            'slip_ratio': '#9D00FF', # Purple
            'force': '#00FF88',      # Mint
            'suspension': '#FF1493', # Pink
            'track': '#444444',      # Dark gray
            'car': '#FF0000',        # Red
            'bg': '#0A0A0A',         # Almost black
            'grid': '#222222',       # Dark gray
            'text': '#FFFFFF',       # White
        }

        self.setup_ui()
        self.connect_events()

    def setup_ui(self) -> None:
        """Create the user interface."""
        # Create figure with dark theme
        plt.style.use('dark_background')
        self.fig = plt.figure(figsize=(16, 10))
        self.fig.patch.set_facecolor(self.colors['bg'])
        self.fig.canvas.manager.set_window_title('Professional Telemetry Viewer')

        # Create grid layout
        gs = GridSpec(4, 3, figure=self.fig, hspace=0.35, wspace=0.3,
                     left=0.05, right=0.98, top=0.96, bottom=0.08)

        # Track map (top left, larger)
        self.ax_track = self.fig.add_subplot(gs[0:2, 0])

        # Info panel (top middle)
        self.ax_info = self.fig.add_subplot(gs[0, 1:])
        self.ax_info.axis('off')

        # Speed plot (top right + middle right)
        self.ax_speed = self.fig.add_subplot(gs[1, 1:])

        # Input plots (middle row)
        self.ax_inputs = self.fig.add_subplot(gs[2, :])

        # Wheel telemetry (bottom row)
        self.ax_slip_angle = self.fig.add_subplot(gs[3, 0])
        self.ax_slip_ratio = self.fig.add_subplot(gs[3, 1])
        self.ax_forces = self.fig.add_subplot(gs[3, 2])

        self.time_series_axes = [self.ax_speed, self.ax_inputs,
                                 self.ax_slip_angle, self.ax_slip_ratio,
                                 self.ax_forces]

        # Plot all data
        self.plot_track_map()
        self.plot_speed()
        self.plot_inputs()
        self.plot_wheel_telemetry()
        self.update_info_panel()

        # Add title
        title = f"Telemetry Viewer - {self.current_data.filename.split('/')[-1]}"
        self.fig.suptitle(title, fontsize=14, fontweight='bold',
                         color=self.colors['text'])

    def plot_track_map(self) -> None:
        """Plot the track map with speed overlay."""
        ax = self.ax_track
        ax.clear()
        ax.set_facecolor(self.colors['bg'])

        data = self.current_data
        x = data.get_column('car_x')
        y = data.get_column('car_y')
        speed = data.get_column('speed_kmh')

        # Plot track with speed colormap
        if len(x) > 1:
            points = np.array([x, y]).T.reshape(-1, 1, 2)
            segments = np.concatenate([points[:-1], points[1:]], axis=1)

            from matplotlib.collections import LineCollection
            lc = LineCollection(segments, cmap='plasma', linewidth=3)
            lc.set_array(speed)
            lc.set_clim(0, np.max(speed))
            ax.add_collection(lc)

            # Add colorbar
            cbar = self.fig.colorbar(lc, ax=ax, fraction=0.046, pad=0.04)
            cbar.set_label('Speed (km/h)', color=self.colors['text'])
            cbar.ax.yaxis.set_tick_params(color=self.colors['text'])
            plt.setp(plt.getp(cbar.ax.axes, 'yticklabels'),
                    color=self.colors['text'])

        # Current position marker
        if len(x) > 0:
            self.car_marker = ax.plot(x[0], y[0], 'o', color=self.colors['car'],
                                     markersize=12, markeredgecolor='white',
                                     markeredgewidth=2, zorder=10)[0]

            # Direction arrow
            if len(x) > 1:
                angle = data.get_column('car_angle')[0]
                arrow_len = (np.max(x) - np.min(x)) * 0.05
                dx = arrow_len * np.cos(angle)
                dy = arrow_len * np.sin(angle)
                self.car_arrow = ax.arrow(x[0], y[0], dx, dy,
                                         head_width=arrow_len*0.5,
                                         head_length=arrow_len*0.5,
                                         fc=self.colors['car'],
                                         ec='white', linewidth=2, zorder=11)

        ax.set_aspect('equal')
        ax.set_title('Track Map', fontsize=11, fontweight='bold', pad=10)
        ax.set_xlabel('X Position (m)', color=self.colors['text'])
        ax.set_ylabel('Y Position (m)', color=self.colors['text'])
        ax.tick_params(colors=self.colors['text'])
        ax.grid(True, alpha=0.2, color=self.colors['grid'])

    def plot_speed(self) -> None:
        """Plot speed over time."""
        ax = self.ax_speed
        ax.clear()
        ax.set_facecolor(self.colors['bg'])

        data = self.current_data
        steps = data.get_column('step')
        speed = data.get_column('speed_kmh')

        ax.plot(steps, speed, color=self.colors['speed'], linewidth=2,
               label='Speed')

        # Current position line
        if len(steps) > 0:
            self.speed_vline = ax.axvline(steps[0], color=self.colors['car'],
                                         linewidth=2, linestyle='--', alpha=0.7)

        ax.set_title('Speed', fontsize=11, fontweight='bold')
        ax.set_ylabel('Speed (km/h)', color=self.colors['text'])
        ax.tick_params(colors=self.colors['text'])
        ax.grid(True, alpha=0.2, color=self.colors['grid'])
        ax.legend(loc='upper right', facecolor=self.colors['bg'],
                 edgecolor=self.colors['grid'])

    def plot_inputs(self) -> None:
        """Plot steering and acceleration inputs."""
        ax = self.ax_inputs
        ax.clear()
        ax.set_facecolor(self.colors['bg'])

        data = self.current_data
        steps = data.get_column('step')
        steering = data.get_column('steering')
        accel = data.get_column('acceleration')

        # Plot steering
        ax.plot(steps, steering, color=self.colors['steering'], linewidth=2,
               label='Steering', alpha=0.8)

        # Plot acceleration with different colors for gas/brake
        gas_mask = accel >= 0
        brake_mask = accel < 0

        ax.fill_between(steps, 0, accel, where=gas_mask,
                       color=self.colors['accel'], alpha=0.5, label='Gas')
        ax.fill_between(steps, 0, accel, where=brake_mask,
                       color=self.colors['brake'], alpha=0.5, label='Brake')

        # Current position line
        if len(steps) > 0:
            self.inputs_vline = ax.axvline(steps[0], color=self.colors['car'],
                                          linewidth=2, linestyle='--', alpha=0.7)

        ax.set_title('Driver Inputs', fontsize=11, fontweight='bold')
        ax.set_ylabel('Input [-1, 1]', color=self.colors['text'])
        ax.set_xlabel('Step', color=self.colors['text'])
        ax.tick_params(colors=self.colors['text'])
        ax.grid(True, alpha=0.2, color=self.colors['grid'])
        ax.legend(loc='upper right', facecolor=self.colors['bg'],
                 edgecolor=self.colors['grid'], ncol=3)
        ax.set_ylim(-1.1, 1.1)

    def plot_wheel_telemetry(self) -> None:
        """Plot wheel slip angles, slip ratios, and forces."""
        data = self.current_data
        steps = data.get_column('step')
        wheels = ['fl', 'fr', 'rl', 'rr']
        wheel_names = ['FL', 'FR', 'RL', 'RR']
        wheel_colors = ['#FF6B6B', '#4ECDC4', '#FFE66D', '#95E1D3']

        # Slip Angles
        ax = self.ax_slip_angle
        ax.clear()
        ax.set_facecolor(self.colors['bg'])
        for wheel, name, color in zip(wheels, wheel_names, wheel_colors):
            sa = data.get_column(f'{wheel}_slip_angle') * 180 / math.pi
            ax.plot(steps, sa, color=color, linewidth=1.5, label=name, alpha=0.8)

        if len(steps) > 0:
            self.sa_vline = ax.axvline(steps[0], color=self.colors['car'],
                                      linewidth=2, linestyle='--', alpha=0.7)

        ax.set_title('Slip Angles', fontsize=11, fontweight='bold')
        ax.set_ylabel('Angle (deg)', color=self.colors['text'])
        ax.set_xlabel('Step', color=self.colors['text'])
        ax.tick_params(colors=self.colors['text'])
        ax.grid(True, alpha=0.2, color=self.colors['grid'])
        ax.legend(loc='upper right', facecolor=self.colors['bg'],
                 edgecolor=self.colors['grid'], ncol=4)

        # Slip Ratios
        ax = self.ax_slip_ratio
        ax.clear()
        ax.set_facecolor(self.colors['bg'])
        for wheel, name, color in zip(wheels, wheel_names, wheel_colors):
            sr = data.get_column(f'{wheel}_slip_ratio')
            ax.plot(steps, sr, color=color, linewidth=1.5, label=name, alpha=0.8)

        if len(steps) > 0:
            self.sr_vline = ax.axvline(steps[0], color=self.colors['car'],
                                      linewidth=2, linestyle='--', alpha=0.7)

        ax.set_title('Slip Ratios', fontsize=11, fontweight='bold')
        ax.set_ylabel('Ratio', color=self.colors['text'])
        ax.set_xlabel('Step', color=self.colors['text'])
        ax.tick_params(colors=self.colors['text'])
        ax.grid(True, alpha=0.2, color=self.colors['grid'])
        ax.legend(loc='upper right', facecolor=self.colors['bg'],
                 edgecolor=self.colors['grid'], ncol=4)

        # Normal Forces
        ax = self.ax_forces
        ax.clear()
        ax.set_facecolor(self.colors['bg'])
        for wheel, name, color in zip(wheels, wheel_names, wheel_colors):
            force = data.get_column(f'{wheel}_normal_force') / 1000.0  # Convert to kN
            ax.plot(steps, force, color=color, linewidth=1.5, label=name, alpha=0.8)

        if len(steps) > 0:
            self.force_vline = ax.axvline(steps[0], color=self.colors['car'],
                                         linewidth=2, linestyle='--', alpha=0.7)

        ax.set_title('Normal Forces', fontsize=11, fontweight='bold')
        ax.set_ylabel('Force (kN)', color=self.colors['text'])
        ax.set_xlabel('Step', color=self.colors['text'])
        ax.tick_params(colors=self.colors['text'])
        ax.grid(True, alpha=0.2, color=self.colors['grid'])
        ax.legend(loc='upper right', facecolor=self.colors['bg'],
                 edgecolor=self.colors['grid'], ncol=4)

    def update_info_panel(self) -> None:
        """Update the information panel with current data."""
        ax = self.ax_info
        ax.clear()
        ax.axis('off')
        ax.set_facecolor(self.colors['bg'])

        row = self.current_data.get_value_at_index(self.current_index)
        if row is None:
            return

        # Create info text
        info_text = (
            f"Step: {int(row['step']):6d}  |  "
            f"Episode: {int(row['episode'])}  |  "
            f"Speed: {row['speed_kmh']:6.1f} km/h  |  "
            f"Steering: {row['steering']:+.3f}  |  "
            f"Accel: {row['acceleration']:+.3f}  |  "
            f"Reward: {row['reward']:.3f}\n\n"
            f"Position: ({row['car_x']:.1f}, {row['car_y']:.1f})  |  "
            f"Angle: {row['car_angle']*180/math.pi:.1f}°  |  "
            f"Vel: ({row['car_vx']:.2f}, {row['car_vy']:.2f}) m/s  |  "
            f"Yaw Rate: {row['car_yaw_rate']*180/math.pi:.1f} °/s"
        )

        ax.text(0.5, 0.5, info_text, transform=ax.transAxes,
               fontsize=10, fontfamily='monospace',
               verticalalignment='center', horizontalalignment='center',
               color=self.colors['text'], bbox=dict(boxstyle='round',
               facecolor=self.colors['bg'], edgecolor=self.colors['grid'],
               linewidth=2, pad=10))

    def update_position(self, index: int) -> None:
        """Update the current position marker across all plots."""
        self.current_index = int(index)
        row = self.current_data.get_value_at_index(self.current_index)

        if row is None:
            return

        # Update track map car position
        x = row['car_x']
        y = row['car_y']
        angle = row['car_angle']

        self.car_marker.set_data([x], [y])

        # Update direction arrow
        if hasattr(self, 'car_arrow') and self.car_arrow:
            self.car_arrow.remove()

        data = self.current_data
        all_x = data.get_column('car_x')
        arrow_len = (np.max(all_x) - np.min(all_x)) * 0.05
        dx = arrow_len * np.cos(angle)
        dy = arrow_len * np.sin(angle)

        self.car_arrow = self.ax_track.arrow(x, y, dx, dy,
                                            head_width=arrow_len*0.5,
                                            head_length=arrow_len*0.5,
                                            fc=self.colors['car'],
                                            ec='white', linewidth=2, zorder=11)

        # Update vertical lines on time series plots
        step = row['step']
        self.speed_vline.set_xdata([step])
        self.inputs_vline.set_xdata([step])
        self.sa_vline.set_xdata([step])
        self.sr_vline.set_xdata([step])
        self.force_vline.set_xdata([step])

        # Update info panel
        self.update_info_panel()

        self.fig.canvas.draw_idle()

    def connect_events(self) -> None:
        """Connect keyboard and mouse events."""
        self.fig.canvas.mpl_connect('key_press_event', self.on_key_press)
        self.fig.canvas.mpl_connect('button_press_event', self.on_click)

    def on_key_press(self, event: Any) -> None:
        """Handle keyboard input."""
        if event.key == 'q':
            plt.close(self.fig)
        elif event.key == 'right':
            self.step_forward()
        elif event.key == 'left':
            self.step_backward()
        elif event.key == ' ':
            self.toggle_playback()
        elif event.key == 'r':
            self.reset_zoom()

    def on_click(self, event: Any) -> None:
        """Handle mouse clicks on plots."""
        # Check if click is on a time series plot
        for ax in self.time_series_axes:
            if event.inaxes == ax and event.xdata is not None:
                # Find closest data point
                steps = self.current_data.get_column('step')
                idx = np.argmin(np.abs(steps - event.xdata))
                self.update_position(idx)
                break

    def step_forward(self) -> None:
        """Move one step forward in time."""
        if self.current_index < len(self.current_data.data) - 1:
            self.update_position(self.current_index + 1)

    def step_backward(self) -> None:
        """Move one step backward in time."""
        if self.current_index > 0:
            self.update_position(self.current_index - 1)

    def toggle_playback(self) -> None:
        """Toggle animation playback."""
        self.playing = not self.playing
        if self.playing:
            self.animate()
        else:
            if self.animation_timer:
                self.animation_timer.stop()

    def animate(self) -> None:
        """Animate through the data."""
        if self.playing and self.current_index < len(self.current_data.data) - 1:
            self.step_forward()
            self.animation_timer = self.fig.canvas.new_timer(interval=50)
            self.animation_timer.add_callback(self.animate)
            self.animation_timer.start()
        else:
            self.playing = False

    def reset_zoom(self) -> None:
        """Reset zoom on all plots."""
        for ax in [self.ax_track, self.ax_speed, self.ax_inputs,
                   self.ax_slip_angle, self.ax_slip_ratio, self.ax_forces]:
            ax.relim()
            ax.autoscale()
        self.fig.canvas.draw_idle()

    def show(self) -> None:
        """Display the viewer."""
        print("\n" + "="*60)
        print("CONTROLS:")
        print("  Left/Right Arrow : Navigate through time")
        print("  Space            : Play/Pause animation")
        print("  Click on plot    : Jump to that time")
        print("  Mouse wheel      : Zoom")
        print("  R                : Reset zoom")
        print("  Q                : Quit")
        print("="*60 + "\n")

        plt.show()


def main() -> None:
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description='Professional Interactive Telemetry Viewer',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python telemetry_viewer.py telemetry_20250113_123456.csv
  python telemetry_viewer.py session.csv

Controls:
  Left/Right arrows : Navigate through time
  Space            : Play/Pause animation
  Click on plot    : Jump to that time
  R                : Reset zoom
  Q                : Quit
        """)

    parser.add_argument('files', nargs='+', metavar='FILE',
                       help='Telemetry CSV file(s) to visualize')

    args: argparse.Namespace = parser.parse_args()

    # Create and show viewer
    viewer = TelemetryViewer(args.files)
    viewer.show()


if __name__ == '__main__':
    main()
