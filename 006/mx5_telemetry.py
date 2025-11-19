#!/usr/bin/env python3
"""
Professional MX-5 Powertrain Telemetry Display
===============================================

A real-time telemetry visualization system inspired by professional motorsport
data acquisition systems (Motec i2, Cosworth Toolbox, AIM Race Studio).

Features:
- Real-time RPM tachometer with redline warning
- Digital speed display
- Large gear indicator
- Throttle and brake position bars
- Engine temperature and oil pressure gauges
- Live time-series charts for RPM, speed, throttle/brake
- Professional dark theme with motorsport aesthetics
- Interactive controls for engine and gearbox (AZERTY keyboard)

Interactive Controls (AZERTY keyboard):
    Z          - Accelerate (increase throttle)
    S          - Brake (apply brakes)
    A          - Shift down
    E          - Shift up
    1-6        - Select gear directly (1st through 6th)
    N          - Neutral
    Space      - Release throttle and brake
    ESC        - Quit

Usage:
    # Interactive mode with manual controls
    python mx5_telemetry.py

    # Integration with powertrain simulation
    from mx5_telemetry import MX5TelemetryDisplay
    from mx5_powertrain import MX5Powertrain

    powertrain = MX5Powertrain()
    telemetry = MX5TelemetryDisplay()

    # Update loop
    while running:
        state = powertrain.get_state()
        telemetry.update(state)

Requirements:
    pip install matplotlib numpy
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
from matplotlib.patches import Wedge, Rectangle, Circle, FancyBboxPatch
import matplotlib.animation as animation
from collections import deque
import math


class Gauge:
    """Base class for circular gauge displays."""

    def __init__(self, ax, min_val, max_val, units, redline=None):
        """
        Initialize gauge.

        Args:
            ax: Matplotlib axis
            min_val: Minimum value
            max_val: Maximum value
            units: Unit string (e.g., "RPM", "km/h")
            redline: Redline value (optional)
        """
        self.ax = ax
        self.min_val = min_val
        self.max_val = max_val
        self.units = units
        self.redline = redline
        self.current_value = min_val

        self.ax.set_xlim(-1.2, 1.2)
        self.ax.set_ylim(-1.2, 1.2)
        self.ax.set_aspect('equal')
        self.ax.axis('off')

        self._draw_gauge()

    def _draw_gauge(self):
        """Draw the static gauge elements."""
        # Background circle
        bg_circle = Circle((0, 0), 1.0, color='#1a1a1a', zorder=1)
        self.ax.add_patch(bg_circle)

        # Outer ring
        outer_ring = Circle((0, 0), 1.0, fill=False, edgecolor='#333333',
                           linewidth=3, zorder=2)
        self.ax.add_patch(outer_ring)

        # Draw tick marks and labels
        self._draw_ticks()

        # Needle (will be updated)
        self.needle = self.ax.plot([0, 0], [0, 0.8], color='#00D9FF',
                                   linewidth=4, zorder=10)[0]
        self.needle_tip = Circle((0, 0.8), 0.05, color='#00D9FF', zorder=11)
        self.ax.add_patch(self.needle_tip)

        # Center cap
        center = Circle((0, 0), 0.1, color='#333333', zorder=12)
        self.ax.add_patch(center)

        # Digital readout
        self.text = self.ax.text(0, -0.4, '0', ha='center', va='center',
                                fontsize=28, fontweight='bold', color='white',
                                family='monospace', zorder=15)

        # Units label
        self.ax.text(0, -0.6, self.units, ha='center', va='center',
                    fontsize=12, color='#888888', family='sans-serif')

    def _draw_ticks(self):
        """Draw tick marks and value labels around the gauge."""
        # Gauge arc: 225° to -45° (270° total, like a car tachometer)
        start_angle = 225  # degrees
        end_angle = -45
        total_angle = start_angle - end_angle  # 270°

        # Determine tick positions
        value_range = self.max_val - self.min_val
        num_major_ticks = 8
        major_tick_values = np.linspace(self.min_val, self.max_val, num_major_ticks)

        for i, value in enumerate(major_tick_values):
            # Calculate angle for this tick
            fraction = (value - self.min_val) / value_range
            angle_deg = start_angle - (fraction * total_angle)
            angle_rad = math.radians(angle_deg)

            # Tick mark
            r_inner = 0.85
            r_outer = 0.95
            x_inner = r_inner * math.cos(angle_rad)
            y_inner = r_inner * math.sin(angle_rad)
            x_outer = r_outer * math.cos(angle_rad)
            y_outer = r_outer * math.sin(angle_rad)

            # Color tick red if in redline zone
            tick_color = '#FF4444' if self.redline and value >= self.redline else '#666666'

            self.ax.plot([x_inner, x_outer], [y_inner, y_outer],
                        color=tick_color, linewidth=2, zorder=3)

            # Value label
            r_label = 0.72
            x_label = r_label * math.cos(angle_rad)
            y_label = r_label * math.sin(angle_rad)

            # Format value (remove decimals for whole numbers)
            if value >= 1000:
                label_text = f'{int(value/1000)}k'
            else:
                label_text = f'{int(value)}'

            self.ax.text(x_label, y_label, label_text, ha='center', va='center',
                        fontsize=9, color=tick_color, fontweight='bold')

        # Redline arc
        if self.redline:
            redline_fraction = (self.redline - self.min_val) / value_range
            redline_angle = start_angle - (redline_fraction * total_angle)

            theta1 = redline_angle
            theta2 = end_angle

            wedge = Wedge((0, 0), 0.95, theta2, theta1, width=0.05,
                         facecolor='#FF4444', edgecolor='none', alpha=0.3, zorder=2)
            self.ax.add_patch(wedge)

    def update(self, value):
        """
        Update gauge to show new value.

        Args:
            value: New value to display
        """
        self.current_value = np.clip(value, self.min_val, self.max_val)

        # Calculate needle angle
        value_range = self.max_val - self.min_val
        fraction = (self.current_value - self.min_val) / value_range

        start_angle = 225
        end_angle = -45
        total_angle = start_angle - end_angle

        angle_deg = start_angle - (fraction * total_angle)
        angle_rad = math.radians(angle_deg)

        # Update needle
        needle_length = 0.8
        x = needle_length * math.cos(angle_rad)
        y = needle_length * math.sin(angle_rad)

        self.needle.set_data([0, x], [0, y])
        self.needle_tip.center = (x, y)

        # Flash red if in redline
        if self.redline and self.current_value >= self.redline:
            self.needle.set_color('#FF4444')
            self.needle_tip.set_color('#FF4444')
        else:
            self.needle.set_color('#00D9FF')
            self.needle_tip.set_color('#00D9FF')

        # Update digital readout
        self.text.set_text(f'{int(self.current_value)}')


class BarGauge:
    """Horizontal or vertical bar gauge."""

    def __init__(self, ax, label, color, orientation='horizontal'):
        """
        Initialize bar gauge.

        Args:
            ax: Matplotlib axis
            label: Label text
            color: Bar color
            orientation: 'horizontal' or 'vertical'
        """
        self.ax = ax
        self.label = label
        self.color = color
        self.orientation = orientation
        self.current_value = 0.0

        self.ax.set_xlim(0, 1)
        self.ax.set_ylim(0, 1)
        self.ax.axis('off')

        self._draw_bar()

    def _draw_bar(self):
        """Draw the static bar elements."""
        # Background
        bg = Rectangle((0.1, 0.3), 0.8, 0.4, facecolor='#1a1a1a',
                      edgecolor='#333333', linewidth=2)
        self.ax.add_patch(bg)

        # Bar (will be updated)
        self.bar = Rectangle((0.1, 0.3), 0.0, 0.4, facecolor=self.color)
        self.ax.add_patch(self.bar)

        # Label
        self.ax.text(0.5, 0.1, self.label, ha='center', va='center',
                    fontsize=12, color='white', fontweight='bold')

    def update(self, value):
        """
        Update bar to show new value [0.0 - 1.0].

        Args:
            value: New value (0.0 to 1.0)
        """
        self.current_value = np.clip(value, 0.0, 1.0)
        self.bar.set_width(0.8 * self.current_value)


class MX5TelemetryDisplay:
    """
    Professional real-time telemetry display for MX-5 powertrain.

    Displays engine RPM, gear, speed, throttle/brake positions, and vital
    parameters in a professional motorsport-style interface.
    """

    def __init__(self, history_length=500):
        """
        Initialize telemetry display.

        Args:
            history_length: Number of data points to keep in history
        """
        self.history_length = history_length

        # Data buffers
        self.time_buffer = deque(maxlen=history_length)
        self.rpm_buffer = deque(maxlen=history_length)
        self.speed_buffer = deque(maxlen=history_length)
        self.throttle_buffer = deque(maxlen=history_length)
        self.brake_buffer = deque(maxlen=history_length)
        self.gear_buffer = deque(maxlen=history_length)

        self.current_time = 0.0

        # Current values
        self.current_rpm = 0
        self.current_speed = 0
        self.current_gear = 1
        self.current_throttle = 0.0
        self.current_brake = 0.0
        self.current_temp = 90.0
        self.current_oil_pressure = 0.0

        self._setup_display()

    def _setup_display(self):
        """Setup the display layout."""
        # Create figure with dark theme
        plt.style.use('dark_background')
        self.fig = plt.figure(figsize=(16, 10))
        self.fig.patch.set_facecolor('#0a0a0a')
        self.fig.canvas.manager.set_window_title('MX-5 Powertrain Telemetry')

        # Create grid layout
        gs = GridSpec(3, 4, figure=self.fig, hspace=0.4, wspace=0.3,
                     left=0.05, right=0.97, top=0.93, bottom=0.07)

        # Title
        self.fig.suptitle('MAZDA MX-5 POWERTRAIN TELEMETRY', fontsize=18,
                         fontweight='bold', color='#00D9FF', y=0.97)

        # Top row: Gauges
        self.ax_rpm = self.fig.add_subplot(gs[0, 0])
        self.ax_speed = self.fig.add_subplot(gs[0, 1])
        self.ax_gear = self.fig.add_subplot(gs[0, 2])
        self.ax_temp = self.fig.add_subplot(gs[0, 3])

        # Middle row: Input bars and vitals
        self.ax_throttle = self.fig.add_subplot(gs[1, 0])
        self.ax_brake = self.fig.add_subplot(gs[1, 1])
        self.ax_oil = self.fig.add_subplot(gs[1, 2])
        self.ax_info = self.fig.add_subplot(gs[1, 3])

        # Bottom row: Time series charts
        self.ax_rpm_chart = self.fig.add_subplot(gs[2, 0:2])
        self.ax_speed_chart = self.fig.add_subplot(gs[2, 2:4])

        # Create gauges
        self.rpm_gauge = Gauge(self.ax_rpm, 0, 8000, 'RPM', redline=7500)
        self.speed_gauge = Gauge(self.ax_speed, 0, 260, 'km/h')

        # Temperature gauge (60-120°C)
        self.temp_gauge = Gauge(self.ax_temp, 60, 120, '°C', redline=105)

        # Gear indicator
        self._setup_gear_indicator()

        # Input bars
        self.throttle_bar = BarGauge(self.ax_throttle, 'THROTTLE', '#00FF00')
        self.brake_bar = BarGauge(self.ax_brake, 'BRAKE', '#FF4444')
        self.oil_bar = BarGauge(self.ax_oil, 'OIL (bar)', '#FFD700')

        # Info panel
        self._setup_info_panel()

        # Time series charts
        self._setup_charts()

    def _setup_gear_indicator(self):
        """Setup large gear indicator display."""
        ax = self.ax_gear
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis('off')

        # Background
        bg = FancyBboxPatch((0.1, 0.2), 0.8, 0.6, boxstyle="round,pad=0.05",
                           facecolor='#1a1a1a', edgecolor='#333333', linewidth=3)
        ax.add_patch(bg)

        # Gear text
        self.gear_text = ax.text(0.5, 0.55, '1', ha='center', va='center',
                                fontsize=80, fontweight='bold', color='#00D9FF',
                                family='sans-serif')

        # Label
        ax.text(0.5, 0.15, 'GEAR', ha='center', va='center',
               fontsize=14, color='#888888', fontweight='bold')

    def _setup_info_panel(self):
        """Setup information panel."""
        ax = self.ax_info
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis('off')

        # Create text fields
        self.info_text = ax.text(0.5, 0.5, '', ha='center', va='center',
                                fontsize=11, color='white', family='monospace',
                                linespacing=1.8)

    def _setup_charts(self):
        """Setup time series charts."""
        # RPM chart
        ax = self.ax_rpm_chart
        ax.set_facecolor('#1a1a1a')
        ax.set_title('ENGINE RPM', fontsize=12, fontweight='bold', color='#00D9FF', pad=10)
        ax.set_ylabel('RPM', color='white')
        ax.set_xlabel('Time (s)', color='white')
        ax.tick_params(colors='white')
        ax.grid(True, alpha=0.2, color='#333333')
        ax.set_ylim(0, 8000)

        self.rpm_line, = ax.plot([], [], color='#00D9FF', linewidth=2)
        ax.axhline(7500, color='#FF4444', linestyle='--', alpha=0.5, linewidth=1.5)

        # Speed chart
        ax = self.ax_speed_chart
        ax.set_facecolor('#1a1a1a')
        ax.set_title('VEHICLE SPEED', fontsize=12, fontweight='bold', color='#00FF88', pad=10)
        ax.set_ylabel('km/h', color='white')
        ax.set_xlabel('Time (s)', color='white')
        ax.tick_params(colors='white')
        ax.grid(True, alpha=0.2, color='#333333')
        ax.set_ylim(0, 260)

        self.speed_line, = ax.plot([], [], color='#00FF88', linewidth=2)

        # Throttle/brake overlay on speed chart
        ax2 = ax.twinx()
        ax2.set_ylabel('Input', color='white')
        ax2.set_ylim(-1.1, 1.1)
        ax2.tick_params(colors='white')

        self.throttle_line, = ax2.plot([], [], color='#00FF00', linewidth=1.5,
                                       alpha=0.6, label='Throttle')
        self.brake_line, = ax2.plot([], [], color='#FF4444', linewidth=1.5,
                                    alpha=0.6, label='Brake')
        ax2.legend(loc='upper right', facecolor='#1a1a1a', edgecolor='#333333')

    def update(self, state):
        """
        Update telemetry display with new data.

        Args:
            state (dict): Powertrain state dictionary with keys:
                - engine_rpm
                - current_gear
                - speed_kmh (optional)
                - throttle (optional, 0-1)
                - brake (optional, 0-1)
                - engine_temp_c
                - oil_pressure_bar
        """
        # Extract values
        self.current_rpm = state.get('engine_rpm', 0)
        self.current_gear = state.get('current_gear', 1)
        self.current_speed = state.get('speed_kmh', 0)
        self.current_throttle = state.get('throttle', 0.0)
        self.current_brake = state.get('brake', 0.0)
        self.current_temp = state.get('engine_temp_c', 90.0)
        self.current_oil_pressure = state.get('oil_pressure_bar', 0.0)

        # Update buffers
        self.time_buffer.append(self.current_time)
        self.rpm_buffer.append(self.current_rpm)
        self.speed_buffer.append(self.current_speed)
        self.throttle_buffer.append(self.current_throttle)
        self.brake_buffer.append(self.current_brake)
        self.gear_buffer.append(self.current_gear)

        self.current_time += 0.02  # Assume 50Hz update rate

        # Update gauges
        self.rpm_gauge.update(self.current_rpm)
        self.speed_gauge.update(self.current_speed)
        self.temp_gauge.update(self.current_temp)

        # Update gear indicator
        gear_text = 'N' if self.current_gear == 0 else str(self.current_gear)
        self.gear_text.set_text(gear_text)

        # Flash gear indicator if shifting
        if state.get('is_shifting', False):
            self.gear_text.set_color('#FFD700')
        else:
            self.gear_text.set_color('#00D9FF')

        # Update bars
        self.throttle_bar.update(self.current_throttle)
        self.brake_bar.update(self.current_brake)
        self.oil_bar.update(self.current_oil_pressure / 6.0)  # Normalize to 0-6 bar

        # Update info panel
        info_str = (
            f"Power:  {state.get('engine_power_hp', 0):.0f} hp\n"
            f"Torque: {state.get('engine_torque_nm', 0):.0f} Nm\n"
            f"Ratio:  {state.get('gear_ratio', 0):.3f}\n"
            f"Fuel:   {'CUT' if state.get('fuel_cut_active', False) else 'OK'}"
        )
        self.info_text.set_text(info_str)

        # Update charts
        if len(self.time_buffer) > 1:
            times = np.array(self.time_buffer)
            rpms = np.array(self.rpm_buffer)
            speeds = np.array(self.speed_buffer)
            throttles = np.array(self.throttle_buffer)
            brakes = np.array(self.brake_buffer)

            self.rpm_line.set_data(times, rpms)
            self.speed_line.set_data(times, speeds)
            self.throttle_line.set_data(times, throttles)
            self.brake_line.set_data(times, -brakes)  # Negative for visual separation

            # Auto-scale x-axis
            time_window = 10.0  # Show last 10 seconds
            if times[-1] > time_window:
                self.ax_rpm_chart.set_xlim(times[-1] - time_window, times[-1])
                self.ax_speed_chart.set_xlim(times[-1] - time_window, times[-1])
            else:
                self.ax_rpm_chart.set_xlim(0, time_window)
                self.ax_speed_chart.set_xlim(0, time_window)

        self.fig.canvas.draw_idle()
        self.fig.canvas.flush_events()

    def show(self, block=False):
        """
        Display the telemetry window.

        Args:
            block: If True, blocks until window is closed
        """
        plt.show(block=block)

    def close(self):
        """Close the telemetry display."""
        plt.close(self.fig)


class InteractiveDriveSimulator:
    """
    Interactive driving simulator with manual controls.

    Allows user to control throttle, brake, and gearbox using keyboard.
    """

    def __init__(self):
        """Initialize interactive simulator."""
        # Import powertrain
        try:
            from mx5_powertrain import MX5Powertrain
        except ImportError:
            print("ERROR: Cannot import mx5_powertrain.py")
            print("Make sure mx5_powertrain.py is in the same directory.")
            raise

        # Create telemetry and powertrain
        self.telemetry = MX5TelemetryDisplay()
        self.powertrain = MX5Powertrain()

        # Control inputs
        self.throttle = 0.0
        self.brake = 0.0
        self.throttle_increment = 0.05
        self.brake_increment = 0.05

        # Vehicle state
        self.speed_kmh = 0.0
        self.wheel_rpm = 0.0

        # Physics constants
        self.wheel_radius = 0.305  # meters
        self.vehicle_mass = 1062  # kg
        self.drag_coefficient = 0.5  # Simplified drag

        # Simulation
        self.dt = 0.02  # 50Hz update rate
        self.time = 0.0
        self.running = False

        # Key press state tracking
        self.keys_pressed = set()

    def on_key_press(self, event):
        """Handle key press events (AZERTY layout)."""
        key = event.key.lower()

        # ESC to quit
        if key == 'escape':
            print("\n[ESC] Exiting...")
            self.running = False
            plt.close(self.telemetry.fig)
            return

        # Add to pressed keys
        self.keys_pressed.add(key)

        # Gear shifting
        if key == 'a':  # Shift down (AZERTY: A key)
            self.powertrain.shift_down()
            print(f"[A] Shifted DOWN to gear {self.powertrain.gearbox.current_gear}")

        elif key == 'e':  # Shift up (AZERTY: E key)
            self.powertrain.shift_up()
            print(f"[E] Shifted UP to gear {self.powertrain.gearbox.current_gear}")

        # Direct gear selection
        elif key in ['1', '2', '3', '4', '5', '6']:
            gear = int(key)
            self.powertrain.shift_to(gear)
            print(f"[{key}] Selected gear {gear}")

        elif key == 'n':  # Neutral
            self.powertrain.shift_to(0)
            print(f"[N] Neutral")

        # Space to release all inputs
        elif key == ' ':
            self.throttle = 0.0
            self.brake = 0.0
            print("[SPACE] Released throttle and brake")

    def on_key_release(self, event):
        """Handle key release events."""
        key = event.key.lower()
        if key in self.keys_pressed:
            self.keys_pressed.remove(key)

    def update_inputs(self):
        """Update throttle and brake based on held keys."""
        # Z key - Accelerate (AZERTY layout)
        if 'z' in self.keys_pressed:
            self.throttle = min(1.0, self.throttle + self.throttle_increment)
            self.brake = 0.0  # Release brake when accelerating

        # S key - Brake (AZERTY layout)
        elif 's' in self.keys_pressed:
            self.brake = min(1.0, self.brake + self.brake_increment)
            self.throttle = 0.0  # Release throttle when braking

        # Natural throttle/brake decay when keys released
        else:
            # Smooth release
            self.throttle *= 0.95
            self.brake *= 0.95

            # Snap to zero when very small
            if self.throttle < 0.01:
                self.throttle = 0.0
            if self.brake < 0.01:
                self.brake = 0.0

    def update_physics(self):
        """Update vehicle physics."""
        # Get wheel torque from powertrain (includes engine braking when throttle released)
        wheel_torque = self.powertrain.get_wheel_torque(self.throttle, self.wheel_rpm)
        wheel_force = wheel_torque / self.wheel_radius

        # Update powertrain
        self.powertrain.update(self.dt, self.wheel_rpm)

        # Braking force (only when brake pedal is pressed)
        brake_force = -self.brake * 8000.0  # Newtons

        # Drag force (F = 0.5 * rho * Cd * A * v^2)
        speed_ms = self.speed_kmh / 3.6
        drag_force = -self.drag_coefficient * speed_ms * abs(speed_ms)

        # Total force
        total_force = wheel_force + brake_force + drag_force

        # Acceleration (F = ma)
        acceleration = total_force / self.vehicle_mass

        # Update speed
        speed_ms += acceleration * self.dt
        speed_ms = max(0.0, speed_ms)  # Can't go backwards
        self.speed_kmh = speed_ms * 3.6

        # Calculate wheel RPM
        wheel_angular_velocity = speed_ms / self.wheel_radius  # rad/s
        self.wheel_rpm = wheel_angular_velocity * 60 / (2 * math.pi)

    def run(self):
        """Run the interactive simulator."""
        print("="*70)
        print("MX-5 Powertrain Telemetry - INTERACTIVE MODE")
        print("="*70)
        print("\nControls (AZERTY keyboard):")
        print("  Z          - Accelerate (hold to increase throttle)")
        print("  S          - Brake (hold to apply brakes)")
        print("  A          - Shift DOWN")
        print("  E          - Shift UP")
        print("  1-6        - Select gear directly")
        print("  N          - Neutral")
        print("  Space      - Release throttle and brake")
        print("  ESC        - Quit")
        print("\nStarting in gear 1. Press Z to accelerate!")
        print("="*70 + "\n")

        # Connect keyboard events
        self.telemetry.fig.canvas.mpl_connect('key_press_event', self.on_key_press)
        self.telemetry.fig.canvas.mpl_connect('key_release_event', self.on_key_release)

        # Show display (non-blocking)
        plt.ion()
        self.telemetry.show(block=False)

        self.running = True

        try:
            while self.running and plt.fignum_exists(self.telemetry.fig.number):
                # Update inputs based on held keys
                self.update_inputs()

                # Update physics
                self.update_physics()

                # Get powertrain state
                state = self.powertrain.get_state()
                state['speed_kmh'] = self.speed_kmh
                state['throttle'] = self.throttle
                state['brake'] = self.brake

                # Update telemetry display
                self.telemetry.update(state)

                # Increment time
                self.time += self.dt

                # Control update rate
                plt.pause(self.dt)

        except KeyboardInterrupt:
            print("\nInterrupted by user.")

        print("\nSession complete!")
        print(f"Total time: {self.time:.1f}s")
        print(f"Max speed: {max(self.telemetry.speed_buffer) if self.telemetry.speed_buffer else 0:.1f} km/h")

        plt.ioff()
        plt.show()  # Block to keep window open


def demo():
    """
    Interactive demonstration mode with manual controls.

    Allows user to control the MX-5 using keyboard inputs.
    """
    try:
        simulator = InteractiveDriveSimulator()
        simulator.run()
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    demo()
