#!/usr/bin/env python3
"""
MX-5 Engine & Gearbox Telemetry (Engine-Only Mode)
==================================================

Simplified telemetry focused purely on engine and gearbox control.
No vehicle physics - just direct engine throttle and RPM control.

Perfect for:
- Engine testing and tuning
- Understanding gear ratios
- Learning the powerband
- Practicing manual transmission shifts

Interactive Controls (AZERTY keyboard):
    Z          - Increase throttle
    S          - Decrease throttle
    A          - Shift down
    E          - Shift up
    1-6        - Select gear directly (1st through 6th)
    N          - Neutral
    Space      - Cut throttle to idle
    ESC        - Quit

Usage:
    python mx5_engine_telemetry.py

Requirements:
    pip install matplotlib numpy
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
from matplotlib.patches import Wedge, Rectangle, Circle, FancyBboxPatch
from collections import deque
import math


class EngineOnlyTelemetry:
    """
    Engine and gearbox telemetry display without vehicle physics.

    Controls engine throttle directly and shows RPM, gear, and power output.
    """

    def __init__(self, history_length=500):
        """
        Initialize engine-only telemetry.

        Args:
            history_length: Number of data points to keep in history
        """
        # Import powertrain
        try:
            from mx5_powertrain import MX5Powertrain
        except ImportError:
            print("ERROR: Cannot import mx5_powertrain.py")
            print("Make sure mx5_powertrain.py is in the same directory.")
            raise

        self.powertrain = MX5Powertrain()
        self.history_length = history_length

        # Data buffers
        self.time_buffer = deque(maxlen=history_length)
        self.rpm_buffer = deque(maxlen=history_length)
        self.throttle_buffer = deque(maxlen=history_length)
        self.torque_buffer = deque(maxlen=history_length)
        self.power_buffer = deque(maxlen=history_length)
        self.gear_buffer = deque(maxlen=history_length)

        self.current_time = 0.0

        # Control inputs
        self.throttle = 0.0
        self.throttle_increment = 0.02

        # Simulation
        self.dt = 0.02  # 50Hz
        self.running = False

        # Key press state
        self.keys_pressed = set()

        # Create display
        self._setup_display()

    def _setup_display(self):
        """Setup the telemetry display."""
        plt.style.use('dark_background')
        self.fig = plt.figure(figsize=(16, 10))
        self.fig.patch.set_facecolor('#0a0a0a')
        self.fig.canvas.manager.set_window_title('MX-5 Engine & Gearbox Telemetry')

        # Create grid layout
        gs = GridSpec(3, 4, figure=self.fig, hspace=0.4, wspace=0.3,
                     left=0.05, right=0.97, top=0.93, bottom=0.07)

        # Title
        self.fig.suptitle('MAZDA MX-5 ENGINE & GEARBOX TELEMETRY (Engine-Only Mode)',
                         fontsize=16, fontweight='bold', color='#00D9FF', y=0.97)

        # Top row: Gauges
        self.ax_rpm = self.fig.add_subplot(gs[0, 0])
        self.ax_gear = self.fig.add_subplot(gs[0, 1])
        self.ax_power = self.fig.add_subplot(gs[0, 2])
        self.ax_torque = self.fig.add_subplot(gs[0, 3])

        # Middle row: Throttle bar and info
        self.ax_throttle = self.fig.add_subplot(gs[1, 0:2])
        self.ax_info = self.fig.add_subplot(gs[1, 2:4])

        # Bottom row: Time series charts
        self.ax_rpm_chart = self.fig.add_subplot(gs[2, 0:2])
        self.ax_power_chart = self.fig.add_subplot(gs[2, 2:4])

        # Create gauges
        self._create_rpm_gauge()
        self._create_gear_display()
        self._create_power_gauge()
        self._create_torque_gauge()
        self._create_throttle_bar()
        self._create_info_panel()
        self._create_charts()

    def _create_rpm_gauge(self):
        """Create RPM gauge."""
        ax = self.ax_rpm
        ax.set_xlim(-1.2, 1.2)
        ax.set_ylim(-1.2, 1.2)
        ax.set_aspect('equal')
        ax.axis('off')

        # Background
        bg = Circle((0, 0), 1.0, color='#1a1a1a', zorder=1)
        ax.add_patch(bg)
        outer = Circle((0, 0), 1.0, fill=False, edgecolor='#333333', linewidth=3, zorder=2)
        ax.add_patch(outer)

        # Redline arc (7500 RPM)
        wedge = Wedge((0, 0), 0.95, -45, 0, width=0.05,
                     facecolor='#FF4444', edgecolor='none', alpha=0.3, zorder=2)
        ax.add_patch(wedge)

        # Needle
        self.rpm_needle = ax.plot([0, 0], [0, 0.8], color='#00D9FF',
                                  linewidth=4, zorder=10)[0]
        self.rpm_needle_tip = Circle((0, 0.8), 0.05, color='#00D9FF', zorder=11)
        ax.add_patch(self.rpm_needle_tip)

        # Center cap
        center = Circle((0, 0), 0.1, color='#333333', zorder=12)
        ax.add_patch(center)

        # Digital readout
        self.rpm_text = ax.text(0, -0.4, '800', ha='center', va='center',
                               fontsize=28, fontweight='bold', color='white',
                               family='monospace', zorder=15)

        # Label
        ax.text(0, -0.6, 'RPM', ha='center', va='center',
               fontsize=12, color='#888888')

        ax.set_title('ENGINE RPM', fontsize=11, fontweight='bold', color='white', pad=10)

    def _create_gear_display(self):
        """Create large gear display."""
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
                                fontsize=80, fontweight='bold', color='#00D9FF')

        # Label
        ax.text(0.5, 0.15, 'GEAR', ha='center', va='center',
               fontsize=14, color='#888888', fontweight='bold')

    def _create_power_gauge(self):
        """Create power gauge."""
        ax = self.ax_power
        ax.set_xlim(-1.2, 1.2)
        ax.set_ylim(-1.2, 1.2)
        ax.set_aspect('equal')
        ax.axis('off')

        # Background
        bg = Circle((0, 0), 1.0, color='#1a1a1a', zorder=1)
        ax.add_patch(bg)
        outer = Circle((0, 0), 1.0, fill=False, edgecolor='#333333', linewidth=3, zorder=2)
        ax.add_patch(outer)

        # Needle
        self.power_needle = ax.plot([0, 0], [0, 0.8], color='#00FF88',
                                    linewidth=4, zorder=10)[0]
        self.power_needle_tip = Circle((0, 0.8), 0.05, color='#00FF88', zorder=11)
        ax.add_patch(self.power_needle_tip)

        # Center
        center = Circle((0, 0), 0.1, color='#333333', zorder=12)
        ax.add_patch(center)

        # Digital readout
        self.power_text = ax.text(0, -0.4, '0', ha='center', va='center',
                                 fontsize=28, fontweight='bold', color='white',
                                 family='monospace', zorder=15)

        # Label
        ax.text(0, -0.6, 'hp', ha='center', va='center',
               fontsize=12, color='#888888')

        ax.set_title('POWER', fontsize=11, fontweight='bold', color='white', pad=10)

    def _create_torque_gauge(self):
        """Create torque gauge."""
        ax = self.ax_torque
        ax.set_xlim(-1.2, 1.2)
        ax.set_ylim(-1.2, 1.2)
        ax.set_aspect('equal')
        ax.axis('off')

        # Background
        bg = Circle((0, 0), 1.0, color='#1a1a1a', zorder=1)
        ax.add_patch(bg)
        outer = Circle((0, 0), 1.0, fill=False, edgecolor='#333333', linewidth=3, zorder=2)
        ax.add_patch(outer)

        # Needle
        self.torque_needle = ax.plot([0, 0], [0, 0.8], color='#FFD700',
                                     linewidth=4, zorder=10)[0]
        self.torque_needle_tip = Circle((0, 0.8), 0.05, color='#FFD700', zorder=11)
        ax.add_patch(self.torque_needle_tip)

        # Center
        center = Circle((0, 0), 0.1, color='#333333', zorder=12)
        ax.add_patch(center)

        # Digital readout
        self.torque_text = ax.text(0, -0.4, '0', ha='center', va='center',
                                  fontsize=28, fontweight='bold', color='white',
                                  family='monospace', zorder=15)

        # Label
        ax.text(0, -0.6, 'Nm', ha='center', va='center',
               fontsize=12, color='#888888')

        ax.set_title('TORQUE', fontsize=11, fontweight='bold', color='white', pad=10)

    def _create_throttle_bar(self):
        """Create throttle position bar."""
        ax = self.ax_throttle
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis('off')

        # Background
        bg = Rectangle((0.05, 0.3), 0.9, 0.4, facecolor='#1a1a1a',
                      edgecolor='#333333', linewidth=3)
        ax.add_patch(bg)

        # Throttle bar
        self.throttle_bar = Rectangle((0.05, 0.3), 0.0, 0.4, facecolor='#00FF00')
        ax.add_patch(self.throttle_bar)

        # Label
        ax.text(0.5, 0.1, 'THROTTLE POSITION', ha='center', va='center',
               fontsize=14, color='white', fontweight='bold')

        # Percentage text
        self.throttle_pct_text = ax.text(0.5, 0.5, '0%', ha='center', va='center',
                                         fontsize=24, color='white', fontweight='bold')

    def _create_info_panel(self):
        """Create info panel."""
        ax = self.ax_info
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis('off')

        # Background
        bg = FancyBboxPatch((0.05, 0.05), 0.9, 0.9, boxstyle="round,pad=0.05",
                           facecolor='#1a1a1a', edgecolor='#00D9FF', linewidth=2)
        ax.add_patch(bg)

        # Info text
        self.info_text = ax.text(0.5, 0.5, '', ha='center', va='center',
                                fontsize=11, color='white', family='monospace',
                                linespacing=1.8, fontweight='bold')

    def _create_charts(self):
        """Create time series charts."""
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
        ax.axhline(7500, color='#FF4444', linestyle='--', alpha=0.5, linewidth=1.5, label='Redline')

        # Throttle overlay
        ax2 = ax.twinx()
        ax2.set_ylabel('Throttle %', color='#00FF00')
        ax2.set_ylim(0, 100)
        ax2.tick_params(colors='#00FF00')
        self.throttle_line, = ax2.plot([], [], color='#00FF00', linewidth=1.5, alpha=0.6)

        # Power chart
        ax = self.ax_power_chart
        ax.set_facecolor('#1a1a1a')
        ax.set_title('POWER & TORQUE', fontsize=12, fontweight='bold', color='#00FF88', pad=10)
        ax.set_ylabel('Power (hp)', color='white')
        ax.set_xlabel('Time (s)', color='white')
        ax.tick_params(colors='white')
        ax.grid(True, alpha=0.2, color='#333333')
        ax.set_ylim(0, 200)

        self.power_line, = ax.plot([], [], color='#00FF88', linewidth=2, label='Power')

        # Torque overlay
        ax3 = ax.twinx()
        ax3.set_ylabel('Torque (Nm)', color='#FFD700')
        ax3.set_ylim(0, 250)
        ax3.tick_params(colors='#FFD700')
        self.torque_line, = ax3.plot([], [], color='#FFD700', linewidth=2, alpha=0.8, label='Torque')

    def update_gauge_needle(self, needle, needle_tip, value, min_val, max_val):
        """Update a gauge needle position."""
        value = np.clip(value, min_val, max_val)
        fraction = (value - min_val) / (max_val - min_val)

        angle_deg = 225 - (fraction * 270)  # 225° to -45°
        angle_rad = math.radians(angle_deg)

        x = 0.8 * math.cos(angle_rad)
        y = 0.8 * math.sin(angle_rad)

        needle.set_data([0, x], [0, y])
        needle_tip.center = (x, y)

    def update_display(self):
        """Update all display elements."""
        state = self.powertrain.get_state(throttle=self.throttle)

        rpm = state['engine_rpm']
        gear = state['current_gear']
        power_hp = state['engine_power_hp']
        torque_nm = state['engine_torque_nm']

        # Update gauges
        self.update_gauge_needle(self.rpm_needle, self.rpm_needle_tip, rpm, 0, 8000)
        self.rpm_text.set_text(f'{int(rpm)}')

        if rpm >= 7500:
            self.rpm_needle.set_color('#FF4444')
            self.rpm_needle_tip.set_color('#FF4444')
        else:
            self.rpm_needle.set_color('#00D9FF')
            self.rpm_needle_tip.set_color('#00D9FF')

        self.update_gauge_needle(self.power_needle, self.power_needle_tip, power_hp, 0, 200)
        self.power_text.set_text(f'{int(power_hp)}')

        # Torque gauge handles positive and negative (engine braking)
        # Clamp to display range but show actual value in text
        torque_display = np.clip(torque_nm, -80, 250)
        self.update_gauge_needle(self.torque_needle, self.torque_needle_tip, torque_display, -80, 250)
        self.torque_text.set_text(f'{int(torque_nm)}')

        # Change needle color for negative torque (engine braking)
        if torque_nm < 0:
            self.torque_needle.set_color('#FF4444')  # Red for braking
            self.torque_needle_tip.set_color('#FF4444')
        else:
            self.torque_needle.set_color('#FFD700')  # Gold for power
            self.torque_needle_tip.set_color('#FFD700')

        # Gear display
        gear_text = 'N' if gear == 0 else str(gear)
        self.gear_text.set_text(gear_text)
        if state['is_shifting']:
            self.gear_text.set_color('#FFD700')
        else:
            self.gear_text.set_color('#00D9FF')

        # Throttle bar
        self.throttle_bar.set_width(0.9 * self.throttle)
        self.throttle_pct_text.set_text(f'{int(self.throttle * 100)}%')

        # Info panel
        info_str = (
            f"Engine RPM:  {rpm:.0f}\n"
            f"Gear:        {gear_text}\n"
            f"Gear Ratio:  {state['gear_ratio']:.3f}\n"
            f"Overall:     {state['overall_ratio']:.3f}\n"
            f"\n"
            f"Power:       {power_hp:.1f} hp\n"
            f"Torque:      {torque_nm:.1f} Nm\n"
            f"Throttle:    {self.throttle*100:.0f}%\n"
            f"\n"
            f"Fuel Cut:    {'YES' if state['fuel_cut_active'] else 'NO'}"
        )
        self.info_text.set_text(info_str)

        # Update buffers
        self.time_buffer.append(self.current_time)
        self.rpm_buffer.append(rpm)
        self.throttle_buffer.append(self.throttle * 100)
        self.torque_buffer.append(torque_nm)
        self.power_buffer.append(power_hp)
        self.gear_buffer.append(gear)

        # Update charts
        if len(self.time_buffer) > 1:
            times = np.array(self.time_buffer)
            rpms = np.array(self.rpm_buffer)
            throttles = np.array(self.throttle_buffer)
            torques = np.array(self.torque_buffer)
            powers = np.array(self.power_buffer)

            self.rpm_line.set_data(times, rpms)
            self.throttle_line.set_data(times, throttles)
            self.power_line.set_data(times, powers)
            self.torque_line.set_data(times, torques)

            # Auto-scale x-axis
            time_window = 10.0
            if times[-1] > time_window:
                self.ax_rpm_chart.set_xlim(times[-1] - time_window, times[-1])
                self.ax_power_chart.set_xlim(times[-1] - time_window, times[-1])
            else:
                self.ax_rpm_chart.set_xlim(0, time_window)
                self.ax_power_chart.set_xlim(0, time_window)

        self.fig.canvas.draw_idle()
        self.fig.canvas.flush_events()

    def on_key_press(self, event):
        """Handle key press (AZERTY)."""
        key = event.key.lower()

        if key == 'escape':
            print("\n[ESC] Exiting...")
            self.running = False
            plt.close(self.fig)
            return

        self.keys_pressed.add(key)

        # Gear shifting
        if key == 'a':
            self.powertrain.shift_down()
            print(f"[A] Shifted DOWN to gear {self.powertrain.gearbox.current_gear}")
        elif key == 'e':
            self.powertrain.shift_up()
            print(f"[E] Shifted UP to gear {self.powertrain.gearbox.current_gear}")
        elif key in ['1', '2', '3', '4', '5', '6']:
            gear = int(key)
            self.powertrain.shift_to(gear)
            print(f"[{key}] Selected gear {gear}")
        elif key == 'n':
            self.powertrain.shift_to(0)
            print(f"[N] Neutral")
        elif key == ' ':
            self.throttle = 0.0
            print("[SPACE] Throttle cut")

    def on_key_release(self, event):
        """Handle key release."""
        key = event.key.lower()
        if key in self.keys_pressed:
            self.keys_pressed.remove(key)

    def update_inputs(self):
        """Update throttle based on keys."""
        if 'z' in self.keys_pressed:
            self.throttle = min(1.0, self.throttle + self.throttle_increment)
        elif 's' in self.keys_pressed:
            self.throttle = max(0.0, self.throttle - self.throttle_increment)

    def run(self):
        """Run the engine telemetry."""
        print("="*70)
        print("MX-5 Engine & Gearbox Telemetry - ENGINE-ONLY MODE")
        print("="*70)
        print("\nControls (AZERTY keyboard):")
        print("  Z          - Increase throttle")
        print("  S          - Decrease throttle")
        print("  A          - Shift DOWN")
        print("  E          - Shift UP")
        print("  1-6        - Select gear directly")
        print("  N          - Neutral")
        print("  Space      - Cut throttle")
        print("  ESC        - Quit")
        print("\nStarting in gear 1 at idle.")
        print("="*70 + "\n")

        # Connect events
        self.fig.canvas.mpl_connect('key_press_event', self.on_key_press)
        self.fig.canvas.mpl_connect('key_release_event', self.on_key_release)

        # Show display
        plt.ion()
        plt.show(block=False)

        self.running = True

        try:
            while self.running and plt.fignum_exists(self.fig.number):
                # Update inputs
                self.update_inputs()

                # Get engine torque at current throttle (respects fuel cut at 7500 RPM!)
                engine_torque = self.powertrain.engine.get_torque(self.throttle)

                # Simulate engine acceleration based on torque
                # Positive torque = RPM increases, zero torque (fuel cut) = RPM decreases
                if engine_torque > 0:
                    # Engine producing power - RPM increases
                    # More torque = faster RPM increase
                    # At full throttle (205 Nm), should reach 7500 RPM in ~5-6 seconds
                    rpm_acceleration = engine_torque * 20.0  # RPM/s increase rate
                    self.powertrain.engine.rpm += rpm_acceleration * self.dt
                else:
                    # Fuel cut active or no throttle - RPM decreases (engine braking)
                    # Decay back toward idle (slower decay to simulate flywheel inertia)
                    target_rpm = 800  # Idle RPM
                    rpm_diff = target_rpm - self.powertrain.engine.rpm
                    self.powertrain.engine.rpm += rpm_diff * 0.08 * self.dt * 50  # Smooth decay

                # Clamp RPM to valid range (can't go below idle, respects max RPM)
                self.powertrain.engine.rpm = np.clip(
                    self.powertrain.engine.rpm,
                    self.powertrain.engine.IDLE_RPM,
                    self.powertrain.engine.MAX_RPM
                )

                # Update gearbox
                self.powertrain.gearbox.update(self.dt)

                # Update display
                self.update_display()

                # Increment time
                self.current_time += self.dt

                # Control update rate
                plt.pause(self.dt)

        except KeyboardInterrupt:
            print("\nInterrupted by user.")

        print("\nSession complete!")
        print(f"Total time: {self.current_time:.1f}s")
        print(f"Max RPM: {max(self.rpm_buffer) if self.rpm_buffer else 0:.0f}")

        plt.ioff()
        plt.show()


def main():
    """Main entry point."""
    try:
        telemetry = EngineOnlyTelemetry()
        telemetry.run()
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    main()
