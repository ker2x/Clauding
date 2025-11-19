#!/usr/bin/env python3
"""
Professional Engine Dynamometer (Dyno) for MX5 SKYACTIV-G Engine
=================================================================

A comprehensive engine dyno testing and visualization tool that produces
professional dyno sheets similar to Dynojet, Mustang Dyno, and other
chassis/engine dynamometer systems.

Features:
- Full RPM sweep testing (idle to redline)
- Torque and power curve plotting
- Peak performance metrics
- Comparison with factory specifications
- Throttle response testing
- Engine efficiency analysis
- Power-to-weight calculations
- Professional dyno sheet output

Usage:
    python mx5_dyno.py

    # Custom throttle test
    python mx5_dyno.py --throttle 0.8

    # Save dyno sheet
    python mx5_dyno.py --save dyno_sheet.png

Requirements:
    pip install matplotlib numpy
"""

import argparse
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.patches import Rectangle, FancyBboxPatch
import datetime


class EngineDyno:
    """
    Professional engine dynamometer for testing and analysis.

    Simulates a chassis dyno or engine dyno test, measuring torque and power
    across the full RPM range at various throttle positions.
    """

    def __init__(self, engine):
        """
        Initialize dyno with an engine to test.

        Args:
            engine: MX5Engine instance
        """
        self.engine = engine
        self.test_results = []

        # Factory specifications for comparison
        self.factory_specs = {
            'peak_power_hp': 181,
            'peak_power_rpm': 7000,
            'peak_torque_nm': 205,
            'peak_torque_rpm': 4000,
            'redline_rpm': 7500,
        }

        # Vehicle specs for performance calculations
        self.vehicle_mass_kg = 1062  # MX-5 ND curb weight

    def run_test(self, throttle=1.0, rpm_start=1000, rpm_end=8000, rpm_step=100):
        """
        Run a full dyno test across RPM range.

        Args:
            throttle: Throttle position [0.0 - 1.0]
            rpm_start: Starting RPM
            rpm_end: Ending RPM
            rpm_step: RPM increment

        Returns:
            dict: Test results with RPM, torque, power data
        """
        print(f"Running dyno test @ {throttle*100:.0f}% throttle...")
        print(f"RPM Range: {rpm_start} - {rpm_end} RPM")

        rpm_range = np.arange(rpm_start, rpm_end + rpm_step, rpm_step)

        results = {
            'throttle': throttle,
            'rpm': [],
            'torque_nm': [],
            'power_kw': [],
            'power_hp': [],
            'bsfc': [],  # Brake Specific Fuel Consumption (placeholder)
        }

        for rpm in rpm_range:
            # Set engine RPM
            self.engine.rpm = rpm

            # Measure torque
            torque = self.engine.get_torque(throttle)

            # Calculate power
            # Power (W) = Torque (Nm) × Angular velocity (rad/s)
            # Angular velocity = 2π × RPM / 60
            angular_velocity = (2 * np.pi * rpm) / 60.0
            power_w = torque * angular_velocity
            power_kw = power_w / 1000.0
            power_hp = power_kw * 1.341  # 1 kW = 1.341 hp

            # Store results
            results['rpm'].append(rpm)
            results['torque_nm'].append(torque)
            results['power_kw'].append(power_kw)
            results['power_hp'].append(power_hp)

            # Simple BSFC model (lower is better, typical: 200-300 g/kWh)
            # At peak efficiency ~250 g/kWh, worse at high/low RPM
            if power_kw > 0.1:
                efficiency = 1.0 - abs(rpm - 4500) / 10000.0  # Best at 4500 RPM
                efficiency = min(1.0, max(0.6, efficiency))
                bsfc = 240 / efficiency  # g/kWh
            else:
                bsfc = 0
            results['bsfc'].append(bsfc)

        # Convert to numpy arrays
        for key in ['rpm', 'torque_nm', 'power_kw', 'power_hp', 'bsfc']:
            results[key] = np.array(results[key])

        self.test_results.append(results)

        # Calculate and print summary
        self._print_summary(results)

        return results

    def _print_summary(self, results):
        """Print test summary statistics."""
        if len(results['rpm']) == 0:
            return

        # Find peaks
        peak_torque = np.max(results['torque_nm'])
        peak_torque_rpm = results['rpm'][np.argmax(results['torque_nm'])]

        peak_power_hp = np.max(results['power_hp'])
        peak_power_rpm = results['rpm'][np.argmax(results['power_hp'])]
        peak_power_kw = np.max(results['power_kw'])

        print("\n" + "="*60)
        print("DYNO TEST RESULTS")
        print("="*60)
        print(f"Peak Torque:  {peak_torque:.1f} Nm @ {peak_torque_rpm:.0f} RPM")
        print(f"Peak Power:   {peak_power_hp:.1f} hp ({peak_power_kw:.1f} kW) @ {peak_power_rpm:.0f} RPM")
        print(f"Power/Weight: {peak_power_hp / (self.vehicle_mass_kg / 453.592):.1f} hp/ton")
        print(f"              {peak_power_kw / (self.vehicle_mass_kg / 1000.0):.1f} kW/tonne")

        # Compare to factory specs
        torque_diff = peak_torque - self.factory_specs['peak_torque_nm']
        power_diff = peak_power_hp - self.factory_specs['peak_power_hp']

        print("\nComparison to Factory Specs:")
        print(f"  Torque: {torque_diff:+.1f} Nm ({torque_diff/self.factory_specs['peak_torque_nm']*100:+.1f}%)")
        print(f"  Power:  {power_diff:+.1f} hp ({power_diff/self.factory_specs['peak_power_hp']*100:+.1f}%)")
        print("="*60 + "\n")

    def run_throttle_sweep(self, throttle_positions=None):
        """
        Run multiple tests at different throttle positions.

        Args:
            throttle_positions: List of throttle positions to test
        """
        if throttle_positions is None:
            throttle_positions = [0.25, 0.50, 0.75, 1.0]

        for throttle in throttle_positions:
            self.run_test(throttle=throttle)

    def estimate_acceleration(self):
        """
        Estimate 0-60 mph and 0-100 km/h times based on power curve.

        This is a simplified model assuming:
        - Perfect traction
        - No drivetrain loss (already accounted in power)
        - Aerodynamic drag
        - Rolling resistance

        Returns:
            dict: Estimated acceleration times
        """
        if len(self.test_results) == 0:
            return None

        # Use full throttle results
        results = self.test_results[-1]

        # Simple physics-based model
        # This is very simplified - real acceleration depends on many factors

        # Average power from 2000-7000 RPM (typical acceleration range)
        mask = (results['rpm'] >= 2000) & (results['rpm'] <= 7000)
        avg_power_kw = np.mean(results['power_kw'][mask])

        # Simplified acceleration time estimation
        # Real MX-5 ND: 0-60 mph in ~6.0 seconds
        # This is a rough approximation
        mass = self.vehicle_mass_kg
        target_speed_ms = 26.82  # 60 mph in m/s

        # Average force (simplified, assuming 80% efficiency)
        avg_force = (avg_power_kw * 1000 * 0.8) / (target_speed_ms / 2)

        # Average acceleration
        avg_accel = avg_force / mass

        # Time (simplified kinematic equation)
        # This will underestimate real time (perfect traction, no shift time)
        time_0_60 = target_speed_ms / avg_accel * 1.4  # Fudge factor for reality

        # 0-100 km/h (27.78 m/s)
        time_0_100 = 27.78 / avg_accel * 1.4

        return {
            '0-60_mph': time_0_60,
            '0-100_kmh': time_0_100,
            'avg_power_kw': avg_power_kw,
        }


class DynoVisualization:
    """Professional dyno sheet visualization."""

    def __init__(self):
        """Initialize visualization."""
        self.fig = None

    def create_dyno_sheet(self, dyno, save_path=None):
        """
        Create a professional dyno sheet similar to Dynojet/Mustang Dyno.

        Args:
            dyno: EngineDyno instance with test results
            save_path: Optional path to save figure
        """
        if len(dyno.test_results) == 0:
            print("No test results to visualize!")
            return

        # Create figure
        plt.style.use('dark_background')
        self.fig = plt.figure(figsize=(16, 10))
        self.fig.patch.set_facecolor('#0f0f0f')

        # Create grid layout
        gs = GridSpec(3, 3, figure=self.fig, hspace=0.35, wspace=0.3,
                     left=0.06, right=0.96, top=0.92, bottom=0.08)

        # Main plot: Torque & Power curves
        ax_main = self.fig.add_subplot(gs[0:2, :])

        # Info panels
        ax_info = self.fig.add_subplot(gs[2, 0])
        ax_specs = self.fig.add_subplot(gs[2, 1])
        ax_perf = self.fig.add_subplot(gs[2, 2])

        # Draw dyno sheet
        self._draw_header(dyno)
        self._draw_main_plot(ax_main, dyno)
        self._draw_info_panel(ax_info, dyno)
        self._draw_specs_panel(ax_specs, dyno)
        self._draw_performance_panel(ax_perf, dyno)

        # Save if requested
        if save_path:
            print(f"Saving dyno sheet to: {save_path}")
            self.fig.savefig(save_path, dpi=150, facecolor='#0f0f0f',
                           edgecolor='none', bbox_inches='tight')

        plt.show()

    def _draw_header(self, dyno):
        """Draw dyno sheet header."""
        # Get current date
        date_str = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")

        # Title
        title = "MAZDA MX-5 ND - ENGINE DYNAMOMETER TEST"
        self.fig.suptitle(title, fontsize=18, fontweight='bold',
                         color='#00D9FF', y=0.97)

        # Subtitle
        subtitle = f"2.0L SKYACTIV-G I4 Engine  |  Test Date: {date_str}"
        self.fig.text(0.5, 0.94, subtitle, ha='center', fontsize=11,
                     color='#888888')

    def _draw_main_plot(self, ax, dyno):
        """Draw main torque and power curves."""
        ax.set_facecolor('#1a1a1a')

        # Colors
        color_torque = '#FFD700'  # Gold
        color_power = '#00D9FF'   # Cyan
        color_redline = '#FF4444'  # Red

        # Plot each test result
        for i, results in enumerate(dyno.test_results):
            throttle = results['throttle']
            alpha = 0.4 + (throttle * 0.6)  # Fade for partial throttle

            label_suffix = f" ({throttle*100:.0f}%)" if throttle < 1.0 else ""

            # Torque curve (left y-axis)
            ax.plot(results['rpm'], results['torque_nm'],
                   color=color_torque, linewidth=3, alpha=alpha,
                   label=f'Torque{label_suffix}')

            # Power curve (right y-axis) - scale to match torque visually
            # Typical: 200 Nm peak torque, 135 kW (181 hp) peak power
            # Scale: power_hp / 1.5 to match torque axis roughly
            ax.plot(results['rpm'], results['power_hp'] * 1.2,
                   color=color_power, linewidth=3, alpha=alpha,
                   label=f'Power{label_suffix}')

        # Redline marker
        redline = dyno.factory_specs['redline_rpm']
        ax.axvline(redline, color=color_redline, linestyle='--',
                  linewidth=2, alpha=0.7, label='Redline')

        # Styling
        ax.set_xlabel('Engine Speed (RPM)', fontsize=13, fontweight='bold',
                     color='white')
        ax.set_ylabel('Torque (Nm) / Power (hp × 1.2)', fontsize=13,
                     fontweight='bold', color='white')

        ax.set_xlim(1000, 8000)
        ax.set_ylim(0, 280)

        # Grid
        ax.grid(True, alpha=0.3, color='#333333', linestyle='-', linewidth=0.5)
        ax.grid(True, alpha=0.15, color='#333333', linestyle=':', linewidth=0.5,
               which='minor')
        ax.minorticks_on()

        # Legend
        ax.legend(loc='upper left', fontsize=10, facecolor='#0f0f0f',
                 edgecolor='#333333', framealpha=0.95)

        # Title
        ax.set_title('TORQUE & POWER CURVES', fontsize=14, fontweight='bold',
                    color='white', pad=15)

        # Tick styling
        ax.tick_params(colors='white', which='both', labelsize=10)

        # Add secondary y-axis for power (actual values)
        ax2 = ax.twinx()
        ax2.set_ylabel('Power (hp)', fontsize=13, fontweight='bold',
                      color=color_power)
        ax2.set_ylim(0, 280 / 1.2)  # Undo the scaling
        ax2.tick_params(colors=color_power, labelsize=10)
        ax2.spines['right'].set_color(color_power)
        ax2.spines['right'].set_linewidth(2)

    def _draw_info_panel(self, ax, dyno):
        """Draw test information panel."""
        ax.axis('off')
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)

        # Get full throttle results
        results = dyno.test_results[-1]

        # Calculate metrics
        peak_torque = np.max(results['torque_nm'])
        peak_torque_rpm = results['rpm'][np.argmax(results['torque_nm'])]

        peak_power_hp = np.max(results['power_hp'])
        peak_power_rpm = results['rpm'][np.argmax(results['power_hp'])]
        peak_power_kw = np.max(results['power_kw'])

        # Text content
        info_text = (
            "MEASURED VALUES\n"
            "─────────────────────\n"
            f"Peak Torque:\n"
            f"  {peak_torque:.1f} Nm\n"
            f"  @ {peak_torque_rpm:.0f} RPM\n"
            f"\n"
            f"Peak Power:\n"
            f"  {peak_power_hp:.1f} hp\n"
            f"  {peak_power_kw:.1f} kW\n"
            f"  @ {peak_power_rpm:.0f} RPM"
        )

        # Background box
        box = FancyBboxPatch((0.05, 0.05), 0.9, 0.9,
                            boxstyle="round,pad=0.05",
                            facecolor='#1a1a1a', edgecolor='#FFD700',
                            linewidth=2)
        ax.add_patch(box)

        # Text
        ax.text(0.5, 0.5, info_text, ha='center', va='center',
               fontsize=10, color='white', family='monospace',
               linespacing=1.6, fontweight='bold')

    def _draw_specs_panel(self, ax, dyno):
        """Draw specifications comparison panel."""
        ax.axis('off')
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)

        # Get full throttle results
        results = dyno.test_results[-1]

        # Calculate differences
        peak_torque = np.max(results['torque_nm'])
        peak_power_hp = np.max(results['power_hp'])

        torque_diff = peak_torque - dyno.factory_specs['peak_torque_nm']
        power_diff = peak_power_hp - dyno.factory_specs['peak_power_hp']

        torque_pct = (torque_diff / dyno.factory_specs['peak_torque_nm']) * 100
        power_pct = (power_diff / dyno.factory_specs['peak_power_hp']) * 100

        # Text content
        specs_text = (
            "FACTORY COMPARISON\n"
            "─────────────────────\n"
            f"Torque:\n"
            f"  Factory: {dyno.factory_specs['peak_torque_nm']:.0f} Nm\n"
            f"  Actual:  {peak_torque:.1f} Nm\n"
            f"  Diff:    {torque_diff:+.1f} Nm ({torque_pct:+.1f}%)\n"
            f"\n"
            f"Power:\n"
            f"  Factory: {dyno.factory_specs['peak_power_hp']:.0f} hp\n"
            f"  Actual:  {peak_power_hp:.1f} hp\n"
            f"  Diff:    {power_diff:+.1f} hp ({power_pct:+.1f}%)"
        )

        # Background box
        box = FancyBboxPatch((0.05, 0.05), 0.9, 0.9,
                            boxstyle="round,pad=0.05",
                            facecolor='#1a1a1a', edgecolor='#00D9FF',
                            linewidth=2)
        ax.add_patch(box)

        # Text
        ax.text(0.5, 0.5, specs_text, ha='center', va='center',
               fontsize=10, color='white', family='monospace',
               linespacing=1.6, fontweight='bold')

    def _draw_performance_panel(self, ax, dyno):
        """Draw performance estimates panel."""
        ax.axis('off')
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)

        # Get full throttle results
        results = dyno.test_results[-1]
        peak_power_hp = np.max(results['power_hp'])
        peak_power_kw = np.max(results['power_kw'])

        # Calculate power-to-weight
        pw_ratio_hp = peak_power_hp / (dyno.vehicle_mass_kg / 453.592)  # hp/ton
        pw_ratio_kw = peak_power_kw / (dyno.vehicle_mass_kg / 1000.0)  # kW/tonne

        # Estimate acceleration
        accel = dyno.estimate_acceleration()

        # Text content
        perf_text = (
            "PERFORMANCE EST.\n"
            "─────────────────────\n"
            f"Power/Weight:\n"
            f"  {pw_ratio_hp:.1f} hp/ton\n"
            f"  {pw_ratio_kw:.1f} kW/tonne\n"
            f"\n"
            f"Acceleration:\n"
            f"  0-60 mph: {accel['0-60_mph']:.1f}s\n"
            f"  0-100 km/h: {accel['0-100_kmh']:.1f}s\n"
            f"\n"
            f"Vehicle Mass:\n"
            f"  {dyno.vehicle_mass_kg} kg"
        )

        # Background box
        box = FancyBboxPatch((0.05, 0.05), 0.9, 0.9,
                            boxstyle="round,pad=0.05",
                            facecolor='#1a1a1a', edgecolor='#00FF88',
                            linewidth=2)
        ax.add_patch(box)

        # Text
        ax.text(0.5, 0.5, perf_text, ha='center', va='center',
               fontsize=10, color='white', family='monospace',
               linespacing=1.6, fontweight='bold')


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description='Professional Engine Dynamometer for MX-5 SKYACTIV-G',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python mx5_dyno.py
  python mx5_dyno.py --throttle 0.8
  python mx5_dyno.py --save dyno_sheet.png
  python mx5_dyno.py --sweep
        """)

    parser.add_argument('--throttle', type=float, default=1.0,
                       help='Throttle position for test (0.0-1.0, default: 1.0)')
    parser.add_argument('--save', type=str, default=None,
                       help='Save dyno sheet to file (e.g., dyno_sheet.png)')
    parser.add_argument('--sweep', action='store_true',
                       help='Run throttle sweep (25%%, 50%%, 75%%, 100%%)')

    args = parser.parse_args()

    print("="*70)
    print("MX-5 SKYACTIV-G ENGINE DYNAMOMETER")
    print("="*70)
    print()

    # Import engine
    try:
        from env.mx5_powertrain import MX5Engine
    except ImportError:
        print("ERROR: Cannot import env.mx5_powertrain.py")
        print("Make sure mx5_powertrain.py is in the env/ directory.")
        return

    # Create engine and dyno
    engine = MX5Engine()
    dyno = EngineDyno(engine)

    # Run tests
    if args.sweep:
        print("Running throttle sweep test...")
        dyno.run_throttle_sweep()
    else:
        dyno.run_test(throttle=args.throttle)

    # Create visualization
    viz = DynoVisualization()
    viz.create_dyno_sheet(dyno, save_path=args.save)


if __name__ == '__main__':
    main()
