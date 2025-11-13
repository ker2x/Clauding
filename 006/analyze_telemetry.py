"""
Analyze telemetry data logged from play_human_gui.py

This script reads CSV telemetry files and provides analysis.
Useful for debugging vehicle dynamics issues or understanding driving behavior.

Usage:
    python analyze_telemetry.py telemetry_20250113_123456.csv

Requirements:
    pip install pandas  (optional but recommended for better analysis)
"""

import argparse
import csv
import statistics


def load_telemetry(filename):
    """Load telemetry CSV file."""
    try:
        data = []
        with open(filename, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                # Convert numeric fields
                for key in row:
                    try:
                        if key != 'timestamp':
                            row[key] = float(row[key])
                    except (ValueError, KeyError):
                        pass
                data.append(row)

        print(f"✓ Loaded telemetry: {filename}")
        print(f"  Total frames: {len(data)}")
        episodes = set(row['episode'] for row in data)
        print(f"  Episodes: {sorted(episodes)}")
        max_step = max(row['step'] for row in data)
        print(f"  Duration: {int(max_step)} steps")
        return data
    except FileNotFoundError:
        print(f"✗ File not found: {filename}")
        return None
    except Exception as e:
        print(f"✗ Error loading file: {e}")
        return None


def analyze_summary(data):
    """Print summary statistics."""
    print("\n" + "="*60)
    print("SUMMARY STATISTICS")
    print("="*60)

    # Speed
    speeds = [row['speed_kmh'] for row in data]
    print(f"\nSpeed (km/h):")
    print(f"  Mean: {statistics.mean(speeds):.2f}")
    print(f"  Max: {max(speeds):.2f}")
    print(f"  Min: {min(speeds):.2f}")

    # Rewards
    rewards = [row['reward'] for row in data]
    print(f"\nRewards:")
    print(f"  Mean per step: {statistics.mean(rewards):.4f}")
    print(f"  Total: {data[-1]['total_reward']:.2f}")

    # Steering
    steering = [row['steering'] for row in data]
    print(f"\nSteering:")
    print(f"  Mean: {statistics.mean(steering):.4f}")
    print(f"  Std: {statistics.stdev(steering) if len(steering) > 1 else 0:.4f}")
    print(f"  Max left: {min(steering):.4f}")
    print(f"  Max right: {max(steering):.4f}")

    # Acceleration
    accel = [row['acceleration'] for row in data]
    gas_count = sum(1 for a in accel if a > 0)
    brake_count = sum(1 for a in accel if a < 0)
    coast_count = sum(1 for a in accel if a == 0)
    print(f"\nAcceleration:")
    print(f"  Mean: {statistics.mean(accel):.4f}")
    print(f"  Gas frames: {gas_count} ({gas_count/len(accel)*100:.1f}%)")
    print(f"  Brake frames: {brake_count} ({brake_count/len(accel)*100:.1f}%)")
    print(f"  Coast frames: {coast_count} ({coast_count/len(accel)*100:.1f}%)")


def analyze_wheels(data):
    """Analyze wheel telemetry."""
    print("\n" + "="*60)
    print("WHEEL ANALYSIS")
    print("="*60)

    import math

    wheels = ['fl', 'fr', 'rl', 'rr']
    names = ['Front Left', 'Front Right', 'Rear Left', 'Rear Right']

    for wheel, name in zip(wheels, names):
        print(f"\n{name} ({wheel.upper()}):")

        # Slip angle
        sa_col = f'{wheel}_slip_angle'
        sa_values = [abs(row[sa_col]) * 180 / math.pi for row in data if sa_col in row]
        if sa_values:
            print(f"  Slip Angle: mean={statistics.mean(sa_values):.2f}°, max={max(sa_values):.2f}°")

        # Slip ratio
        sr_col = f'{wheel}_slip_ratio'
        sr_values = [abs(row[sr_col]) for row in data if sr_col in row]
        if sr_values:
            print(f"  Slip Ratio: mean={statistics.mean(sr_values):.3f}, max={max(sr_values):.3f}")

        # Normal force
        nf_col = f'{wheel}_normal_force'
        nf_values = [row[nf_col] / 1000.0 for row in data if nf_col in row]
        if nf_values:
            print(f"  Load: mean={statistics.mean(nf_values):.2f}kN, max={max(nf_values):.2f}kN, min={min(nf_values):.2f}kN")

        # Suspension travel
        susp_col = f'{wheel}_suspension'
        susp_values = [row[susp_col] * 1000.0 for row in data if susp_col in row]
        if susp_values:
            print(f"  Suspension: mean={statistics.mean(susp_values):.1f}mm, max={max(susp_values):.1f}mm, min={min(susp_values):.1f}mm")


def analyze_suspension(data):
    """Analyze suspension behavior."""
    print("\n" + "="*60)
    print("SUSPENSION DYNAMICS")
    print("="*60)

    wheels = ['fl', 'fr', 'rl', 'rr']

    # Calculate roll and pitch
    front_roll = [(row['fr_suspension'] - row['fl_suspension']) * 1000.0 / 2.0 for row in data
                  if 'fr_suspension' in row and 'fl_suspension' in row]
    rear_roll = [(row['rr_suspension'] - row['rl_suspension']) * 1000.0 / 2.0 for row in data
                 if 'rr_suspension' in row and 'rl_suspension' in row]

    if front_roll and rear_roll:
        print(f"\nBody Roll (suspension difference):")
        print(f"  Front: mean={statistics.mean(front_roll):.2f}mm, max={max(front_roll):.2f}mm")
        print(f"  Rear: mean={statistics.mean(rear_roll):.2f}mm, max={max(rear_roll):.2f}mm")

        # Pitch
        pitch_diff = [((row['fl_suspension'] + row['fr_suspension']) / 2.0 -
                      (row['rl_suspension'] + row['rr_suspension']) / 2.0) * 1000.0
                     for row in data if all(f'{w}_suspension' in row for w in wheels)]

        if pitch_diff:
            print(f"\nBody Pitch (front-rear difference):")
            print(f"  Mean: {statistics.mean(pitch_diff):.2f}mm")
            print(f"  Max nose-down: {min(pitch_diff):.2f}mm")
            print(f"  Max nose-up: {max(pitch_diff):.2f}mm")


def find_interesting_moments(data):
    """Find interesting moments in the telemetry."""
    print("\n" + "="*60)
    print("INTERESTING MOMENTS")
    print("="*60)

    import math

    # Max speed
    max_speed_row = max(data, key=lambda r: r['speed_kmh'])
    print(f"\nMax Speed: {max_speed_row['speed_kmh']:.2f} km/h")
    print(f"  Step: {int(max_speed_row['step'])}")
    print(f"  Steering: {max_speed_row['steering']:.3f}")

    # Max slip angles
    wheels = ['fl', 'fr', 'rl', 'rr']
    for wheel in wheels:
        sa_col = f'{wheel}_slip_angle'
        if sa_col in data[0]:
            max_sa_row = max(data, key=lambda r: abs(r[sa_col]))
            max_sa = max_sa_row[sa_col] * 180 / math.pi
            if abs(max_sa) > 5:
                print(f"\nMax Slip Angle ({wheel.upper()}): {max_sa:+.2f}°")
                print(f"  Step: {int(max_sa_row['step'])}")
                print(f"  Speed: {max_sa_row['speed_kmh']:.2f} km/h")

    # Min normal force
    for wheel in wheels:
        nf_col = f'{wheel}_normal_force'
        if nf_col in data[0]:
            min_nf_row = min(data, key=lambda r: r[nf_col])
            min_nf = min_nf_row[nf_col] / 1000.0
            if min_nf < 1.5:
                print(f"\nMin Load ({wheel.upper()}): {min_nf:.2f} kN")
                print(f"  Step: {int(min_nf_row['step'])}")
                print(f"  Speed: {min_nf_row['speed_kmh']:.2f} km/h")


def main():
    parser = argparse.ArgumentParser(description='Analyze telemetry data from play_human_gui.py')
    parser.add_argument('filename', type=str, help='CSV telemetry file to analyze')
    parser.add_argument('--summary', action='store_true', help='Show summary statistics (default if no flags)')
    parser.add_argument('--wheels', action='store_true', help='Show wheel analysis')
    parser.add_argument('--suspension', action='store_true', help='Show suspension dynamics')
    parser.add_argument('--moments', action='store_true', help='Find interesting moments')
    parser.add_argument('--all', action='store_true', help='Show all analyses')

    args = parser.parse_args()

    # Load data
    data = load_telemetry(args.filename)
    if data is None:
        return

    # If no specific analysis requested, show summary
    show_all = args.all or not (args.summary or args.wheels or args.suspension or args.moments)

    if show_all or args.summary:
        analyze_summary(data)

    if show_all or args.wheels:
        analyze_wheels(data)

    if show_all or args.suspension:
        analyze_suspension(data)

    if show_all or args.moments:
        find_interesting_moments(data)

    print("\n" + "="*60)
    print("Analysis complete!")
    print("="*60)


if __name__ == '__main__':
    main()
