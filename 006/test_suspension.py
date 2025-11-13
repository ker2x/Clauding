"""
Test script for suspension system.

Demonstrates how to use different suspension modes and compare their behavior.

Usage:
    python test_suspension.py
"""

import numpy as np
from env.suspension_config import SuspensionConfig, SuspensionMode, SuspensionPresets
from env.car_dynamics import Car


def print_suspension_info(car, mode_name):
    """Print suspension configuration and state."""
    print(f"\n{'='*60}")
    print(f"Suspension Mode: {mode_name}")
    print(f"{'='*60}")

    config = car.suspension_config
    print(f"Mode: {config['mode']}")

    if config['mode'] == SuspensionMode.VIRTUAL:
        print(f"  Natural frequency: {config['natural_frequency']:.2f} Hz")
        print(f"  Lerp factor: {config['lerp_factor']:.4f}")
        print(f"  Lateral factor: {config['lateral_factor']:.2f}")
        print(f"  Longitudinal factor: {config['longitudinal_factor']:.4f}")

    elif config['mode'] in [SuspensionMode.QUARTER_CAR, SuspensionMode.FULL]:
        print(f"  Preset: {config.get('preset', 'custom')}")
        print(f"  Description: {config.get('description', 'N/A')}")
        print(f"  Spring rate: {config['spring_rate']:.0f} N/m")
        print(f"  Damping: {config['damping']:.0f} N·s/m")
        print(f"  Natural frequency: {config['natural_frequency']:.2f} Hz")
        print(f"  Damping ratio: {config['damping_ratio']:.2f}")
        print(f"  Ride height: {config['ride_height']:.3f} m")
        print(f"  Max travel: +{config['max_compression']:.3f}/-{config['max_extension']:.3f} m")

        if config['mode'] == SuspensionMode.FULL:
            print(f"  ARB Front: {config['arb_front']:.0f} N·m/rad")
            print(f"  ARB Rear: {config['arb_rear']:.0f} N·m/rad")
            print(f"  ARB Ratio (F/R): {config['arb_front']/config['arb_rear']:.2f}")

    print()


def simulate_cornering(car, mode_name, duration=2.0, dt=0.02):
    """
    Simulate cornering and measure suspension response.

    Args:
        car: Car instance
        mode_name: Name of suspension mode
        duration: Simulation duration (seconds)
        dt: Timestep (seconds)
    """
    print(f"\nSimulating cornering for {mode_name}...")

    # Apply constant steering and throttle (simulate steady-state corner)
    car.steer(0.5)  # 50% steering
    car.gas(0.3)    # 30% throttle

    # Simulate
    steps = int(duration / dt)
    max_lateral_accel = 0.0

    for step in range(steps):
        # Step physics
        result = car.step(dt)

        # Track maximum lateral acceleration
        lateral_accel = abs(car.vx * car.yaw_rate)
        max_lateral_accel = max(max_lateral_accel, lateral_accel)

    # Print results
    print(f"  Final speed: {np.sqrt(car.vx**2 + car.vy**2):.2f} m/s")
    print(f"  Max lateral accel: {max_lateral_accel/9.81:.2f}g")

    if hasattr(car, 'suspension_travel'):
        print(f"  Suspension travel [FL, FR, RL, RR]: [{', '.join(f'{t*1000:.1f}' for t in car.suspension_travel)}] mm")
        # Calculate roll (difference between left and right)
        front_roll = (car.suspension_travel[1] - car.suspension_travel[0]) * 1000
        rear_roll = (car.suspension_travel[3] - car.suspension_travel[2]) * 1000
        print(f"  Roll (compression diff): Front={front_roll:.1f}mm, Rear={rear_roll:.1f}mm")
    else:
        print(f"  Smoothed lateral accel: {car.smoothed_lateral_accel:.2f} m/s²")


def test_all_modes():
    """Test all suspension modes."""
    print("SUSPENSION SYSTEM TEST")
    print("=" * 60)
    print("\nThis test demonstrates the hybrid suspension system with")
    print("three different modes: Virtual, Quarter-Car, and Full.\n")

    # Test configurations
    configs = [
        ("Virtual (Original)", SuspensionPresets.VIRTUAL),
        ("Quarter-Car Stock", SuspensionPresets.QUARTER_CAR_STOCK),
        ("Quarter-Car Sport", SuspensionPresets.QUARTER_CAR_SPORT),
        ("Quarter-Car Track", SuspensionPresets.QUARTER_CAR_TRACK),
        ("Full Stock", SuspensionPresets.FULL_STOCK),
        ("Full Sport", SuspensionPresets.FULL_SPORT),
        ("Full Track", SuspensionPresets.FULL_TRACK),
        ("Full Drift", SuspensionPresets.FULL_DRIFT),
    ]

    for name, config in configs:
        # Create car with suspension config
        car = Car(
            world=None,
            init_angle=0.0,
            init_x=100.0,
            init_y=100.0,
            suspension_config=config
        )

        # Print configuration
        print_suspension_info(car, name)

        # Simulate cornering
        simulate_cornering(car, name)

    print("\n" + "=" * 60)
    print("TEST COMPLETE")
    print("=" * 60)


def compare_modes():
    """
    Compare virtual vs physical suspension side-by-side.
    """
    print("\nCOMPARISON: Virtual vs Quarter-Car vs Full")
    print("=" * 60)

    # Create three cars with different suspension modes
    cars = [
        ("Virtual", Car(None, 0, 100, 100, SuspensionPresets.VIRTUAL)),
        ("Quarter-Car", Car(None, 0, 100, 100, SuspensionPresets.QUARTER_CAR_SPORT)),
        ("Full", Car(None, 0, 100, 100, SuspensionPresets.FULL_SPORT)),
    ]

    # Simulate same maneuver for all
    duration = 1.0
    dt = 0.02
    steps = int(duration / dt)

    print(f"\nSimulating {duration}s cornering maneuver...")
    print(f"{'Mode':<15} {'Speed (m/s)':<12} {'Lat Accel (g)':<15} {'Roll Front (mm)':<15} {'Roll Rear (mm)'}")
    print("-" * 70)

    for name, car in cars:
        car.steer(0.5)
        car.gas(0.4)

        max_lat_accel = 0.0
        for _ in range(steps):
            car.step(dt)
            lat_accel = abs(car.vx * car.yaw_rate)
            max_lat_accel = max(max_lat_accel, lat_accel)

        speed = np.sqrt(car.vx**2 + car.vy**2)

        if hasattr(car, 'suspension_travel'):
            front_roll = (car.suspension_travel[1] - car.suspension_travel[0]) * 1000
            rear_roll = (car.suspension_travel[3] - car.suspension_travel[2]) * 1000
            print(f"{name:<15} {speed:<12.2f} {max_lat_accel/9.81:<15.2f} {front_roll:<15.1f} {rear_roll:.1f}")
        else:
            print(f"{name:<15} {speed:<12.2f} {max_lat_accel/9.81:<15.2f} {'N/A':<15} {'N/A'}")


if __name__ == '__main__':
    import sys

    # Check if numpy is available
    try:
        import numpy as np
    except ImportError:
        print("ERROR: NumPy not installed. Please install requirements:")
        print("  pip install -r requirements.txt")
        sys.exit(1)

    print("Suspension System Test Suite")
    print("=" * 60)
    print()

    # Run tests
    test_all_modes()
    compare_modes()

    print("\n" + "=" * 60)
    print("All tests completed successfully!")
    print("\nNext steps:")
    print("1. Try different suspension modes in watch_agent.py")
    print("2. Train agents with different suspension configs")
    print("3. Compare performance and lap times")
    print("=" * 60)
