#!/usr/bin/env python3
"""
Test script to verify wheel dynamics produce realistic slip ratios.
"""

import sys
import numpy as np
from env.car_dynamics import Car

def test_acceleration_slip():
    """Test that acceleration produces reasonable slip ratios."""
    print("=" * 60)
    print("Testing Acceleration Slip Ratios")
    print("=" * 60)

    # Create a car
    car = Car(world=None, init_angle=0, init_x=50, init_y=50)

    # Apply full throttle
    car.gas(1.0)

    # Simulate for 1 second (50 timesteps at 0.02s each)
    dt = 0.02
    max_slip = 0.0
    slip_history = []

    print("\nTime(s) | Speed(m/s) | RL Slip | RR Slip | Max Slip")
    print("-" * 60)

    for step in range(50):
        # Step the simulation
        result = car.step(dt)
        forces = result['tire_forces']

        # Get slip ratios for rear wheels (driven)
        rl_slip = forces[2]['slip_ratio']  # RL
        rr_slip = forces[3]['slip_ratio']  # RR

        # Track maximum slip
        current_max_slip = max(abs(rl_slip), abs(rr_slip))
        max_slip = max(max_slip, current_max_slip)
        slip_history.append(current_max_slip)

        # Calculate speed
        speed = np.sqrt(car.vx**2 + car.vy**2)

        # Print every 10 steps
        if step % 10 == 0:
            print(f"{step * dt:6.2f}  | {speed:10.2f} | {rl_slip:7.3f} | {rr_slip:7.3f} | {current_max_slip:8.3f}")

    print("\n" + "=" * 60)
    print(f"Maximum slip ratio during acceleration: {max_slip:.3f}")

    # For an MX-5, typical slip ratio during hard acceleration should be:
    # - 0.05-0.15 for good traction
    # - 0.20-0.30 for aggressive acceleration (some wheelspin)
    # - >0.50 means significant wheelspin (burnout)

    if max_slip < 0.40:
        print("✓ PASS: Slip ratios are realistic")
        return True
    else:
        print("✗ FAIL: Slip ratios are too high (unrealistic wheel spin)")
        return False


def test_braking_slip():
    """Test that braking produces reasonable slip ratios."""
    print("\n" + "=" * 60)
    print("Testing Braking Slip Ratios")
    print("=" * 60)

    # Create a car
    car = Car(world=None, init_angle=0, init_x=50, init_y=50)

    # First, accelerate to 20 m/s
    car.gas(1.0)
    dt = 0.02
    for _ in range(100):
        car.step(dt)

    # Now brake hard
    car.gas(0.0)
    car.brake(1.0)

    max_slip = 0.0
    slip_history = []

    print("\nTime(s) | Speed(m/s) | FL Slip | FR Slip | Max Slip")
    print("-" * 60)

    for step in range(50):
        # Step the simulation
        result = car.step(dt)
        forces = result['tire_forces']

        # Get slip ratios for front wheels (more braking force)
        fl_slip = forces[0]['slip_ratio']  # FL
        fr_slip = forces[1]['slip_ratio']  # FR

        # Track maximum slip
        current_max_slip = max(abs(fl_slip), abs(fr_slip))
        max_slip = max(max_slip, current_max_slip)
        slip_history.append(current_max_slip)

        # Calculate speed
        speed = np.sqrt(car.vx**2 + car.vy**2)

        # Print every 10 steps
        if step % 10 == 0:
            print(f"{step * dt:6.2f}  | {speed:10.2f} | {fl_slip:7.3f} | {fr_slip:7.3f} | {current_max_slip:8.3f}")

        # Stop if we've come to a halt
        if speed < 0.1:
            print(f"\nCar stopped at t={step * dt:.2f}s")
            break

    print("\n" + "=" * 60)
    print(f"Maximum slip ratio during braking: {max_slip:.3f}")

    # For braking, typical slip ratio should be:
    # - 0.10-0.20 for good braking (ABS-like)
    # - 0.30-0.50 for hard braking without ABS
    # - 0.80-1.00 means locked wheels

    if max_slip < 0.70:
        print("✓ PASS: Slip ratios are realistic")
        return True
    else:
        print("✗ FAIL: Slip ratios are too high (wheels locking)")
        return False


if __name__ == "__main__":
    accel_pass = test_acceleration_slip()
    brake_pass = test_braking_slip()

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Acceleration test: {'PASS ✓' if accel_pass else 'FAIL ✗'}")
    print(f"Braking test: {'PASS ✓' if brake_pass else 'FAIL ✗'}")

    if accel_pass and brake_pass:
        print("\n🎉 All tests passed! Wheel dynamics are realistic.")
        sys.exit(0)
    else:
        print("\n❌ Some tests failed. Wheel dynamics need adjustment.")
        sys.exit(1)
