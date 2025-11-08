#!/usr/bin/env python3
"""
Test script to verify load transfer physics are working correctly.
"""

import sys
import numpy as np
from env.car_dynamics import Car

def test_longitudinal_load_transfer():
    """Test that acceleration/braking shifts weight correctly."""
    print("=" * 70)
    print("Testing Longitudinal Load Transfer (Acceleration)")
    print("=" * 70)

    # Create a car
    car = Car(world=None, init_angle=0, init_x=50, init_y=50)

    # Apply full throttle
    car.gas(1.0)

    dt = 0.02
    print("\nTime(s) | Speed  | FL Load | FR Load | RL Load | RR Load | Slip RL | Slip RR")
    print("-" * 70)

    for step in range(30):
        result = car.step(dt)
        forces = result['tire_forces']

        # Get normal forces
        fl_load = forces[0]['normal_force']
        fr_load = forces[1]['normal_force']
        rl_load = forces[2]['normal_force']
        rr_load = forces[3]['normal_force']

        # Get slip ratios (rear wheels are driven)
        rl_slip = forces[2]['slip_ratio']
        rr_slip = forces[3]['slip_ratio']

        speed = np.sqrt(car.vx**2 + car.vy**2)

        if step % 5 == 0:
            print(f"{step * dt:6.2f}  | {speed:6.2f} | {fl_load:7.1f} | {fr_load:7.1f} | "
                  f"{rl_load:7.1f} | {rr_load:7.1f} | {rl_slip:7.3f} | {rr_slip:7.3f}")

    # During acceleration, rear wheels should have MORE load than front
    avg_rear_load = (rl_load + rr_load) / 2
    avg_front_load = (fl_load + fr_load) / 2

    print("\n" + "=" * 70)
    print(f"Average rear load: {avg_rear_load:.1f} N")
    print(f"Average front load: {avg_front_load:.1f} N")
    print(f"Load transfer (rear - front): {avg_rear_load - avg_front_load:.1f} N")

    if avg_rear_load > avg_front_load:
        print("✓ PASS: Weight shifted to rear during acceleration")
        accel_pass = True
    else:
        print("✗ FAIL: Weight did not shift to rear")
        accel_pass = False

    # Check slip ratios are reasonable
    max_slip = max(abs(rl_slip), abs(rr_slip))
    print(f"\nFinal slip ratio: {max_slip:.3f}")
    if max_slip < 0.50:
        print("✓ PASS: Slip ratios are reasonable")
        slip_pass = True
    else:
        print("✗ FAIL: Slip ratios still too high")
        slip_pass = False

    return accel_pass and slip_pass


def test_braking_load_transfer():
    """Test that braking shifts weight forward."""
    print("\n" + "=" * 70)
    print("Testing Longitudinal Load Transfer (Braking)")
    print("=" * 70)

    # Create a car and accelerate to speed
    car = Car(world=None, init_angle=0, init_x=50, init_y=50)
    car.gas(1.0)
    dt = 0.02
    for _ in range(100):
        car.step(dt)

    # Now brake
    car.gas(0.0)
    car.brake(1.0)

    print("\nTime(s) | Speed  | FL Load | FR Load | RL Load | RR Load | Slip FL | Slip FR")
    print("-" * 70)

    for step in range(30):
        result = car.step(dt)
        forces = result['tire_forces']

        # Get normal forces
        fl_load = forces[0]['normal_force']
        fr_load = forces[1]['normal_force']
        rl_load = forces[2]['normal_force']
        rr_load = forces[3]['normal_force']

        # Get slip ratios (front wheels have more braking)
        fl_slip = forces[0]['slip_ratio']
        fr_slip = forces[1]['slip_ratio']

        speed = np.sqrt(car.vx**2 + car.vy**2)

        if step % 5 == 0:
            print(f"{step * dt:6.2f}  | {speed:6.2f} | {fl_load:7.1f} | {fr_load:7.1f} | "
                  f"{rl_load:7.1f} | {rr_load:7.1f} | {fl_slip:7.3f} | {fr_slip:7.3f}")

        if speed < 0.1:
            break

    # During braking, front wheels should have MORE load than rear
    avg_front_load = (fl_load + fr_load) / 2
    avg_rear_load = (rl_load + rr_load) / 2

    print("\n" + "=" * 70)
    print(f"Average front load: {avg_front_load:.1f} N")
    print(f"Average rear load: {avg_rear_load:.1f} N")
    print(f"Load transfer (front - rear): {avg_front_load - avg_rear_load:.1f} N")

    if avg_front_load > avg_rear_load:
        print("✓ PASS: Weight shifted to front during braking")
        brake_pass = True
    else:
        print("✗ FAIL: Weight did not shift to front")
        brake_pass = False

    # Check slip ratios are reasonable
    max_slip = max(abs(fl_slip), abs(fr_slip))
    print(f"\nFinal slip ratio: {max_slip:.3f}")
    if max_slip < 0.70:
        print("✓ PASS: Slip ratios are reasonable")
        slip_pass = True
    else:
        print("✗ FAIL: Slip ratios too high (wheels locking)")
        slip_pass = False

    return brake_pass and slip_pass


def test_oscillation_check():
    """Check for oscillating slip ratios."""
    print("\n" + "=" * 70)
    print("Testing for Slip Ratio Oscillations")
    print("=" * 70)

    car = Car(world=None, init_angle=0, init_x=50, init_y=50)
    car.gas(0.5)  # Moderate throttle

    dt = 0.02
    slip_history_rl = []
    slip_history_rr = []

    for step in range(100):
        result = car.step(dt)
        forces = result['tire_forces']

        rl_slip = forces[2]['slip_ratio']
        rr_slip = forces[3]['slip_ratio']

        slip_history_rl.append(rl_slip)
        slip_history_rr.append(rr_slip)

    # Check for sign changes (oscillations)
    sign_changes_rl = sum(1 for i in range(1, len(slip_history_rl))
                          if np.sign(slip_history_rl[i]) != np.sign(slip_history_rl[i-1])
                          and abs(slip_history_rl[i]) > 0.01)

    sign_changes_rr = sum(1 for i in range(1, len(slip_history_rr))
                          if np.sign(slip_history_rr[i]) != np.sign(slip_history_rr[i-1])
                          and abs(slip_history_rr[i]) > 0.01)

    print(f"\nRL wheel sign changes: {sign_changes_rl}")
    print(f"RR wheel sign changes: {sign_changes_rr}")

    # During constant throttle, slip should not oscillate between positive/negative
    if sign_changes_rl < 5 and sign_changes_rr < 5:
        print("✓ PASS: No significant oscillations detected")
        return True
    else:
        print("✗ FAIL: Slip ratio oscillating between positive and negative")
        return False


if __name__ == "__main__":
    test1 = test_longitudinal_load_transfer()
    test2 = test_braking_load_transfer()
    test3 = test_oscillation_check()

    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"Acceleration load transfer: {'PASS ✓' if test1 else 'FAIL ✗'}")
    print(f"Braking load transfer: {'PASS ✓' if test2 else 'FAIL ✗'}")
    print(f"Oscillation check: {'PASS ✓' if test3 else 'FAIL ✗'}")

    if test1 and test2 and test3:
        print("\n🎉 All tests passed! Load transfer is working correctly.")
        sys.exit(0)
    else:
        print("\n❌ Some tests failed. Check load transfer implementation.")
        sys.exit(1)
