#!/usr/bin/env python3
"""
Test that the car CANNOT drive backward (no reverse gear).
Brakes should only slow the car to zero, not push it backward.
"""

import numpy as np
from preprocessing import make_carracing_env

def test_no_reverse_gear():
    """Test that car cannot drive backward by braking."""
    print("=" * 70)
    print("Testing No Reverse Gear (Brake Cannot Make Car Go Backward)")
    print("=" * 70)

    # Create environment
    env = make_carracing_env(
        stack_size=4,
        terminate_stationary=False,
        stationary_patience=100,
        render_mode=None,
        state_mode='vector'
    )

    print("\nTest 1: Braking from stationary should NOT make car move backward")
    print("-" * 70)

    # Reset to stationary
    state, _ = env.reset()

    # Apply full brake for 30 steps while stationary
    action_brake = np.array([0.0, -1.0], dtype=np.float32)  # Full brake

    print("Applying full brake to stationary car for 30 steps...")

    velocities = []
    wheel_speeds = []

    for step in range(30):
        state, _, _, _, _ = env.step(action_brake)
        car = env.unwrapped.car
        velocities.append(car.vx)
        wheel_speeds.append(car.wheels[0].omega)  # Front left wheel

        if step % 10 == 0:
            print(f"  Step {step}: vx = {car.vx:+.4f} m/s, wheel omega = {car.wheels[0].omega:+.4f} rad/s")

    final_vx = velocities[-1]
    final_omega = wheel_speeds[-1]
    min_vx = min(velocities)
    min_omega = min(wheel_speeds)

    print(f"\nFinal velocity: {final_vx:+.4f} m/s")
    print(f"Final wheel speed: {final_omega:+.4f} rad/s")
    print(f"Min velocity during test: {min_vx:+.4f} m/s")
    print(f"Min wheel speed during test: {min_omega:+.4f} rad/s")

    test1_pass = (min_vx >= -0.01 and min_omega >= -0.001)
    if test1_pass:
        print("✓ PASS: Car did not move backward")
    else:
        print(f"✗ FAIL: Car moved backward (vx={min_vx:.4f}, omega={min_omega:.4f})")

    print("\n\nTest 2: Braking from forward motion should stop at zero")
    print("-" * 70)

    # Reset and accelerate forward
    env.reset()

    print("Accelerating forward for 20 steps...")
    action_accel = np.array([0.0, 0.8], dtype=np.float32)
    for step in range(20):
        state, _, _, _, _ = env.step(action_accel)

    car = env.unwrapped.car
    initial_vx = car.vx
    print(f"Initial forward velocity: {initial_vx:.2f} m/s")

    # Now brake hard
    print("\nBraking hard for 50 steps...")
    velocities = []
    wheel_speeds = []

    for step in range(50):
        state, _, _, _, _ = env.step(action_brake)
        car = env.unwrapped.car
        velocities.append(car.vx)
        wheel_speeds.append(car.wheels[2].omega)  # Rear left wheel

        if step % 10 == 0:
            print(f"  Step {step}: vx = {car.vx:+.4f} m/s, wheel omega = {car.wheels[2].omega:+.4f} rad/s")

    final_vx = velocities[-1]
    final_omega = wheel_speeds[-1]
    min_vx = min(velocities)
    min_omega = min(wheel_speeds)

    print(f"\nFinal velocity: {final_vx:+.4f} m/s")
    print(f"Final wheel speed: {final_omega:+.4f} rad/s")
    print(f"Min velocity during braking: {min_vx:+.4f} m/s")
    print(f"Min wheel speed during braking: {min_omega:+.4f} rad/s")

    test2_pass = (min_vx >= -0.01 and min_omega >= -0.001)
    if test2_pass:
        print("✓ PASS: Car stopped at zero, did not reverse")
    else:
        print(f"✗ FAIL: Car went backward (vx={min_vx:.4f}, omega={min_omega:.4f})")

    env.close()

    # Overall result
    print("\n" + "=" * 70)
    print("OVERALL RESULT")
    print("=" * 70)

    if test1_pass and test2_pass:
        print("\n🎉 SUCCESS: Car has no reverse gear!")
        print("   - Braking from stationary: stays at zero ✓")
        print("   - Braking from forward motion: stops at zero ✓")
        print("   - Wheels never spin backward ✓")
        return True
    else:
        print("\n❌ FAILURE: Car can move backward (has reverse gear bug)")
        if not test1_pass:
            print("   - Failed: Braking from stationary made car go backward")
        if not test2_pass:
            print("   - Failed: Braking from forward motion made car reverse")
        return False

if __name__ == "__main__":
    import sys
    success = test_no_reverse_gear()
    sys.exit(0 if success else 1)
