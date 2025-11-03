"""
Test that the stationary car loophole is fixed.

The agent was exploiting the fact that wiggling the steering wheel
while stationary didn't trigger early termination. This test verifies
that the fix (checking velocity in addition to new tiles) works.
"""

import numpy as np
from preprocessing import make_carracing_env


def test_wiggling_stationary_car():
    """Test that wiggling in place triggers stationary termination."""
    print("=" * 60)
    print("Testing Stationary Loophole Fix")
    print("=" * 60)
    print("\nTest: Wiggling steering wheel while stationary")
    print("Expected: Episode should terminate after ~100 frames")
    print()

    # Create environment with stationary termination
    env = make_carracing_env(
        stack_size=4,
        discretize_actions=True,
        steering_bins=3,
        gas_brake_bins=3,
        terminate_stationary=True,
        stationary_patience=100,
        render_mode=None,
        state_mode='snapshot'  # Test with snapshot mode
    )

    state, _ = env.reset()
    step = 0
    done = False

    # Action mapping: 0=LEFT+BRAKE, 1=STRAIGHT+BRAKE, 2=RIGHT+BRAKE
    # Alternating between LEFT+BRAKE and RIGHT+BRAKE (wiggling without gas)
    wiggle_actions = [0, 2, 0, 2, 0, 2]  # LEFT, RIGHT, LEFT, RIGHT, ...

    while not done and step < 500:
        # Wiggle steering wheel without gas (should stay mostly stationary)
        action = wiggle_actions[step % len(wiggle_actions)]
        next_state, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        step += 1

        if step % 50 == 0:
            print(f"Step {step}: reward={reward:.2f}, done={done}")

    env.close()

    print(f"\nResult:")
    print(f"  Total steps: {step}")
    print(f"  Terminated: {terminated}")
    print(f"  Truncated: {truncated}")

    if 'stationary_termination' in info:
        print(f"  Stationary termination: {info['stationary_termination']}")

    # Verify the fix works
    if step < 200 and truncated:
        print(f"\n✅ SUCCESS: Episode terminated early (step {step})")
        print("   The loophole is FIXED! Agent can't exploit wiggling.")
        return True
    else:
        print(f"\n❌ FAILURE: Episode didn't terminate early (step {step})")
        print("   The loophole still exists!")
        return False


def test_moving_car():
    """Test that moving forward doesn't trigger premature termination."""
    print("\n" + "=" * 60)
    print("Testing Normal Movement (shouldn't terminate early)")
    print("=" * 60)
    print("\nTest: Moving forward with gas")
    print("Expected: Episode should NOT terminate via stationary logic")
    print()

    # Create environment with stationary termination
    env = make_carracing_env(
        stack_size=4,
        discretize_actions=True,
        steering_bins=3,
        gas_brake_bins=3,
        terminate_stationary=True,
        stationary_patience=100,
        render_mode=None,
        state_mode='snapshot'  # Test with snapshot mode
    )

    state, _ = env.reset()
    step = 0
    done = False

    # Action 7 = STRAIGHT + GAS (should move forward until hitting curve/off-track)
    action = 7

    while not done and step < 500:
        next_state, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        step += 1

        if step % 50 == 0:
            print(f"Step {step}: reward={reward:.2f}, done={done}")

    env.close()

    print(f"\nResult:")
    print(f"  Total steps: {step}")
    print(f"  Terminated: {terminated}")
    print(f"  Truncated: {truncated}")

    # Check if stationary termination was triggered
    stationary_term = info.get('stationary_termination', False)
    off_track = info.get('off_track', False)

    if stationary_term:
        print(f"\n❌ FAILURE: Stationary termination triggered for moving car")
        return False
    elif off_track:
        print(f"\n✅ SUCCESS: Car went off-track at step {step} (expected)")
        print("   Stationary logic did NOT trigger false positives.")
        return True
    else:
        print(f"\n✅ SUCCESS: Episode ended normally at step {step}")
        print("   Stationary logic did NOT trigger false positives.")
        return True


if __name__ == "__main__":
    test1 = test_wiggling_stationary_car()
    test2 = test_moving_car()

    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)
    if test1 and test2:
        print("✅ ALL TESTS PASSED - Loophole is FIXED!")
    else:
        print("❌ SOME TESTS FAILED - Loophole may still exist")
    print("=" * 60)
