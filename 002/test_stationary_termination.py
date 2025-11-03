"""
Test script to verify that stationary car termination works as expected.
"""

from preprocessing import make_carracing_env

def test_stationary_termination():
    """Test that the car terminates after staying stationary."""
    print("Testing stationary car termination...")
    print("=" * 60)

    # Create environment with short patience for quick testing
    env = make_carracing_env(
        render_mode=None,
        terminate_stationary=True,
        stationary_patience=50,  # Short patience for testing
        stationary_min_steps=20   # Low minimum for testing
    )

    obs, info = env.reset()
    print("Environment reset. Testing with NO_OP action (coast)...")

    total_reward = 0
    step_count = 0
    done = False

    # Take NO_OP actions (coast - action 4)
    no_op_action = 4  # STRAIGHT+COAST

    while not done and step_count < 200:
        obs, reward, terminated, truncated, info = env.step(no_op_action)
        total_reward += reward
        step_count += 1
        done = terminated or truncated

        if done:
            print(f"\nEpisode terminated at step {step_count}")
            print(f"Total reward: {total_reward:.2f}")
            print(f"Terminated: {terminated}")
            print(f"Truncated: {truncated}")
            print(f"Info: {info}")

            if truncated and info.get('stationary_termination'):
                print("\n✓ SUCCESS: Episode was truncated due to stationary car!")
                return True
            elif terminated and info.get('off_track'):
                print("\n✗ Car went off track before stationary termination")
                return False
            else:
                print("\n✗ Episode ended for unexpected reason")
                return False

    print(f"\n✗ FAIL: Episode did not terminate after {step_count} steps")
    return False

def test_normal_movement():
    """Test that the car does NOT terminate when making progress."""
    print("\n" + "=" * 60)
    print("Testing normal movement (should NOT terminate early)...")
    print("=" * 60)

    env = make_carracing_env(
        render_mode=None,
        terminate_stationary=True,
        stationary_patience=50,
        stationary_min_steps=20
    )

    obs, info = env.reset()
    print("Environment reset. Testing with GAS action...")

    total_reward = 0
    step_count = 0
    done = False

    # Take gas actions (STRAIGHT+GAS - action 7)
    gas_action = 7  # STRAIGHT+GAS

    while not done and step_count < 200:
        obs, reward, terminated, truncated, info = env.step(gas_action)
        total_reward += reward
        step_count += 1
        done = terminated or truncated

        if step_count % 50 == 0:
            print(f"  Step {step_count}: reward = {total_reward:.2f}")

    print(f"\nEpisode ended at step {step_count}")
    print(f"Total reward: {total_reward:.2f}")
    print(f"Terminated: {terminated}")
    print(f"Truncated: {truncated}")
    print(f"Info: {info}")

    if truncated and info.get('stationary_termination'):
        print("\n✗ FAIL: Episode was truncated due to stationary car (should not happen with gas)")
        return False
    else:
        print("\n✓ SUCCESS: Episode did not terminate early due to stationary car")
        return True

def test_disable_stationary():
    """Test that stationary termination can be disabled."""
    print("\n" + "=" * 60)
    print("Testing with stationary termination DISABLED...")
    print("=" * 60)

    env = make_carracing_env(
        render_mode=None,
        terminate_stationary=False  # Disabled
    )

    obs, info = env.reset()
    print("Environment reset. Testing with NO_OP action (should not terminate early)...")

    step_count = 0
    done = False
    no_op_action = 4  # STRAIGHT+COAST

    # Run for 150 steps (more than patience=100)
    while not done and step_count < 150:
        obs, reward, terminated, truncated, info = env.step(no_op_action)
        step_count += 1
        done = terminated or truncated

    print(f"\nRan for {step_count} steps")

    if step_count >= 150:
        print("✓ SUCCESS: Episode did not terminate early (stationary termination disabled)")
        return True
    else:
        print(f"✗ Episode ended early at step {step_count}")
        if info.get('stationary_termination'):
            print("✗ FAIL: Stationary termination occurred even though it was disabled!")
            return False
        else:
            print("  (ended for other reason, likely off-track)")
            return True

if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("STATIONARY CAR TERMINATION TEST SUITE")
    print("=" * 60 + "\n")

    test1 = test_stationary_termination()
    test2 = test_normal_movement()
    test3 = test_disable_stationary()

    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)
    print(f"Test 1 (Stationary terminates): {'PASS ✓' if test1 else 'FAIL ✗'}")
    print(f"Test 2 (Movement continues): {'PASS ✓' if test2 else 'FAIL ✗'}")
    print(f"Test 3 (Disable works): {'PASS ✓' if test3 else 'FAIL ✗'}")

    if test1 and test2 and test3:
        print("\n✓ ALL TESTS PASSED!")
    else:
        print("\n✗ SOME TESTS FAILED")

    print("=" * 60)
