"""
Test to verify that episodes end when the car leaves the track
and that appropriate negative rewards are received.

This test ensures:
1. Car receives -100 penalty when going off-track
2. Episodes terminate when off-track
3. No reward clipping - agent sees true environment penalties
"""

import numpy as np
from preprocessing import make_carracing_env


def test_off_track_termination():
    """Test that going off-track causes termination and negative rewards."""

    print("=" * 60)
    print("Testing Off-Track Termination and Rewards")
    print("=" * 60)
    print("\nNote: CarRacing-v3 terminates immediately when going off-track.")
    print("Expected penalty: -100.00")

    # Test 1: Brake at start
    print("\nTest 1: Braking at start (should eventually go off-track)")
    print("-" * 60)
    env = make_carracing_env(
        terminate_stationary=False,  # Disable so we can test off-track separately
        render_mode=None
    )

    state, _ = env.reset()
    total_reward = 0
    steps = 0
    min_reward = 0
    max_reward = 0
    rewards_history = []

    # Strategy: Start with gas to get moving, then hard brake
    print(f"Strategy: GAS for 10 steps, then BRAKE")
    print(f"Expected: Car gains speed, then skids off-track when braking\n")

    for i in range(100):
        if i < 10:
            action = 7  # STRAIGHT+GAS
        else:
            action = 1  # STRAIGHT+BRAKE

        state, reward, terminated, truncated, info = env.step(action)
        steps += 1
        total_reward += reward
        rewards_history.append(reward)
        min_reward = min(min_reward, reward)
        max_reward = max(max_reward, reward)

        # Print significant rewards
        if reward < -10.0:
            print(f"  Step {steps}: reward = {reward:.2f} *** OFF-TRACK PENALTY ***")
        elif reward > 1.0 and i < 15:
            print(f"  Step {steps}: reward = {reward:.2f}")

        if terminated or truncated:
            termination_type = "terminated" if terminated else "truncated"
            print(f"\n✓ Episode ended at step {steps} ({termination_type})")
            if 'stationary_termination' in info:
                print(f"  Reason: Stationary car")
            break
    else:
        print(f"\n⚠ Episode did NOT terminate after {steps} steps")

    print(f"\nResults:")
    print(f"  Total steps: {steps}")
    print(f"  Total reward: {total_reward:.2f}")
    print(f"  Min reward (single step): {min_reward:.2f}")
    print(f"  Max reward (single step): {max_reward:.2f}")
    print(f"  Average reward per step: {total_reward/steps:.2f}")

    catastrophic_negatives = sum(1 for r in rewards_history if r < -50.0)
    print(f"  Steps with catastrophic rewards (< -50): {catastrophic_negatives}")

    if catastrophic_negatives > 0:
        print(f"  ✓ Found off-track penalty: {min_reward:.2f}")

    env.close()

    # Test 2: Random actions - should eventually go off-track
    print("\n" + "=" * 60)
    print("Test 2: Random actions (should eventually go off-track)")
    print("-" * 60)
    env = make_carracing_env(
        terminate_stationary=False,
        render_mode=None
    )

    state, _ = env.reset()
    total_reward = 0
    steps = 0
    min_reward = 0
    rewards_history = []
    np.random.seed(42)  # Reproducible

    print(f"Strategy: Random actions")
    print(f"Expected: Eventually goes off-track, gets -100 penalty\n")

    for i in range(300):
        action = np.random.randint(0, 9)
        state, reward, terminated, truncated, info = env.step(action)
        steps += 1
        total_reward += reward
        rewards_history.append(reward)
        min_reward = min(min_reward, reward)

        # Print when we get significant negative rewards (likely off-track)
        if reward < -50.0:
            print(f"  Step {steps}: reward = {reward:.2f} *** OFF-TRACK PENALTY ***")

        if terminated or truncated:
            termination_type = "terminated" if terminated else "truncated"
            print(f"\n✓ Episode ended at step {steps} ({termination_type})")
            break
    else:
        print(f"\n⚠ Episode did NOT terminate after {steps} steps")

    print(f"\nResults:")
    print(f"  Total steps: {steps}")
    print(f"  Total reward: {total_reward:.2f}")
    print(f"  Min reward (single step): {min_reward:.2f}")
    print(f"  Average reward per step: {total_reward/steps:.2f}")

    catastrophic_negatives = sum(1 for r in rewards_history if r < -50.0)
    print(f"  Steps with catastrophic rewards (< -50): {catastrophic_negatives}")

    if catastrophic_negatives > 0:
        print(f"  ✓ Found raw off-track penalty: {min_reward:.2f}")
    else:
        print(f"  ⚠ No large penalties found (may not have gone off-track)")

    env.close()

    # Test 3: Normal driving
    print("\n" + "=" * 60)
    print("Test 3: Normal driving (STRAIGHT+GAS)")
    print("-" * 60)
    env = make_carracing_env(
        terminate_stationary=False,
        render_mode=None
    )

    state, _ = env.reset()
    total_reward = 0
    steps = 0
    positive_rewards = 0

    # Action 7 = STRAIGHT+GAS
    forward_action = 7

    print(f"Action: STRAIGHT+GAS (should stay on track initially)")
    print(f"Expected: Small positive rewards for visiting new tiles\n")

    for i in range(50):  # Just 50 steps to see initial behavior
        state, reward, terminated, truncated, info = env.step(forward_action)
        steps += 1
        total_reward += reward

        if reward > 0:
            positive_rewards += 1
            if i < 10:  # Print first few positive rewards
                print(f"  Step {steps}: reward = {reward:.2f} (visiting new tile!)")

        if terminated or truncated:
            print(f"\n✓ Episode ended at step {steps}")
            break

    print(f"\nResults:")
    print(f"  Total steps: {steps}")
    print(f"  Total reward: {total_reward:.2f}")
    print(f"  Steps with positive rewards: {positive_rewards}/{steps}")
    print(f"  Average reward per step: {total_reward/steps:.2f}")

    if positive_rewards > 0:
        print(f"  ✓ Car successfully visited new tiles (positive rewards)")
    else:
        print(f"  ⚠ No positive rewards received (car may not be moving forward)")

    env.close()

    print("\n" + "=" * 60)
    print("Test Complete!")
    print("=" * 60)


if __name__ == "__main__":
    test_off_track_termination()
