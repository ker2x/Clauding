"""
Test to verify that episodes end when the car leaves the track
and that appropriate negative rewards are received.

This test ensures:
1. Car receives negative rewards when going off-track
2. Episodes terminate when off-track for too long
3. RewardShaping wrapper clips extreme negative rewards appropriately
"""

import numpy as np
from preprocessing import make_carracing_env


def test_off_track_termination():
    """Test that going off-track causes termination and negative rewards."""

    print("=" * 60)
    print("Testing Off-Track Termination and Rewards")
    print("=" * 60)
    print("\nNote: CarRacing-v3 may not auto-terminate on off-track.")
    print("Instead, it gives negative rewards and requires no progress for 100 frames.")

    # Create environment with reward shaping (clips to -5.0)
    print("\nTest 1: Braking at start (should go off-track via grass)")
    print("-" * 60)
    env = make_carracing_env(
        terminate_stationary=False,  # Disable so we can test off-track separately
        shape_rewards=True,          # Enable reward clipping
        render_mode=None
    )

    state, _ = env.reset()
    total_reward = 0
    steps = 0
    min_reward = 0
    max_reward = 0
    rewards_history = []

    # Strategy: Start with gas to get moving, then hard brake
    # This should cause the car to skid off track
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
        if reward < -1.0 or (reward > 1.0 and i >= 10):
            print(f"  Step {steps}: reward = {reward:.2f} {'(brake+off-track?)' if reward < 0 else ''}")

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

    # Count reward patterns
    large_negatives = sum(1 for r in rewards_history if r < -1.0)
    very_large_negatives = sum(1 for r in rewards_history if r <= -5.0)
    print(f"  Steps with large negative rewards (< -1.0): {large_negatives}")
    print(f"  Steps with very large negative rewards (<= -5.0): {very_large_negatives}")

    if very_large_negatives > 0:
        print(f"  ✓ Reward clipping triggered! (found -5.0 rewards)")

    if min_reward >= -5.0:
        print(f"  ✓ Reward clipping working (min reward >= -5.0)")
    else:
        print(f"  ⚠ WARNING: Found reward {min_reward:.2f} < -5.0 (clipping failed)")

    env.close()

    # Test 2: Random actions - should eventually go off-track
    print("\n" + "=" * 60)
    print("Test 2: Random actions (should eventually go off-track)")
    print("-" * 60)
    env = make_carracing_env(
        terminate_stationary=False,
        shape_rewards=False,  # Disable reward clipping to see raw rewards
        render_mode=None
    )

    state, _ = env.reset()
    total_reward = 0
    steps = 0
    min_reward = 0
    rewards_history = []
    np.random.seed(42)  # Reproducible

    print(f"Strategy: Random actions")
    print(f"Expected: Eventually goes off-track, shows raw negative rewards\n")

    for i in range(300):
        action = np.random.randint(0, 9)
        state, reward, terminated, truncated, info = env.step(action)
        steps += 1
        total_reward += reward
        rewards_history.append(reward)
        min_reward = min(min_reward, reward)

        # Print when we get significant negative rewards (likely off-track)
        if reward < -5.0:
            print(f"  Step {steps}: reward = {reward:.2f} (RAW off-track penalty!)")

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

    very_large_negatives = sum(1 for r in rewards_history if r < -5.0)
    extreme_negatives = sum(1 for r in rewards_history if r < -10.0)
    print(f"  Steps with very large negative rewards (< -5.0): {very_large_negatives}")
    print(f"  Steps with extreme negative rewards (< -10.0): {extreme_negatives}")

    if very_large_negatives > 0:
        print(f"  ✓ Found raw off-track penalties (< -5.0)")
    else:
        print(f"  ⚠ No large penalties found (may not have gone off-track)")

    env.close()

    # Test 3: Verify normal driving doesn't trigger early termination
    print("\n" + "=" * 60)
    print("Test 3: Normal driving (STRAIGHT+GAS)")
    print("-" * 60)
    env = make_carracing_env(
        terminate_stationary=False,
        shape_rewards=True,
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
