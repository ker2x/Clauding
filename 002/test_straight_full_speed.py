"""
Simple test: Go straight at full speed.
The track always has curves, so the car should go off-track quickly.

This tests:
1. Off-track detection and negative rewards (should be -100)
2. Episode termination when off-track
"""

import gymnasium as gym
import numpy as np
from preprocessing import make_carracing_env


def test_straight_full_speed():
    """Go straight at full speed to force off-track."""

    print("=" * 60)
    print("Test: STRAIGHT + FULL GAS (will hit curve and go off-track)")
    print("=" * 60)

    print("\nGoing straight at full gas - should go off-track and receive -100 penalty")
    print("-" * 60)
    env = make_carracing_env(
        terminate_stationary=False,
        render_mode=None
    )

    state, _ = env.reset()
    total_reward = 0
    steps = 0
    rewards_history = []
    min_reward = 0

    # Action 7 = STRAIGHT+GAS
    action = 7

    print("Going STRAIGHT+GAS until episode ends or 500 steps...")
    print("Expected: Hit curve around step 50-150, go off-track, get -100 penalty\n")

    for i in range(500):
        state, reward, terminated, truncated, info = env.step(action)
        steps += 1
        total_reward += reward
        rewards_history.append(reward)
        min_reward = min(min_reward, reward)

        # Print first 20 rewards to see pattern
        if i < 20:
            print(f"  Step {steps}: reward = {reward:7.2f}")
        # Then print only significant events
        elif reward < -10.0:
            print(f"  Step {steps}: reward = {reward:7.2f} *** OFF-TRACK PENALTY ***")
        elif reward > 5.0 and i % 10 == 0:
            print(f"  Step {steps}: reward = {reward:7.2f} (still on track)")

        if terminated or truncated:
            termination_type = "terminated" if terminated else "truncated"
            print(f"\n✓ Episode ended at step {steps} ({termination_type})")
            break
    else:
        print(f"\n⚠ Episode did NOT terminate after {steps} steps")

    print(f"\n{'='*60}")
    print("Results:")
    print(f"{'='*60}")
    print(f"  Total steps: {steps}")
    print(f"  Total reward: {total_reward:.2f}")
    print(f"  Min reward (single step): {min_reward:.2f}")
    print(f"  Max reward (single step): {max(rewards_history):.2f}")
    print(f"  Average reward per step: {total_reward/steps:.2f}")

    large_negatives = sum(1 for r in rewards_history if r < -10.0)
    catastrophic_negatives = sum(1 for r in rewards_history if r < -50.0)

    print(f"  Steps with large negative rewards (< -10.0): {large_negatives}")
    print(f"  Steps with catastrophic negative rewards (< -50.0): {catastrophic_negatives}")

    if catastrophic_negatives > 0:
        print(f"\n  ✓ CONFIRMED: Off-track gives catastrophic penalties")
        print(f"     Raw penalty magnitude: {min_reward:.2f}")
    elif large_negatives > 0:
        print(f"\n  ✓ Found off-track penalties: {min_reward:.2f}")
    else:
        print(f"\n  ⚠ No large off-track penalties found (min: {min_reward:.2f})")

    env.close()

    print("\n" + "=" * 60)
    print("Test Complete!")
    print("=" * 60)


if __name__ == "__main__":
    test_straight_full_speed()
