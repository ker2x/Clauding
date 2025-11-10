#!/usr/bin/env python3
"""
Quick test script for the new waypoint-based reward system.
Tests that waypoints are properly initialized and rewards are calculated.
"""

import numpy as np
from env.car_racing import CarRacing

def test_waypoint_initialization():
    """Test that waypoints are properly initialized."""
    print("Testing waypoint initialization...")
    env = CarRacing(verbose=True, state_mode="vector")
    obs, info = env.reset()

    print(f"\n✓ Environment created successfully")
    print(f"  Track length: {len(env.track)} tiles")
    print(f"  Number of waypoints: {env.num_waypoints}")
    print(f"  Waypoint spacing: ~{len(env.track) // env.num_waypoints} tiles")
    print(f"  Waypoint indices: {env.waypoint_indices[:5]}... (showing first 5)")
    print(f"  Next waypoint to reach: {env.next_waypoint_index}")
    print(f"  Waypoint reward: {env.waypoint_reward}")
    print(f"  Waypoint distance threshold: {env.waypoint_distance_threshold}m")

    return env

def test_basic_episode():
    """Run a short episode and check reward structure."""
    print("\n" + "="*70)
    print("Testing basic episode with random actions...")
    print("="*70)

    env = CarRacing(verbose=False, state_mode="vector")
    obs, info = env.reset()

    total_reward = 0
    waypoints_reached = 0
    steps = 0
    max_steps = 100

    print(f"\nRunning {max_steps} steps with random actions...")

    for step in range(max_steps):
        # Random action [steering, gas/brake]
        action = np.random.uniform(-1, 1, size=2)
        obs, reward, terminated, truncated, info = env.step(action)

        # Check if waypoint was reached this step
        if reward > 10:  # Likely a waypoint reward (50 - penalties)
            waypoints_reached += 1
            print(f"  Step {step}: Waypoint {env.next_waypoint_index}/{env.num_waypoints} "
                  f"reached! Reward: {reward:.2f}")

        total_reward += reward
        steps += 1

        if terminated or truncated:
            print(f"\nEpisode ended at step {steps}")
            if terminated and info.get("lap_finished"):
                print("  ✓ Lap completed!")
            elif info.get("off_track"):
                print("  ✗ Went off track")
            break

    print(f"\n{'='*70}")
    print("Episode Summary:")
    print(f"  Steps: {steps}")
    print(f"  Waypoints reached: {waypoints_reached}/{env.num_waypoints}")
    print(f"  Total reward: {total_reward:.2f}")
    print(f"  Average reward per step: {total_reward/steps:.2f}")
    print(f"{'='*70}")

    env.close()

def test_reward_components():
    """Test individual reward components."""
    print("\n" + "="*70)
    print("Testing reward components...")
    print("="*70)

    env = CarRacing(verbose=False, state_mode="vector")
    obs, info = env.reset()

    print(f"\nReward configuration:")
    print(f"  WAYPOINT_REWARD: {env.waypoint_reward}")
    print(f"  LAP_COMPLETION_REWARD: {env.env.lap_completion_reward if hasattr(env, 'env') else 'N/A'}")
    print(f"  STEP_PENALTY: -1.0")
    print(f"  OFFTRACK_PENALTY: -2.0")
    print(f"  OFFTRACK_THRESHOLD: 2 wheels")

    print(f"\nExpected rewards for complete lap:")
    print(f"  Waypoints: {env.num_waypoints} × {env.waypoint_reward} = {env.num_waypoints * env.waypoint_reward}")
    print(f"  Lap completion: 500")
    print(f"  Total positive: {env.num_waypoints * env.waypoint_reward + 500}")
    print(f"  Minus step penalties: depends on speed (~-600 to -1500)")
    print(f"  Net expected: ~{env.num_waypoints * env.waypoint_reward + 500 - 1000} (for medium speed)")

    env.close()

if __name__ == "__main__":
    print("="*70)
    print("WAYPOINT-BASED REWARD SYSTEM TEST")
    print("="*70)

    # Test 1: Initialization
    env = test_waypoint_initialization()
    env.close()

    # Test 2: Basic episode
    test_basic_episode()

    # Test 3: Reward components
    test_reward_components()

    print("\n" + "="*70)
    print("✓ All tests completed!")
    print("="*70)
