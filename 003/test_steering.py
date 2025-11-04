#!/usr/bin/env python3
"""
Test steering responsiveness by manually controlling the car.
"""

import numpy as np
from preprocessing import make_carracing_env

def test_steering():
    """Test if steering actually affects the car's position."""
    print("Testing steering responsiveness...")
    print("="*60)

    # Create environment
    env = make_carracing_env(
        state_mode='vector',
        render_mode=None,
        terminate_stationary=False,  # Disable early termination
    )

    # Test 1: No steering, full gas
    print("\nTest 1: No steering, full gas (should go straight)")
    state, _ = env.reset()

    straight_positions = []
    for step in range(200):
        action = np.array([0.0, 1.0, 0.0])  # No steering, full gas, no brake
        state, reward, terminated, truncated, info = env.step(action)

        # Extract car position from state if available
        # State format: [car_state (11D), track_info (5D), lookahead (20D)]
        # Car state should include position/velocity
        if step % 50 == 0:
            straight_positions.append(state[:2] if len(state) >= 2 else None)

        if terminated or truncated:
            print(f"  Episode ended at step {step}")
            break

    print(f"  Completed {step} steps with no steering")

    # Test 2: Left steering, full gas
    print("\nTest 2: Left steering, full gas (should turn left)")
    state, _ = env.reset()

    left_positions = []
    for step in range(200):
        action = np.array([-0.5, 1.0, 0.0])  # Left steering, full gas, no brake
        state, reward, terminated, truncated, info = env.step(action)

        if step % 50 == 0:
            left_positions.append(state[:2] if len(state) >= 2 else None)

        if terminated or truncated:
            print(f"  Episode ended at step {step}")
            break

    print(f"  Completed {step} steps with left steering")

    # Test 3: Right steering, full gas
    print("\nTest 3: Right steering, full gas (should turn right)")
    state, _ = env.reset()

    right_positions = []
    for step in range(200):
        action = np.array([0.5, 1.0, 0.0])  # Right steering, full gas, no brake
        state, reward, terminated, truncated, info = env.step(action)

        if step % 50 == 0:
            right_positions.append(state[:2] if len(state) >= 2 else None)

        if terminated or truncated:
            print(f"  Episode ended at step {step}")
            break

    print(f"  Completed {step} steps with right steering")

    print("\n" + "="*60)
    print("ANALYSIS:")
    print("="*60)
    print("\nIf steering works properly:")
    print("  - Left steering should move in a different direction than straight")
    print("  - Right steering should move in the opposite direction of left")
    print("\nIf steering is broken:")
    print("  - All three tests would produce similar paths")
    print("  - The car would go straight regardless of steering input")

    env.close()

if __name__ == "__main__":
    test_steering()
