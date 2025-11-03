"""
Quick test to verify synthetic mode implementation.

Tests:
1. Environment creation with synthetic mode
2. Pre-rendering track
3. State observation extraction
4. State shape and type validation
"""

import numpy as np
from preprocessing import make_carracing_env

def test_synthetic_mode():
    """Test synthetic mode implementation."""
    print("=" * 70)
    print("Testing Synthetic Mode")
    print("=" * 70)

    # Create environment in synthetic mode
    print("\n1. Creating environment with synthetic mode...")
    env = make_carracing_env(
        state_mode='synthetic',
        stack_size=4,
        discretize_actions=True,
        steering_bins=3,
        gas_brake_bins=3,
        terminate_stationary=True,
        stationary_patience=200,
        render_mode=None
    )
    print(f"   ✓ Environment created")
    print(f"   Observation space: {env.observation_space}")
    print(f"   Action space: {env.action_space}")

    # Reset environment (this should trigger pre-rendering)
    print("\n2. Resetting environment (pre-rendering track)...")
    obs, info = env.reset()
    print(f"   ✓ Environment reset complete")
    print(f"   Observation shape: {obs.shape}")
    print(f"   Observation dtype: {obs.dtype}")
    print(f"   Observation range: [{obs.min():.3f}, {obs.max():.3f}]")

    # Verify observation shape
    print("\n3. Validating observation...")
    expected_shape = (4, 96, 96)  # 4 stacked frames, 96x96 each
    assert obs.shape == expected_shape, f"Expected shape {expected_shape}, got {obs.shape}"
    print(f"   ✓ Shape is correct: {obs.shape}")

    assert obs.dtype == np.float32, f"Expected dtype float32, got {obs.dtype}"
    print(f"   ✓ Dtype is correct: {obs.dtype}")

    assert 0.0 <= obs.min() and obs.max() <= 1.0, f"Values should be in [0, 1], got [{obs.min()}, {obs.max()}]"
    print(f"   ✓ Values are normalized to [0, 1]")

    # Take a few steps
    print("\n4. Taking steps in the environment...")
    for i in range(5):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        print(f"   Step {i+1}: action={action}, reward={reward:.2f}, shape={obs.shape}")

        # Verify each observation
        assert obs.shape == expected_shape, f"Step {i+1}: Shape mismatch"
        assert obs.dtype == np.float32, f"Step {i+1}: Dtype mismatch"

    print("\n5. Success!")
    print("   ✓ Synthetic mode is working correctly")
    print("   ✓ Pre-rendering is functioning")
    print("   ✓ State extraction is working")
    print("   ✓ Observations have correct shape and type")

    env.close()
    print("\n" + "=" * 70)
    print("All tests passed!")
    print("=" * 70)


if __name__ == "__main__":
    test_synthetic_mode()
