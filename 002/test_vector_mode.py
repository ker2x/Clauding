"""
Test script to verify vector mode optimization.

This script tests both visual and vector modes to ensure:
1. Vector mode works correctly
2. Vector mode is significantly faster than visual mode
3. Both modes produce valid states
"""

import time
import numpy as np
from preprocessing import make_carracing_env
from ddqn_agent import DDQNAgent


def test_mode(state_mode, n_steps=100):
    """
    Test a specific state mode.

    Args:
        state_mode: 'visual' or 'vector'
        n_steps: Number of steps to run

    Returns:
        Average time per step
    """
    print(f"\n{'='*60}")
    print(f"Testing {state_mode.upper()} mode")
    print(f"{'='*60}")

    # Create environment
    env = make_carracing_env(
        stack_size=4,
        discretize_actions=True,
        terminate_stationary=True,
        render_mode=None,
        state_mode=state_mode
    )

    n_actions = env.action_space.n
    state_shape = env.observation_space.shape

    print(f"State shape: {state_shape}")
    print(f"Number of actions: {n_actions}")

    # Create agent
    agent = DDQNAgent(
        state_shape=state_shape,
        n_actions=n_actions,
        state_mode=state_mode
    )

    # Run steps and measure time
    state, _ = env.reset()
    print(f"Initial state shape: {state.shape}")
    print(f"Initial state dtype: {state.dtype}")
    if state_mode == 'vector':
        print(f"Initial state sample: {state[:5]}")  # First 5 values

    print(f"\nRunning {n_steps} steps...")
    start_time = time.time()

    for step in range(n_steps):
        action = agent.select_action(state, training=True)
        next_state, reward, terminated, truncated, _ = env.step(action)

        if terminated or truncated:
            state, _ = env.reset()
        else:
            state = next_state

    elapsed_time = time.time() - start_time
    avg_time_per_step = elapsed_time / n_steps
    steps_per_second = n_steps / elapsed_time

    print(f"\nResults:")
    print(f"  Total time: {elapsed_time:.2f} seconds")
    print(f"  Time per step: {avg_time_per_step*1000:.2f} ms")
    print(f"  Steps per second: {steps_per_second:.1f}")

    env.close()

    return avg_time_per_step


def main():
    """Main test function."""
    print("="*60)
    print("Vector Mode Optimization Test")
    print("="*60)
    print("\nThis test compares the performance of visual vs vector modes.")
    print("Vector mode should be 3-5x faster than visual mode.")

    n_steps = 200  # Enough steps to get reliable timing

    # Test vector mode (should be fast)
    vector_time = test_mode('vector', n_steps)

    # Test visual mode (will be slower)
    visual_time = test_mode('visual', n_steps)

    # Compare
    print(f"\n{'='*60}")
    print("COMPARISON")
    print(f"{'='*60}")
    print(f"Visual mode: {visual_time*1000:.2f} ms/step")
    print(f"Vector mode: {vector_time*1000:.2f} ms/step")
    speedup = visual_time / vector_time
    print(f"\nSpeedup: {speedup:.2f}x faster with vector mode")

    if speedup >= 2.5:
        print("✓ SUCCESS: Vector mode is significantly faster!")
    elif speedup >= 1.5:
        print("⚠ WARNING: Vector mode is faster but not as much as expected")
    else:
        print("✗ FAIL: Vector mode is not showing expected speedup")

    print(f"\n{'='*60}")
    print("For training, use: python train.py --state-mode vector")
    print("For watching, scripts automatically use visual mode")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
