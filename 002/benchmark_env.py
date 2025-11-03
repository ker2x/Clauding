"""
Benchmark script to measure environment performance improvements.

Compares:
1. Optimized headless mode (render_mode=None)
2. Standard rendering mode (render_mode='rgb_array')
"""

import time
import numpy as np
from preprocessing import make_carracing_env


def benchmark_env(render_mode, n_steps=1000):
    """
    Benchmark environment performance.

    Args:
        render_mode: 'rgb_array', None
        n_steps: Number of steps to run

    Returns:
        Steps per second
    """
    env = make_carracing_env(
        stack_size=4,
        discretize_actions=True,
        steering_bins=3,
        gas_brake_bins=3,
        terminate_stationary=False,  # No early termination for fair comparison
        render_mode=render_mode
    )

    # Warmup
    env.reset()
    for _ in range(50):
        action = env.action_space.sample()
        env.step(action)

    # Benchmark
    env.reset()
    start_time = time.time()

    for _ in range(n_steps):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)

        if terminated or truncated:
            env.reset()

    elapsed = time.time() - start_time
    steps_per_sec = n_steps / elapsed

    env.close()
    return steps_per_sec


if __name__ == "__main__":
    print("=" * 60)
    print("CarRacing Environment Performance Benchmark")
    print("=" * 60)

    n_steps = 1000
    print(f"\nRunning {n_steps} steps for each mode...\n")

    # Benchmark headless mode (optimized)
    print("Testing OPTIMIZED headless mode (render_mode=None)...")
    headless_fps = benchmark_env(render_mode=None, n_steps=n_steps)
    print(f"  ✓ Headless mode: {headless_fps:.1f} steps/sec")

    # Benchmark standard rendering
    print("\nTesting standard rendering (render_mode='rgb_array')...")
    rendering_fps = benchmark_env(render_mode='rgb_array', n_steps=n_steps)
    print(f"  ✓ Rendering mode: {rendering_fps:.1f} steps/sec")

    # Calculate speedup
    speedup = headless_fps / rendering_fps
    print("\n" + "=" * 60)
    print(f"SPEEDUP: {speedup:.2f}x faster with optimized headless mode")
    print("=" * 60)

    # Time estimates
    print("\nTime to run 1000 training episodes (~500 steps each):")
    print(f"  Standard rendering: {500_000 / rendering_fps / 3600:.1f} hours")
    print(f"  Optimized headless: {500_000 / headless_fps / 3600:.1f} hours")
    print(f"  Time saved: {(500_000 / rendering_fps - 500_000 / headless_fps) / 3600:.1f} hours")
