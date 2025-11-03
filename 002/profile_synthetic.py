"""
Profile synthetic mode to find bottlenecks.
"""

import time
import numpy as np
from preprocessing import make_carracing_env

def profile_env_step():
    """Profile a single environment step."""
    env = make_carracing_env(
        state_mode='synthetic',
        stack_size=4,
        render_mode=None
    )

    env.reset()

    # Warm up
    for _ in range(10):
        env.step(env.action_space.sample())

    # Profile 100 steps
    n_steps = 100
    times = []

    for _ in range(n_steps):
        start = time.perf_counter()
        obs, reward, terminated, truncated, _ = env.step(env.action_space.sample())
        elapsed = time.perf_counter() - start
        times.append(elapsed)

        if terminated or truncated:
            env.reset()

    avg_time = np.mean(times) * 1000  # milliseconds
    min_time = np.min(times) * 1000
    max_time = np.max(times) * 1000

    print(f"Environment step time:")
    print(f"  Average: {avg_time:.2f} ms")
    print(f"  Min: {min_time:.2f} ms")
    print(f"  Max: {max_time:.2f} ms")
    print(f"  Steps/sec (from avg): {1000/avg_time:.1f}")

    env.close()

if __name__ == "__main__":
    profile_env_step()
