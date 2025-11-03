"""
Quick speed test for synthetic mode.
Tests how many steps/second we can achieve.
"""

import time
import numpy as np
from preprocessing import make_carracing_env

def test_speed(n_steps=500):
    """Test speed over n_steps."""
    print(f"Testing synthetic mode speed over {n_steps} steps...")

    env = make_carracing_env(
        state_mode='synthetic',
        stack_size=4,
        render_mode=None
    )

    env.reset()

    start_time = time.time()

    for i in range(n_steps):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, _ = env.step(action)

        if terminated or truncated:
            env.reset()

    elapsed = time.time() - start_time
    steps_per_sec = n_steps / elapsed

    print(f"Completed {n_steps} steps in {elapsed:.2f} seconds")
    print(f"Speed: {steps_per_sec:.1f} steps/second")

    env.close()
    return steps_per_sec

if __name__ == "__main__":
    speed = test_speed(500)

    print("\nComparison:")
    print(f"  Visual mode:   ~57 steps/sec")
    print(f"  Vector mode:   ~313 steps/sec")
    print(f"  Synthetic mode: {speed:.1f} steps/sec")

    if speed > 100:
        print(f"\n✅ Synthetic mode is {speed/57:.1f}x faster than visual!")
    else:
        print(f"\n❌ Synthetic mode is still slow ({speed:.1f} steps/sec)")
