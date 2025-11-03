"""
Check if there's a dtype issue causing slowdowns.
"""

from preprocessing import make_carracing_env
import numpy as np

def check_state_properties():
    """Check state properties for each mode."""

    modes = ['visual', 'synthetic', 'vector']

    for mode in modes:
        print(f"\n{mode.upper()} MODE:")
        print("=" * 50)

        if mode == 'vector':
            env = make_carracing_env(state_mode=mode, render_mode=None)
        else:
            env = make_carracing_env(state_mode=mode, stack_size=4, render_mode=None)

        obs, _ = env.reset()

        print(f"  Shape: {obs.shape}")
        print(f"  Dtype: {obs.dtype}")
        print(f"  Memory: {obs.nbytes / 1024:.2f} KB")
        print(f"  Range: [{obs.min():.3f}, {obs.max():.3f}]")
        print(f"  Is contiguous: {obs.flags['C_CONTIGUOUS']}")
        print(f"  Is F-contiguous: {obs.flags['F_CONTIGUOUS']}")

        # Take a step
        obs, _, _, _, _ = env.step(env.action_space.sample())
        print(f"\n  After step:")
        print(f"    Shape: {obs.shape}")
        print(f"    Dtype: {obs.dtype}")
        print(f"    Is contiguous: {obs.flags['C_CONTIGUOUS']}")

        env.close()

if __name__ == "__main__":
    check_state_properties()
