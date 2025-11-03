"""
Debug script to visualize what synthetic mode is actually showing the agent.
"""

import numpy as np
import matplotlib.pyplot as plt
from preprocessing import make_carracing_env

def visualize_synthetic_state():
    """Visualize what the synthetic mode shows."""
    print("Creating synthetic mode environment...")
    env = make_carracing_env(
        state_mode='synthetic',
        stack_size=4,
        render_mode=None
    )

    print("Resetting environment...")
    obs, _ = env.reset()

    print(f"Observation shape: {obs.shape}")
    print(f"Observation range: [{obs.min():.3f}, {obs.max():.3f}]")

    # Take a few steps to get some variation
    print("\nTaking steps to collect frames...")
    frames = [obs[0]]  # First frame from stack

    for i in range(10):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, _ = env.step(action)
        frames.append(obs[0])  # Latest frame from stack
        if terminated or truncated:
            break

    # Visualize frames
    print(f"\nVisualizing {len(frames)} frames...")
    fig, axes = plt.subplots(2, 5, figsize=(15, 6))
    axes = axes.flatten()

    for i, frame in enumerate(frames[:10]):
        ax = axes[i]
        ax.imshow(frame, cmap='gray', vmin=0, vmax=1)
        ax.set_title(f'Frame {i}')
        ax.axis('off')

    plt.tight_layout()
    plt.savefig('logs/synthetic_debug.png', dpi=150)
    print("Saved visualization to: logs/synthetic_debug.png")

    # Print statistics
    print("\nFrame statistics:")
    for i, frame in enumerate(frames[:5]):
        unique_vals = np.unique(frame)
        print(f"  Frame {i}: {len(unique_vals)} unique values, range [{frame.min():.3f}, {frame.max():.3f}]")

    env.close()
    print("\nDone! Check logs/synthetic_debug.png to see what the agent sees.")


if __name__ == "__main__":
    visualize_synthetic_state()
