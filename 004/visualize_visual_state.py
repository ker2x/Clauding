"""
Visualize what the model sees in visual state mode.

This script demonstrates the visual preprocessing pipeline:
1. RGB headless rendering (96×96×3) - no GUI/telemetry overlays
2. Grayscale conversion (96×96)
3. Normalization [0, 255] → [0.0, 1.0]
4. Frame stacking (4 frames → 4×96×96)

IMPORTANT: Uses headless rendering (render_mode=None) to show the exact same
view the model sees during training - pure camera view without any telemetry bars.

The output shows all 4 stacked frames side by side, which is what the
visual actor network receives as input.

Usage:
    # Generate visualization with random actions
    python visualize_visual_state.py

    # Take more steps to see frame changes
    python visualize_visual_state.py --steps 50

    # Save to custom location
    python visualize_visual_state.py --output my_visual_state.png
"""

import argparse
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

from preprocessing import make_carracing_env


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description='Visualize what the model sees in visual state mode'
    )

    parser.add_argument('--steps', type=int, default=20,
                        help='Number of steps to take before capturing (default: 20)')
    parser.add_argument('--output', type=str, default='visual_state_output.png',
                        help='Output file path (default: visual_state_output.png)')
    parser.add_argument('--seed', type=int, default=None,
                        help='Random seed for reproducibility')
    parser.add_argument('--display', action='store_true',
                        help='Display the plot interactively (default: just save)')

    return parser.parse_args()


def visualize_visual_state(state, output_path, display=False):
    """
    Visualize the 4-frame stacked visual state.

    Args:
        state: Numpy array of shape (4, 96, 96) - 4 stacked grayscale frames
        output_path: Where to save the visualization
        display: Whether to display the plot interactively
    """
    assert state.shape == (4, 96, 96), f"Expected shape (4, 96, 96), got {state.shape}"

    # Create figure with custom layout
    fig = plt.figure(figsize=(16, 5))
    gs = GridSpec(2, 5, figure=fig, hspace=0.3, wspace=0.3)

    # Main title
    fig.suptitle('Visual State Mode: What the Model Sees', fontsize=16, fontweight='bold')

    # Plot all 4 frames
    for i in range(4):
        ax = fig.add_subplot(gs[0, i])
        ax.imshow(state[i], cmap='gray', vmin=0.0, vmax=1.0)
        ax.set_title(f'Frame t-{3-i}' if i < 3 else 'Frame t (current)', fontsize=10)
        ax.axis('off')

    # Plot combined view (average of all frames)
    ax_avg = fig.add_subplot(gs[0, 4])
    avg_frame = np.mean(state, axis=0)
    ax_avg.imshow(avg_frame, cmap='viridis', vmin=0.0, vmax=1.0)
    ax_avg.set_title('Average of 4 Frames', fontsize=10)
    ax_avg.axis('off')

    # Add detailed stats
    stats_text = (
        f"State Shape: {state.shape}\n"
        f"Data Type: {state.dtype}\n"
        f"Value Range: [{state.min():.3f}, {state.max():.3f}]\n"
        f"Mean: {state.mean():.3f}, Std: {state.std():.3f}\n\n"
        f"Preprocessing Pipeline:\n"
        f"  1. RGB headless render (96×96×3) - no telemetry!\n"
        f"  2. Grayscale conversion\n"
        f"  3. Normalize to [0.0, 1.0]\n"
        f"  4. Stack 4 frames\n\n"
        f"Why 4 frames?\n"
        f"  • Captures motion/velocity\n"
        f"  • Temporal context for CNN\n"
        f"  • Standard for Atari/vision RL"
    )

    ax_text = fig.add_subplot(gs[1, :])
    ax_text.text(0.05, 0.5, stats_text, fontsize=10, family='monospace',
                 verticalalignment='center', transform=ax_text.transAxes)
    ax_text.axis('off')

    # Save figure
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"✓ Visualization saved to: {output_path}")

    # Display if requested
    if display:
        plt.show()
    else:
        plt.close(fig)


def main():
    """Generate and visualize visual state."""
    args = parse_args()

    print("=" * 60)
    print("Visual State Mode Visualization")
    print("=" * 60)

    # Create environment in visual mode
    print("\nCreating CarRacing environment in visual mode...")
    env = make_carracing_env(
        stack_size=4,
        terminate_stationary=True,
        render_mode=None,  # Use headless rendering (same as training) - no telemetry
        state_mode='visual'
    )

    print(f"✓ Environment created")
    print(f"  State shape: {env.observation_space.shape}")
    print(f"  Action space: {env.action_space.shape}")

    # Reset environment
    if args.seed is not None:
        print(f"\nUsing random seed: {args.seed}")
        state, _ = env.reset(seed=args.seed)
    else:
        state, _ = env.reset()

    print(f"\n✓ Environment reset")
    print(f"  Initial state shape: {state.shape}")
    print(f"  Initial state dtype: {state.dtype}")
    print(f"  Initial value range: [{state.min():.3f}, {state.max():.3f}]")

    # Take some steps to get interesting frames
    print(f"\nTaking {args.steps} steps with random actions...")
    for step in range(args.steps):
        action = env.action_space.sample()
        state, reward, terminated, truncated, _ = env.step(action)

        if terminated or truncated:
            print(f"  Episode terminated at step {step + 1}, resetting...")
            state, _ = env.reset()

    print(f"✓ Steps completed")

    # Visualize the state
    print(f"\nGenerating visualization...")
    visualize_visual_state(state, args.output, display=args.display)

    # Cleanup
    env.close()

    print("\n" + "=" * 60)
    print("Done!")
    print("=" * 60)
    print(f"\nThe visualization shows:")
    print(f"  • 4 grayscale frames (96×96 each)")
    print(f"  • Values normalized to [0.0, 1.0]")
    print(f"  • Frame stacking provides temporal information")
    print(f"  • This (4×96×96) tensor is fed to the VisualActor CNN")


if __name__ == "__main__":
    main()
