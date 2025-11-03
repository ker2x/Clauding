"""
Visualize Atari Preprocessing

This script helps you understand what the agent actually "sees" after preprocessing.
It's useful for:
1. Understanding why we preprocess frames
2. Debugging preprocessing issues
3. Learning about RL input representations
"""

import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
from preprocessing import make_atari_env

# Register ALE environments (required for gymnasium 1.0+)
try:
    import ale_py
    gym.register_envs(ale_py)
except Exception:
    pass


def visualize_preprocessing(env_name='ALE/Breakout-v5', num_steps=10):
    """
    Visualize the preprocessing pipeline

    Shows:
    1. Original frame (what we see)
    2. Processed frame (what the agent sees)
    3. Stacked frames (4 frames for motion information)
    """
    # Create raw environment (no preprocessing)
    raw_env = gym.make(env_name, render_mode='rgb_array')

    # Create preprocessed environment
    processed_env = make_atari_env(env_name, clip_rewards=False)

    # Reset both environments with same seed for consistency
    seed = 42
    raw_obs, _ = raw_env.reset(seed=seed)
    processed_obs, _ = processed_env.reset(seed=seed)

    # Take a few random steps to get interesting frames
    # Sample from processed_env since it has fewer actions (3 vs 4)
    for _ in range(num_steps):
        action = processed_env.action_space.sample()
        # Map processed action back to raw action space
        # Processed: 0=NOOP, 1=RIGHT, 2=LEFT
        # Raw: 0=NOOP, 1=FIRE, 2=RIGHT, 3=LEFT
        raw_action_mapping = {0: 0, 1: 2, 2: 3}
        raw_action = raw_action_mapping[action]

        raw_obs, _, terminated, truncated, _ = raw_env.step(raw_action)
        processed_obs, _, terminated2, truncated2, _ = processed_env.step(action)

        if terminated or truncated:
            raw_obs, _ = raw_env.reset()
            processed_obs, _ = processed_env.reset()

    # Visualize
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle('Atari Preprocessing Visualization', fontsize=16, fontweight='bold')

    # Original frame
    axes[0, 0].imshow(raw_obs)
    axes[0, 0].set_title(f'Original Frame\nShape: {raw_obs.shape}\nRGB, High Resolution')
    axes[0, 0].axis('off')

    # Info text
    info_text = (
        "Raw Atari Output:\n"
        f"• Shape: {raw_obs.shape}\n"
        "• 210x160 RGB\n"
        "• 60 Hz frame rate\n"
        "• Full color information\n\n"
        "Why preprocess?\n"
        "• Reduce computation\n"
        "• Remove unnecessary info\n"
        "• Capture temporal info"
    )
    axes[0, 1].text(0.1, 0.5, info_text, fontsize=11, verticalalignment='center',
                    family='monospace', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    axes[0, 1].axis('off')

    # Processed frame (just one frame from the stack)
    current_frame = processed_obs[0]  # Most recent frame
    axes[0, 2].imshow(current_frame, cmap='gray')
    axes[0, 2].set_title(f'Processed Frame\nShape: {current_frame.shape}\nGrayscale, 84x84')
    axes[0, 2].axis('off')

    # Show all 4 stacked frames
    for i in range(4):
        axes[1, 0] if i < 2 else axes[1, 1]
        ax = axes[1, i // 2] if i < 2 else axes[1, 1 + (i - 2) // 2]

        if i < 2:
            ax = axes[1, 0] if i == 0 else axes[1, 1]
        else:
            ax = axes[1, 2]
            if i == 2:
                # For the last plot, show all 4 frames side by side
                combined = np.concatenate([processed_obs[j] for j in range(4)], axis=1)
                ax.imshow(combined, cmap='gray')
                ax.set_title('All 4 Stacked Frames\n(captures motion over time)')
                ax.axis('off')
                break

        ax.imshow(processed_obs[i], cmap='gray')
        ax.set_title(f'Frame t-{3-i}{"" if i == 0 else f" (older)"}')
        ax.axis('off')

    # Info about frame stacking
    stack_info = (
        f"Frame Stack:\n"
        f"• Shape: {processed_obs.shape}\n"
        f"• 4 consecutive frames\n"
        f"• Provides motion information\n"
        f"• Agent can infer velocity\n"
        f"  and direction from stack\n\n"
        f"Example:\n"
        f"• Ball moving right:\n"
        f"  Position shifts across frames\n"
        f"• Paddle velocity:\n"
        f"  Visible from frame differences"
    )
    axes[1, 2].text(0.1, 0.5, stack_info, fontsize=10, verticalalignment='center',
                    family='monospace', bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5))
    axes[1, 2].axis('off')

    # Add better layout for frame visualization
    fig.delaxes(axes[1, 2])

    # Create new axes for the 4 frames
    for i in range(4):
        ax = fig.add_subplot(2, 6, 7 + i)
        ax.imshow(processed_obs[i], cmap='gray')
        ax.set_title(f't-{3-i}', fontsize=10)
        ax.axis('off')

    # Add info box - using subplot2grid for proper spanning
    ax = plt.subplot2grid((2, 6), (1, 4), colspan=2, fig=fig)
    stack_info = (
        "Frame Stacking:\n"
        "• 4 frames capture motion\n"
        "• Ball trajectory visible\n"
        "• Paddle velocity apparent"
    )
    ax.text(0.1, 0.5, stack_info, fontsize=10, verticalalignment='center',
            family='monospace', bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5))
    ax.axis('off')

    plt.tight_layout()
    plt.savefig('preprocessing_visualization.png', dpi=150, bbox_inches='tight')
    print("Visualization saved to 'preprocessing_visualization.png'")
    plt.show()

    raw_env.close()
    processed_env.close()


def compare_sizes():
    """
    Print comparison of data sizes before and after preprocessing
    Helps understand the computational benefits
    """
    print("\n" + "="*60)
    print("PREPROCESSING BENEFITS")
    print("="*60)

    # Raw frame
    raw_size = 210 * 160 * 3  # RGB
    print(f"\nRaw Frame:")
    print(f"  Shape: (210, 160, 3)")
    print(f"  Total values: {raw_size:,}")
    print(f"  Memory: ~{raw_size / 1024:.1f} KB per frame")

    # Processed frame
    processed_size = 84 * 84  # Grayscale
    print(f"\nProcessed Frame:")
    print(f"  Shape: (84, 84)")
    print(f"  Total values: {processed_size:,}")
    print(f"  Memory: ~{processed_size / 1024:.1f} KB per frame")

    # Reduction
    reduction = (1 - processed_size / raw_size) * 100
    print(f"\nReduction: {reduction:.1f}%")

    # Stack
    stack_size = processed_size * 4
    print(f"\nWith Frame Stacking (4 frames):")
    print(f"  Shape: (4, 84, 84)")
    print(f"  Total values: {stack_size:,}")
    print(f"  Memory: ~{stack_size / 1024:.1f} KB per observation")

    print("\n" + "="*60)


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Visualize Atari preprocessing')
    parser.add_argument('--env', type=str, default='ALE/Breakout-v5',
                       help='Atari environment name')
    parser.add_argument('--steps', type=int, default=10,
                       help='Number of random steps to take before visualization')
    parser.add_argument('--compare-sizes', action='store_true',
                       help='Print size comparison')
    args = parser.parse_args()

    if args.compare_sizes:
        compare_sizes()

    visualize_preprocessing(args.env, args.steps)
