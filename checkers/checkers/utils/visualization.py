"""
Visualization utilities for training metrics.
"""

import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Optional


def plot_training_metrics(
    log_file: str = "logs/training_log.csv",
    save_dir: Optional[str] = None
):
    """
    Plot training metrics from CSV log file.

    Args:
        log_file: Path to training log CSV
        save_dir: Directory to save plots (if None, just display)
    """
    # Read log file
    log_path = Path(log_file)
    if not log_path.exists():
        print(f"Log file not found: {log_file}")
        return

    df = pd.read_csv(log_path)

    if len(df) == 0:
        print("No data in log file")
        return

    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Training Metrics', fontsize=16, fontweight='bold')

    # Plot 1: Losses
    ax = axes[0, 0]
    ax.plot(df['iteration'], df['total_loss'], label='Total Loss', linewidth=2)
    ax.plot(df['iteration'], df['policy_loss'], label='Policy Loss', alpha=0.7)
    ax.plot(df['iteration'], df['value_loss'], label='Value Loss', alpha=0.7)
    ax.set_xlabel('Iteration')
    ax.set_ylabel('Loss')
    ax.set_title('Training Losses')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Plot 2: Win Rate
    ax = axes[0, 1]
    eval_data = df[df['eval_win_rate'] >= 0]
    if len(eval_data) > 0:
        ax.plot(eval_data['iteration'], eval_data['eval_win_rate'],
                marker='o', linewidth=2, markersize=6)
        ax.axhline(y=0.55, color='r', linestyle='--', label='Promotion Threshold', alpha=0.7)
        ax.axhline(y=0.5, color='gray', linestyle=':', label='Even', alpha=0.5)
    ax.set_xlabel('Iteration')
    ax.set_ylabel('Win Rate')
    ax.set_title('Evaluation Win Rate vs Best Model')
    ax.set_ylim([0, 1])
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Plot 3: Buffer Size
    ax = axes[1, 0]
    ax.plot(df['iteration'], df['buffer_size'], linewidth=2, color='green')
    ax.set_xlabel('Iteration')
    ax.set_ylabel('Number of Examples')
    ax.set_title('Replay Buffer Size')
    ax.grid(True, alpha=0.3)

    # Plot 4: Timing
    ax = axes[1, 1]
    ax.plot(df['iteration'], df['time_selfplay'], label='Self-Play', alpha=0.7)
    ax.plot(df['iteration'], df['time_training'], label='Training', alpha=0.7)
    eval_times = df[df['time_eval'] > 0]
    if len(eval_times) > 0:
        ax.scatter(eval_times['iteration'], eval_times['time_eval'],
                  label='Evaluation', alpha=0.7, s=50)
    ax.set_xlabel('Iteration')
    ax.set_ylabel('Time (seconds)')
    ax.set_title('Iteration Timing')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    # Save or display
    if save_dir:
        save_path = Path(save_dir)
        save_path.mkdir(parents=True, exist_ok=True)
        output_file = save_path / "training_metrics.png"
        plt.savefig(output_file, dpi=150, bbox_inches='tight')
        print(f"Saved plot to: {output_file}")
    else:
        plt.show()

    plt.close()


def print_training_summary(log_file: str = "logs/training_log.csv"):
    """
    Print summary statistics from training log.

    Args:
        log_file: Path to training log CSV
    """
    log_path = Path(log_file)
    if not log_path.exists():
        print(f"Log file not found: {log_file}")
        return

    df = pd.read_csv(log_path)

    if len(df) == 0:
        print("No data in log file")
        return

    print("\n" + "=" * 60)
    print("Training Summary")
    print("=" * 60)

    print(f"\nIterations completed: {len(df)}")
    print(f"Total games played: {df['games_played'].sum()}")
    print(f"Final buffer size: {df['buffer_size'].iloc[-1]:,}")

    print(f"\nFinal losses:")
    print(f"  Total: {df['total_loss'].iloc[-1]:.4f}")
    print(f"  Policy: {df['policy_loss'].iloc[-1]:.4f}")
    print(f"  Value: {df['value_loss'].iloc[-1]:.4f}")

    eval_data = df[df['eval_win_rate'] >= 0]
    if len(eval_data) > 0:
        print(f"\nEvaluation results:")
        print(f"  Evaluations: {len(eval_data)}")
        print(f"  Best win rate: {eval_data['eval_win_rate'].max():.2%}")
        print(f"  Latest win rate: {eval_data['eval_win_rate'].iloc[-1]:.2%}")

        promotions = (eval_data['eval_win_rate'] >= 0.55).sum()
        print(f"  Model promotions: {promotions}")

    print(f"\nAverage times per iteration:")
    print(f"  Self-play: {df['time_selfplay'].mean():.1f}s")
    print(f"  Training: {df['time_training'].mean():.1f}s")
    if 'time_eval' in df.columns:
        eval_times = df[df['time_eval'] > 0]['time_eval']
        if len(eval_times) > 0:
            print(f"  Evaluation: {eval_times.mean():.1f}s")

    total_time = (df['time_selfplay'].sum() + df['time_training'].sum() +
                  df['time_eval'].sum())
    print(f"\nTotal training time: {total_time/3600:.2f} hours")

    print("=" * 60)


# Example usage
if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        log_file = sys.argv[1]
    else:
        log_file = "logs/training_log.csv"

    # Print summary
    print_training_summary(log_file)

    # Create plots
    print("\nGenerating plots...")
    plot_training_metrics(log_file, save_dir="logs/plots")

    print("\n✓ Visualization complete!")
