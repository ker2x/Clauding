"""
Demo script to show the new progress monitoring features.

This runs a quick 1-iteration training to demonstrate the progress bars.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import torch
import tempfile
import shutil
from config import Config
from checkers.network.resnet import CheckersNetwork, initialize_weights
from checkers.training.trainer import Trainer


def demo_progress_monitoring():
    """Demonstrate the progress monitoring features."""
    print("\n" + "=" * 70)
    print("📊 Progress Monitoring Demo")
    print("=" * 70)
    print("\nThis demo shows the new real-time progress features:")
    print("  ✓ Progress bars for self-play games")
    print("  ✓ Progress bars for training steps")
    print("  ✓ Live metrics updates (loss, win rate, ETA)")
    print("  ✓ Trend indicators (↑↓→)")
    print("  ✓ Visual progress bar")
    print("\nRunning 1 training iteration with reduced settings...")
    print("=" * 70)

    # Create temporary directories
    temp_dir = tempfile.mkdtemp(prefix="checkers_demo_")
    Config.CHECKPOINT_DIR = os.path.join(temp_dir, "checkpoints")
    Config.LOG_DIR = os.path.join(temp_dir, "logs")

    try:
        # Reduce scale for quick demo
        Config.GAMES_PER_ITERATION = 5
        Config.TRAINING_STEPS_PER_ITERATION = 20
        Config.MCTS_SIMS_SELFPLAY = 50  # Faster
        Config.BUFFER_SIZE = 10000
        Config.EVAL_FREQUENCY = 1  # Evaluate on first iteration
        Config.EVAL_GAMES = 2
        Config.SAVE_FREQUENCY = 1

        # Get device
        device = Config.get_device()

        # Create network
        network = CheckersNetwork(
            num_filters=Config.NUM_FILTERS,
            num_res_blocks=Config.NUM_RES_BLOCKS,
            policy_size=Config.POLICY_SIZE
        )
        network.apply(initialize_weights)

        # Create trainer
        trainer = Trainer(
            network=network,
            config=Config,
            device=device,
            resume_from=None
        )

        # Run 1 training iteration
        trainer.train(num_iterations=1)

        print("\n" + "=" * 70)
        print("✓ Demo Complete!")
        print("=" * 70)
        print("\nYou can see:")
        print("  • Progress bars showing game/step completion")
        print("  • Real-time loss updates during training")
        print("  • ETA calculation based on iteration speed")
        print("  • Overall progress percentage")
        print("  • Trend indicators for metrics")
        print("\nFor full training, run:")
        print("  python scripts/train.py --iterations 100")

    finally:
        # Cleanup
        print(f"\nCleaning up: {temp_dir}")
        shutil.rmtree(temp_dir, ignore_errors=True)


if __name__ == "__main__":
    demo_progress_monitoring()
