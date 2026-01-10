"""
Training script for 8x8 American Checkers with Fixed Action Space.
"""

import sys
import os
import argparse
import torch
import random
import numpy as np

# Add parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from config8x8 import Config
from checkers8x8.network.resnet import CheckersNetwork, count_parameters
from checkers8x8.training.trainer import Trainer
from checkers8x8.utils.game_visualizer import GameVisualizer


def set_seed(seed: int):
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)


def main():
    parser = argparse.ArgumentParser(description='Train 8x8 Checkers AI')
    parser.add_argument('--iterations', type=int, default=100,
                       help='Target iteration number (when resuming, must be > current iteration)')
    parser.add_argument('--seed', type=int, default=None,
                       help='Random seed for reproducibility (default: random)')
    parser.add_argument('--visualize', action='store_true',
                       help='Enable real-time visualization of training')
    parser.add_argument('--resume', type=str, nargs='?', const='auto', default=None,
                       help='Resume from checkpoint (default: auto-detect latest.pt)')

    args = parser.parse_args()

    # Set seed (random by default, or use --seed for reproducibility)
    if args.seed is not None:
        seed = args.seed
        print(f"Using specified seed: {seed}")
    else:
        import time
        seed = int(time.time() * 1000) % (2**32)  # Random seed from current time
        print(f"Using random seed: {seed}")
    
    set_seed(seed)
    print(f"Random seed set to {seed}\n")

    # Get devices
    device = Config.get_device()
    selfplay_device = Config.get_selfplay_device()

    print(f"Training device: {device}")
    print(f"Self-play device: {selfplay_device}\n")

    # Print configuration
    Config.print_config()

    # Initialize network
    print("\nInitializing network...")
    network = CheckersNetwork(
        num_filters=Config.NUM_FILTERS,
        num_res_blocks=Config.NUM_RES_BLOCKS,
        policy_size=Config.POLICY_SIZE
    )

    # Initialize weights
    def init_weights(m):
        if isinstance(m, torch.nn.Conv2d) or isinstance(m, torch.nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                torch.nn.init.zeros_(m.bias)

    network.apply(init_weights)
    print("Initialized network weights")

    print(f"Network parameters: {count_parameters(network):,}")

    # Create visualizers if requested
    game_visualizer = None
    if args.visualize:
        print("\nInitializing visualizer...")
        # Consolidated game and metrics visualizer (pygame)
        game_visualizer = GameVisualizer(square_size=70, max_metrics=args.iterations)
        print("✓ Integrated Visualizer ready (board and metrics in one window)")

    # Create trainer
    print("\nCreating trainer...")
    trainer = Trainer(network, Config, device, selfplay_device,
                     game_visualizer=game_visualizer)

    # Resume from checkpoint if requested
    if args.resume:
        if args.resume == 'auto':
            # Auto-detect latest checkpoint
            from pathlib import Path
            latest_path = Path(Config.CHECKPOINT_DIR) / 'latest.pt'
            if latest_path.exists():
                print(f"\n{'='*70}")
                trainer.load_checkpoint(str(latest_path))
                print(f"{'='*70}\n")
            else:
                print(f"\n⚠ No checkpoint found at {latest_path}, starting from scratch\n")
        else:
            # Load specific checkpoint
            print(f"\n{'='*70}")
            trainer.load_checkpoint(args.resume)
            print(f"{'='*70}\n")

    # Start training
    print(f"\nStarting training for {args.iterations} iterations...")
    print(f"Checkpoints will be saved to: {Config.CHECKPOINT_DIR}")
    print(f"Logs will be saved to: {Config.LOG_DIR}")

    try:
        trainer.train(num_iterations=args.iterations)
    finally:
        if game_visualizer:
            print("Closing visualizer...")
            game_visualizer.close()

    print("\n" + "=" * 70)
    print("Training script finished!")
    print("=" * 70)


if __name__ == "__main__":
    main()
