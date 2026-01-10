"""
Main training script for AlphaZero-style checkers.

Usage:
    python scripts/train.py --iterations 100
    python scripts/train.py --resume checkpoints/checkpoint_iter_50.pt
"""

import sys
import os
import argparse
import torch

# Add parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from config import Config
from checkers.network.resnet import CheckersNetwork, initialize_weights
from checkers.training.trainer import Trainer


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Train checkers AI with AlphaZero")

    parser.add_argument(
        '--iterations',
        type=int,
        default=100,
        help='Number of training iterations (default: 100)'
    )

    parser.add_argument(
        '--resume',
        type=str,
        default=None,
        help='Path to checkpoint to resume from'
    )

    parser.add_argument(
        '--seed',
        type=int,
        default=None,
        help='Random seed (default: from config)'
    )

    parser.add_argument(
        '--device',
        type=str,
        choices=['cpu', 'cuda', 'mps'],
        default=None,
        help='Device to use (default: from config)'
    )

    parser.add_argument(
        '--visualize',
        action='store_true',
        help='Enable real-time pygame visualization'
    )

    return parser.parse_args()


def set_seed(seed: int):
    """Set random seed for reproducibility."""
    import random
    import numpy as np

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    print(f"Set random seed to {seed}")


def main():
    """Main training function."""
    args = parse_args()

    # Print header
    print("\n" + "=" * 60)
    print("AlphaZero-Style Checkers Training")
    print("=" * 60)

    # Set seed
    # Set seed
    if args.seed is not None:
        seed = args.seed
    else:
        import random
        seed = random.randint(1, 100000)
    
    # Update config so it prints correctly
    Config.SEED = seed
    set_seed(seed)

    # Get device
    if args.device:
        Config.DEVICE = args.device

    # Enable visualization if requested
    if args.visualize:
        Config.VISUALIZE_TRAINING = True

    device = Config.get_device()
    print(f"\nUsing device: {device}")

    # Print configuration
    Config.print_config()

    # Create network
    print("\nInitializing network...")
    network = CheckersNetwork(
        num_filters=Config.NUM_FILTERS,
        num_res_blocks=Config.NUM_RES_BLOCKS,
        policy_size=Config.POLICY_SIZE
    )

    # Initialize weights if not resuming
    if args.resume is None:
        network.apply(initialize_weights)
        print("Initialized network weights")

    # Count parameters
    total_params = sum(p.numel() for p in network.parameters())
    print(f"Network parameters: {total_params:,}")

    # Create trainer
    print("\nCreating trainer...")
    trainer = Trainer(
        network=network,
        config=Config,
        device=device,
        resume_from=args.resume
    )

    # Start training
    print(f"\nStarting training for {args.iterations} iterations...")
    print(f"Checkpoints will be saved to: {Config.CHECKPOINT_DIR}")
    print(f"Logs will be saved to: {Config.LOG_DIR}")

    try:
        trainer.train(num_iterations=args.iterations)
        
        # Save final checkpoint
        print("\nSaving final checkpoint...")
        trainer.checkpoint_manager.save_checkpoint(
            trainer.network,
            trainer.optimizer,
            trainer.start_iteration + args.iterations,
            {'final': True},
            is_best=False
        )
        print("Final checkpoint saved!")

    except KeyboardInterrupt:
        print("\n\nTraining interrupted by user")
        print("Saving checkpoint...")

        # Save interrupt checkpoint
        trainer.checkpoint_manager.save_checkpoint(
            trainer.network,
            trainer.optimizer,
            trainer.replay_buffer.current_generation,
            {'interrupted': True},
            is_best=False
        )

        print("Checkpoint saved. You can resume with --resume")

    print("\n" + "=" * 60)
    print("Training script finished!")
    print("=" * 60)


if __name__ == "__main__":
    main()
