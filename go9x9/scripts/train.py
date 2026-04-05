#!/usr/bin/env python3
"""
Training entry point for 9x9 Go AlphaZero.

Usage:
    ../.venv/bin/python scripts/train.py --iterations 100
    ../.venv/bin/python scripts/train.py --resume --iterations 150
    ../.venv/bin/python scripts/train.py --resume checkpoints/checkpoint_iter_50.pt --iterations 150
"""

import argparse
import sys
import os
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import torch
import torch.multiprocessing as mp

from going.network.resnet import GoNetwork, count_parameters
from going.training.trainer import Trainer
from config import Config


def main():
    parser = argparse.ArgumentParser(description="Train 9x9 Go AlphaZero")
    parser.add_argument("--iterations", type=int, default=100,
                        help="Number of training iterations")
    parser.add_argument("--resume", nargs="?", const="auto", default=None,
                        help="Resume from checkpoint (auto or path)")
    parser.add_argument("--clear-buffer", action="store_true",
                        help="Clear replay buffer (useful after config changes)")
    parser.add_argument("--seed", type=int, default=None,
                        help="Random seed for reproducibility")
    args = parser.parse_args()

    # Set working directory to project root
    os.chdir(project_root)

    # Set seed
    if args.seed is not None:
        import numpy as np
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)

    # Set multiprocessing start method
    try:
        mp.set_start_method("spawn")
    except RuntimeError:
        pass  # Already set

    # Print config
    Config.print_config()

    # Get devices
    device = Config.get_device()
    selfplay_device = Config.get_selfplay_device()

    # Create network
    network = GoNetwork(
        num_filters=Config.NUM_FILTERS,
        num_res_blocks=Config.NUM_RES_BLOCKS,
        policy_size=Config.POLICY_SIZE,
        input_planes=Config.INPUT_PLANES,
        global_pool_freq=Config.GLOBAL_POOL_FREQ,
    )
    print(f"\nNetwork parameters: {count_parameters(network):,}")

    # Create trainer
    trainer = Trainer(network, Config, device, selfplay_device)

    # Resume from checkpoint
    if args.resume is not None:
        if args.resume == "auto":
            latest_path = Path(Config.CHECKPOINT_DIR) / "latest.pt"
            if latest_path.exists():
                trainer.load_checkpoint(str(latest_path))
            else:
                print("No checkpoint found, starting fresh")
        else:
            trainer.load_checkpoint(args.resume)

    # Clear buffer if requested
    if args.clear_buffer:
        trainer.replay_buffer.size = 0
        trainer.replay_buffer.position = 0
        print("Replay buffer cleared.")

    # Train
    trainer.train(args.iterations)


if __name__ == "__main__":
    main()
