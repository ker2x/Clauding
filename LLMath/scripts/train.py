#!/usr/bin/env python3
"""CLI: Fine-tune student model with mlx-lm LoRA."""

import argparse
import sys
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent.parent))

from config import Config
from llmath.train import train


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--resume", default=None, help="Adapter file to resume from (e.g. adapters/adapters.safetensors)")
    parser.add_argument("--iters", type=int, default=None, help="Override MLX_TRAIN_ITERS from config")
    parser.add_argument("--lr", type=float, default=None, help="Override MLX_LEARNING_RATE from config")
    args = parser.parse_args()

    config = Config()
    if args.iters is not None:
        config.MLX_TRAIN_ITERS = args.iters
    if args.lr is not None:
        config.MLX_LEARNING_RATE = args.lr

    train(config, resume_from=args.resume)


if __name__ == "__main__":
    main()
