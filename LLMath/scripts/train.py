#!/usr/bin/env python3
"""CLI: Fine-tune student model on distilled dataset."""

import argparse
import sys
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent.parent))

from config import Config
from llmath.train import train


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--resume", default=None, help="Checkpoint path to resume from (e.g. checkpoints/final)")
    args = parser.parse_args()

    config = Config()
    train(config, resume_from=args.resume)


if __name__ == "__main__":
    main()
