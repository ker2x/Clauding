#!/usr/bin/env python3
"""CLI: Benchmark model on arithmetic expressions."""

import argparse
import sys
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent.parent))

from config import Config
from llmath.benchmark import benchmark


def main():
    parser = argparse.ArgumentParser(description="Benchmark model on arithmetic expressions")
    parser.add_argument("--model", default=None, help="Path to fine-tuned checkpoint (default: base model)")
    parser.add_argument("--max-per-tier", type=int, default=0, help="Max expressions per tier (0 = all)")
    args = parser.parse_args()

    config = Config()
    benchmark(config, model_path=args.model, max_per_tier=args.max_per_tier)


if __name__ == "__main__":
    main()
