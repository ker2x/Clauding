#!/usr/bin/env python3
"""CLI: Generate arithmetic expressions."""

import argparse
import json
import sys
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent.parent))

from config import Config
from llmath.generate import generate_expressions, save_expressions


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--append", action="store_true", help="Append to existing expressions file")
    parser.add_argument("--seed", type=int, default=None, help="Override seed (useful for generating different data)")
    parser.add_argument("--count", type=int, default=None, help="Expressions per tier (overrides config)")
    args = parser.parse_args()

    config = Config()
    if args.count is not None:
        for tier in config.TIERS:
            tier.count = args.count

    # When appending, load existing expressions to avoid duplicates
    existing = set()
    if args.append and config.EXPRESSIONS_PATH.exists():
        with open(config.EXPRESSIONS_PATH) as f:
            for line in f:
                line = line.strip()
                if line:
                    existing.add(json.loads(line)["expression"])
        print(f"Loaded {len(existing)} existing expressions")

    expressions = generate_expressions(config, seed=args.seed, existing=existing)
    save_expressions(expressions, config.EXPRESSIONS_PATH, append=args.append)


if __name__ == "__main__":
    main()
