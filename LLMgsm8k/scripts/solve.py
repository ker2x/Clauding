#!/usr/bin/env python3
"""Run vLLM solver on GSM8K questions."""

import argparse
import asyncio
import sys
sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent.parent))

from config import Config
from gsm8k.solve import solve

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", type=str, default="both", help="test, train, or both (default: both)")
    args = parser.parse_args()
    asyncio.run(solve(Config(), split=args.split))
