#!/usr/bin/env python3
"""Run evaluation passes on solver traces."""

import argparse
import asyncio
import sys
sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent.parent))

from config import Config
from gsm8k.evaluate import evaluate, rescore

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--self", dest="run_self", action="store_true", help="Run self-eval pass")
    parser.add_argument("--smart", dest="run_smart", action="store_true", help="Run smart-eval pass")
    parser.add_argument("--rescore", action="store_true", help="Re-extract ratings from saved raw responses")
    args = parser.parse_args()

    if args.rescore:
        rescore(Config())
    else:
        # Default: run both if neither specified
        if not args.run_self and not args.run_smart:
            args.run_self = True
            args.run_smart = True
        asyncio.run(evaluate(Config(), run_self=args.run_self, run_smart=args.run_smart))
