#!/usr/bin/env python3
"""CLI: Run teacher model on expressions (resumable)."""

import sys
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent.parent))

from config import Config
from llmath.oracle import run_oracle


def main():
    config = Config()
    run_oracle(config)


if __name__ == "__main__":
    main()
