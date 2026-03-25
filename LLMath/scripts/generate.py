#!/usr/bin/env python3
"""CLI: Generate arithmetic expressions."""

import sys
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent.parent))

from config import Config
from llmath.generate import generate_expressions, save_expressions


def main():
    config = Config()
    expressions = generate_expressions(config)
    save_expressions(expressions, config.EXPRESSIONS_PATH)


if __name__ == "__main__":
    main()
