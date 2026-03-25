#!/usr/bin/env python3
"""CLI: Filter correct traces and format training dataset."""

import sys
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent.parent))

from config import Config
from llmath.filter import filter_and_format


def main():
    config = Config()
    filter_and_format(config)


if __name__ == "__main__":
    main()
