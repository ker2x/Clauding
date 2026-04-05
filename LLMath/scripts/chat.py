#!/usr/bin/env python3
"""REPL: Chat with the student model."""

import argparse
import sys
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent.parent))

from config import Config
from llmath.chat import chat


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=None, help="Model path (default: MLX_MODEL from config)")
    parser.add_argument("--max-tokens", type=int, default=512)
    args = parser.parse_args()

    config = Config()
    chat(config, model_path=args.model, max_tokens=args.max_tokens)


if __name__ == "__main__":
    main()
