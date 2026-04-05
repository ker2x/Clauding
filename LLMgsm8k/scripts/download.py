#!/usr/bin/env python3
"""Download GSM8K dataset and convert to JSONL."""

import sys
sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent.parent))

from config import Config
from gsm8k.download import download

if __name__ == "__main__":
    download(Config())
