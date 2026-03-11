#!/usr/bin/env python3
"""LLM Debate Arena — entry point.

Usage:
    ../.venv/bin/python scripts/main.py
    ../.venv/bin/python scripts/main.py --config examples/debate.yaml
    ../.venv/bin/python scripts/main.py --topic "Tabs vs spaces" --rounds 3
    ../.venv/bin/python scripts/main.py --model-a mistral --model-b llama3.2 --moderator
"""

import argparse
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from config import Config
from arena.moderator import make_moderator, make_participant
from arena.tui.app import DebateApp


def main():
    parser = argparse.ArgumentParser(description="LLM Debate Arena")
    parser.add_argument("--config", type=str, help="YAML config file path")
    parser.add_argument("--topic", type=str, help="Debate topic")
    parser.add_argument("--rounds", type=int, help="Number of rounds")
    parser.add_argument("--model-a", type=str, dest="model_a", help="Model for debater A")
    parser.add_argument("--model-b", type=str, dest="model_b", help="Model for debater B")
    parser.add_argument("--moderator", action="store_true", default=None,
                        help="Enable moderator")
    parser.add_argument("--host", type=str, help="Ollama host URL")
    parser.add_argument("--max-tokens", type=int, dest="max_tokens",
                        help="Max tokens per turn")
    parser.add_argument("--think", action="store_true", default=None,
                        help="Enable thinking mode (for Qwen3 etc.)")
    args = parser.parse_args()

    if args.config:
        Config.from_yaml(args.config)
    Config.apply_cli_overrides(args)
    Config.print_config()

    a = make_participant("A", Config)
    b = make_participant("B", Config)
    mod = make_moderator(Config)

    app = DebateApp(
        participant_a=a,
        participant_b=b,
        moderator=mod,
        config=Config,
    )
    app.run()


if __name__ == "__main__":
    main()
