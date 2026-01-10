# Antigravity Project Instructions

These instructions are for the Antigravity agent when working on this checkers reinforcement learning project.

## Core Environment Rule

> [!CRITICAL]
> **Always use the Python interpreter located at `../.venv/bin/python`.**
> Never use the system python or any other virtual environment unless explicitly told to. This venv contains all necessary dependencies like PyTorch (with MPS support) and Pygame.

## Project Context

- This is an AlphaZero-style training system for 8x8 checkers (currently being migrated/refined from a 10x10 base).
- **Core Engine**: Bitboard-based for high performance.
- **Visualization**: Consolidated Pygame visualizer that shows the board and training metrics (loss graphs) in a single window.
- **Hardware**: Running on Mac (M1/M2/M3) using `mps` for training acceleration.

## Preferred Workflow

1. Always check `config8x8.py` for current hyperparameters before modifying training logic.
2. When testing UI changes, use `../.venv/bin/python checkers8x8/utils/game_visualizer.py` to run the standalone test block.
3. Ensure all new scripts follow the same `on_move` callback pattern for real-time visualization.

## Common Command Shorthands

- **Train**: `../.venv/bin/python scripts8x8/train.py --visualize`
- **Test Engine**: `../.venv/bin/python tests/test_engine.py`
