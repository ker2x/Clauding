# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

AlphaZero-style 9x9 Go engine in Python/PyTorch. Trains a ResNet via self-play with MCTS, and exposes a GTP interface for play in GUI clients like Sabaki.

## Commands

```bash
# Run from the go9x9/ directory. Uses the shared venv one level up.

# Training
../.venv/bin/python scripts/train.py --iterations 100
../.venv/bin/python scripts/train.py --resume --iterations 150        # auto-resume from latest checkpoint
../.venv/bin/python scripts/train.py --resume checkpoints/checkpoint_iter_50.pt --iterations 150

# GTP play (use in Sabaki or gogui-twogtp)
../.venv/bin/python scripts/gtp_play.py --model checkpoints/best_model.pt
../.venv/bin/python scripts/gtp_play.py --model checkpoints/best_model.pt --simulations 400 --device mps

# Tests (plain scripts, no test framework)
../.venv/bin/python going/tests/test_rules.py
../.venv/bin/python going/tests/test_integration.py
```

## Architecture

**`going/engine/`** — Go rules and game state (pure numpy, no torch dependency)
- `board.py`: Flat 81-int board array (`0`=empty, `1`=black, `2`=white). Precomputed `NEIGHBORS` table. Core functions: `apply_move`, `get_legal_moves`, `is_suicide`, `find_group`.
- `game.py`: `GoGame` class wrapping board + zobrist superko + 8-frame board history. `to_neural_input()` produces `(17, 9, 9)` planes (8 history × 2 colors + color-to-play). Result from current player's perspective.
- `action_encoder.py`: Action space is 82 (81 intersections + pass=81). GTP coordinate conversion (I column skipped per Go convention).
- `zobrist.py`: Zobrist hashing for positional superko detection.
- `scoring.py`: Chinese-style area scoring with territory flood-fill.

**`going/network/resnet.py`** — `GoNetwork`: ResNet with dual policy+value heads. Input `(batch, 17, 9, 9)` → policy logits `(batch, 82)` + value `(batch, 1)` in `[-1, 1]`. Default: 128 filters, 10 residual blocks. `predict()` applies legal-action masking and softmax.

**`going/mcts/`** — MCTS with neural network evaluation. Dirichlet noise at root for exploration. `search()` returns visit-count policy over 82 actions.

**`going/training/`** — AlphaZero training loop.
- `trainer.py`: Orchestrates self-play → replay buffer → SGD → evaluation → checkpoint cycle. Dynamic sample reuse scaling based on buffer saturation.
- `self_play.py`: `SelfPlayGame` + parallel worker pool (`mp.Process` with spawn). Temperature schedule: exploration early, greedy after `TEMPERATURE_THRESHOLD` moves.
- `replay_buffer.py`: Circular buffer with recency-weighted sampling.
- `evaluation.py`: Pits current model vs best model; promotes on win-loss differential > 0.

**`going/gtp/`** — GTP v2 protocol implementation for external GUI integration.

**`config.py`** — All hyperparameters in one `Config` class. Training on MPS (Apple Silicon), self-play on CPU. Key: `MCTS_SIMS_SELFPLAY=200`, `GAMES_PER_ITERATION=20`, `BUFFER_SIZE=50_000`.

## Key Conventions

- Board positions are flat indices 0-80 (`row * 9 + col`). Actions 0-80 map directly to positions; action 81 is pass.
- Neural input is always relative to current player ("my stones" / "opponent stones"), but the board itself stays in absolute coordinates.
- Game results are from the perspective of the player to move when `get_result()` is called.
- Self-play uses `spawn` multiprocessing with `share_memory()` on the network.
- Tests are standalone scripts with `assert`-based checks (no pytest/unittest).
