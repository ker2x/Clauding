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

# Plot training curves (watch mode auto-refreshes)
../.venv/bin/python scripts/plot_training.py
../.venv/bin/python scripts/plot_training.py --watch --interval 60

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

**`going/network/resnet.py`** — `GoNetwork`: ResNet with triple heads (policy + value + ownership). Input `(batch, 17, 9, 9)` → policy logits `(batch, 82)` + value `(batch, 1)` in `[-1, 1]` + ownership `(batch, 81)` in `[0, 1]`. 128 filters, 10 residual blocks, ~800K params. GlobalPoolBias layers every 3 blocks inject board-wide context. `predict()` applies legal-action masking and softmax.

**`going/mcts/`** — MCTS with neural network evaluation and batched leaf expansion.
- `node.py`: MCTSNode with PUCT selection (negamax Q-value convention), virtual loss for parallel traversal, Dirichlet noise at root.
- `mcts.py`: `search()` returns visit-count policy over 82 actions. Batched NN evaluation of leaves.

**`going/training/`** — AlphaZero training loop.
- `trainer.py`: Orchestrates self-play → replay buffer → SGD → evaluation → checkpoint cycle. Dynamic sample reuse scaling based on buffer saturation. Loss = policy CE + value MSE + ownership BCE (weighted).
- `self_play.py`: `SelfPlayGame` + parallel worker pool. KataGo-style playout cap randomization (75% fast/25% full moves).
- `inference_server.py`: When self-play device is GPU/MPS, spawns a dedicated inference server process. Workers use `RemoteNetwork` proxy for batched cross-process inference.
- `replay_buffer.py`: Circular buffer with recency-weighted sampling (tau controls staleness decay).
- `evaluation.py`: Pits current model vs best model; promotes on win-loss differential > 0.

**`going/gtp/`** — GTP v2 protocol. `engine.py` handles commands, `controller.py` manages I/O. Stdin pre-buffered in a thread to avoid Sabaki handshake timeouts.

**`config.py`** — All hyperparameters in one `Config` class. Key values: `MCTS_SIMS_SELFPLAY=200`, `MCTS_SIMS_FAST=50`, `GAMES_PER_ITERATION=50`, `BUFFER_SIZE=100_000`, LR via ReduceLROnPlateau (starts at `0.002`, halves on plateau, min `1e-4`). Both training and self-play on MPS (Apple Silicon).

## Key Conventions

- Board positions are flat indices 0-80 (`row * 9 + col`). Actions 0-80 map directly to positions; action 81 is pass.
- Neural input is always relative to current player ("my stones" / "opponent stones"), but the board itself stays in absolute coordinates.
- Game results are from the perspective of the player to move when `get_result()` is called.
- MCTS uses negamax backup: each node's value is from its own player's perspective. `select_child` negates child Q-values to convert to parent's perspective.
- Self-play uses `spawn` multiprocessing. On MPS, an inference server batches NN calls across workers.
- Tests are standalone scripts with `assert`-based checks (no pytest/unittest).
