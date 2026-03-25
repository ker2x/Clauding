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
../.venv/bin/python scripts/train.py --resume --iterations 200 --clear-buffer  # wipe buffer after config change

# Distributed self-play server (run on remote machines or localhost)
../.venv/bin/python scripts/selfplay_server.py --device mps --port 9377
../.venv/bin/python scripts/selfplay_server.py --device cpu --workers 8 --games 50 --port 9377

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
- `game.py`: `GoGame` class wrapping board + zobrist superko + board history. `to_neural_input()` produces `(INPUT_PLANES, 9, 9)` planes (history × 2 colors, no color-to-play — network is color-agnostic). Result from current player's perspective.
- `dihedral.py`: Dihedral group (8 symmetries of the square) for data augmentation.
- `action_encoder.py`: Action space is 82 (81 intersections + pass=81). GTP coordinate conversion (I column skipped per Go convention).
- `zobrist.py`: Zobrist hashing for positional superko detection.
- `scoring.py`: Chinese-style area scoring with territory flood-fill.

**`going/network/resnet.py`** — `GoNetwork`: ResNet with triple heads (policy + value + ownership). Input `(batch, INPUT_PLANES, 9, 9)` → policy logits `(batch, 82)` + value `(batch, 1)` in `[-1, 1]` + ownership `(batch, 81)` in `[0, 1]`. 128 filters, 6 residual blocks. GlobalPoolBias layers every 3 blocks inject board-wide context. `predict()` applies legal-action masking and softmax.

**`going/mcts/`** — MCTS with neural network evaluation and batched leaf expansion.
- `node.py`: MCTSNode with PUCT selection (negamax Q-value convention, `sqrt(1+N)` parent term), virtual loss for parallel traversal, Dirichlet noise at root.
- `mcts.py`: `search()` returns visit-count policy over 82 actions. Batched NN evaluation of leaves. Early termination stops search when the leading move's visit gap exceeds remaining sims (controlled by `MCTS_EARLY_TERM` threshold).

**`going/training/`** — AlphaZero training loop.
- `trainer.py`: Orchestrates self-play → replay buffer → SGD → evaluation → checkpoint cycle. Dynamic sample reuse scaling based on buffer saturation. Loss = policy CE + value MSE + ownership BCE (weighted).
- `self_play.py`: `SelfPlayGame` + parallel worker pool. KataGo-style playout cap randomization (`P_FAST_MOVE` fraction use fewer sims and aren't stored as training data).
- `inference_server.py`: When self-play device is GPU/MPS, spawns a dedicated inference server process. Workers use `RemoteNetwork` proxy for batched cross-process inference.
- `distributed.py`: TCP protocol for multi-machine self-play. Length-prefixed pickle messages (stdlib only, no ZMQ). Trainer fans out to all `SELFPLAY_SERVERS` in parallel via threads, each server plays `GAMES_PER_ITERATION` games independently.
- `replay_buffer.py`: Circular buffer with recency-weighted sampling (tau controls staleness decay).
- `evaluation.py`: Pits current model vs best model; promotes on win-loss differential > 0.

**`going/gtp/`** — GTP v2 protocol. `engine.py` handles commands, `controller.py` manages I/O. Stdin pre-buffered in a thread to avoid Sabaki handshake timeouts.

**`config.py`** — All hyperparameters in one `Config` class. `BUFFER_SIZE=100_000`. `SELFPLAY_SERVERS` lists `"host:port"` strings for distributed self-play (empty = local only). Three-phase curriculum plan: phase 1 (LR 1e-3, 200 sims, 30 games), phase 2 (LR 3e-4, 200 sims, 50 games), phase 3 (LR 1e-4, 400 sims, 50 games).

## Key Conventions

- Board positions are flat indices 0-80 (`row * 9 + col`). Actions 0-80 map directly to positions; action 81 is pass.
- Neural input is always relative to current player ("my stones" / "opponent stones"), but the board itself stays in absolute coordinates.
- Game results are from the perspective of the player to move when `get_result()` is called.
- MCTS uses negamax backup: each node's value is from its own player's perspective. `select_child` negates child Q-values to convert to parent's perspective.
- Self-play uses `spawn` multiprocessing. On MPS, an inference server batches NN calls across workers.
- Tests are standalone scripts with `assert`-based checks (no pytest/unittest).
