# 9x9 Go Engine — AlphaZero Style

A neural network–based 9x9 Go engine trained via self-play with MCTS (Monte Carlo Tree Search). Inspired by AlphaGo Zero and AlphaZero.

## Features

- **ResNet-based policy/value network** — 128 filters, 10 residual blocks, ~800K parameters
- **MCTS self-play** — Parallel games with neural evaluation and Dirichlet exploration noise
- **Playout cap randomization** — KataGo-style speed optimization (3× faster self-play)
- **Positional superko detection** — Zobrist hashing for legal move validation
- **GTP interface** — Play in GUI clients like Sabaki or gogui-twogtp
- **Territory prediction** — Ownership auxiliary head for board understanding

## Quick Start

```bash
# From go9x9/ directory

# Training (1000 iterations, ~28 hours on Apple Silicon)
../.venv/bin/python scripts/train.py --iterations 1000

# Resume from checkpoint
../.venv/bin/python scripts/train.py --resume --iterations 1500

# Play against the engine in Sabaki/GoGUI
../.venv/bin/python scripts/gtp_play.py --model checkpoints/best_model.pt

# Run tests
../.venv/bin/python going/tests/test_rules.py
../.venv/bin/python going/tests/test_integration.py
```

## Architecture

### `going/engine/`
- **board.py** — Flat 81-element board array, legal move generation, capture logic
- **game.py** — `GoGame` class, 8-frame board history (17 planes for network input), zobrist superko
- **scoring.py** — Chinese-style area scoring with territory flood-fill
- **zobrist.py** — Zobrist hashing for positional superko detection
- **action_encoder.py** — Action space (82 = 81 positions + pass)

### `going/network/`
- **resnet.py** — `GoNetwork`: dual-head network with policy (82 logits) + value (∈[-1,1]) + ownership (territory prediction)

### `going/mcts/`
- **mcts.py** — MCTS tree search with neural network evaluation, Dirichlet noise, virtual loss

### `going/training/`
- **trainer.py** — Main training loop: self-play → buffer → SGD → evaluation → checkpointing
- **self_play.py** — Parallel self-play workers, playout cap randomization (fast/slow moves), temperature schedule
- **replay_buffer.py** — Circular buffer with recency + surprise-weighted sampling
- **evaluation.py** — Model comparison (current vs. best), promotion on win-loss differential
- **inference_server.py** — MPS GPU inference server for accelerated network forward passes

### `going/gtp/`
- **controller.py** — GTP v2 protocol for GUI integration

## Configuration

All hyperparameters in `config.py`:

- **Network**: `NUM_RES_BLOCKS=10`, `NUM_FILTERS=128`, `INPUT_PLANES=17`
- **MCTS**: `MCTS_SIMS_SELFPLAY=400` (slow), `MCTS_SIMS_FAST=50` (fast, not stored)
- **Training**: `GAMES_PER_ITERATION=20`, `BUFFER_SIZE=50_000`, `BATCH_SIZE=256`
- **Learning**: `LEARNING_RATE=0.002 → 1e-5` (cosine annealing)
- **Device**: `DEVICE="mps"` (training), `SELFPLAY_DEVICE="mps"` (self-play with inference server)

## Logs & Checkpoints

- `checkpoints/` — Model snapshots and best model
- `logs/training_log.csv` — Per-iteration metrics (loss, buffer size, learning rate, etc.)

## Performance

- ~100s/iteration at iter 0 (growing network, increasing MCTS depth)
- ~28 hours for 1000 iterations on Apple Silicon M1/M2
- Policy loss trajectory: ~4.3 → ~3.0+ by iter 100

## References

- AlphaGo Zero (Silver et al., 2017)
- AlphaZero (Silver et al., 2018)
- KataGo (Lightvector et al., 2019) — playout cap randomization technique
