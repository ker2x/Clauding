# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is an **AlphaZero-style reinforcement learning system** for 10x10 International Draughts (checkers). The AI learns entirely through self-play with no initial training data, using Monte Carlo Tree Search (MCTS) guided by a ResNet neural network.

**Key Architecture**: Bitboard game engine → ResNet neural network → MCTS search → Self-play training → Iterative improvement through evaluation tournaments.

## Environment & Setup

> [!IMPORTANT]
> Always use the python interpreter located in `../.venv/bin/python` for running all scripts and commands in this project.
> Example: `../.venv/bin/python scripts8x8/train.py`

## Common Commands

### Testing
```bash
# Test game engine (core rules and bitboard logic)
../.venv/bin/python tests/test_engine.py

# Integration test (full pipeline with reduced settings)
../.venv/bin/python tests/test_integration.py

# Test individual components
../.venv/bin/python checkers/network/resnet.py          # Neural network
../.venv/bin/python checkers/training/replay_buffer.py  # Replay buffer
../.venv/bin/python checkers/utils/checkpoint.py        # Checkpoint system
```

### Training
```bash
# Start training from scratch
../.venv/bin/python scripts/train.py --iterations 100

# Resume from checkpoint
../.venv/bin/python scripts/train.py --resume checkpoints/checkpoint_iter_50.pt --iterations 50

# Quick progress demo (1-2 minutes, shows all monitoring features)
../.venv/bin/python scripts/demo_progress.py
```

### Monitoring
```bash
# View training metrics and plots
../.venv/bin/python checkers/utils/visualization.py logs/training_log.csv

# Print training summary
../.venv/bin/python -c "from checkers.utils.visualization import print_training_summary; print_training_summary()"

# View config
../.venv/bin/python config.py
```

## Architecture

### Core Data Flow

1. **Game State** → 8-plane tensor (8×10×10) representing board, pieces, legal moves, repetitions
2. **Neural Network** → Policy logits (150 actions) + Value scalar (-1 to +1)
3. **MCTS** → Search tree using PUCT algorithm, guided by network predictions
4. **Self-Play** → Generate games using MCTS, store (state, policy, value) tuples
5. **Replay Buffer** → Store 500K positions with recency-biased sampling
6. **Training** → Sample batches, apply 8× data augmentation, optimize network
7. **Evaluation** → Tournament between current and best model, promote if win rate > 55%

### Critical Components

**Bitboard Representation** (`checkers/engine/bitboard.py`)
- 10×10 board = 50 playable squares (dark squares only)
- State stored as 4 uint64 integers: player_men, player_kings, opponent_men, opponent_kings
- Perspective switching: After each move, board is flipped and player/opponent are swapped
- This ensures current player is always at bottom, simplifying neural network input

**Move Generation** (`checkers/engine/moves.py`)
- International Draughts rules: forced captures, chain captures, flying kings
- `get_legal_moves()` returns captures if any exist (mandatory), otherwise quiet moves
- Capture chains are fully enumerated as single Move objects
- Move encoding: `Move(from_square, to_square, captured_squares, promotes_to_king)`

**Game State Management** (`checkers/engine/game.py`)
- `CheckersGame.make_move()` applies move, then swaps perspective (flips board, swaps players)
- Position history tracks repetitions via hash for draw detection
- `to_neural_input()` converts bitboards to 8-plane network input

**Neural Network** (`checkers/network/resnet.py`)
- Input: (batch, 8, 10, 10) - 8 feature planes
- Architecture: Initial conv → 6 residual blocks (128 filters) → Policy head (150) + Value head (1)
- **Critical**: Policy head outputs raw logits; legal move masking applied before softmax
- 2.7M parameters, optimized for CPU training

**MCTS** (`checkers/mcts/mcts.py`)
- PUCT selection: `score = Q + c_puct × P × sqrt(N_parent) / (1 + N_child)`
- Root noise: Dirichlet(α=0.3) mixed with ε=0.25 for exploration during self-play
- Temperature schedule: τ=1.0 for first 15 moves (exploration), τ=0 after (greedy)
- Returns visit-count-based policy distribution for training

**Replay Buffer** (`checkers/training/replay_buffer.py`)
- Stores numpy arrays: states (8,10,10), policies (150), values (scalar), generations
- **Recency bias**: Sampling weight = exp(-(current_gen - position_gen) / 50)
- This prevents overfitting to weak early-game positions

**Training Loop** (`checkers/training/trainer.py`)
- Each iteration: Self-play → Training → Evaluation (every 10) → Checkpoint
- Self-play generates ~3000-5000 positions per 100 games
- Training: 100 gradient steps, 8× data augmentation per sample
- Loss = policy_cross_entropy + value_MSE

## Configuration

**All hyperparameters are in `config.py`**. Key settings:

```python
MCTS_SIMS_SELFPLAY = 300   # Reduce to 200 for 2× speedup
GAMES_PER_ITERATION = 100  # Reduce to 50 for 2× speedup
NUM_WORKERS = 6            # Adjust for CPU core count
BUFFER_SIZE = 500_000      # Reduce to 250_000 if memory limited
AUGMENTATION = True        # 8× data multiplication
```

**Device Selection**: Automatically falls back to CPU if GPU unavailable. Set `Config.DEVICE = "cpu"/"cuda"/"mps"`.

## Development Notes

### Adding New Features

**New network architecture**:
- Modify `checkers/network/resnet.py`
- Update `Config.NUM_FILTERS` and `Config.NUM_RES_BLOCKS`
- Network must output `(policy_logits, value)` where policy is (batch, 150)

**New MCTS variant**:
- Modify `checkers/mcts/mcts.py`
- Key method: `_simulate()` implements selection-expansion-evaluation-backup
- Ensure legal move masking is preserved

**New training algorithm**:
- Modify `checkers/training/trainer.py`
- Keep replay buffer interface: `sample()` returns (states, policies, values)
- Maintain checkpoint format for resumption

### Common Issues

**Perspective confusion**: The game state always represents the current player's view (pieces at bottom). After `make_move()`, the board is flipped and players swapped. This is critical for MCTS to work correctly.

**Legal move masking**: Policy logits must be masked with `-inf` for illegal moves before softmax. This happens in:
- `CheckersNetwork.predict()` - during inference
- MCTS policy evaluation - before UCB calculation

**Memory usage**: Replay buffer dominates memory (~1.5GB for 500K). Bitboards (200 bytes) are stored, not neural inputs (3KB), for efficiency.

**Training instability**: If loss diverges or NaN appears:
- Check gradient clipping (GRAD_CLIP = 5.0)
- Verify legal move masking is applied
- Ensure game rules are correct (test with `test_engine.py`)

### Testing Strategy

**Unit tests** focus on game correctness:
- Move generation (captures, chains, forced moves)
- Game termination (wins, draws, repetitions)
- Perspective switching (board flips correctly)

**Integration test** (`tests/test_integration.py`) runs 2 iterations with reduced settings to verify:
- Self-play generates valid games
- Training loop completes without errors
- Checkpoints save/load correctly
- Logging writes to CSV

Run integration test before long training runs to catch issues early.

### Performance Optimization

**CPU-optimized**:
- Bitboards for fast move generation (2-5× faster than arrays)
- Batch processing in MCTS (accumulate evaluations, not implemented yet but planned)
- NumPy storage in replay buffer (16× more memory efficient)
- 6 parallel workers for self-play (configurable)

**Bottlenecks**:
- MCTS dominates time (~80% of iteration)
- Reduce `MCTS_SIMS_SELFPLAY` for faster iterations
- Neural network forward pass is fast on CPU (~60ms/batch)

## Progress Monitoring

Training includes real-time progress display with:
- Progress bars for self-play and training phases
- ETA calculation based on last 10 iterations
- Trend indicators (↑↓→) for loss and win rate
- Buffer fill percentage
- Promotion alerts (⭐) when model improves

See `PROGRESS_MONITORING.md` for details.

## Checkpoints and Resumption

**Checkpoint format**:
```python
{
    'iteration': int,
    'network_state_dict': OrderedDict,
    'optimizer_state_dict': dict,
    'metrics': dict
}
```

**Files**:
- `checkpoints/latest_model.pt` - most recent iteration
- `checkpoints/best_model.pt` - highest win rate model
- `checkpoints/checkpoint_iter_N.pt` - iteration-specific saves

**Resume training**: `python scripts/train.py --resume checkpoints/checkpoint_iter_50.pt --iterations 50`

## Expected Training Behavior

**Iterations 1-50**: Loss decreases rapidly (10 → 3), network learns basic rules
**Iterations 50-100**: Win rate vs self starts showing variation, tactics emerge
**Iterations 100-200**: Clear improvement visible, win rate climbs toward 60%+
**Iterations 200+**: Advanced tactics, strong endgame play

**Typical metrics after 100 iterations**:
- Policy loss: 2-3
- Value loss: 0.5-1.0
- Win rate: 50-60% (against earlier checkpoints)

If metrics don't follow this pattern, check game rules and MCTS implementation.
