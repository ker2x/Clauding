# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is an **AlphaZero-style reinforcement learning system** for 8x8 American Checkers. The AI learns entirely through self-play with no initial training data, using Monte Carlo Tree Search (MCTS) guided by a ResNet neural network.

**Key Innovation**: Fixed action space (128 actions) where each action index always represents the same move, enabling the network to learn spatial patterns rather than just relative preferences.

**Key Architecture**: Bitboard game engine → ResNet neural network → MCTS search → Self-play training → Iterative improvement through evaluation tournaments.

## Environment & Setup

> [!IMPORTANT]
> Always use the python interpreter located in `../.venv/bin/python` for running all scripts and commands in this project.
> Example: `../.venv/bin/python scripts8x8/train.py`

## Common Commands

### Training
```bash
# Start training from scratch
../.venv/bin/python scripts8x8/train.py --iterations 100

# Resume from checkpoint (auto-detect latest)
../.venv/bin/python scripts8x8/train.py --resume --iterations 150

# Resume from specific checkpoint
../.venv/bin/python scripts8x8/train.py --resume checkpoints8x8/checkpoint_iter_50.pt --iterations 150

# Train with visualization
../.venv/bin/python scripts8x8/train.py --iterations 50 --visualize

# Train with specific seed for reproducibility
../.venv/bin/python scripts8x8/train.py --iterations 100 --seed 42
```

### Testing
```bash
# Integration test (full pipeline with reduced settings)
../.venv/bin/python checkers8x8/tests/test_integration.py

# Test individual components
../.venv/bin/python checkers8x8/network/resnet.py          # Neural network
../.venv/bin/python checkers8x8/training/replay_buffer.py  # Replay buffer
../.venv/bin/python checkers8x8/engine/game.py             # Game engine
../.venv/bin/python checkers8x8/engine/action_encoder.py   # Action encoding
../.venv/bin/python checkers8x8/mcts/mcts.py              # MCTS
```

### Monitoring
```bash
# View config
../.venv/bin/python config8x8.py

# Check logs
cat logs8x8/training_log.csv
```

### C++ Extension (Optional Performance)
```bash
# Build C++ extension for faster MCTS
../.venv/bin/python setup.py build_ext --inplace

# The extension provides checkers_cpp module with optimized game and MCTS
```

## Architecture

### Core Data Flow

1. **Game State** → 8-plane tensor (8×8×8) representing board, pieces, legal moves, game state
2. **Neural Network** → Policy logits (128 actions, fixed space) + Value scalar (-1 to +1)
3. **MCTS** → Search tree using PUCT algorithm, guided by network predictions
4. **Self-Play** → Generate games using MCTS, store (state, policy, value) tuples
5. **Replay Buffer** → Store positions with recency-biased sampling and checkpointing
6. **Training** → Sample batches with dynamic reuse ratio, optimize network
7. **Evaluation** → Tournament between current and best model, promote if wins > losses

### Critical Components

**Bitboard Representation** (`checkers8x8/engine/bitboard.py`)
- 8×8 board = 32 playable squares (dark squares only)
- State stored as 4 uint32 integers: player_men, player_kings, opponent_men, opponent_kings
- Perspective switching: After each move, board is flipped and player/opponent are swapped
- This ensures current player is always at bottom (rows 5-7), simplifying neural network input

**Action Encoding** (`checkers8x8/engine/action_encoder.py`)
- **Fixed action space**: 32 squares × 4 directions = 128 actions
- `action = from_square * 4 + direction`
- Direction: 0=NW, 1=NE, 2=SW, 3=SE
- Each action index ALWAYS means the same move, unlike dynamic action spaces
- Enables network to learn spatial patterns (e.g., "moving from center NE is strong")

**Move Generation** (`checkers8x8/engine/moves.py`)
- American Checkers rules: forced captures, chain captures, men move forward only (capture backward)
- Kings move 1 square in any direction (not flying like International Draughts)
- `get_legal_moves()` returns captures if any exist (mandatory), otherwise quiet moves
- Capture chains are fully enumerated as single Move objects

**Game State Management** (`checkers8x8/engine/game.py`)
- `CheckersGame.make_move()` applies move, then swaps perspective (flips board, swaps players)
- `to_neural_input()` converts bitboards to 8-plane network input
- `get_legal_action_mask()` returns boolean mask of 128 actions for legal move masking

**Neural Network** (`checkers8x8/network/resnet.py`)
- Input: (batch, 8, 8, 8) - 8 feature planes
- Architecture: Initial conv → 6 residual blocks (128 filters) → Policy head (128) + Value head (1)
- **Critical**: Policy head outputs raw logits; legal move masking applied before softmax
- 2.3M parameters, optimized for CPU/MPS training

**MCTS** (`checkers8x8/mcts/mcts.py`)
- PUCT selection: `score = Q + c_puct × P × sqrt(N_parent) / (1 + N_child)`
- Root noise: Dirichlet(α=0.3) mixed with ε=0.25 for exploration during self-play
- Temperature schedule: τ=1.0 for first 15 moves (exploration), τ=0 after (greedy)
- Returns visit-count-based policy distribution for training
- Direct action masking with fixed action space (no move list mapping needed)

**Replay Buffer** (`checkers8x8/training/replay_buffer.py`)
- Stores numpy arrays: states (8,8,8), policies (128), values (scalar), generations
- **Recency bias**: Sampling weight = exp(-(current_gen - position_gen) / 50)
- This prevents overfitting to weak early-game positions
- **Checkpointed**: Buffer state is saved and restored with checkpoints (unlike 10x10 version)

**Training Loop** (`checkers8x8/training/trainer.py`)
- Each iteration: Self-play → Training → Evaluation (every 5 iterations) → Checkpoint
- Self-play generates positions using parallel workers
- Training: Dynamic steps based on sample reuse ratio (scales 10-30× based on buffer fullness)
- Loss = policy_cross_entropy + value_MSE
- Evaluation: Current model vs best model, promote if wins > losses (ignores draws)

## Configuration

**All hyperparameters are in `config8x8.py`**. Key settings:

```python
MCTS_SIMS_SELFPLAY = 100   # Reduce to 50 for 2× speedup
GAMES_PER_ITERATION = 20   # Reduce to 10 for faster iterations
NUM_WORKERS = 2            # Adjust for CPU core count
BUFFER_SIZE = 20_000       # Increase if memory allows (buffer is checkpointed!)
BATCH_SIZE = 256           # Reduce if memory limited
MIN_SAMPLE_REUSE = 10      # Training passes per state (early training)
MAX_SAMPLE_REUSE = 30      # Training passes per state (mature training)
```

**Device Selection**:
- Training device: `Config.DEVICE = "mps"/"cuda"/"cpu"` (default: "mps" for Apple Silicon)
- Self-play device: `Config.SELFPLAY_DEVICE = "cpu"/"mps"` (default: "cpu" - MCTS faster on CPU)

## Development Notes

### Adding New Features

**New network architecture**:
- Modify `checkers8x8/network/resnet.py`
- Update `Config.NUM_FILTERS` and `Config.NUM_RES_BLOCKS`
- Network must output `(policy_logits, value)` where policy is (batch, 128)

**New MCTS variant**:
- Modify `checkers8x8/mcts/mcts.py`
- Key method: `search()` implements PUCT selection-expansion-evaluation-backup
- Ensure legal action masking is preserved (128-dim boolean mask)

**New training algorithm**:
- Modify `checkers8x8/training/trainer.py`
- Keep replay buffer interface: `sample()` returns (states, policies, values)
- Maintain checkpoint format for resumption (includes replay buffer state!)

### Common Issues

**Perspective confusion**: The game state always represents the current player's view (pieces at bottom rows 5-7). After `make_move()`, the board is flipped and players swapped. This is critical for MCTS to work correctly.

**Legal action masking**: Policy logits must be masked with `-inf` for illegal actions before softmax. The fixed action space has 128 actions, but only ~5-15 are legal in any position.

**Fixed action space**: Unlike dynamic action spaces, action 53 ALWAYS means "Square 13, Direction NE". This allows the network to learn spatial patterns.

**Memory usage**: Replay buffer is checkpointed (saved with training state). With BUFFER_SIZE=20K, buffer uses ~80MB. Increasing to 100K uses ~400MB but is safe.

**Training instability**: If loss diverges or NaN appears:
- Check gradient clipping (GRAD_CLIP = 5.0)
- Verify legal action masking is applied
- Ensure game rules are correct (test with `test_integration.py`)
- Check learning rate (default: 0.001)

### Testing Strategy

**Integration test** (`checkers8x8/tests/test_integration.py`) verifies:
- Game engine generates valid moves
- Action encoding covers all 128 actions
- Neural network outputs correct shapes
- MCTS completes search without errors
- Self-play generates valid games
- Replay buffer sampling works
- Training step completes
- Checkpointing works (including buffer state)

Run integration test before long training runs to catch issues early.

### Performance Optimization

**CPU-optimized**:
- Bitboards for fast move generation (2-5× faster than arrays)
- MCTS runs on CPU (faster than MPS for single-game tree search)
- NumPy storage in replay buffer (memory efficient)
- Parallel workers for self-play (default: 2 workers)

**Optional C++ extension** (`checkers8x8/cpp/`):
- Provides `checkers_cpp` module with optimized game and MCTS
- Build with `setup.py build_ext --inplace`
- ~10× faster than pure Python for MCTS

**Bottlenecks**:
- MCTS dominates time (~80-90% of iteration)
- Reduce `MCTS_SIMS_SELFPLAY` for faster iterations
- Neural network forward pass is fast (~10-20ms/batch on MPS)

## Checkpoints and Resumption

**Checkpoint format**:
```python
{
    'iteration': int,
    'network_state_dict': OrderedDict,
    'optimizer_state_dict': dict,
    'replay_buffer_state_dict': dict,  # Buffer is saved!
}
```

**Files**:
- `checkpoints8x8/latest.pt` - most recent iteration
- `checkpoints8x8/best_model.pt` - highest win rate model (for evaluation)
- `checkpoints8x8/checkpoint_iter_N.pt` - iteration-specific saves (every 10 iterations)

**Resume training**:
```bash
# Auto-detect latest
../.venv/bin/python scripts8x8/train.py --resume --iterations 150

# Specific checkpoint
../.venv/bin/python scripts8x8/train.py --resume checkpoints8x8/checkpoint_iter_50.pt --iterations 150
```

**Buffer size changes**: You can increase `BUFFER_SIZE` when resuming. The old buffer data is preserved, and new capacity is added. A warning is shown but training continues normally.

## Expected Training Behavior

**Iterations 1-20**: Loss decreases rapidly (4.0 → 1.5), network learns basic rules and legal moves
**Iterations 20-50**: Tactics emerge (captures, forks), loss continues down to ~1.0
**Iterations 50-100**: Strategic patterns appear, win rate against best model starts varying
**Iterations 100+**: Strong tactical play, endgame technique

**Typical metrics after 100 iterations**:
- Total loss: 0.8-1.5
- Policy loss: 0.6-1.2
- Value loss: 0.2-0.5
- Buffer: Full or near capacity
- Win rate: Should see promotions every 20-30 iterations

If metrics don't follow this pattern, check game rules and MCTS implementation.

## Real-Time Visualization

Training includes optional real-time visualization with `--visualize` flag:
- Live board display with piece rendering
- Move-by-move policy heatmap during self-play
- Training metrics graphs (loss, buffer size)
- Uses pygame for integrated window

## Project Structure

```
checkers8x8/
├── engine/
│   ├── bitboard.py          # 32-square bitboard
│   ├── action_encoder.py    # Fixed 128-action space
│   ├── moves.py             # American Checkers rules
│   └── game.py              # Game state manager
├── network/
│   └── resnet.py            # ResNet (8×8×8 → 128 + value)
├── mcts/
│   ├── node.py              # MCTS tree node
│   └── mcts.py              # Monte Carlo Tree Search
├── training/
│   ├── replay_buffer.py     # Checkpointed replay buffer
│   ├── self_play.py         # Parallel self-play
│   ├── evaluation.py        # Model tournaments
│   └── trainer.py           # Main training loop
├── utils/
│   └── game_visualizer.py   # Real-time visualization
├── cpp/                     # Optional C++ optimizations
│   ├── game.cpp             # Fast game engine
│   ├── mcts.cpp             # Fast MCTS
│   └── bindings.cpp         # Python bindings
└── tests/
    └── test_integration.py  # Full system test

config8x8.py                 # All hyperparameters
scripts8x8/
├── train.py                 # Training entry point
└── benchmark.py             # Performance testing
setup.py                     # C++ extension build
```
