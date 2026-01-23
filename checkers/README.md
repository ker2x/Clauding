# AlphaZero Checkers

**Train a checkers AI from scratch using AlphaZero-style reinforcement learning.**

Learn entirely through self-play with no initial training data. Watch your AI evolve from random moves to tactical mastery.

## Key Features

- **Fixed Action Space**: Each action index always represents the same move, enabling spatial pattern learning
- **8x8 American Checkers**: Standard rules with forced captures and chain captures
- **ResNet Neural Network**: 2.3M parameters, optimized for CPU/GPU training
- **Monte Carlo Tree Search**: PUCT algorithm with 100 simulations per move
- **Parallel Self-Play**: Multi-worker game generation
- **Checkpointed Training**: Resume from any iteration with full replay buffer state
- **Real-Time Visualization**: Watch training live with pygame

## Quick Start

### Installation

```bash
# Clone repository
git clone <repository-url>
cd checkers

# Install dependencies (PyTorch, NumPy)
pip install -r requirements.txt
```

### Train Your First Model

```bash
# Train for 100 iterations (~3-4 hours on M1 Mac)
python scripts8x8/train.py --iterations 100

# Resume from checkpoint
python scripts8x8/train.py --resume --iterations 150

# Train with live visualization
python scripts8x8/train.py --iterations 50 --visualize
```

### Monitor Progress

Training metrics are logged to `logs8x8/training_log.csv`:
- Loss (policy and value)
- Buffer size and utilization
- Self-play and training time per iteration

Checkpoints saved to `checkpoints8x8/`:
- `latest.pt` - most recent iteration
- `best_model.pt` - highest win rate model
- `checkpoint_iter_N.pt` - iteration-specific saves

## How It Works

### AlphaZero Training Loop

```
┌─────────────────────────────────────────────────────────┐
│  1. Self-Play: Generate games using MCTS + Neural Net  │
│  2. Store: Save (state, policy, value) to replay buffer│
│  3. Train: Sample batches and optimize neural network  │
│  4. Evaluate: Play tournament, promote if improved     │
│  5. Repeat: Virtuous cycle of self-improvement         │
└─────────────────────────────────────────────────────────┘
```

### Fixed Action Space Innovation

Unlike traditional dynamic action spaces, each of the 128 actions has a **fixed meaning**:

```python
Action 53 = Square 13, Direction NE  (always!)
Action 81 = Square 20, Direction NE  (always!)
```

This allows the network to learn **spatial patterns** like:
- "Moving from center toward NE is strong in middlegame"
- "Backrow moves are weak unless forced"

Instead of just relative preferences like:
- "In this position, the 3rd legal move is usually good"

## Architecture

### Components

- **Game Engine** (`checkers8x8/engine/`): Bitboard representation, move generation, American Checkers rules
- **Neural Network** (`checkers8x8/network/`): ResNet with 6 residual blocks, 128 filters
- **MCTS** (`checkers8x8/mcts/`): Monte Carlo Tree Search with PUCT selection
- **Training** (`checkers8x8/training/`): Self-play, replay buffer, training loop, evaluation
- **Visualization** (`checkers8x8/utils/`): Real-time pygame visualization

### Neural Network

**Input**: 8 planes × 8 × 8
- Player pieces (men, kings)
- Opponent pieces (men, kings)
- Legal move indicators
- Game state features

**Output**:
- Policy head: 128 logits (one per action)
- Value head: scalar in [-1, 1] (win/loss prediction)

**Architecture**: Conv → 6 ResNet blocks → Policy head + Value head

### MCTS

**Selection**: PUCT formula balances exploration vs exploitation
**Expansion**: Add new nodes when visited
**Evaluation**: Neural network predicts policy and value
**Backup**: Propagate values up the tree
**Action Selection**: Visit counts → training policy

## Configuration

Edit `config8x8.py` to customize hyperparameters:

```python
# Speed vs quality tradeoff
MCTS_SIMS_SELFPLAY = 100      # Simulations per move (50-200)
GAMES_PER_ITERATION = 20      # Games per iteration (10-50)

# Resources
NUM_WORKERS = 2               # Parallel self-play workers
BUFFER_SIZE = 20_000          # Replay buffer capacity (10K-100K)
BATCH_SIZE = 256              # Training batch size

# Devices
DEVICE = "mps"                # Training: "mps"/"cuda"/"cpu"
SELFPLAY_DEVICE = "cpu"       # Self-play: "cpu" (faster for MCTS)

# Learning
LEARNING_RATE = 0.001
MIN_SAMPLE_REUSE = 10         # Early training passes per state
MAX_SAMPLE_REUSE = 30         # Mature training passes per state
```

## Performance

### Training Speed (M1 Pro)

- **Self-play** (20 games, CPU): ~90s
- **Training** (dynamic steps, MPS): ~15s
- **Total**: ~105s per iteration

### Expected Learning Curve

| Iterations | Loss | Behavior |
|------------|------|----------|
| 1-20 | 4.0 → 1.5 | Learns legal moves, basic captures |
| 20-50 | 1.5 → 1.0 | Develops tactics (forks, pins) |
| 50-100 | 1.0 → 0.8 | Strategic patterns emerge |
| 100+ | 0.8 → 0.6 | Strong tactical play, endgame skill |

### Memory Usage

- Network: 8.8 MB
- Replay buffer (20K): ~80 MB
- Total: ~100 MB (very efficient!)

## Advanced Usage

### Resume from Checkpoint

```bash
# Auto-detect latest checkpoint
python scripts8x8/train.py --resume --iterations 150

# Load specific checkpoint
python scripts8x8/train.py --resume checkpoints8x8/checkpoint_iter_50.pt --iterations 150
```

### Change Buffer Size Mid-Training

You can increase `BUFFER_SIZE` when resuming. Existing data is preserved:

```python
# In config8x8.py
BUFFER_SIZE = 50_000  # Increased from 20_000
```

Then resume normally - a warning appears but training continues with larger capacity.

### Run Tests

```bash
# Full integration test
python checkers8x8/tests/test_integration.py

# Test individual components
python checkers8x8/network/resnet.py
python checkers8x8/engine/game.py
python checkers8x8/mcts/mcts.py
```

### Build C++ Extension (Optional)

For ~10× faster MCTS:

```bash
python setup.py build_ext --inplace
```

## Troubleshooting

**MPS/GPU errors**: Set `DEVICE = "cpu"` in `config8x8.py`

**Out of memory**: Reduce `BUFFER_SIZE` or `BATCH_SIZE` in `config8x8.py`

**Slow training**: Reduce `MCTS_SIMS_SELFPLAY` to 50 (2× speedup)

**Loss diverging**: Lower learning rate to 0.0005 or increase `GRAD_CLIP`

**Import errors**: Ensure you're using the correct Python environment with PyTorch installed

## Project Structure

```
checkers/
├── config8x8.py              # All hyperparameters
├── scripts8x8/
│   └── train.py              # Training entry point
├── checkers8x8/              # Main implementation
│   ├── engine/               # Game rules and bitboards
│   ├── network/              # Neural network (ResNet)
│   ├── mcts/                 # Monte Carlo Tree Search
│   ├── training/             # Self-play, replay buffer, trainer
│   ├── utils/                # Visualization
│   └── tests/                # Integration tests
├── checkpoints8x8/           # Saved models
└── logs8x8/                  # Training metrics
```

## Technical Details

### Perspective Switching

The current player is always at the bottom (rows 5-7). After each move:
1. Apply move to bitboards
2. Flip board vertically
3. Swap player/opponent bitboards

This ensures consistent neural network input representation.

### Action Encoding

Fixed action space with consistent meaning:

```python
action_index = from_square * 4 + direction

Directions:
  0 = NW (Northwest)
  1 = NE (Northeast)
  2 = SW (Southwest)
  3 = SE (Southeast)

Total: 32 squares × 4 directions = 128 actions
```

### Legal Action Masking

Policy logits are masked before softmax:

```python
masked_logits = logits.clone()
masked_logits[~legal_mask] = -inf  # Mask illegal actions
policy = softmax(masked_logits)     # Only legal actions have probability
```

## Why This Works

### Traditional (Dynamic) Action Space

```
Position A: legal_moves = [m1, m2, m3]
action_0 → m1
action_1 → m2

Position B: legal_moves = [m4, m5]
action_0 → m4  (different move!)
action_1 → m5
```

Network learns: "In positions like A, prefer action_1 over action_0"

### Fixed Action Space

```
Position A: legal_actions = {53: m1, 81: m2, 92: m3}
action_53 → m1 (Square 13, NE)
action_81 → m2 (Square 20, NE)

Position B: legal_actions = {53: m4, 67: m5}
action_53 → m4 (SAME SQUARE, SAME DIRECTION!)
action_67 → m5
```

Network learns: "Moving from square 13 toward NE (action 53) is strong in these types of positions"

This enables **transfer learning** across similar positions and **spatial pattern recognition**.

## Success Metrics

After **100 iterations**, expect:
- Loss: 0.8-1.5
- Games lasting 30-60 moves (structured play)
- Captures prioritized correctly
- Basic tactical awareness

After **500 iterations**, expect:
- Loss: 0.6-0.8
- Strong tactical play
- Opening principles
- Endgame technique
- Beats random player 95%+ of games

## References

- [AlphaZero Paper (Silver et al., 2017)](https://arxiv.org/abs/1712.01815)
- [AlphaGo Zero Paper (Silver et al., 2017)](https://www.nature.com/articles/nature24270)
- [MCTS Survey](https://ieeexplore.ieee.org/document/6145622)

## License

MIT License - See LICENSE file for details

---

**Built with PyTorch, NumPy, and Python 3.13**

Train your AI from zero to hero through pure self-play! 🚀
