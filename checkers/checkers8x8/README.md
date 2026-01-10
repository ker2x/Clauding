# 8x8 American Checkers with Fixed Action Space

**A complete AlphaZero-style RL implementation designed for optimal learning.**

## 🎯 Key Innovation: Fixed Action Space

### The Problem (Old 10x10 System)

In the previous 10x10 implementation, action indices had **dynamic meaning**:

```python
# Position A
legal_moves = [Move(5→10), Move(7→12)]
action_0 = Move(5→10)  # "First legal move"

# Position B
legal_moves = [Move(3→8), Move(9→14)]
action_0 = Move(3→8)   # DIFFERENT MOVE!
```

**Result**: Network couldn't learn move patterns, only relative preferences.

### The Solution (New 8x8 System)

**Fixed action encoding**: Each action index always means the same move.

```python
Action = from_square * 4 + direction

action_53 = Square 13, Direction NE  # ALWAYS!
action_81 = Square 20, Direction NE  # ALWAYS!
```

**Total action space**: 32 squares × 4 directions = **128 actions**

**Result**: Network learns spatial patterns and tactical concepts.

---

## 🏗️ Architecture

### Game Engine
- **Bitboard representation**: 32 playable squares on 8×8 board
- **American Checkers rules**:
  - Men move forward 1 square, can capture backward
  - Kings move 1 square in any direction (no flying)
  - Forced captures
  - Capture chains
  - Promotion at opposite end

### Neural Network
- **Input**: 8 planes × 8 × 8 (pieces, legal moves, game state)
- **Architecture**: ResNet with 6 residual blocks, 128 filters
- **Output**:
  - Policy head: 128 logits (fixed action space)
  - Value head: Scalar in [-1, 1]
- **Parameters**: 2.3M

### MCTS
- **PUCT algorithm** with neural network guidance
- **100 simulations** per move during self-play
- **Direct action masking**: No move list mapping needed
- **Dirichlet noise** at root for exploration

### Training Loop
```
Self-play (CPU) → Replay Buffer → Training (MPS) → Repeat

1. Generate games with MCTS (on CPU - faster for small batches)
2. Store (state, policy, value) in replay buffer
3. Train network on sampled batches (on MPS - faster for gradients)
4. Repeat
```

---

## 🚀 Quick Start

### Installation

```bash
# Already set up in your existing environment
# Requires: torch, numpy
```

### Run Training

```bash
# Train for 100 iterations
python scripts8x8/train.py --iterations 100

# Custom seed
python scripts8x8/train.py --iterations 50 --seed 123
```

### Configuration

Edit `config8x8.py` to customize:
- `MCTS_SIMS_SELFPLAY`: Simulations per move (default: 100)
- `GAMES_PER_ITERATION`: Games per iteration (default: 10)
- `TRAINING_STEPS_PER_ITERATION`: Gradient steps (default: 100)
- `BATCH_SIZE`: Mini-batch size (default: 256)
- `DEVICE`: Training device (`"mps"`, `"cuda"`, `"cpu"`)
- `SELFPLAY_DEVICE`: Self-play device (default: `"cpu"`)

---

## 📊 Performance

### Training Speed (1 iteration, M1 Pro)
- Self-play (10 games, CPU): ~99s
- Training (100 steps, MPS): ~10s
- **Total**: ~110s per iteration

### Memory Usage
- Network: 8.8 MB
- Replay buffer (100K samples): ~200 MB
- **Total**: ~210 MB

### Expected Learning
- **Iterations 1-20**: Learns basic moves (loss: 4.0 → 1.5)
- **Iterations 20-50**: Learns captures and tactics
- **Iterations 50-100**: Strategic play emerges
- **Iterations 100+**: Strong tactical understanding

---

## 🧪 Testing

### Run Integration Test

```bash
cd checkers8x8
python tests/test_integration.py
```

Tests all components:
- ✓ Game engine
- ✓ Action encoding (all 128 actions)
- ✓ Neural network
- ✓ MCTS search
- ✓ Self-play generation
- ✓ Replay buffer
- ✓ Training step

### Run Individual Components

```bash
# Test bitboard
python engine/bitboard.py

# Test action encoder
python engine/action_encoder.py

# Test move generation
python engine/moves.py

# Test game state
python engine/game.py

# Test network
python network/resnet.py

# Test MCTS
python mcts/mcts.py
```

---

## 📁 Project Structure

```
checkers8x8/
├── engine/
│   ├── bitboard.py          # 32-square bitboard representation
│   ├── action_encoder.py    # Fixed action space (128 actions)
│   ├── moves.py             # Move generation (American rules)
│   └── game.py              # Game state manager
├── network/
│   └── resnet.py            # ResNet (8×8×8 → 128 + value)
├── mcts/
│   ├── node.py              # MCTS tree node
│   └── mcts.py              # Monte Carlo Tree Search
├── training/
│   ├── replay_buffer.py     # Recency-weighted replay buffer
│   ├── self_play.py         # Self-play game generation
│   └── trainer.py           # Main training loop
└── tests/
    └── test_integration.py  # Full system test

config8x8.py                 # Configuration
scripts8x8/
└── train.py                 # Training entry point
```

---

## 🎓 How It Learns

### Fixed Action Space Benefits

**Old (dynamic)**: Network learns *"In positions like this, the 3rd legal move is usually good"*

**New (fixed)**: Network learns *"Moving from square 13 toward NE (action 53) is strong in middlegame positions"*

### What the Network Learns

1. **Positional concepts**: Center control, piece activity
2. **Tactical patterns**: Forks, pins, skewers
3. **Strategic plans**: Pawn structure, king safety
4. **Endgame technique**: King + men vs king

### Training Signal

```
MCTS (100 simulations) → Improved policy π
Network policy p ← learns to match π
Value V ← learns game outcomes
```

The network becomes a better "intuition" for MCTS, which becomes better search, which trains better network = virtuous cycle!

---

## 🔬 Technical Details

### Perspective Switching

Current player is **always at bottom** (rows 5-7). After each move:
1. Apply move
2. Flip board vertically
3. Swap player/opponent bitboards

This ensures consistent neural input representation.

### Action Encoding Examples

```
action_0   = Square 0,  Direction NW
action_1   = Square 0,  Direction NE
action_53  = Square 13, Direction NE
action_127 = Square 31, Direction SE
```

### Legal Action Masking

```python
# Network outputs 128 logits
masked_logits = logits.clone()
masked_logits[~legal_mask] = -inf  # Mask illegal actions
policy = softmax(masked_logits)    # Only legal actions have probability
```

---

## 🆚 Comparison: 8×8 vs 10×10

| Aspect | 10×10 (Old) | 8×8 (New) |
|--------|-------------|-----------|
| **Action Space** | Dynamic (150 max) | Fixed (128 always) |
| **Action Meaning** | Changes per position | Always consistent |
| **Board Size** | 50 squares | 32 squares |
| **Kings** | Flying (multi-square) | Normal (1 square) |
| **Complexity** | Very high | Moderate |
| **Learning** | Relative preferences | Spatial patterns |
| **Network Size** | 2.7M params | 2.3M params |
| **Training Speed** | Slower | Faster |

---

## 📈 Monitoring Training

### Logs

Training metrics are saved to `logs8x8/training_log.csv`:
- iteration, total_loss, policy_loss, value_loss
- buffer_size, games_played
- time_selfplay, time_training

### Checkpoints

Saved to `checkpoints8x8/`:
- `checkpoint_iter_N.pt` - Every 10 iterations
- Contains: network weights, optimizer state, iteration number

### Loss Interpretation

- **Policy loss**: How well network predicts MCTS policy (target: < 1.0)
- **Value loss**: How well network predicts game outcomes (target: < 0.3)
- **Total loss**: Sum of both (typical: 4.0 → 1.5 → 0.8 over 100 iterations)

---

## 🎮 Next Steps

### Immediate
1. **Train for 100 iterations** - See basic tactics emerge
2. **Evaluate against random** - Check playing strength
3. **Tune hyperparameters** - Optimize learning speed

### Advanced
1. **Add evaluation tournaments** - Best model vs new model
2. **Implement parallel self-play** - 8× faster data generation
3. **Add arena competition** - Test against baselines
4. **Export to ONNX** - Deploy for fast inference
5. **Build web UI** - Play against your AI

---

## 🐛 Troubleshooting

### MPS/GPU Errors
Set `DEVICE = "cpu"` and `SELFPLAY_DEVICE = "cpu"` in `config8x8.py`

### Out of Memory
Reduce `BUFFER_SIZE` or `BATCH_SIZE` in `config8x8.py`

### Slow Training
Reduce `MCTS_SIMS_SELFPLAY` (try 50 instead of 100)

### Loss Diverging
Check learning rate (try 0.0005) or increase `GRAD_CLIP`

---

## 🏆 Success Metrics

After 100 iterations of training, you should see:
- ✅ Loss below 1.5
- ✅ Games lasting 30-50 moves (not random)
- ✅ Captures being prioritized
- ✅ Some basic tactical awareness

After 500 iterations:
- ✅ Strong tactical play
- ✅ Opening knowledge
- ✅ Endgame technique
- ✅ Beats random player 100% of games

---

## 📚 References

- AlphaZero: [Silver et al., 2017](https://arxiv.org/abs/1712.01815)
- MCTS + Neural Networks: [Silver et al., 2016](https://www.nature.com/articles/nature16961)
- American Checkers Rules: [World Checkers Federation](https://www.fmjd.org/)

---

## ✨ Built With

- **PyTorch** - Neural network framework
- **NumPy** - Numerical computing
- **Python 3.13** - Language

---

**Happy Training! 🚀**

Your network is now learning actual chess-like spatial patterns instead of just relative move preferences. Watch it grow from random moves to tactical mastery!
