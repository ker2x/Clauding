# AlphaZero-Style Checkers AI

A complete implementation of AlphaZero-style reinforcement learning for 10x10 International Draughts, trained purely through self-play with no initial data.

## Features

- **Complete Game Engine**: Bitboard-based 10x10 International Draughts with full rules
- **ResNet Neural Network**: 6 residual blocks, 128 filters (~2.7M parameters)
- **Monte Carlo Tree Search**: PUCT algorithm with neural network guidance
- **Self-Play Training**: AlphaZero-style learning from scratch
- **Optimized for CPU**: Efficient implementation for 8-core CPUs
- **Comprehensive Logging**: CSV metrics and matplotlib visualizations

## System Requirements

- Python 3.8+
- PyTorch 2.0+
- 8-core CPU recommended (configurable for 4 or 12+ cores)
- ~2GB RAM (for 500K replay buffer)
- ~10GB disk space (for checkpoints and logs)

## Installation

```bash
# Install dependencies
pip install torch numpy gymnasium matplotlib pandas

# Verify installation
python -c "import torch; print(f'PyTorch {torch.__version__}')"
```

## Quick Start

### 1. Run Tests

```bash
# Test game engine
python tests/test_engine.py

# Test network
python checkers/network/resnet.py

# Test MCTS (quick test with 50 simulations)
python -c "import sys; sys.path.insert(0, 'checkers'); from checkers.mcts.mcts import MCTS; from checkers.network.resnet import CheckersNetwork; from checkers.engine.game import CheckersGame; net = CheckersNetwork(); game = CheckersGame(); mcts = MCTS(net, num_simulations=50); policy = mcts.search(game); print(f'Policy shape: {policy.shape}, sum: {policy.sum():.4f}')"
```

### 2. Start Training

```bash
# Start training from scratch (100 iterations)
python scripts/train.py --iterations 100

# Resume from checkpoint
python scripts/train.py --resume checkpoints/checkpoint_iter_50.pt --iterations 50

# Use specific device
python scripts/train.py --device cpu --iterations 100
```

### 3. Monitor Training

```bash
# View training metrics
python checkers/utils/visualization.py logs/training_log.csv

# Or directly:
python -c "from checkers.utils.visualization import print_training_summary; print_training_summary()"
```

## Configuration

Edit `config.py` to customize hyperparameters:

```python
# Key settings
MCTS_SIMS_SELFPLAY = 300      # MCTS simulations per move
GAMES_PER_ITERATION = 100      # Self-play games per iteration
NUM_WORKERS = 6                # Parallel game generation
BATCH_SIZE = 256               # Training batch size
LEARNING_RATE = 0.001          # Adam learning rate
```

## Training Timeline

**Hardware**: 8-core CPU (M1/M2 Mac or equivalent)

| Iterations | Time (days) | Skill Level |
|------------|-------------|-------------|
| 50         | 2.2         | Learns basics |
| 100        | 4.4         | Intermediate tactics |
| 200        | 8.8         | Strong tactical play |
| 500        | 22          | Advanced/expert level |

**Estimated times per iteration** (~63 minutes):
- Self-play: 50 minutes (100 games)
- Training: 10 minutes (100 gradient steps)
- Evaluation: 2.5 minutes (every 10 iterations)

## Project Structure

```
checkers/
├── checkers/
│   ├── engine/          # Game logic (bitboards, moves, rules)
│   ├── env/             # Gymnasium environment
│   ├── network/         # ResNet architecture & augmentation
│   ├── mcts/            # Monte Carlo Tree Search
│   ├── training/        # Training loop, self-play, evaluation
│   └── utils/           # Checkpoints, visualization, logging
├── scripts/
│   └── train.py         # Main training script
├── tests/               # Unit tests
├── config.py            # Hyperparameters
├── checkpoints/         # Saved models
└── logs/                # Training metrics (CSV)
```

## Architecture Details

### Neural Network
```
Input: (8, 10, 10) - 8 feature planes
  ├─ Plane 0-1: Current player pieces (men, kings)
  ├─ Plane 2-3: Opponent pieces (men, kings)
  ├─ Plane 4: Legal move destinations
  ├─ Plane 5-6: Repetition indicators
  └─ Plane 7: Constant bias

ResNet Tower:
  ├─ Initial Conv: 8 → 128 filters
  ├─ 6 Residual Blocks (128 filters each)
  ├─ Policy Head → 150 action logits
  └─ Value Head → scalar value (-1 to +1)

Parameters: 2,682,071
```

### MCTS Configuration
- **Simulations**: 300 (self-play), 400 (evaluation)
- **Exploration**: c_puct = 1.0
- **Temperature**: 1.0 (first 15 moves), 0.0 (greedy after)
- **Dirichlet noise**: α=0.3, ε=0.25 (root exploration)

### Training Pipeline
1. **Self-Play**: 100 games with MCTS (6 workers)
2. **Replay Buffer**: 500K capacity, recency-biased sampling
3. **Training**: 100 gradient steps, 8x data augmentation
4. **Evaluation**: Every 10 iterations (50 games)
5. **Promotion**: If win rate > 55%, replace best model

## Usage Examples

### Play a Single Game

```python
from checkers.engine.game import CheckersGame

game = CheckersGame()
print(game.render())

while not game.is_terminal():
    legal_moves = game.get_legal_moves()
    print(f"\n{len(legal_moves)} legal moves")

    # Random move
    import random
    move = random.choice(legal_moves)
    game.make_move(move)
    print(game.render())

print(f"Winner: {game.get_winner()}")
```

### Use MCTS for Move Selection

```python
from checkers.engine.game import CheckersGame
from checkers.network.resnet import CheckersNetwork
from checkers.mcts.mcts import MCTS
import torch

# Setup
network = CheckersNetwork()
network.eval()

game = CheckersGame()
mcts = MCTS(network, num_simulations=300)

# Get move
policy = mcts.search(game)
best_action = mcts.get_best_action()

legal_moves = game.get_legal_moves()
game.make_move(legal_moves[best_action])
```

### Load and Use Trained Model

```python
from checkers.network.resnet import CheckersNetwork
from checkers.utils.checkpoint import load_model_for_inference

network = CheckersNetwork()
load_model_for_inference(network, "checkpoints/best_model.pt")
network.eval()

# Use for MCTS...
```

## Performance Optimization

### CPU Optimization
- **Threads**: Set `NUM_THREADS = 8` in config
- **Workers**: 6 workers for 8-core, 3 for 4-core, 10 for 12+ core
- **Batch size**: 256 (can reduce to 128 if memory limited)

### Speed vs Quality Trade-offs
- Reduce `MCTS_SIMS_SELFPLAY` to 200 → 2x faster
- Reduce `GAMES_PER_ITERATION` to 50 → 2x faster
- Disable augmentation → 8x less data, faster training

### Memory Usage
- Network: ~10 MB
- Replay buffer: ~1.5 GB (500K samples)
- Reduce `BUFFER_SIZE` to 250_000 if needed

## Troubleshooting

### Training is too slow
- Reduce MCTS simulations: `MCTS_SIMS_SELFPLAY = 200`
- Reduce games per iteration: `GAMES_PER_ITERATION = 50`
- Reduce workers if CPU overhead is high

### Out of memory
- Reduce buffer size: `BUFFER_SIZE = 250_000`
- Reduce batch size: `BATCH_SIZE = 128`
- Disable augmentation temporarily

### Loss not decreasing
- Check buffer has enough samples (wait for 10K+)
- Verify legal move masking is working
- Check for NaN values in gradients

### Win rate stuck at 50%
- This is expected early in training
- Models need 50-100 iterations to differentiate
- Ensure evaluation uses different random seeds

## Advanced Features

### Custom Network Architecture
Modify `config.py`:
```python
NUM_FILTERS = 256  # Increase from 128
NUM_RES_BLOCKS = 10  # Increase from 6
```

### Resume Training
```bash
python scripts/train.py --resume checkpoints/checkpoint_iter_100.pt --iterations 100
```

### Distributed Training
Modify `self_play.py` to use multiprocessing for parallel game generation across multiple machines.

## Citation

This implementation is based on the AlphaZero algorithm:

```
Silver, D., Hubert, T., Schrittwieser, J. et al.
A general reinforcement learning algorithm that masters chess, shogi, and Go through self-play.
Science 362, 1140-1144 (2018).
```

## License

MIT License - See LICENSE file for details

## Contributing

Contributions welcome! Areas for improvement:
- Distributed training support
- GUI for human vs AI play
- Opening book generation
- Stronger evaluation metrics (ELO tracking)
- Multi-GPU support

## Acknowledgments

- AlphaGo Zero / AlphaZero papers
- PyTorch team
- Gymnasium environment framework
