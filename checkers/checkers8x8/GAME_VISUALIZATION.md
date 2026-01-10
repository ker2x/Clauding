# Real-Time Game Visualization During Self-Play

Watch your AI learn checkers in real-time with pygame visualization!

## Features

### Pygame Window Shows:
1. **8x8 Checkerboard** - Classic checkers board layout
2. **Pieces** - Red (Player 1) and Black (Player 2), with golden crowns for kings
3. **Policy Heatmap** - Blue overlay showing where AI is thinking of moving
4. **Move Counter** - Current move number
5. **Player Info** - Whose turn it is
6. **Legend** - What each piece/color means

## Usage

```bash
# Train with visualization
python scripts8x8/train.py --iterations 5 --visualize

# Without visualization (faster)
python scripts8x8/train.py --iterations 5
```

## What You'll See

### During Self-Play

**Pygame Window:**
- Board updates in real-time as AI plays
- Blue heatmap shows AI's policy (where it's considering moves)
- Brighter blue = higher probability
- Watch pieces move as MCTS selects actions
- ~30 FPS smooth animation

**Matplotlib Window** (separate):
- Training metrics (loss curves)
- Updates after each iteration completes

### Visual Legend

**Pieces:**
- 🔴 Red circle = Player 1 man
- ⚫ Black circle = Player 2 man
- 🔴👑 Red with gold crown = Player 1 king
- ⚫👑 Black with gold crown = Player 2 king

**Overlay:**
- 💙 Blue highlight = AI policy (thinking)
- Brighter = higher probability

## Example Game Progression

### Early Game (Moves 1-10)
- Pieces advance forward
- Blue heatmap shows uniform exploration
- Network is still learning

### Mid Game (Moves 10-30)
- Captures start happening
- Blue heatmap concentrates on tactical moves
- Network shows preference for center control

### Late Game (Moves 30+)
- Kings dominate the board
- Blue heatmap shows endgame technique
- Network demonstrates learned patterns

## Performance

- **FPS**: 30 (smooth animation)
- **Overhead**: ~5% slower than no visualization
- **Window Size**: 940×640 pixels (board + info panel)
- **Updates**: Real-time during MCTS moves

## Tips

### Best Practice
1. **Position windows**: Pygame on left, matplotlib on right
2. **Watch first game** of each iteration to see learning progress
3. **Look for patterns** in policy heatmap over time
4. **Compare early vs late** iterations

### What to Watch For

**Signs of Learning:**
- ✅ Policy heatmap becomes more focused over iterations
- ✅ Captures are prioritized (forced by rules, but network learns value)
- ✅ Center control emerges
- ✅ King promotion is valued

**Signs of Problems:**
- ❌ Random-looking moves after many iterations
- ❌ No change in policy focus over time
- ❌ Repeated positions (might indicate draw loops)

## Technical Details

### Visualization Pipeline

```
Game State (Bitboards)
    ↓
to_neural_input() → 8×8×8 tensor
    ↓
Network → Policy (128 actions)
    ↓
Action Decoder → Board squares
    ↓
Pygame Renderer → Visual display
```

### Policy Heatmap Calculation

The 128 actions (32 squares × 4 directions) are aggregated by destination square:

```python
For each action with probability > 0.01:
    from_square = action // 4
    direction = action % 4
    to_square = from_square + direction_offset
    heatmap[to_square] += policy[action]
```

This shows "where the AI wants to move to" rather than "which specific action."

### Coordinate System

- **Board**: Row 0 (top) to Row 7 (bottom)
- **Squares**: 0-31 (dark squares only, row-major order)
- **Actions**: 0-127 (from_square × 4 + direction)

### Performance Optimization

- Rendering happens in main thread (non-blocking)
- 30 FPS cap prevents excessive CPU usage
- Only updates during move selection (~100 times per game)
- Can be disabled for faster training

## Keyboard Controls

- **Close window** → Visualization stops, training continues
- **Ctrl+C** → Stop everything
- **No interaction needed** → Just watch!

## Comparison with 10x10 Version

| Feature | 10x10 (Old) | 8x8 (New) |
|---------|-------------|-----------|
| Board size | 10×10 (50 squares) | 8×8 (32 squares) |
| Window | Larger | Compact |
| Piece types | Men + flying kings | Men + regular kings |
| Policy overlay | 150 actions | 128 actions |
| Clarity | More complex | Clearer patterns |

---

## Troubleshooting

### "pygame.error: No available video device"
You're running on a headless server. Remove `--visualize` flag.

### Window appears but is black
Wait a few seconds - first render takes time.

### Window freezes
This is normal during MCTS search (can take 10-30 seconds per move).

### No policy overlay visible
Network might be outputting uniform probabilities (early in training).

---

**Watch your AI evolve from random player to tactical master! 🎮**

The visualization makes the learning process tangible and exciting.
