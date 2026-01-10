# Training Visualization

Real-time matplotlib-based visualization for monitoring training progress.

## Usage

```bash
# Train with visualization
python scripts8x8/train.py --iterations 50 --visualize

# Without visualization (faster)
python scripts8x8/train.py --iterations 50
```

## What It Shows

The visualizer displays 4 real-time plots:

### 1. Total Loss (Top Left)
- Combined policy + value loss
- **Target**: Decreasing from ~4.0 to < 1.0

### 2. Policy Loss (Top Right)
- Cross-entropy loss for action predictions
- **Target**: Decreasing from ~3.0 to < 0.8

### 3. Value Loss (Bottom Left)
- MSE loss for position evaluation
- **Target**: Decreasing from ~1.0 to < 0.3

### 4. Buffer Size (Bottom Right)
- Number of training examples in replay buffer
- **Target**: Growing to ~50,000-100,000

## Expected Behavior

### Iterations 1-10
- **Total Loss**: 4.0 → 2.0 (rapid initial learning)
- **Buffer**: 0 → 5,000 samples

### Iterations 10-30
- **Total Loss**: 2.0 → 1.2 (learning tactics)
- **Buffer**: 5,000 → 20,000 samples

### Iterations 30-50
- **Total Loss**: 1.2 → 0.8 (refinement)
- **Buffer**: 20,000 → 50,000 samples

## Tips

1. **Keep window visible** - Updates happen after each iteration
2. **Smooth curves** - Training is working well
3. **Spiky curves** - Normal, especially early on
4. **Plateau** - May need to adjust hyperparameters

## Keyboard Shortcuts

- **Close window** - Stops visualization but training continues
- **Ctrl+C** - Stop training and close everything

## Technical Details

- Updates every iteration (~2 minutes)
- Shows last N iterations (default: all)
- Auto-scales axes for best view
- Non-blocking (training runs in main thread)

## Troubleshooting

### "FigureCanvas is non-interactive"
This is normal when matplotlib can't find a display. Visualization will be disabled but training continues.

### Window not updating
Click on the window to bring it to focus. Updates happen automatically.

### Out of memory
Disable visualization: remove `--visualize` flag

---

**Pro tip**: Keep the visualization window on a second monitor while training!
