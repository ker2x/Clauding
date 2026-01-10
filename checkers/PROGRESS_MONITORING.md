# Real-Time Progress Monitoring

The training system now includes comprehensive real-time progress monitoring to help you track training progress.

## Features

### 🎯 **Iteration Progress Display**

Each iteration shows:
```
======================================================================
Iteration 42/100 (42.0%)
======================================================================
⏱️  Elapsed: 45.2h | ETA: 62.1h
⏳ Iteration time: 63.5s (self-play: 50.2s, training: 10.8s, eval: 2.5s)
📉 Loss: 2.3451 ↓ (policy: 1.5234, value: 0.8217)
💾 Buffer: 250,000 samples (50.0% full)
🎯 Win rate: 58.2% ↑ ⭐ PROMOTED!
[████████████████████░░░░░░░░░░] 42.0%
```

### 📊 **Progress Bars**

**Self-Play Progress:**
```
[1/4] Self-play (100 games)...
  [████████████████████████████████████████] 100/100 games | 4523 examples
```

**Training Progress:**
```
[2/4] Training (100 steps)...
  [████████████████████████████████████████] 100/100 steps | loss: 2.3451
```

### 📈 **Trend Indicators**

- **↑** - Metric improving (or increasing)
- **↓** - Metric decreasing (good for losses)
- **→** - Metric stable

The system compares recent averages to older averages to detect trends.

### ⏱️ **Time Tracking**

- **Elapsed**: Total time since training started
- **ETA**: Estimated time remaining (based on last 10 iterations)
- **Iteration breakdown**: Time for each phase

### 💾 **Buffer Status**

Shows current replay buffer size and percentage full:
- Important for knowing when buffer is saturated
- Training quality improves as buffer fills

### 🎯 **Evaluation Updates**

When evaluation runs (every 10 iterations):
- Win rate against best model
- Promotion indicator (⭐) when model improves
- Trend arrow showing if win rate is improving

## Usage

### Standard Training

```bash
python scripts/train.py --iterations 100
```

You'll see real-time progress updates automatically!

### Demo (Quick Preview)

```bash
python scripts/demo_progress.py
```

Runs a quick 1-iteration demo to show all progress features.

### View Historical Progress

```bash
# Print training summary
python checkers/utils/visualization.py logs/training_log.csv

# Or directly:
python -c "from checkers.utils.visualization import print_training_summary; print_training_summary()"
```

## Example Output

```
======================================================================
Iteration 1/100 (1.0%)
======================================================================
⏱️  Elapsed: 1.1m | ETA: calculating...
⏳ Iteration time: 63.2s (self-play: 50.5s, training: 10.2s, eval: 2.5s)
📉 Loss: 10.2345 (policy: 8.1234, value: 2.1111)
💾 Buffer: 4523 samples (0.9% full)
[█░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░] 1.0%

======================================================================
Iteration 10/100 (10.0%)
======================================================================
⏱️  Elapsed: 10.5m | ETA: 94.5m
⏳ Iteration time: 62.8s (self-play: 49.8s, training: 10.5s, eval: 2.5s)
📉 Loss: 5.6789 ↓ (policy: 4.2345, value: 1.4444)
💾 Buffer: 45,230 samples (9.0% full)
🎯 Win rate: 52.0% ↑
[████░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░] 10.0%

...

======================================================================
Iteration 100/100 (100.0%)
======================================================================
⏱️  Elapsed: 105.2h | ETA: 0.0s
⏳ Iteration time: 63.1s (self-play: 50.1s, training: 10.4s, eval: 2.6s)
📉 Loss: 1.8765 ↓ (policy: 1.2345, value: 0.6420)
💾 Buffer: 452,300 samples (90.5% full)
🎯 Win rate: 67.5% ↑ ⭐ PROMOTED!
[████████████████████████████████████████] 100.0%

======================================================================
🎉 TRAINING COMPLETE!
======================================================================
Total time: 105.2h
Final loss: 1.8765 (improved 81.7%)
Best win rate: 67.5%
======================================================================
```

## Technical Details

### ProgressMonitor Class

Located in `checkers/utils/progress.py`:

- **update()**: Called after each iteration with metrics
- **print_progress()**: Formats and displays progress
- **_get_trend()**: Calculates trend indicators
- **print_summary()**: Final training summary

### Metrics Tracked

- Total loss, policy loss, value loss
- Buffer size and utilization
- Win rate vs best model
- Iteration timing breakdown
- ETA based on rolling average

### Performance Impact

- Minimal overhead (~0.1% of iteration time)
- Progress updates don't slow down training
- All calculations are lightweight

## Customization

### Disable Progress Bars

If you prefer minimal output, you can comment out the progress bar prints in:
- `checkers/training/self_play.py` (line 190-207)
- `checkers/training/trainer.py` (line 257-310)

### Change Update Frequency

Training progress bar updates every 10 steps by default. To change:

```python
# In trainer.py, line 302
if (step + 1) % 10 == 0:  # Change 10 to desired frequency
```

### Customize Display

Edit `checkers/utils/progress.py` to:
- Change bar width (default: 50 characters)
- Modify emoji icons
- Add/remove metrics
- Change trend calculation window

## Troubleshooting

### Progress bars not updating

- Make sure terminal supports ANSI codes
- Try using `flush=True` in prints
- Check that stdout is not buffered

### ETA shows "calculating..."

- Normal for first few iterations
- Needs 2+ iterations to estimate
- Will stabilize after ~5 iterations

### Trends not showing

- Needs 5+ data points
- Compare window: last 5 vs previous 5
- Threshold: 5% change required

## Next Steps

After seeing progress in action:

1. **Monitor trends** - Watch for loss decreasing and win rate increasing
2. **Check ETA** - Plan when to check results
3. **Track promotions** - Each ⭐ means the model improved
4. **View plots** - Use visualization.py for detailed charts

Happy training! 🚀
