# Training Logging Guide

This document explains the comprehensive logging system added to the training script.

## Overview

All training runs now generate detailed logs in the `logs/` directory. These logs allow you to:
- Monitor training progress in real-time
- Analyze training metrics programmatically
- Debug issues during long training runs
- Reproduce experiments with exact configurations

## Log Files Created

### 1. `logs/training_metrics.csv`
**Per-episode training metrics** - Written after every episode

**Columns**:
- `episode`: Episode number (1, 2, 3, ...)
- `total_steps`: Cumulative steps across all episodes
- `episode_steps`: Steps in this specific episode
- `reward`: Episode total reward
- `avg_loss`: Average loss for this episode
- `epsilon`: Current exploration rate
- `buffer_size`: Number of experiences in replay buffer
- `elapsed_time_sec`: Time since training started (seconds)
- `avg_reward_100`: Moving average of last 100 episodes
- `timestamp`: When this episode completed

**Example**:
```csv
episode,total_steps,episode_steps,reward,avg_loss,epsilon,buffer_size,elapsed_time_sec,avg_reward_100,timestamp
1,1828,1828,-204.69,0.2418,0.9982,1828,7.22,-204.69,2025-11-03 19:58:29
2,3721,1893,-214.52,2.9262,0.9963,3721,14.29,-209.61,2025-11-03 19:58:36
```

**Usage**:
```bash
# View last 20 episodes
tail -20 logs/training_metrics.csv

# Parse with Python
python -c "
import pandas as pd
df = pd.read_csv('logs/training_metrics.csv')
print(df[['episode', 'total_steps', 'reward', 'epsilon']].tail())
"

# Get current training state
python -c "
import pandas as pd
df = pd.read_csv('logs/training_metrics.csv')
last = df.iloc[-1]
print(f'Episode: {last.episode}, Steps: {last.total_steps}, Epsilon: {last.epsilon:.4f}')
print(f'Reward: {last.reward:.2f}, Avg(100): {last.avg_reward_100:.2f}')
"
```

### 2. `logs/evaluation_metrics.csv`
**Evaluation results** - Written after each evaluation run

**Columns**:
- `episode`: Episode when evaluation occurred
- `total_steps`: Total training steps at evaluation time
- `eval_mean_reward`: Mean reward over evaluation episodes
- `eval_std_reward`: Standard deviation of evaluation rewards
- `eval_rewards`: List of all individual episode rewards
- `is_best`: 1 if this is the best model so far, 0 otherwise
- `elapsed_time_sec`: Time since training started
- `timestamp`: When evaluation completed

**Example**:
```csv
episode,total_steps,eval_mean_reward,eval_std_reward,eval_rewards,is_best,elapsed_time_sec,timestamp
100,45231,150.25,23.45,"[-145.2, 132.8, 167.3, 155.6, 150.4]",1,1234.56,2025-11-03 20:15:32
200,89456,234.67,18.92,"[220.5, 245.3, 238.9, 229.1, 240.5]",1,2456.78,2025-11-03 20:45:18
```

**Usage**:
```bash
# View all evaluations
cat logs/evaluation_metrics.csv

# Plot evaluation progress
python -c "
import pandas as pd
import matplotlib.pyplot as plt
df = pd.read_csv('logs/evaluation_metrics.csv')
plt.plot(df['episode'], df['eval_mean_reward'])
plt.xlabel('Episode')
plt.ylabel('Evaluation Reward')
plt.title('Evaluation Progress')
plt.grid(True)
plt.savefig('logs/eval_progress.png')
print('Saved to logs/eval_progress.png')
"
```

### 3. `logs/training.log`
**Human-readable timestamped log** - Complete training narrative

**Content**:
- Training session start with configuration
- Learning start milestone (when `learning_starts` reached)
- Episode progress (every 10 episodes)
- All evaluation episodes with individual rewards
- Best model saves and checkpoint saves
- Training session summary

**Example**:
```
======================================================================
Training Session Started
======================================================================
Timestamp: 2025-11-03 19:58:22
Device: mps
State mode: vector
Episodes: 2000
Learning starts: 10000 steps
======================================================================

[2025-11-03 19:58:29] >>> Learning started (reached 10000 steps)
[2025-11-03 19:58:36] Episode 10 | Steps: 452 (total: 4521) | Reward: -35.67 | Loss: 0.0987 | Epsilon: 0.9955 | Avg(100): -42.34
[2025-11-03 20:15:00] ==================================================
[2025-11-03 20:15:00] Evaluation at episode 100
[2025-11-03 20:15:05]   Eval episode 1/5: reward = 145.20
[2025-11-03 20:15:25] Evaluation complete | Mean: 150.25 | Std: 23.45
[2025-11-03 20:15:25] *** NEW BEST MODEL *** Reward: 150.25 (prev: N/A)
[2025-11-03 20:15:25] Saved: checkpoints/best_model.pt
```

**Usage**:
```bash
# Monitor training in real-time (while training is running)
tail -f logs/training.log

# View complete log
cat logs/training.log

# Search for specific events
grep "NEW BEST MODEL" logs/training.log
grep "Learning started" logs/training.log
grep "Checkpoint saved" logs/training.log
```

### 4. `logs/system_info.txt`
**Complete configuration snapshot** - For reproducibility

**Content**:
- Date and device used
- State mode and shape
- Environment configuration
- All hyperparameters
- Training parameters
- Resume settings

**Example**:
```
Training Configuration
======================================================================
Date: 2025-11-03 19:58:22
Device: mps
State mode: vector
State shape: (11,)

Environment:
  Name: CarRacing-v3
  Actions: 9 discrete
  Early termination: True (patience=100)

Agent Hyperparameters:
  Learning rate: 0.00025
  Gamma: 0.99
  Epsilon: 1.0 → 0.01 over 1000000 steps
  Buffer size: 100000
  Batch size: 32
```

**Usage**:
```bash
# Check configuration
cat logs/system_info.txt

# Compare configurations between runs
diff logs/system_info.txt ../other_run/logs/system_info.txt
```

## Real-Time Monitoring

### During Training

```bash
# Terminal 1: Run training
python train.py --episodes 2000 --learning-starts 10000

# Terminal 2: Monitor progress
tail -f logs/training.log
```

### Quick Status Check

```bash
# Get current training state
python -c "
import pandas as pd
df = pd.read_csv('logs/training_metrics.csv')
last = df.iloc[-1]
print(f'Episode {int(last.episode)}/{2000}')
print(f'Steps: {int(last.total_steps)}/1000000 ({last.total_steps/10000:.1f}%)')
print(f'Epsilon: {last.epsilon:.4f}')
print(f'Recent reward (avg 100): {last.avg_reward_100:.2f}')
print(f'Time elapsed: {last.elapsed_time_sec/3600:.2f} hours')
"
```

## Analysis Examples

### Plot Training Curves

```python
import pandas as pd
import matplotlib.pyplot as plt

# Load training data
df = pd.read_csv('logs/training_metrics.csv')

# Create figure with subplots
fig, axes = plt.subplots(3, 1, figsize=(10, 12))

# Plot rewards
axes[0].plot(df['episode'], df['reward'], alpha=0.3, label='Episode Reward')
axes[0].plot(df['episode'], df['avg_reward_100'], linewidth=2, label='100-ep Average')
axes[0].set_xlabel('Episode')
axes[0].set_ylabel('Reward')
axes[0].legend()
axes[0].grid(True)

# Plot loss
axes[1].plot(df['episode'], df['avg_loss'])
axes[1].set_xlabel('Episode')
axes[1].set_ylabel('Average Loss')
axes[1].grid(True)

# Plot epsilon
axes[2].plot(df['episode'], df['epsilon'])
axes[2].set_xlabel('Episode')
axes[2].set_ylabel('Epsilon')
axes[2].grid(True)

plt.tight_layout()
plt.savefig('logs/custom_analysis.png')
print('Saved to logs/custom_analysis.png')
```

### Find Best Episodes

```python
import pandas as pd

df = pd.read_csv('logs/training_metrics.csv')

# Top 10 episodes by reward
top_episodes = df.nlargest(10, 'reward')[['episode', 'reward', 'epsilon', 'total_steps']]
print("Top 10 episodes by reward:")
print(top_episodes.to_string(index=False))

# Check when agent started improving (reward > 0)
positive_rewards = df[df['reward'] > 0]
if len(positive_rewards) > 0:
    first_positive = positive_rewards.iloc[0]
    print(f"\nFirst positive reward at episode {int(first_positive.episode)}")
    print(f"Steps: {int(first_positive.total_steps)}, Epsilon: {first_positive.epsilon:.4f}")
```

### Estimate Time Remaining

```python
import pandas as pd

df = pd.read_csv('logs/training_metrics.csv')
target_episodes = 2000
target_steps = 1000000

last = df.iloc[-1]
current_episode = int(last.episode)
current_steps = int(last.total_steps)
elapsed_hours = last.elapsed_time_sec / 3600

# Estimate based on episodes
episodes_per_hour = current_episode / elapsed_hours
hours_remaining_ep = (target_episodes - current_episode) / episodes_per_hour

# Estimate based on steps
steps_per_hour = current_steps / elapsed_hours
hours_remaining_steps = (target_steps - current_steps) / steps_per_hour

print(f"Current: Episode {current_episode}/{target_episodes}, Steps {current_steps}/{target_steps}")
print(f"Elapsed: {elapsed_hours:.2f} hours")
print(f"\nEstimated time remaining:")
print(f"  By episodes: {hours_remaining_ep:.2f} hours")
print(f"  By steps: {hours_remaining_steps:.2f} hours")
```

## Tips

1. **Real-time monitoring**: Use `tail -f logs/training.log` to watch training progress live
2. **CSV is your friend**: All metrics are in CSV format for easy parsing
3. **System info**: Always check `system_info.txt` to remember your hyperparameters
4. **Pandas**: Use pandas for easy data manipulation and analysis
5. **Timestamps**: All events are timestamped for debugging timing issues
6. **Line buffering**: Logs are written with line buffering for real-time updates

## Troubleshooting

**Q: No log files created?**
A: Make sure you're running a recent version of `train.py` that includes the logging infrastructure.

**Q: CSV file is empty or truncated?**
A: The CSV is written after each episode. If training crashed, you'll have data up to the last completed episode.

**Q: Can't parse CSV with pandas?**
A: Install pandas: `pip install pandas matplotlib`

**Q: Training log not updating in real-time?**
A: The log uses line buffering. Make sure you're using `tail -f`, not just `tail`.
