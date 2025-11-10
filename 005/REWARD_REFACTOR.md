# Reward System Refactor: Checkpoint → Waypoint → Continuous Progress

## Summary

Refactored reward system three times to find working solution:
1. **Checkpoint → Waypoint**: Removed technical debt, implemented clean waypoint system
2. **Waypoint reward math fix**: Fixed negative rewards (increased rewards, reduced penalties)
3. **Waypoint → Continuous Progress**: Waypoints too sparse, switched to dense continuous progress tracking

## Changes Made

### 1. Removed Technical Debt

**Removed workarounds:**
- ✅ Sequential checkpoint enforcement logic
- ✅ Forward velocity check (vx > 0.1) to prevent backward driving
- ✅ Checkpoint wrapping logic
- ✅ Forward velocity reward (FORWARD_VEL_REWARD removed)
- ✅ Complex checkpoint size calculations

**Old system had:**
- 15 checkpoints with ~20 tiles each
- Anti-exploit checks for backward driving
- Forward velocity rewards as "hypothesis"
- Complex sequential enforcement

### 2. New Waypoint System

**Core Design (Option 2 from proposal):**
- 20 waypoints evenly distributed along track (~15 tiles apart)
- Reward when car reaches next waypoint (distance threshold: 5.0m)
- Sequential waypoints (must reach in order)
- No backward driving exploits (distance check naturally prevents it)

**Reward Structure:**
```python
# Per frame
reward = -STEP_PENALTY  # -0.5 per frame (mild time pressure)
reward -= wheels_offtrack * OFFTRACK_PENALTY  # -2.0 per wheel (if > 2 wheels off)

# When waypoint reached
reward += WAYPOINT_REWARD  # +200 per waypoint

# On lap completion
reward += LAP_COMPLETION_REWARD  # +500 bonus
```

**Total possible rewards:**
- Waypoints: 20 × 200 = +4000
- Lap completion: +500
- Total positive: +4500
- Step penalties: depends on speed (~-500 to -1000 for typical lap)
- Net expected: **+3500 to +4000 for successful lap** ✓

**Fixed (2nd commit):** Initial values were too low - step penalties outpaced waypoint rewards,
making crash-early optimal. Increased WAYPOINT_REWARD (50→200) and reduced STEP_PENALTY (1.0→0.5)
so rewards are always positive when making progress.

### 3. Code Changes

**env/car_racing.py:**
- Lines 64-71: Updated reward configuration constants
- Lines 222-232: Removed checkpoint logic from FrictionDetector
- Lines 396-399: Updated __init__ to waypoint system
- Lines 692-694: Waypoint tracking variables in reset()
- Lines 714-720: Waypoint indices calculation
- Lines 827-854: Removed forward velocity reward, added waypoint checking
- Lines 272-291: Updated docstring

**train.py:**
- Lines 61-64: Updated imports
- Lines 240-245: Updated logging

**CLAUDE.md:**
- Lines 335-367: Updated reward tuning guide

### 4. Benefits

**Simplicity:**
- ~50 lines of code removed
- No anti-exploit hacks
- Clear, understandable logic

**Robustness:**
- Distance-based waypoint detection
- Natural prevention of backward driving
- Fits existing waypoint observation system (20 waypoints already in state)

**Tunability:**
- Easy to adjust waypoint count (NUM_WAYPOINTS)
- Easy to adjust rewards (WAYPOINT_REWARD)
- Easy to adjust threshold (WAYPOINT_DISTANCE_THRESHOLD)

### 5. Risk & Fallback

**Risk:** Sparse rewards may slow initial learning

**Mitigation options if needed:**
1. Increase NUM_WAYPOINTS (25-30) for denser rewards
2. Increase WAYPOINT_DISTANCE_THRESHOLD (7.0m) for easier reach
3. Fall back to Option 1 (continuous progress tracking) if needed

**Monitoring:**
- Track waypoints reached per episode
- Monitor exploration behavior
- Compare learning curve to previous version

## Testing

- ✅ Python syntax valid (py_compile)
- ✅ All imports resolve correctly
- ⏳ Full training run needed to validate learning

## Configuration

All parameters at top of `env/car_racing.py`:
```python
NUM_WAYPOINTS = 20
WAYPOINT_REWARD = 200.0          # Fixed: 50.0 → 200.0 (4x increase)
LAP_COMPLETION_REWARD = 500.0
STEP_PENALTY = 0.5               # Fixed: 1.0 → 0.5 (less harsh)
OFFTRACK_PENALTY = 2.0
OFFTRACK_THRESHOLD = 2
WAYPOINT_DISTANCE_THRESHOLD = 5.0
```

## 3rd Refactor: Waypoint → Continuous Progress (3rd Commit)

### Problem with Waypoints
Training results showed waypoint system failed:
- Agent stuck at **124.5 reward every episode** (exactly 1 waypoint + time penalty)
- Reached waypoint 1, then stationary timeout at 150 steps
- **Critic loss >900** (should be <100) - critic couldn't learn value function
- 15-tile gap between waypoints too large for random exploration

### Solution: Continuous Progress Tracking
Switched to dense, continuous progress rewards:

**New Reward Structure:**
```python
# Per frame
reward = -STEP_PENALTY  # -0.5 per frame (mild time pressure)
reward -= wheels_offtrack * OFFTRACK_PENALTY  # -2.0 per wheel (if > 2 wheels off)

# Continuous progress (dense signal)
current_progress = furthest_tile_idx / len(track)  # 0.0 to 1.0
progress_delta = max(0, current_progress - last_progress)  # Only forward
reward += progress_delta * PROGRESS_REWARD_SCALE  # +4000 for full lap

# On lap completion
reward += LAP_COMPLETION_REWARD  # +500 bonus
```

**Benefits:**
- **Dense signal**: Reward every frame car reaches new furthest tile
- **Natural backward prevention**: Only forward progress counts
- **Smooth gradients**: Critic can learn stable value function
- **No local minima**: Can't get stuck between sparse rewards

**Configuration:**
```python
PROGRESS_REWARD_SCALE = 4000.0  # Full lap reward
LAP_COMPLETION_REWARD = 500.0
STEP_PENALTY = 0.5
OFFTRACK_PENALTY = 2.0
OFFTRACK_THRESHOLD = 2
```

**Expected Performance:**
- 50% progress in 500 frames: 0.5 × 4000 - 250 = **+1750** ✓
- Full lap: 4000 + 500 - (500 to 1000) = **+3500 to +4000** ✓
- Smooth learning curve with stable critic losses
