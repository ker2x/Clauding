# Reward System Refactor: Checkpoint → Waypoint

## Summary

Completely refactored the reward system from a checkpoint-based approach with technical debt to a clean, simple waypoint-based system.

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
reward = -STEP_PENALTY  # -1.0 per frame (time pressure)
reward -= wheels_offtrack * OFFTRACK_PENALTY  # -2.0 per wheel (if > 2 wheels off)

# When waypoint reached
reward += WAYPOINT_REWARD  # +50 per waypoint

# On lap completion
reward += LAP_COMPLETION_REWARD  # +500 bonus
```

**Total possible rewards:**
- Waypoints: 20 × 50 = +1000
- Lap completion: +500
- Total positive: +1500
- Step penalties: depends on speed (~-600 to -1500 for typical lap)
- Net expected: ~0 to +900 for successful lap

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
WAYPOINT_REWARD = 50.0
LAP_COMPLETION_REWARD = 500.0
STEP_PENALTY = 1.0
OFFTRACK_PENALTY = 2.0
OFFTRACK_THRESHOLD = 2
WAYPOINT_DISTANCE_THRESHOLD = 5.0
```

## Next Steps

1. Run training and monitor:
   - Waypoints reached per episode
   - Exploration behavior
   - Learning curve stability
2. Tune if needed (see CLAUDE.md tuning guide)
3. Fall back to Option 1 (continuous progress) if sparse rewards too difficult
