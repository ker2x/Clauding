# Vector State Representation (67D)

## Overview

The vector mode uses a 67-dimensional state representation combining car dynamics, track geometry, lookahead waypoints, and opponent information.

## State Vector Breakdown

### Car State (11D)
- `[0-1]` Position (x, y) - normalized by PLAYFIELD
- `[2-3]` Velocity (vx, vy) - body frame
- `[4]` Angle - normalized by 2π
- `[5]` Angular velocity (yaw rate)
- `[6-9]` Wheel contacts [FL, FR, RL, RR] - 1.0=on track, 0.0=off track
- `[10]` Track progress (0.0 to 1.0)

### Track Segment (5D)
- `[11]` Distance to centerline - normalized by track width
- `[12]` Angle to centerline - normalized
- `[13]` Curvature ahead
- `[14]` Distance along current segment - normalized
- `[15]` Current segment length - normalized

### Lookahead Waypoints (40D)
- `[16-55]` 20 waypoints × (x, y) - car-relative coordinates
- Provides preview of upcoming track geometry

### Opponent Information (11D)
- `[56-66]` Nearest competitor state (position, velocity, etc.)
- Used in multi-agent scenarios

## Dynamic Features

The state includes physics-based features derived from the Pacejka Magic Formula tire model:

**Speed and Accelerations:**
- Overall speed magnitude
- Longitudinal acceleration (forward/backward)
- Lateral acceleration (sideways, for cornering)

**Tire Dynamics:**
- Slip angles (4 wheels): Angle between tire heading and actual velocity
- Slip ratios (4 wheels): Wheel rotation speed vs ground speed
- These indicate tire grip limits and sliding behavior

## Benefits

1. **Complete Information**: Full track geometry without rendering
2. **Physics Awareness**: Direct access to tire slip and grip
3. **Fast Training**: 10-50× faster than visual mode
4. **Optimal for RL**: Structured, low-dimensional representation

## Implementation

State is computed in `env/car_racing.py` in the `_create_vector_state()` method. All values are normalized for neural network training.

## Compatibility

- Works with all training scripts (train.py, train_selection_parallel.py, etc.)
- SAC agent automatically adapts to state shape
- Visual mode (96×96 RGB) available as alternative for visualization

---

*For technical details, see `env/car_racing.py:_create_vector_state()`*
