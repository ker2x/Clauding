# Vector Mode Expansion - Enhanced State Representation

## Overview

The vector mode observation space has been expanded from 36 to 47 dimensions to provide the agent with richer information about vehicle dynamics and tire behavior.

## Changes Summary

### Previous State (36 dimensions)
- **Car state (11)**: position, velocities, angle, angular velocity, wheel contacts, progress
- **Track segment info (5)**: distance to center, angle difference, curvature, position along segment
- **Lookahead waypoints (20)**: 10 waypoints × (x, y) in car-relative coordinates

### New State (47 dimensions)
All previous features PLUS:

#### New Dynamic Features (11 dimensions)
1. **Speed (1)**: Magnitude of velocity vector
   - Provides overall speed information that's easier to interpret than separate vx/vy

2. **Longitudinal Acceleration (1)**: Forward/backward acceleration in body frame
   - Computed as `(current_vx - prev_vx) / dt`
   - Helps agent understand traction/braking performance

3. **Lateral Acceleration (1)**: Sideways acceleration in body frame
   - Computed as `(current_vy - prev_vy) / dt`
   - Critical for understanding cornering forces and stability

4. **Slip Angles (4)**: One per wheel [FL, FR, RL, RR]
   - Angle between tire heading and actual velocity direction
   - Key indicator of tire grip and cornering behavior
   - Zero when tire rolls straight, increases during sliding

5. **Slip Ratios (4)**: One per wheel [FL, FR, RL, RR]
   - Ratio of wheel rotation speed vs ground speed
   - Indicates longitudinal tire slip (traction/braking)
   - Zero when wheel rolls perfectly, negative when braking, positive when spinning

## Implementation Details

### State Computation
```python
# Speed
speed = sqrt(vx^2 + vy^2)

# Accelerations (body frame)
ax = (car.vx - prev_vx) / dt  # Longitudinal
ay = (car.vy - prev_vy) / dt  # Lateral

# For each wheel:
# - Compute wheel velocity at contact point
# - Calculate slip angle: arctan2(wheel_vy, wheel_vx) - steering_angle
# - Calculate slip ratio: (wheel_omega * radius - wheel_vx) / max(|wheel_vx|, |wheel_linear_vel|)
```

### Why These Features Matter

**Speed**: Single value representing overall velocity makes it easier for the agent to learn speed control.

**Accelerations**: Direct feedback on how forces are affecting the car, crucial for learning smooth control.

**Slip Angles & Ratios**: These are the fundamental quantities that the Pacejka tire model uses to compute forces. By exposing them to the agent:
- Agent can learn to recognize when tires are approaching their grip limits
- Can understand the difference between understeer (front slip) and oversteer (rear slip)
- Can optimize tire usage for maximum performance
- Can learn recovery behaviors when sliding

## Benefits for Learning

1. **Richer Dynamics Awareness**: Agent has direct access to tire behavior that determines grip and handling
2. **Better Control**: Understanding accelerations helps learn smoother, more efficient control
3. **Grip Optimization**: Slip angle/ratio data enables learning to drive at the edge of tire grip
4. **Stability Understanding**: Can detect and respond to oversteer/understeer conditions
5. **Faster Training**: More informative state should lead to faster policy learning

## State Vector Layout (47 dimensions)

```
[0-1]    Car position (x, y) - normalized by PLAYFIELD
[2-3]    Car velocity (vx, vy) - body frame
[4]      Car angle - normalized by 2π
[5]      Angular velocity (yaw rate)
[6-9]    Wheel contacts [FL, FR, RL, RR] - 1.0=on track, 0.0=off track
[10]     Track progress (0.0 to 1.0)
[11]     Distance to centerline - normalized by track width
[12]     Angle to centerline - normalized
[13]     Curvature ahead
[14]     Distance along current segment - normalized
[15]     Current segment length - normalized
[16-35]  Lookahead waypoints (10 × 2) - car-relative coordinates
[36]     Speed magnitude
[37]     Longitudinal acceleration (body frame)
[38]     Lateral acceleration (body frame)
[39-42]  Slip angles [FL, FR, RL, RR]
[43-46]  Slip ratios [FL, FR, RL, RR]
```

## Compatibility

- SAC agent automatically adapts to new state shape
- Training scripts work without modification
- Visual mode unchanged (still 96×96×3 RGB)

## Notes

- Acceleration values are zero on first step (no previous velocity)
- Slip angles/ratios are zero when stationary (v < 0.5 m/s or wheel speeds < 0.1)
- All slip calculations match the physics engine's Pacejka tire model
