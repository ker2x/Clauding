# Observation Space Enhancement Plan for 006

## Executive Summary

This plan outlines the changes needed to evolve the current observation space to be more aligned with GT SOPHY's approach while maintaining compatibility with the existing SAC implementation. The focus is on improving temporal information and ensuring we capture all the physics data that professional racing AI systems use.

## Current State Analysis

### Current Observation Space (BASE: 33 + NUM_LOOKAHEAD × 2)

**Default configuration:** NUM_LOOKAHEAD=10, WAYPOINT_STRIDE=2, FRAME_STACK=3
- Base dimension: 33 + (10 × 2) = 53
- With frame stacking: 53 × 3 = 159 dimensions

**Components (from `006/env/car_racing.py:_create_vector_state`):**

1. **Car state (11 dims):**
   - Position: x, y (world frame, normalized)
   - Velocity: vx, vy (body frame, normalized)
   - Orientation: angle (normalized by 2π)
   - Angular velocity: yaw_rate (normalized)
   - Wheel contacts: 4 binary flags (on/off track)
   - Track progress: scalar [0, 1]

2. **Track segment info (5 dims):**
   - Distance to center (normalized by track width)
   - Angle difference (relative to track direction)
   - Curvature (normalized)
   - Position on segment (t ∈ [0, 1])
   - Segment length (normalized)

3. **Lookahead waypoints (20 dims with default):**
   - NUM_LOOKAHEAD × 2 (x, y) in car-relative coordinates
   - Configurable stride for horizon tuning

4. **Speed (1 dim):**
   - Velocity magnitude (normalized)

5. **Accelerations (2 dims):**
   - Longitudinal: ax (body frame, normalized)
   - Lateral: ay (body frame, normalized)

6. **Tire data per wheel [FL, FR, RL, RR] (12 dims):**
   - Slip angles (4 dims, normalized by π)
   - Slip ratios (4 dims, normalized)
   - Vertical forces (4 dims, normalized)

7. **Steering state (2 dims):**
   - Current steering angle (normalized)
   - Steering rate (normalized)

### GT SOPHY Reference (from `006/sophy/GT SOPHY.md`)

**SOPHY's observation space includes:**
- ✓ 3D Velocity (we have 2D horizontal)
- ✓ 3D Angular velocity (we have yaw rate only)
- ✓ 3D Acceleration (we have 2D horizontal)
- ✓ Tyre load per tyre (we have vertical forces)
- ✓ Tyre slip angle per tyre (we have this)
- ✓ Tyre slip ratio per tyre (we have this)
- ✓ Track progress (we have this)
- ✗ Surface incline (not applicable - flat track)
- ✓ Orientation relative to track (we have angle_diff)
- ✓ Upcoming course points with ~6s lookahead (we have configurable waypoints)
- ✗ Barrier flag (not applicable)
- ✓ "Off course" flag (we have wheel_contacts)

**Key insight:** We already have most of SOPHY's observation components! The main areas for improvement are:
1. **Frame stacking** (configured but not yet implemented)
2. **Temporal derivatives** (velocity → acceleration → jerk via frame differences)
3. **Extended 3D vectors** (for future extensibility)

## Gap Analysis

### What We Have ✓
- Complete tire physics (slip angles, slip ratios, vertical forces)
- Body-frame velocities and accelerations
- Track-relative information (distance, angle, curvature)
- Configurable waypoint lookahead
- Steering state (angle + rate)
- All physics properly normalized

### What's Missing
1. **Frame stacking implementation** (CRITICAL)
   - Config parameter exists (`FRAME_STACK=3`) but not implemented
   - Would provide temporal context for learning derivatives
   - Essential for capturing dynamics (velocity → acceleration → jerk)

2. **3D vector representation** (NICE TO HAVE)
   - Current: 2D (x, y) for horizontal plane
   - SOPHY: 3D (x, y, z) vectors
   - Benefit: Future-proof for 3D tracks or elevation changes
   - For now: z-components would be zero

3. **Previous action in observation** (RECOMMENDED)
   - Current: steering state is internal to Car class
   - Proposed: Include previous [steering_cmd, throttle_cmd] in observation
   - Benefit: Agent can learn action consistency and smooth control

### What's Already Implemented But Not Active
- `config/physics_config.py:ObservationParams.FRAME_STACK` is defined
- Frame dimension calculation exists: `get_stacked_observation_dim()`
- Infrastructure ready, just needs implementation in `car_racing.py`

## Proposed Changes

### Phase 1: Frame Stacking Implementation (HIGH PRIORITY)

**Goal:** Implement frame stacking to provide temporal context

**Changes needed in `006/env/car_racing.py`:**

1. **Add frame buffer to CarRacing class:**
   ```python
   class CarRacing:
       def __init__(self, ...):
           # Add frame buffer
           self.frame_stack = self.physics_config.observation.FRAME_STACK
           self.frame_buffer = None  # Will hold last N frames
   ```

2. **Initialize buffer in reset():**
   ```python
   def reset(self, ...):
       # ... existing code ...

       # Get initial observation
       initial_obs = self._create_vector_state()

       # Initialize frame buffer with copies of initial state
       if self.frame_stack > 1:
           self.frame_buffer = np.tile(initial_obs, self.frame_stack)
       else:
           self.frame_buffer = initial_obs

       return self.frame_buffer, {}
   ```

3. **Update buffer in step():**
   ```python
   def step(self, action):
       # ... existing physics step ...

       # Get current observation
       current_obs = self._create_vector_state()

       # Update frame buffer
       if self.frame_stack > 1:
           # Shift old frames left, add new frame on right
           base_dim = len(current_obs)
           self.frame_buffer = np.concatenate([
               self.frame_buffer[base_dim:],  # Drop oldest frame
               current_obs                     # Add newest frame
           ])
       else:
           self.frame_buffer = current_obs

       return self.frame_buffer, reward, terminated, truncated, info
   ```

4. **Update observation space dimensions:**
   - Already calculated correctly in `__init__` using `get_stacked_observation_dim()`
   - No changes needed

**Benefits:**
- Temporal derivatives via finite differences:
  - Frame N - Frame N-1 → velocity approximation
  - Frame N - 2×Frame(N-1) + Frame(N-2) → acceleration approximation
- Network learns temporal patterns naturally
- No architecture changes needed (just larger input)

**Testing:**
- Verify observation shape matches `observation_space.shape`
- Verify frame buffer rotates correctly
- Verify reset() initializes buffer properly
- Train small test to verify convergence

### Phase 2: Previous Action in Observation (MEDIUM PRIORITY)

**Goal:** Help agent learn smooth, consistent control

**Changes needed:**

1. **Add to CarRacing class:**
   ```python
   def __init__(self, ...):
       self.prev_action = np.zeros(2, dtype=np.float32)  # [steering, throttle]
   ```

2. **Update observation dimension:**
   ```python
   # In config/physics_config.py
   def get_base_observation_dim(num_lookahead: int) -> int:
       return 35 + (num_lookahead * 2)  # Was 33, now 33+2 for prev action
   ```

3. **Append to state vector in `_create_vector_state()`:**
   ```python
   state = np.array([
       # ... existing 33 features ...
       steering_angle_norm, steering_rate_norm,
       # Previous action (2) - NEW
       self.prev_action[0], self.prev_action[1]
   ], dtype=np.float32)
   ```

4. **Update in step():**
   ```python
   def step(self, action):
       # Store previous action before executing new one
       self.prev_action = np.array(action, dtype=np.float32)
       # ... rest of step ...
   ```

**Benefits:**
- Agent can learn action smoothness
- Helps with control consistency
- Common practice in robotics RL

**Impact:**
- Changes base observation dimension from 33 to 35
- **Requires retraining** (observation space changes)
- Minimal computational overhead

### Phase 3: Extended 3D Vectors (LOW PRIORITY - FUTURE)

**Goal:** Future-proof for 3D tracks or elevation

**Changes needed:**

1. **Expand velocity to 3D:**
   ```python
   # Current: vx, vy (2D)
   # Proposed: vx, vy, vz (3D, where vz=0 for flat track)
   ```

2. **Expand acceleration to 3D:**
   ```python
   # Current: ax, ay (2D)
   # Proposed: ax, ay, az (3D, where az=0 for flat track)
   ```

3. **Expand angular velocity to 3D:**
   ```python
   # Current: yaw_rate (1D)
   # Proposed: roll_rate, pitch_rate, yaw_rate (3D, where roll=pitch=0)
   ```

**Benefits:**
- Future-proof for 3D environments
- Matches GT SOPHY's representation
- No computational overhead if z-components are zero

**Impact:**
- Adds 5 dimensions: +1 velocity, +1 acceleration, +2 angular velocity
- Base dimension: 33 → 38 (or 35 → 40 with prev_action)
- **Requires retraining**
- Should be done **only if** 3D tracks are planned

**Recommendation:** **DEFER** until 3D tracks are actually implemented. Not worth the retraining cost for zero-valued dimensions.

## Implementation Roadmap

### Milestone 1: Frame Stacking (IMMEDIATE)
**Estimated effort:** 2-4 hours
**Risk:** LOW (infrastructure exists, just needs wiring)

- [ ] 1.1: Add `frame_buffer` to `CarRacing.__init__()`
- [ ] 1.2: Initialize buffer in `reset()`
- [ ] 1.3: Update buffer in `step()`
- [ ] 1.4: Write unit tests for frame rotation
- [ ] 1.5: Verify observation space shape
- [ ] 1.6: Run quick training test (500 episodes)
- [ ] 1.7: Update documentation

**Success criteria:**
- Observation shape matches `observation_space.shape`
- Frame buffer rotates correctly (oldest dropped, newest added)
- Training converges with frame stacking enabled

### Milestone 2: Previous Action (OPTIONAL)
**Estimated effort:** 1-2 hours
**Risk:** LOW (simple addition)

- [ ] 2.1: Add `prev_action` tracking
- [ ] 2.2: Update `get_base_observation_dim()` (+2 dims)
- [ ] 2.3: Append prev_action to state vector
- [ ] 2.4: Update documentation
- [ ] 2.5: Full training run and comparison

**Success criteria:**
- Smoother control (lower action variation)
- Similar or better lap times
- Agent learns action consistency

### Milestone 3: 3D Vectors (FUTURE)
**Estimated effort:** 4-6 hours
**Risk:** MEDIUM (requires retraining, may not help)

- [ ] 3.1: Extend velocity to 3D (vz=0)
- [ ] 3.2: Extend acceleration to 3D (az=0)
- [ ] 3.3: Extend angular velocity to 3D (roll=pitch=0)
- [ ] 3.4: Update normalization for new components
- [ ] 3.5: Update dimension calculations
- [ ] 3.6: Full retraining
- [ ] 3.7: Performance comparison

**Success criteria:**
- No regression in performance
- Clean architecture for future 3D expansion

**Recommendation:** Only implement if 3D tracks are planned in next 3-6 months.

## Configuration Management

### Current Configuration (`config/physics_config.py`)

```python
@dataclass
class ObservationParams:
    NUM_LOOKAHEAD: int = 10      # Number of waypoints
    WAYPOINT_STRIDE: int = 2     # Every 2nd waypoint
    FRAME_STACK: int = 3         # 3 frames stacked
```

**Dimension calculation:**
- Base: 33 + (10 × 2) = 53
- Stacked: 53 × 3 = 159 dimensions

### After Phase 1 (Frame Stacking)
No config changes needed - already configured!
Just implement the stacking logic.

### After Phase 2 (Previous Action)
```python
# Base dimension changes from 33 to 35
# Stacked: 55 × 3 = 165 dimensions
```

### After Phase 3 (3D Vectors)
```python
# Base dimension changes from 35 to 40
# Stacked: 60 × 3 = 180 dimensions
```

## Testing Strategy

### Unit Tests
1. **Frame buffer rotation:**
   - Create observation with known values
   - Step environment 5 times
   - Verify oldest frame is dropped, newest is added

2. **Observation shape:**
   - Verify shape matches `observation_space.shape`
   - Verify with different FRAME_STACK values (1, 2, 3, 4)

3. **Previous action tracking:**
   - Verify action is stored correctly
   - Verify action appears in next observation

### Integration Tests
1. **Training stability:**
   - Run 500 episodes with frame stacking
   - Verify no NaN or Inf values
   - Verify reward improves over time

2. **Performance comparison:**
   - Baseline: Current implementation (no stacking)
   - Treatment: With frame stacking
   - Metrics: Average reward, convergence speed, lap completion %

### Validation
1. **Reward comparison:**
   - Should be similar or better than baseline
   - Look for faster convergence (better sample efficiency)

2. **Action smoothness:**
   - Measure action variance over time
   - Should decrease with prev_action in observation

## Risk Assessment

### HIGH RISK
**None identified** - All changes are additive and well-understood

### MEDIUM RISK
1. **Increased training time:**
   - Larger observation → larger networks → slower training
   - Mitigation: Use efficient network architecture, batch normalization
   - Impact: 10-20% slower per episode (acceptable)

2. **Hyperparameter sensitivity:**
   - Frame stacking may require different learning rates
   - Mitigation: Start with current hyperparameters, tune if needed
   - Impact: May need 2-3 training runs to find optimal settings

### LOW RISK
1. **Memory usage:**
   - Frame buffer is small (53 × 3 = 159 floats = ~636 bytes per env)
   - Impact: Negligible

2. **Code complexity:**
   - Frame stacking is straightforward
   - Impact: Minimal maintenance burden

## Performance Expectations

### Frame Stacking (Phase 1)
**Expected improvement:**
- Better temporal understanding → +10-20% sample efficiency
- Faster convergence to good policies
- More robust to noise (temporal filtering effect)

**Trade-offs:**
- 3× larger observation → 10-20% slower training per step
- Slight increase in network size

**Net effect:** **POSITIVE** - Better sample efficiency outweighs slower steps

### Previous Action (Phase 2)
**Expected improvement:**
- Smoother control → better lap times
- Reduced action jitter → more efficient driving
- Better handling of action constraints

**Trade-offs:**
- +2 dimensions (negligible)
- Minimal computational overhead

**Net effect:** **POSITIVE** - Likely improves control quality

### 3D Vectors (Phase 3)
**Expected improvement:**
- None for current 2D environment
- Future-proofing for 3D tracks

**Trade-offs:**
- +5 dimensions → larger network
- Retraining required
- No immediate benefit

**Net effect:** **NEUTRAL to NEGATIVE** for current use case - defer until needed

## Compatibility Considerations

### Backward Compatibility
- **Phase 1:** Backward compatible (controlled by `FRAME_STACK` config)
- **Phase 2:** **NOT backward compatible** (observation space changes)
- **Phase 3:** **NOT backward compatible** (observation space changes)

### Checkpoint Migration
**Phase 1:**
- Old checkpoints work if `FRAME_STACK=1`
- New checkpoints require same `FRAME_STACK` value

**Phases 2-3:**
- **Cannot load old checkpoints** (observation dimension mismatch)
- Need to retrain from scratch or implement checkpoint migration
- Recommend: Archive old checkpoints, start fresh

### Training Script Compatibility
All training scripts should work without modification:
- `train_selection_parallel.py`
- `train.py`
- `watch_agent.py`

Observation space dimension is read from config automatically.

## Conclusion

### Recommended Immediate Action: Implement Phase 1 (Frame Stacking)

**Rationale:**
1. **Already configured** - just needs implementation
2. **High impact** - temporal information is critical for racing
3. **Low risk** - straightforward implementation
4. **Backward compatible** - can toggle via config

**Expected outcome:**
- 10-20% better sample efficiency
- Faster convergence to good policies
- More robust control (temporal filtering)

### Defer Phase 2 (Previous Action) Until:
- Phase 1 is validated and working well
- Training results show need for smoother control
- Ready to retrain from scratch (observation space changes)

### Defer Phase 3 (3D Vectors) Until:
- 3D tracks or elevation changes are planned
- Current 2D implementation is fully optimized
- Clear benefit is demonstrated

### Next Steps
1. **Review this plan** with stakeholders
2. **Implement Phase 1** (frame stacking)
3. **Run validation tests** (500+ episodes)
4. **Measure performance** vs baseline
5. **Decide on Phase 2** based on Phase 1 results

---

**Document Status:** DRAFT
**Last Updated:** 2025-11-22
**Author:** Claude (AI Assistant)
**Review Status:** Pending review
