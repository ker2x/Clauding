# State Modes: Snapshot, Vector, and Visual

This document explains the three state representation modes available in this CarRacing DQN implementation.

## Overview

The CarRacing environment can provide observations in three fundamentally different formats:

1. **Snapshot Mode** (RECOMMENDED): Returns 36-dimensional vector with track geometry and lookahead
2. **Vector Mode** (Limited): Returns 11-dimensional state vector (basic car state only)
3. **Visual Mode** (Slow): Returns 96×96 RGB images (pixel observations)

All modes represent the same environment and use the same action space, but differ in:
- **Representation**: Track geometry vectors vs basic state vs raw images
- **Performance**: Snapshot is 3-5x faster than visual, vector is 6x faster but too limited
- **Learning quality**: Snapshot learns well, vector cannot learn proper driving, visual learns well but slow
- **Network architecture**: MLP for snapshot/vector (different sizes), CNN for visual
- **Use cases**: Snapshot for training (RECOMMENDED), visual for watching, vector not recommended

---

## Snapshot Mode (Track Geometry)

### What It Is
Snapshot mode returns a 36-dimensional vector containing car state, current track segment information, and lookahead waypoints.

### State Representation
A single 36-dimensional vector containing:

**Car State (11 dimensions):**
| Index | Feature | Description | Range |
|-------|---------|-------------|-------|
| 0-1 | `x, y` | Car position (normalized) | [0, 1] |
| 2-3 | `vx, vy` | Velocity components | [-50, +50] m/s |
| 4 | `angle` | Car orientation (normalized) | [-1, +1] |
| 5 | `angular_vel` | Angular velocity | [-5, +5] rad/s |
| 6-9 | `wheel_contacts` | Ground contact (4 wheels) | [0.0, 1.0] each |
| 10 | `track_progress` | Fraction of track completed | [0, 1] |

**Track Segment Info (5 dimensions):**
| Index | Feature | Description |
|-------|---------|-------------|
| 11 | `dist_to_center` | Distance from track centerline (normalized by track width) |
| 12 | `angle_diff` | Angle difference between car and track direction |
| 13 | `curvature` | Current track curvature (change in angle over distance) |
| 14 | `dist_along_segment` | Progress along current segment |
| 15 | `segment_length` | Length of current segment |

**Lookahead Waypoints (20 dimensions):**
| Index | Feature | Description |
|-------|---------|-------------|
| 16-35 | `waypoints` | 10 upcoming waypoints × (x, y) in car-relative coordinates |

**Total state size**: 36 values per state

### Network Architecture
Uses a larger Multi-Layer Perceptron (MLP) than vector mode:
```
Input: 36 dimensions
↓
Linear(36 → 256) → ReLU
↓
Linear(256 → 256) → ReLU
↓
Linear(256 → 128) → ReLU
↓
Linear(128 → 9 actions)
```

### Performance
- **Speed**: ~150-200 steps/second (3-5x faster than visual)
- **Memory**: Low (stores only 36 values per experience)
- **Training time**: 1M steps ≈ 1.5-2 hours (2.5-3x faster than visual)

### When to Use
- ✅ Training agents (RECOMMENDED, default)
- ✅ Hyperparameter tuning
- ✅ Production training scenarios
- ✅ When you need proper racing behavior
- ✅ Best balance of speed and learning quality

### Example Usage
```bash
# Train with snapshot mode (default, recommended)
python train.py --episodes 2000

# Explicitly specify snapshot mode
python train.py --episodes 2000 --state-mode snapshot

# Quick test training
python train.py --episodes 25 --learning-starts 500
```

---

## Visual Mode (Image-Based)

### What It Is
Visual mode returns 96×96×3 RGB images showing a top-down view of the car and track, exactly as a human would see it.

### State Representation
- **Raw observation**: `(96, 96, 3)` RGB image
- **After preprocessing**:
  - Grayscale conversion: `(96, 96)` single channel
  - Normalization: [0, 255] → [0.0, 1.0]
  - Frame stacking: `(4, 96, 96)` - 4 consecutive frames
- **Total state size**: 36,864 values per state (4 × 96 × 96)

### Network Architecture
Uses a Convolutional Neural Network (CNN):
```
Input: (4, 96, 96)
↓
Conv2D(32 filters, 8×8, stride 4) → ReLU
↓
Conv2D(64 filters, 4×4, stride 2) → ReLU
↓
Conv2D(64 filters, 3×3, stride 1) → ReLU
↓
Flatten → 4096 features
↓
Linear(4096 → 512) → ReLU
↓
Linear(512 → 9 actions)
```

### Performance
- **Speed**: ~57 steps/second
- **Memory**: High (stores 36,864 values per experience)
- **Training time**: 1M steps ≈ 4.9 hours

### When to Use
- ✅ Watching the trained agent play (requires rendering)
- ✅ Debugging visual preprocessing pipeline
- ✅ Creating videos or demonstrations
- ✅ Research requiring realistic visual understanding
- ❌ Not recommended for training (too slow)

### Example Usage
```bash
# Watch trained agent (automatically uses visual mode)
python watch_agent.py --checkpoint checkpoints/best_model.pt

# Watch random agent baseline
python watch_random_agent.py --episodes 3

# Train with visual mode (slower, not recommended)
python train.py --episodes 200 --state-mode visual
```

---

## Vector Mode (Basic State - Limited)

### What It Is
Vector mode returns a compact 11-dimensional numerical vector containing only basic car state information without track geometry or lookahead.

### State Representation
A single 11-dimensional vector containing:

| Index | Feature | Description | Range |
|-------|---------|-------------|-------|
| 0 | `x` | Car X position | [0, 1000+] pixels |
| 1 | `y` | Car Y position | [0, 1000+] pixels |
| 2 | `vx` | Velocity X component | [-50, +50] m/s |
| 3 | `vy` | Velocity Y component | [-50, +50] m/s |
| 4 | `angle` | Car orientation | [-π, +π] radians |
| 5 | `angular_vel` | Angular velocity | [-5, +5] rad/s |
| 6-9 | `wheel_contacts` | Ground contact (4 wheels) | [0.0, 1.0] each |
| 10 | `track_progress` | Tiles visited | [0, 100+] count |

**Total state size**: 11 values per state

### Network Architecture
Uses a Multi-Layer Perceptron (MLP):
```
Input: 11 dimensions
↓
Linear(11 → 128) → ReLU
↓
Linear(128 → 128) → ReLU
↓
Linear(128 → 64) → ReLU
↓
Linear(64 → 9 actions)
```

### Performance
- **Speed**: ~313 steps/second (6x faster than visual)
- **Memory**: Very low (stores only 11 values per experience)
- **Training time**: 1M steps ≈ 0.9 hours (5.4x faster than visual)

### When to Use
- ⚠️ **NOT RECOMMENDED** - lacks track information for proper learning
- ❌ Agent cannot learn to drive well (no track geometry or lookahead)
- ❌ Fastest but impractical for real training
- ⚠️ Only use for testing/debugging basic functionality
- ✅ Use **snapshot mode** instead for actual training

### Example Usage
```bash
# NOT RECOMMENDED - use snapshot mode instead
python train.py --episodes 2000 --state-mode vector

# Use snapshot mode for actual training
python train.py --episodes 2000 --state-mode snapshot
```

---

## Performance Comparison

| Metric | Visual Mode | Vector Mode | Snapshot Mode |
|--------|-------------|-------------|---------------|
| Steps/second | ~57 | ~313 | ~150-200 |
| Speedup vs Visual | 1.0x | 5.5x | 3-5x |
| Training time (1M steps) | 4.9 hours | 0.9 hours | 1.5-2 hours |
| Memory per state | 36,864 values | 11 values | 36 values |
| Replay buffer (100k) | ~14 GB | ~4 MB | ~14 MB |
| Network architecture | CNN | MLP (small) | MLP (large) |
| Rendering required | Yes (pygame) | No | No |
| **Learning quality** | ✅ Good | ❌ Poor (limited info) | ✅ Good |
| **Recommended for** | Watching | Not recommended | Training ⭐ |

### Speed Analysis

1. **Vector Mode** (Fastest but limited): No rendering, but lacks track info - agent can't learn properly
2. **Snapshot Mode** (RECOMMENDED): Fast vector-based with track geometry and lookahead - best balance
3. **Visual Mode** (Slowest): Full pygame rendering with texture mapping - only for watching

### Why Snapshot Mode is the Best Choice

1. **Fast**: 3-5x faster than visual mode (no rendering overhead)
2. **Informative**: Includes track geometry and lookahead waypoints
3. **Learning quality**: Agent learns proper racing behavior (unlike vector mode)
4. **Memory efficient**: Only 36 values vs 36,864 for visual
5. **Production ready**: Default for training in this project

---

## Cross-Mode Compatibility

### Can I Train in One Mode and Watch in Another?
**No** - each mode uses a different network architecture, so checkpoints are mode-specific.

- **Snapshot → Visual**: ❌ No (MLP vs CNN architectures)
- **Vector → Visual**: ❌ No (different MLP sizes, different state dimensions)
- **Visual → Snapshot**: ❌ No (CNN vs MLP architectures)

**Solution**: Watch scripts automatically use visual mode for rendering, regardless of training mode. The visual rendering is just for display - the agent still uses its trained network architecture.

### Recommended Workflow

```bash
# Train in snapshot mode (RECOMMENDED)
python train.py --episodes 2000 --state-mode snapshot

# Watch the trained agent (visual mode used for display only)
python watch_agent.py --checkpoint checkpoints/final_model.pt
```

---

## Implementation Details

### How State Mode is Configured

1. **Environment Creation** (`env/car_racing.py`):
```python
env = CarRacing(
    render_mode=None,  # or 'rgb_array' for visual
    state_mode='snapshot'  # 'snapshot', 'vector', or 'visual' (default: snapshot)
)
```

2. **Agent Network Selection** (`ddqn_agent.py`):
```python
if state_mode == 'snapshot':
    self.policy_net = SnapshotDQN(36, n_actions)  # 36D input
elif state_mode == 'vector':
    self.policy_net = VectorDQN(11, n_actions)  # 11D input
elif state_mode == 'visual':
    self.policy_net = DQN(state_shape, n_actions)  # CNN
```

### Snapshot State Generation

The snapshot mode creates compact track geometry vectors:

```python
def _create_snapshot_state(self):
    """Generate 36D vector with car state, track info, and lookahead."""
    # 1. Basic car state (11D)
    car_state = [x, y, vx, vy, angle, angular_vel,
                 wheel_contact[0], wheel_contact[1],
                 wheel_contact[2], wheel_contact[3],
                 track_progress]

    # 2. Find closest track segment
    seg_idx, dist_to_center, closest_point = self._find_closest_track_segment(car_pos)

    # 3. Track segment info (5D)
    track_info = [dist_to_center_norm, angle_diff, curvature,
                  dist_along_segment, segment_length]

    # 4. Lookahead waypoints (20D = 10 waypoints × 2 coords)
    waypoints = []
    for i in range(10):
        wp_idx = (seg_idx + i + 1) % len(self.track)
        wp_x, wp_y = self.track[wp_idx][2:4]
        # Transform to car-relative coordinates
        rel_x, rel_y = transform_to_car_frame(wp_x, wp_y, car_pos, car_angle)
        waypoints.extend([rel_x, rel_y])

    return np.array([*car_state, *track_info, *waypoints], dtype=np.float32)
```

---

## Which Mode Should I Use?

### For Training: **Snapshot Mode** (RECOMMENDED, Default)
- ✅ 3-5x faster than visual mode
- ✅ Low memory usage (36 values vs 36,864 for visual)
- ✅ Agent learns proper racing behavior
- ✅ Includes track geometry and lookahead
- ✅ Best balance of speed and learning quality
- ✅ Production-ready default

### For Training (Not Recommended): **Vector Mode**
- ✅ 6x faster than visual (fastest)
- ✅ Lowest memory usage (11 values)
- ❌ **Agent cannot learn to drive well** (no track info)
- ❌ Too limited for practical use
- ⚠️ Only use for basic functionality testing

### For Watching: **Visual Mode** (Required)
- ✅ See realistic rendering of the agent
- ✅ Debug behavior issues visually
- ✅ Create high-quality demonstrations
- ✅ Automatically used by watch scripts
- ❌ Too slow for training (use snapshot instead)

---

## Common Questions

### Q: Why is snapshot mode recommended over vector mode?
**A:** Vector mode only provides basic car state (position, velocity, etc.) without any track information or lookahead. The agent has no idea where the track goes and cannot learn proper racing behavior. Snapshot mode adds:
- Current track segment info (distance to center, angle, curvature)
- 10 upcoming waypoints in car-relative coordinates
- Only 3x slower than vector, but agent actually learns to drive

### Q: Can snapshot mode learn as well as visual mode?
**A:** Yes! Snapshot mode provides all the geometric information needed for racing:
- Track centerline and curvature
- Lookahead waypoints for planning
- Wheel contact information
The agent learns proper racing behavior without the rendering overhead of visual mode.

### Q: Can I resume training across different modes?
**A:** No - each mode uses a different network architecture:
- Snapshot mode: 36D input → MLP (256→256→128→9)
- Vector mode: 11D input → MLP (128→128→64→9)
- Visual mode: (4, 96, 96) input → CNN

Checkpoints are not compatible across modes.

### Q: When should I use visual mode?
**A:** Only for watching trained agents. Visual mode is too slow for training (3-5x slower than snapshot). Use snapshot mode for training, then watch with visual rendering.

---

## Summary Table

| Aspect | Visual Mode | Vector Mode | Snapshot Mode |
|--------|-------------|-------------|---------------|
| **State size** | 36,864 values | 11 values | 36 values |
| **Network** | CNN | MLP (small) | MLP (large) |
| **Speed** | 57 steps/sec | 313 steps/sec | 150-200 steps/sec |
| **Speedup** | 1.0x baseline | 5.5x faster | 3-5x faster |
| **Training time** | 4.9 hrs (1M) | 0.9 hrs (1M) | 1.5-2 hrs (1M) |
| **Memory** | ~14 GB (100k) | ~4 MB (100k) | ~14 MB (100k) |
| **Rendering** | Pygame (full) | None | None |
| **Learning quality** | ✅ Good | ❌ Poor | ✅ Good |
| **Use case** | Watching only | Not recommended | Training ⭐ |
| **Default for** | watch_*.py | — | train.py |
| **Track info** | Implicit | ❌ None | ✅ Explicit |
| **Lookahead** | Implicit | ❌ None | ✅ 10 waypoints |

---

## Conclusion

**Use snapshot mode for training** - it's the default and recommended choice:
- ✅ 3-5x faster than visual mode
- ✅ Provides track geometry and lookahead waypoints
- ✅ Agent learns proper racing behavior
- ✅ Low memory usage
- ✅ Production-ready

**Don't use vector mode** - too limited:
- ❌ No track information or lookahead
- ❌ Agent cannot learn to drive properly
- ⚠️ Fastest but impractical

**Use visual mode only for watching**:
- ✅ See realistic agent behavior
- ❌ Too slow for training

For most users, the defaults handle everything automatically:
```bash
# Training uses snapshot mode (RECOMMENDED, default)
python train.py --episodes 2000

# Watching uses visual mode (automatic)
python watch_agent.py --checkpoint checkpoints/final_model.pt
```

### Mode Selection Decision Tree

```
Need to train?
├─ Yes → Use snapshot mode (default) ⭐ RECOMMENDED
│        - Fast (3-5x vs visual)
│        - Good learning (track info + lookahead)
│        - Low memory usage
└─ No → Need to watch/visualize?
   └─ Yes → Use visual mode (automatic in watch scripts)
```
