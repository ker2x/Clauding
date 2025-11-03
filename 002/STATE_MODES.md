# State Modes: Visual, Vector, and Synthetic

This document explains the three state representation modes available in this CarRacing DQN implementation.

## Overview

The CarRacing environment can provide observations in three fundamentally different formats:

1. **Visual Mode**: Returns 96×96 RGB images (pixel observations)
2. **Vector Mode**: Returns 11-dimensional state vector (numerical observations)
3. **Synthetic Mode**: Generates 96×96 synthetic images from vector state (rendered vector)

All modes represent the same environment and use the same action space, but differ in:
- **Representation**: Raw images vs numerical vectors vs rendered vectors
- **Performance**: Visual is slowest, vector is fastest (6x), synthetic is in between (3x)
- **Network architecture**: CNN for visual/synthetic, MLP for vector
- **Use cases**: Visual for watching, vector for training, synthetic for visual training without rendering overhead

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

## Vector Mode (State Vector)

### What It Is
Vector mode returns a compact 11-dimensional numerical vector containing all essential state information without rendering images.

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
- ✅ Training agents (default, highly recommended)
- ✅ Hyperparameter tuning (faster iterations)
- ✅ Quick experiments and debugging
- ✅ Any scenario where rendering is not needed
- ❌ Cannot be used for watching (no visual output)

### Example Usage
```bash
# Train with vector mode (default, recommended)
python train.py --episodes 2000

# Explicitly specify vector mode
python train.py --episodes 2000 --state-mode vector

# Quick test training (fast with vector mode)
python train.py --episodes 25 --learning-starts 500
```

---

## Synthetic Mode (Rendered Vector)

### What It Is
Synthetic mode generates 96×96 grayscale images from the 11-dimensional vector state, creating simplified visual representations without the overhead of full pygame rendering.

### State Representation
- **Raw observation**: Generated `(96, 96)` grayscale image from vector state
- **After preprocessing**:
  - Already grayscale: `(96, 96)` single channel
  - Normalization: [0, 255] → [0.0, 1.0]
  - Frame stacking: `(4, 96, 96)` - 4 consecutive frames
- **Total state size**: 36,864 values per state (4 × 96 × 96)

### How It Works
Synthetic mode creates simplified visual representations by:
1. **Track rendering**: Draws track tiles (gray for track, green for grass)
2. **Car rendering**: Draws car body (blue rectangle) based on position and angle
3. **Wheel rendering**: Adds wheel indicators (green = ground contact, red = no contact)

This provides visual structure for CNN training without pygame's rendering overhead.

### Network Architecture
Uses the same CNN architecture as visual mode:
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
- **Speed**: ~165 steps/second (3x faster than visual, 2x slower than vector)
- **Memory**: High (stores 36,864 values per experience, same as visual)
- **Training time**: 1M steps ≈ 1.7 hours (2.9x faster than visual, 1.9x slower than vector)

### When to Use
- ✅ Training CNN-based agents without pygame rendering overhead
- ✅ Research comparing vector information with spatial structure
- ✅ When you need visual processing but want faster training than full rendering
- ✅ Testing visual perception with simplified, consistent representations
- ❌ Not recommended for watching (visual mode has better graphics)
- ❌ Not as fast as vector mode for pure training efficiency

### Example Usage
```bash
# Train with synthetic mode
python train.py --episodes 2000 --state-mode synthetic

# Quick test with synthetic rendering
python train.py --episodes 25 --state-mode synthetic --learning-starts 500
```

---

## Performance Comparison

### Benchmark Results
Based on comprehensive benchmarking (`benchmark_state_modes.py`):

| Metric | Visual Mode | Synthetic Mode | Vector Mode |
|--------|-------------|----------------|-------------|
| Steps/second | 57 | 165 | 313 |
| Speedup vs Visual | 1.0x | **2.9x** | **5.5x** |
| Training time (1M steps) | 4.9 hours | 1.7 hours | 0.9 hours |
| Memory per state | 36,864 values | 36,864 values | 11 values |
| Replay buffer (100k) | ~14 GB | ~14 GB | ~4 MB |
| Network architecture | CNN | CNN | MLP |
| Rendering required | Yes (pygame) | Yes (synthetic) | No |

### Speed Analysis

1. **Vector Mode** (Fastest): No rendering at all, direct state access
2. **Synthetic Mode** (Medium): Lightweight NumPy-based rendering, no pygame overhead
3. **Visual Mode** (Slowest): Full pygame rendering with texture mapping and anti-aliasing

### Why Synthetic Mode is Faster than Visual Mode

1. **No Pygame Overhead**: Uses direct NumPy array manipulation instead of pygame surfaces
2. **Simplified Graphics**: Basic shapes and colors instead of textured rendering
3. **Pre-computed Track**: Can cache track layout to avoid re-rendering
4. **Less Anti-aliasing**: Simple pixel operations without smoothing
5. **Direct Grayscale**: No RGB-to-grayscale conversion needed

---

## Cross-Mode Compatibility

### Can I Train in One Mode and Watch in Another?
**Partially:**

- **Vector → Visual**: ✅ Yes! Train in vector mode, watch in visual mode
- **Vector → Synthetic**: ❌ No (different network architectures: MLP vs CNN)
- **Synthetic → Visual**: ❌ No (checkpoints are mode-specific, but both use CNN)
- **Visual → Synthetic**: ❌ No (checkpoints are mode-specific)

The only fully compatible workflow is training in vector mode and watching in visual mode, because:
1. Both use the same environment physics and reward structure
2. The agent's learned policy transfers directly
3. Watch scripts handle the network architecture difference automatically

### Workflow Examples

```bash
# Recommended: Train in vector, watch in visual
python train.py --episodes 2000 --state-mode vector
python watch_agent.py --checkpoint checkpoints/final_model.pt

# Alternative: Train in synthetic (faster than visual, uses CNN)
python train.py --episodes 2000 --state-mode synthetic
# Note: Cannot directly watch synthetic-trained models in visual mode
```

---

## Implementation Details

### How State Mode is Configured

1. **Environment Creation** (`env/car_racing.py`):
```python
env = CarRacing(
    render_mode=None,  # or 'rgb_array' for visual
    state_mode='vector'  # 'visual', 'vector', or 'synthetic'
)
```

2. **Preprocessing** (`preprocessing.py`):
```python
def make_carracing_env(state_mode='visual', ...):
    env = CarRacing(state_mode=state_mode, ...)

    # Apply visual preprocessing for visual/synthetic modes
    if state_mode == 'visual':
        env = GrayscaleWrapper(env)
        env = NormalizeObservation(env)
        env = FrameStack(env, stack_size=4)
    elif state_mode == 'synthetic':
        # Already grayscale, just normalize and stack
        env = NormalizeObservation(env)
        env = FrameStack(env, stack_size=4)

    return env
```

3. **Agent Network Selection** (`ddqn_agent.py`):
```python
if state_mode == 'vector':
    self.policy_net = VectorDQN(state_dim, n_actions)
elif state_mode in ['visual', 'synthetic']:
    self.policy_net = DQN(state_shape, n_actions)
```

### Synthetic State Generation

The synthetic renderer creates images from vector state:

```python
def _create_synthetic_state(self):
    """Generate 96x96 synthetic image from vector state."""
    # Create blank canvas
    img = np.full((WINDOW_H, WINDOW_W), 107, dtype=np.uint8)  # Gray background

    # Render track tiles (gray for track, green for grass)
    for tile in self.track:
        # Draw tile polygon with appropriate color

    # Render car body (blue rectangle)
    car = self.car
    pos = car.hull.position
    angle = car.hull.angle
    # Draw rotated rectangle at position

    # Render wheels (green=contact, red=no contact)
    for wheel in car.wheels:
        color = (0, 255, 0) if wheel.ground_contact else (255, 0, 0)
        # Draw wheel indicators

    # Scale to 96x96 and return as grayscale
    return cv2.resize(img, (STATE_W, STATE_H))
```

---

## Which Mode Should I Use?

### For Training: **Vector Mode** (Default, Fastest)
- ✅ 6x faster than visual, 2x faster than synthetic
- ✅ Lowest memory usage
- ✅ Fastest hyperparameter tuning
- ✅ Same learning performance
- ✅ Recommended for all training scenarios

### For Training with CNNs: **Synthetic Mode** (Middle Ground)
- ✅ 3x faster than visual mode
- ✅ Trains CNN architecture for visual processing
- ✅ Simplified, consistent visual representations
- ✅ Good for research on visual representations
- ⚠️ Still slower than vector mode
- ⚠️ Uses same memory as visual mode

### For Watching: **Visual Mode** (Required, Slowest)
- ✅ See realistic rendering of the agent
- ✅ Debug behavior issues visually
- ✅ Create high-quality demonstrations
- ✅ Automatically used by watch scripts
- ❌ Too slow for training

### For Research/Analysis
- **Vector mode** if you need maximum training speed
- **Synthetic mode** if you need CNN training without rendering overhead
- **Visual mode** if you need realistic visual perception
- **All three** if comparing representation learning across modalities

---

## Common Questions

### Q: Does synthetic mode learn as well as visual mode?
**A:** Both use the same CNN architecture and should learn similarly. Synthetic mode provides simplified but consistent visual structure, which can be sufficient for the task.

### Q: When should I use synthetic mode instead of vector mode?
**A:** Use synthetic mode when:
- You want to train a CNN (for transfer learning or research)
- You're researching visual representation learning
- You need spatial structure in the input
- Vector mode is "too easy" for your research question

Otherwise, vector mode is faster and more efficient.

### Q: Can I resume training across different modes?
**A:** Only within compatible modes:
- Vector mode: Standalone (MLP architecture)
- Visual mode: Standalone (CNN architecture)
- Synthetic mode: Standalone (CNN architecture, but different preprocessing)

You cannot transfer checkpoints between vector and visual/synthetic modes due to different network architectures.

### Q: Is synthetic mode "cheating" compared to visual mode?
**A:** No more than visual mode itself. Synthetic mode:
- Uses the same vector state that visual mode implicitly contains
- Provides spatial structure for CNN learning
- Removes pygame rendering overhead, not information content
- Is a valid research tool for studying visual processing

### Q: Why use synthetic mode if vector mode is faster?
**A:**
1. Research requiring CNN architectures
2. Studying role of spatial structure in learning
3. Testing transfer learning to visual domain
4. Comparing representation learning across modalities
5. When you need visual processing without rendering overhead

---

## Benchmarking

To compare performance on your system:

```bash
# Run comprehensive benchmark for all modes
python benchmark_state_modes.py --episodes 50

# Test synthetic mode specifically
python test_synthetic_mode.py --episodes 10
```

This will output:
- Steps per second for each mode
- Estimated training time for 1M steps
- Memory usage comparison
- Performance plots and speedup factors

---

## Summary Table

| Aspect | Visual Mode | Synthetic Mode | Vector Mode |
|--------|-------------|----------------|-------------|
| **State size** | 36,864 values | 36,864 values | 11 values |
| **Network** | CNN | CNN | MLP |
| **Speed** | 57 steps/sec | 165 steps/sec | 313 steps/sec |
| **Speedup** | 1.0x baseline | 2.9x faster | 5.5x faster |
| **Training time** | 4.9 hrs (1M) | 1.7 hrs (1M) | 0.9 hrs (1M) |
| **Memory** | ~14 GB (100k) | ~14 GB (100k) | ~4 MB (100k) |
| **Rendering** | Pygame (full) | NumPy (simple) | None |
| **Visual quality** | Realistic | Simplified | N/A |
| **Use case** | Watching, demos | CNN training | Fast training |
| **Default for** | watch_*.py | Research | train.py |
| **Can watch?** | ✅ Yes (best) | ⚠️ Yes (simple) | ❌ No visuals |
| **Learning quality** | Good | Good | Good (same) |

---

## Conclusion

- **Train in vector mode** for maximum speed and efficiency (default)
- **Train in synthetic mode** for CNN-based learning without rendering overhead
- **Watch in visual mode** to see realistic agent behavior (automatic)
- All modes solve the same task with similar learning performance
- Vector mode's 6x speedup makes it the clear choice for pure training efficiency
- Synthetic mode offers a 3x speedup compromise when CNN architecture is needed
- Visual mode is essential for human observation and high-quality demonstrations

For most users, the defaults handle everything automatically:
```bash
# Training automatically uses vector mode (fastest)
python train.py --episodes 2000

# Alternative: Train with synthetic mode (middle ground)
python train.py --episodes 2000 --state-mode synthetic

# Watching automatically uses visual mode (required for rendering)
python watch_agent.py --checkpoint checkpoints/final_model.pt
```

### Mode Selection Decision Tree

```
Need to train?
├─ Yes → Want CNN architecture?
│  ├─ Yes → Use synthetic mode (3x faster than visual)
│  └─ No → Use vector mode (6x faster than visual) ⭐ RECOMMENDED
└─ No → Need to watch/visualize?
   └─ Yes → Use visual mode (realistic rendering)
```
