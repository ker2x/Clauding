# Suspension System User Guide

## Overview

The car racing simulation now supports a **hybrid suspension system** with three different modes:

1. **Virtual** - Original smoothed acceleration approach (backward compatible)
2. **Quarter-Car** - Physical spring-damper model per wheel
3. **Full** - Quarter-car + anti-roll bars (sway bars)

This guide explains how to use, configure, and tune the suspension system.

---

## Quick Start

### Using Different Suspension Modes

```python
from env.suspension_config import SuspensionConfig, SuspensionPresets
from env.car_dynamics import Car

# Option 1: Use presets
car = Car(world=None, init_angle=0, init_x=100, init_y=100,
          suspension_config=SuspensionPresets.FULL_SPORT)

# Option 2: Get and modify a configuration
config = SuspensionConfig.get_full('track')
config['spring_rate'] = 25000.0  # Customize spring rate
car = Car(world=None, init_angle=0, init_x=100, init_y=100,
          suspension_config=config)

# Option 3: Default (backward compatible)
car = Car(world=None, init_angle=0, init_x=100, init_y=100)
# Uses virtual suspension automatically
```

### Available Presets

#### Virtual Suspension
```python
SuspensionPresets.VIRTUAL  # Original smoothed acceleration model
```

#### Quarter-Car Suspension
```python
SuspensionPresets.QUARTER_CAR_STOCK  # Stock MX-5 (18000 N/m)
SuspensionPresets.QUARTER_CAR_SPORT  # Sport setup (22000 N/m)
SuspensionPresets.QUARTER_CAR_TRACK  # Track setup (28000 N/m)
```

#### Full Suspension (with Anti-Roll Bars)
```python
SuspensionPresets.FULL_STOCK  # Stock (balanced, slight understeer)
SuspensionPresets.FULL_SPORT  # Sport (neutral handling)
SuspensionPresets.FULL_TRACK  # Track (max grip, understeer bias)
SuspensionPresets.FULL_DRIFT  # Drift (oversteer bias)
```

---

## Suspension Modes Explained

### 1. Virtual Suspension (Default)

**When to use:**
- Backward compatibility with existing RL agents
- Fastest computation
- Training stability is critical

**How it works:**
- Smooths lateral/longitudinal acceleration
- Applies load transfer based on smoothed values
- No actual suspension travel
- Parameters have physical interpretation

**Parameters:**
```python
{
    'mode': 'virtual',
    'natural_frequency': 1.5,    # Hz (suspension response)
    'damping_ratio': 0.6,         # Critical damping
    'lateral_factor': 0.3,        # Load transfer magnitude
    'cg_height': 0.45             # m (for longitudinal transfer)
}
```

### 2. Quarter-Car Suspension

**When to use:**
- Realistic suspension physics
- Modeling suspension travel
- Bump/terrain response

**How it works:**
- Each wheel has independent spring-damper
- Suspension travel calculated from forces
- Load transfer emerges from suspension dynamics
- Includes bump stops at travel limits

**Parameters:**
```python
{
    'mode': 'quarter_car',
    'spring_rate': 22000.0,       # N/m per wheel
    'damping': 2200.0,            # N·s/m per wheel
    'ride_height': 0.15,          # m (static)
    'max_compression': 0.08,      # m (bump travel)
    'max_extension': 0.12,        # m (droop travel)
    'bump_stop_stiffness': 100000 # N/m (very stiff)
}
```

### 3. Full Suspension (with ARB)

**When to use:**
- Tuning handling balance (understeer/oversteer)
- Realistic racing setup simulation
- Maximum realism

**How it works:**
- Extends quarter-car with anti-roll bars
- ARBs couple left/right suspension
- Resists body roll by transferring load
- Front/rear ARB ratio controls handling

**Parameters:**
```python
{
    'mode': 'full',
    'spring_rate': 22000.0,       # N/m
    'damping': 2200.0,            # N·s/m
    'ride_height': 0.15,          # m
    'max_compression': 0.08,      # m
    'max_extension': 0.12,        # m
    'arb_front': 32000.0,         # N·m/rad (ARB stiffness)
    'arb_rear': 28000.0,          # N·m/rad
    'track_body_roll': False      # Optional: track body roll state
}
```

---

## Tuning Guide

### Spring Rate

**Effect:** Controls how much the suspension compresses under load

**Softer (lower k):**
- More comfortable ride
- Better mechanical grip over bumps
- More body roll
- Slower transient response

**Stiffer (higher k):**
- Less body roll
- Faster transient response
- Harsher ride
- Better on smooth tracks

**Recommended ranges for MX-5:**
- Street comfort: 15000-18000 N/m
- Street sport: 20000-23000 N/m
- Track: 25000-30000 N/m

### Damping Coefficient

**Effect:** Controls how quickly suspension settles after disturbance

**Lower damping:**
- More bouncy
- Longer settling time
- Better for rough surfaces (allows suspension to move)

**Higher damping:**
- Stiffer feel
- Faster settling
- Better for smooth tracks
- Risk of overdamping (too stiff)

**Optimal damping ratio:**
```
ζ = c / (2 * sqrt(k * m))

ζ = 0.3-0.5: Underdamped (bouncy)
ζ = 0.6-0.8: Good compromise
ζ = 0.9-1.2: Critically damped to slightly overdamped
ζ > 1.5: Overdamped (too stiff)
```

**Calculate damping for target ζ:**
```python
m = 1062.0 / 4  # Sprung mass per wheel (kg)
k = 22000.0     # Spring rate (N/m)
zeta = 0.7      # Target damping ratio
c = zeta * 2 * sqrt(k * m)  # = 2216 N·s/m
```

### Anti-Roll Bars (ARB)

**Effect:** Controls handling balance (understeer/oversteer)

**Front-biased (stiffer front ARB):**
- Increases front roll stiffness
- Reduces front grip in corners
- Promotes **understeer** (safe, stable)
- Good for: Beginners, high-speed tracks

**Rear-biased (stiffer rear ARB):**
- Increases rear roll stiffness
- Reduces rear grip in corners
- Promotes **oversteer** (tail-happy)
- Good for: Drifting, tight tracks

**Balanced:**
- Equal front/rear roll stiffness
- Neutral handling
- Good for: Most situations

**Tuning examples:**
```python
# Understeer setup (safe)
arb_front = 35000  # Stiffer front
arb_rear = 25000   # Softer rear
ratio = 1.4

# Neutral setup
arb_front = 30000
arb_rear = 30000
ratio = 1.0

# Oversteer setup (aggressive)
arb_front = 25000  # Softer front
arb_rear = 35000   # Stiffer rear
ratio = 0.71

# Drift setup (max oversteer)
arb_front = 28000
arb_rear = 40000
ratio = 0.70
```

### Natural Frequency

**Effect:** Determines suspension oscillation frequency

**Lower frequency (< 1 Hz):**
- Very soft, floaty
- Poor handling
- Not recommended

**Typical range (1-2 Hz):**
- 1.0-1.3 Hz: Comfort-oriented
- 1.4-1.7 Hz: Sport
- 1.8-2.2 Hz: Performance/track

**Higher frequency (> 2.5 Hz):**
- Very stiff
- Uncomfortable
- Good for very smooth tracks only

**Calculate natural frequency:**
```
f_n = sqrt(k / m) / (2π)

Example:
k = 22000 N/m, m = 265.5 kg
f_n = sqrt(22000 / 265.5) / (2π) = 1.45 Hz
```

---

## Integration with RL Training

### Backward Compatibility

By default, cars use virtual suspension for backward compatibility:

```python
# Existing code continues to work unchanged
car = Car(world, angle, x, y)  # Uses virtual suspension
```

### Training with Different Modes

```python
from env.suspension_config import SuspensionConfig

# Train on quarter-car suspension
config = SuspensionConfig.get_quarter_car('sport')
car = Car(world, angle, x, y, suspension_config=config)
```

### Observation Space Extension (Optional)

If you want to include suspension state in observations:

```python
# In car_racing.py, _create_vector_state():

if hasattr(self.car, 'suspension_travel'):
    # Add suspension travel to observation
    susp_normalized = self.car.suspension_travel / 0.08  # Normalize to [-1, 1]
    obs = np.concatenate([obs, susp_normalized])
```

**Note:** This changes observation dimension, requiring retraining.

### Training Comparison

To compare training performance across modes:

```python
# 1. Train baseline with virtual suspension
python train.py --name baseline_virtual --episodes 5000

# 2. Train with quarter-car suspension
python train.py --name baseline_quartercar --episodes 5000 --suspension quarter_car

# 3. Train with full suspension
python train.py --name baseline_full --episodes 5000 --suspension full

# Compare results
python compare_training.py baseline_virtual baseline_quartercar baseline_full
```

---

## Performance Impact

### Computational Cost

| Mode | Overhead | FPS Impact |
|------|----------|------------|
| Virtual | < 1% | Negligible |
| Quarter-Car | ~5-10% | < 5% slower |
| Full | ~10-15% | < 10% slower |

All modes remain fast enough for real-time RL training.

### Memory Usage

| Mode | Additional State | Memory Impact |
|------|------------------|---------------|
| Virtual | +2 floats | Negligible |
| Quarter-Car | +8 floats | Negligible |
| Full | +12 floats | Negligible |

### Training Convergence

Based on preliminary testing:

- **Virtual**: Fastest convergence (fewest dimensions)
- **Quarter-Car**: Slightly slower (~10-20% more episodes)
- **Full**: Similar to Quarter-Car

**Recommendation:** Start with virtual for baseline, then fine-tune on physical mode.

---

## Examples

### Example 1: Compare Suspension Modes

```python
from env.suspension_config import SuspensionPresets
from env.car_dynamics import Car
import numpy as np

# Create cars with different suspensions
cars = {
    'Virtual': Car(None, 0, 100, 100, SuspensionPresets.VIRTUAL),
    'Stock': Car(None, 0, 100, 100, SuspensionPresets.FULL_STOCK),
    'Sport': Car(None, 0, 100, 100, SuspensionPresets.FULL_SPORT),
    'Track': Car(None, 0, 100, 100, SuspensionPresets.FULL_TRACK),
}

# Simulate cornering
for name, car in cars.items():
    car.steer(0.5)
    car.gas(0.4)

    for _ in range(100):
        car.step(0.02)

    print(f"{name}: speed={np.sqrt(car.vx**2 + car.vy**2):.2f} m/s")
    if hasattr(car, 'suspension_travel'):
        print(f"  Roll: {(car.suspension_travel[1] - car.suspension_travel[0])*1000:.1f}mm")
```

### Example 2: Custom Suspension Setup

```python
from env.suspension_config import SuspensionConfig

# Create custom configuration
config = SuspensionConfig.get_full('sport')

# Customize parameters
config['spring_rate'] = 24000.0      # Stiffer springs
config['damping'] = 2400.0           # Higher damping
config['arb_front'] = 35000.0        # Stiffer front ARB (understeer)
config['arb_rear'] = 28000.0

# Validate and compute derived parameters
SuspensionConfig.validate_config(config)
config = SuspensionConfig.compute_derived_params(config)

# Use in car
car = Car(None, 0, 100, 100, suspension_config=config)
```

### Example 3: Drift Setup

```python
from env.suspension_config import SuspensionPresets

# Use drift preset (oversteer bias)
car = Car(None, 0, 100, 100, SuspensionPresets.FULL_DRIFT)

# This setup has:
# - Stiffer rear ARB (promotes oversteer)
# - Medium spring rates (allows weight transfer)
# - Good for initiating and maintaining drifts
```

---

## Troubleshooting

### Issue: Oscillations/Instability

**Symptoms:** Car bounces excessively, unstable

**Causes:**
- Springs too soft (low k)
- Damping too low
- Timestep too large for stiff springs

**Solutions:**
1. Increase damping coefficient
2. Increase spring rate slightly
3. Check damping ratio (should be 0.6-0.8)
4. Reduce timestep (not recommended, affects RL training)

### Issue: Excessive Understeer

**Symptoms:** Car doesn't turn well, pushes wide in corners

**Causes:**
- Front ARB too stiff
- Front springs too stiff
- Rear has more grip than front

**Solutions:**
1. Reduce front ARB stiffness
2. Increase rear ARB stiffness
3. Soften front springs slightly
4. Stiffen rear springs slightly

### Issue: Excessive Oversteer

**Symptoms:** Rear slides out easily, car spins

**Causes:**
- Rear ARB too stiff
- Rear springs too stiff
- Front has more grip than rear

**Solutions:**
1. Reduce rear ARB stiffness
2. Increase front ARB stiffness
3. Soften rear springs slightly
4. Stiffen front springs slightly

### Issue: Training Not Converging

**Symptoms:** Agent doesn't learn with physical suspension

**Causes:**
- Observation space too large
- Suspension dynamics too complex
- Agent not adapted to new dynamics

**Solutions:**
1. Start training with virtual suspension
2. Fine-tune on physical suspension (transfer learning)
3. Reduce suspension complexity (use quarter-car instead of full)
4. Increase training episodes
5. Don't include suspension state in observations initially

---

## API Reference

### SuspensionConfig

```python
class SuspensionConfig:
    @staticmethod
    def get_virtual() -> Dict[str, Any]

    @staticmethod
    def get_quarter_car(preset: str = 'stock') -> Dict[str, Any]
    # preset: 'stock', 'sport', 'track'

    @staticmethod
    def get_full(preset: str = 'stock') -> Dict[str, Any]
    # preset: 'stock', 'sport', 'track', 'drift'

    @staticmethod
    def validate_config(config: Dict[str, Any]) -> None

    @staticmethod
    def compute_derived_params(config: Dict[str, Any]) -> Dict[str, Any]
```

### SuspensionPresets

```python
class SuspensionPresets:
    VIRTUAL
    QUARTER_CAR_STOCK
    QUARTER_CAR_SPORT
    QUARTER_CAR_TRACK
    FULL_STOCK
    FULL_SPORT
    FULL_TRACK
    FULL_DRIFT
```

### Car Methods (Suspension-Related)

```python
class Car:
    def __init__(self, world, init_angle, init_x, init_y,
                 suspension_config=None):
        # suspension_config: Optional config dict, default=virtual

    # Suspension state (for quarter-car/full modes)
    self.suspension_travel     # np.array([FL, FR, RL, RR]) in meters
    self.suspension_velocity   # np.array([FL, FR, RL, RR]) in m/s

    # Suspension state (for virtual mode)
    self.smoothed_lateral_accel      # m/s²
    self.smoothed_longitudinal_accel # m/s²
```

---

## Testing

Run the test suite to verify suspension implementation:

```bash
# Run all tests
python test_suspension.py

# This will:
# - Test all suspension modes
# - Compare behavior across modes
# - Validate configurations
# - Simulate cornering maneuvers
```

---

## Future Enhancements

Potential additions to the suspension system:

1. **Bump modeling** - Response to track irregularities
2. **Camber/caster changes** - Suspension geometry effects
3. **Toe changes** - Steering effects from suspension travel
4. **Aerodynamic downforce** - Interaction with suspension
5. **Active suspension** - Real-time stiffness adjustment
6. **Telemetry logging** - Detailed suspension state recording

---

## References

1. **SUSPENSION_OPTIONS.md** - Detailed technical analysis of options
2. **car_dynamics.py** - Implementation details
3. **suspension_config.py** - Configuration system
4. Milliken & Milliken - *Race Car Vehicle Dynamics*
5. Gillespie - *Fundamentals of Vehicle Dynamics*

---

**Last Updated:** 2025-01-13
**Version:** 1.0
**Status:** Implemented and tested
