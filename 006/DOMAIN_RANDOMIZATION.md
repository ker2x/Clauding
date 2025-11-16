# Domain Randomization for CarRacing Environment

This document describes the domain randomization feature for the CarRacing environment, which helps train more robust and generalizable reinforcement learning policies.

## Table of Contents
1. [Overview](#overview)
2. [Quick Start](#quick-start)
3. [Configuration](#configuration)
4. [Preset Configurations](#preset-configurations)
5. [Training Integration](#training-integration)
6. [Parameter Reference](#parameter-reference)
7. [Best Practices](#best-practices)

## Overview

Domain randomization is a technique that varies simulation parameters across episodes to:
- **Improve generalization**: Policies learn to handle a range of conditions
- **Increase robustness**: Trained agents work across different scenarios
- **Reduce overfitting**: Prevents memorization of specific track/vehicle parameters
- **Sim-to-real transfer**: Helps bridge the gap between simulation and reality

### What Gets Randomized?

The system can randomize:
- **Vehicle parameters**: Mass, dimensions, center of gravity, weight distribution
- **Tire properties**: Grip levels (Pacejka parameters), rolling resistance
- **Drivetrain**: Engine power, torque, braking force
- **Aerodynamics**: Drag coefficient, frontal area
- **Track**: Surface friction, track width
- **Visual**: Colors and appearance (useful for vision-based training)

## Quick Start

### Basic Usage

```python
from env.car_racing import CarRacing
from config.domain_randomization import conservative_randomization

# Create environment with domain randomization
config = conservative_randomization()
env = CarRacing(
    domain_randomization_config=config,
    verbose=True  # See randomization parameters
)

# Use normally - randomization happens on reset()
obs, info = env.reset()
```

### Running Examples

Try the included examples:
```bash
cd 006
python example_domain_randomization.py
```

## Configuration

### Creating Custom Configurations

```python
from config.domain_randomization import (
    DomainRandomizationConfig,
    VehicleRandomization,
    TireRandomization,
    TrackRandomization,
)

config = DomainRandomizationConfig(
    enabled=True,
    vehicle=VehicleRandomization(
        mass_range=(0.9, 1.1),        # ±10% mass variation
        cg_height_range=(0.95, 1.05), # ±5% CG height
    ),
    tire=TireRandomization(
        pacejka_d_lat_range=(0.85, 1.15),  # ±15% lateral grip
        pacejka_d_lon_range=(0.85, 1.15),  # ±15% longitudinal grip
    ),
    track=TrackRandomization(
        surface_friction_range=(0.9, 1.1),  # ±10% surface friction
    ),
)

env = CarRacing(domain_randomization_config=config)
```

### Range Specification

All randomization ranges use **multipliers** applied to base values:
- `(1.0, 1.0)`: No randomization (default)
- `(0.9, 1.1)`: ±10% variation
- `(0.8, 1.2)`: ±20% variation

For example:
- Base mass = 1062 kg
- `mass_range=(0.9, 1.1)` → Random mass between 956-1168 kg

## Preset Configurations

The system includes several preset configurations for common use cases:

### 1. Conservative Randomization
**Good for: Initial training, beginners**

```python
from config.domain_randomization import conservative_randomization
config = conservative_randomization()
```

**Variations:**
- Mass: ±5%
- Grip: ±5%
- Surface friction: ±5%
- CG height: ±5%

### 2. Moderate Randomization
**Good for: Intermediate training, balanced robustness**

```python
from config.domain_randomization import moderate_randomization
config = moderate_randomization()
```

**Variations:**
- Mass: ±10%
- Grip: ±10%
- Surface friction: ±10%
- Weight distribution: 45/55 to 55/45
- Engine power: ±10%
- Tire stiffness: ±5%

### 3. Aggressive Randomization
**Good for: Maximum robustness, advanced training**

```python
from config.domain_randomization import aggressive_randomization
config = aggressive_randomization()
```

**Variations:**
- Mass: ±15%
- Grip: ±20%
- Surface friction: ±15%
- Weight distribution: 40/60 to 60/40
- Engine power: ±15%
- Tire parameters: ±5-10%
- Dimensions: ±5%

### 4. Wet Surface Conditions
**Good for: Training on slippery surfaces**

```python
from config.domain_randomization import wet_surface_conditions
config = wet_surface_conditions()
```

**Variations:**
- Lateral grip: 60-80% (wet tires)
- Longitudinal grip: 70-90%
- Surface friction: 60-80% (wet surface)
- Tire response: softer

## Training Integration

### Basic Training

```python
from config.domain_randomization import moderate_randomization

# Training with randomization
config = moderate_randomization()
train_env = CarRacing(
    state_mode="vector",
    max_episode_steps=2500,
    domain_randomization_config=config,
)

# Evaluation without randomization (on nominal parameters)
eval_env = CarRacing(
    state_mode="vector",
    max_episode_steps=2500,
    domain_randomization_config=DomainRandomizationConfig(enabled=False),
)
```

### Curriculum Learning

Start with easy randomization and gradually increase difficulty:

```python
def get_curriculum_config(episode):
    """Progressive difficulty curriculum."""
    if episode < 100:
        return conservative_randomization()
    elif episode < 500:
        return moderate_randomization()
    else:
        return aggressive_randomization()

# In training loop
for episode in range(num_episodes):
    config = get_curriculum_config(episode)
    env = CarRacing(domain_randomization_config=config)
    # ... train ...
```

### Parallel Training

For parallel training (e.g., `train_selection_parallel.py`):

```python
from config.domain_randomization import moderate_randomization

# Create config once
domain_rand_config = moderate_randomization()

# Each worker gets same config (randomization happens on reset())
def make_env():
    return CarRacing(
        domain_randomization_config=domain_rand_config
    )
```

## Parameter Reference

### VehicleRandomization

| Parameter | Description | Base Value | Typical Range |
|-----------|-------------|------------|---------------|
| `mass_range` | Vehicle mass multiplier | 1062 kg | (0.85, 1.15) |
| `wheelbase_range` | Wheelbase multiplier | 2.31 m | (0.95, 1.05) |
| `track_width_range` | Track width multiplier | 1.50 m | (0.95, 1.05) |
| `cg_height_range` | Center of gravity height | 0.46 m | (0.85, 1.15) |
| `weight_distribution_range` | Front/rear weight split | 0.5 (50/50) | (0.40, 0.60) |

### TireRandomization

| Parameter | Description | Typical Range |
|-----------|-------------|---------------|
| `pacejka_d_lat_range` | Lateral (cornering) grip | (0.80, 1.20) |
| `pacejka_d_lon_range` | Longitudinal (accel/brake) grip | (0.80, 1.20) |
| `pacejka_b_lat_range` | Lateral stiffness | (0.90, 1.10) |
| `pacejka_b_lon_range` | Longitudinal stiffness | (0.90, 1.10) |
| `pacejka_c_lat_range` | Lateral curve shape | (0.95, 1.05) |
| `pacejka_c_lon_range` | Longitudinal curve shape | (0.95, 1.05) |
| `pacejka_e_lat_range` | Lateral falloff | (0.95, 1.05) |
| `pacejka_e_lon_range` | Longitudinal falloff | (0.95, 1.05) |
| `rolling_resistance_range` | Rolling resistance | (0.90, 1.10) |

**Key Pacejka Parameters:**
- **D (Peak)**: Controls maximum grip level (most important for randomization)
- **B (Stiffness)**: Controls initial tire response and overall force levels
- **C (Shape)**: Controls curve peakiness
- **E (Curvature)**: Controls falloff after peak slip

### DrivetrainRandomization

| Parameter | Description | Typical Range |
|-----------|-------------|---------------|
| `engine_power_range` | Engine power | (0.85, 1.15) |
| `max_torque_range` | Max torque per wheel | (0.85, 1.15) |
| `brake_torque_front_range` | Front brake torque | (0.90, 1.10) |
| `brake_torque_rear_range` | Rear brake torque | (0.90, 1.10) |

### AerodynamicsRandomization

| Parameter | Description | Typical Range |
|-----------|-------------|---------------|
| `drag_coefficient_range` | Drag coefficient | (0.90, 1.10) |
| `frontal_area_range` | Frontal area | (0.95, 1.05) |

### TrackRandomization

| Parameter | Description | Typical Range |
|-----------|-------------|---------------|
| `surface_friction_range` | Surface friction multiplier | (0.85, 1.15) |
| `track_width_range` | Track width multiplier | (0.95, 1.05) |

Note: Track width randomization affects difficulty significantly. Use conservatively.

### VisualRandomization

| Parameter | Description | Use Case |
|-----------|-------------|----------|
| `randomize_track_color` | Randomize track color | Vision-based policies |
| `track_color_noise_std` | RGB noise std deviation | Typical: 10-20 |
| `randomize_bg_color` | Randomize background | Vision-based policies |
| `bg_color_noise_std` | RGB noise std deviation | Typical: 10-20 |
| `randomize_car_color` | Randomize car color | Vision-based policies |

## Best Practices

### 1. Start Conservative
Begin with small variations (±5-10%) and gradually increase:
```python
# Week 1: Conservative
config = conservative_randomization()

# Week 2-3: Moderate
config = moderate_randomization()

# Week 4+: Aggressive
config = aggressive_randomization()
```

### 2. Evaluate Without Randomization
Always evaluate policies on nominal (non-randomized) parameters:
```python
# Training
train_env = CarRacing(domain_randomization_config=moderate_randomization())

# Evaluation
eval_env = CarRacing(
    domain_randomization_config=DomainRandomizationConfig(enabled=False)
)
```

### 3. Monitor Training
Domain randomization can slow initial learning. Track:
- Episode reward variance (should be higher with randomization)
- Convergence time (may be slower but results more robust)
- Evaluation performance (should improve over time)

### 4. Randomize Key Parameters First
Focus on parameters that matter most:
1. **Grip levels** (`pacejka_d_lat`, `pacejka_d_lon`) - Most important
2. **Mass** - Affects handling significantly
3. **Surface friction** - Changes overall difficulty
4. **Engine power** - Affects acceleration

### 5. Use Appropriate Ranges
- **Too little randomization**: Policy may overfit
- **Too much randomization**: Learning may be too slow or unstable

Recommended starting point: **moderate_randomization()**

### 6. Curriculum Learning
Gradually increase randomization difficulty:
```python
def adaptive_randomization(episode, success_rate):
    """Increase randomization as policy improves."""
    if success_rate < 0.3:
        return conservative_randomization()
    elif success_rate < 0.6:
        return moderate_randomization()
    else:
        return aggressive_randomization()
```

### 7. Log Randomization Parameters
Track which parameters were used in each episode:
```python
obs, info = env.reset()
rand_info = env.domain_randomizer.get_info_dict()
# Log rand_info with episode data
```

## Performance Considerations

Domain randomization has minimal computational overhead:
- Randomization happens only on `reset()` (once per episode)
- No performance impact during `step()`
- Memory overhead: negligible

## Troubleshooting

### Training is unstable
- Reduce randomization ranges (use conservative preset)
- Ensure base parameters are working well first
- Check that learning rate is appropriate

### Policy not improving
- May need more episodes (randomization increases variance)
- Try curriculum learning (start conservative)
- Verify base environment works without randomization

### Too slow convergence
- Reduce randomization ranges
- Use curriculum learning
- Focus on key parameters (grip, mass) first

## Examples

See `example_domain_randomization.py` for complete examples:
```bash
python example_domain_randomization.py
```

## Integration with Existing Scripts

### train.py
```python
# Add at top
from config.domain_randomization import moderate_randomization

# Modify environment creation
env = CarRacing(
    state_mode="vector",
    domain_randomization_config=moderate_randomization(),
)
```

### train_selection_parallel.py
```python
# Add at top
from config.domain_randomization import moderate_randomization

# In worker function
def worker(...):
    env = CarRacing(
        domain_randomization_config=moderate_randomization(),
    )
```

## References

- [Domain Randomization for Sim-to-Real Transfer (Tobin et al., 2017)](https://arxiv.org/abs/1703.06907)
- [Sim-to-Real Transfer in Deep RL (Peng et al., 2018)](https://arxiv.org/abs/1710.06537)
- [Pacejka Tire Model Documentation](../TIRE_PARAMETERS.md)

## Future Enhancements

Potential improvements:
- [ ] Automatic Domain Randomization (ADR)
- [ ] Adversarial Domain Randomization
- [ ] Track geometry randomization
- [ ] Weather effects (rain, wind)
- [ ] Time-varying parameters (tire degradation)
