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

The system includes four preset configurations designed for different training scenarios and skill levels. Each preset balances learning difficulty with policy robustness.

### 1. Conservative Randomization
**Good for: Initial training, learning fundamentals, beginners**

```python
from config.domain_randomization import conservative_randomization
config = conservative_randomization()
```

**What it does:**
Conservative randomization introduces small variations (±5%) to help the policy generalize without significantly increasing training difficulty. This is the gentlest introduction to domain randomization.

**Parameter Variations:**
- Mass: ±5% (1009-1115 kg vs base 1062 kg)
- Lateral grip: ±5% (0.90-1.00 vs base 0.95)
- Longitudinal grip: ±5% (1.28-1.42 vs base 1.35)
- Surface friction: ±5% (0.95-1.05 multiplier)
- CG height: ±5% (0.437-0.483 m vs base 0.46 m)

**When to use:**
- **Starting out**: Your first attempt at domain randomization
- **Debugging**: When testing if your setup works correctly
- **Baseline comparison**: Establish performance with minimal randomization
- **Fine-tuning**: When you have a working policy and want to add robustness without disrupting it

**Expected behavior:**
- Training speed: Nearly same as no randomization (~5-10% slower convergence)
- Reward variance: Slightly higher episode-to-episode variance
- Policy robustness: Modest improvement in handling parameter uncertainty
- Success rate: Should maintain >90% of baseline performance

**Tips:**
- Start here if you're new to domain randomization
- Use this for initial prototyping before scaling up
- Good for verifying that randomization doesn't break your training
- Can be used as a "sanity check" configuration

---

### 2. Moderate Randomization
**Good for: Intermediate training, balanced robustness, recommended default**

```python
from config.domain_randomization import moderate_randomization
config = moderate_randomization()
```

**What it does:**
Moderate randomization provides a balanced trade-off between training difficulty and policy robustness. This is the **recommended starting point** for most users who want meaningful generalization improvements.

**Parameter Variations:**
- Mass: ±10% (956-1168 kg vs base 1062 kg)
- Lateral grip: ±10% (0.86-1.05 vs base 0.95)
- Longitudinal grip: ±10% (1.22-1.49 vs base 1.35)
- Surface friction: ±10% (0.90-1.10 multiplier)
- Weight distribution: 45/55 to 55/45 (vs base 50/50)
- Engine power: ±10% (121.5-148.5 kW vs base 135 kW)
- Tire stiffness (B): ±5% (affects initial tire response)
- Drag coefficient: ±5% (affects top speed)

**When to use:**
- **Production training**: When you want a robust policy for actual use
- **General-purpose agents**: Training for unknown/varying conditions
- **After conservative**: Next step after validating with conservative preset
- **Default choice**: When unsure which preset to use

**Expected behavior:**
- Training speed: 15-25% slower convergence than no randomization
- Reward variance: Moderate episode-to-episode variance
- Policy robustness: Significant improvement in handling parameter changes
- Success rate: Policies work well across 80-90% of randomized conditions

**What you'll notice:**
- Car feels "different" each episode (varying grip, power, handling)
- Policy learns more cautious, adaptive driving style
- Better recovery from mistakes and unexpected situations
- More consistent performance across different tracks

**Tips:**
- This is the sweet spot for most use cases
- Expect initial learning to be slower but final policy to be more robust
- Monitor training - if it's struggling, drop back to conservative
- Good for curriculum learning (start conservative, progress to moderate)

---

### 3. Aggressive Randomization
**Good for: Maximum robustness, advanced training, challenging conditions**

```python
from config.domain_randomization import aggressive_randomization
config = aggressive_randomization()
```

**What it does:**
Aggressive randomization creates highly variable conditions with large parameter swings (±15-25%). This forces the policy to be extremely robust but makes learning significantly harder. Only recommended for advanced users or when maximum robustness is critical.

**Parameter Variations:**
- Mass: ±15% (903-1221 kg vs base 1062 kg) - *Dramatic handling changes*
- Lateral grip: ±20% (0.76-1.14 vs base 0.95) - *From slippery to very grippy*
- Longitudinal grip: ±20% (1.08-1.62 vs base 1.35) - *Major braking/acceleration differences*
- Surface friction: ±15% (0.85-1.15 multiplier)
- Weight distribution: 40/60 to 60/40 - *Front-heavy to rear-heavy*
- Engine power: ±15% (114.8-155.3 kW vs base 135 kW)
- Tire stiffness (B): ±10% (affects tire force curve)
- Tire shape (C): ±5% (affects peak grip characteristics)
- Wheelbase: ±5% (affects stability)
- Track width: ±5% (affects cornering)
- Rolling resistance: ±10% (affects efficiency)
- Drag: ±10% (affects top speed significantly)

**When to use:**
- **Maximum robustness needed**: Sim-to-real transfer, unknown deployment conditions
- **Stress testing**: Evaluate how well your policy handles extreme variations
- **Advanced research**: Studying policy adaptation and robustness limits
- **After mastering moderate**: When moderate randomization seems too easy

**Expected behavior:**
- Training speed: 30-50% slower convergence, may need 2-3× more episodes
- Reward variance: High episode-to-episode variance
- Policy robustness: Excellent - handles nearly any reasonable parameter variation
- Success rate: Policies adapt to 70-85% of randomized conditions
- Learning difficulty: Significantly harder, may fail without good hyperparameters

**What you'll notice:**
- Episodes feel dramatically different from each other
- Some episodes are very difficult (low grip + heavy car)
- Some episodes are very easy (high grip + powerful engine)
- Policy learns defensive, highly adaptive driving
- May see bimodal reward distributions (easy vs hard conditions)

**Challenges:**
- Initial learning can be very slow or unstable
- May need larger networks or more training time
- Requires good hyperparameter tuning (learning rates, etc.)
- Risk of policy getting stuck in local optima

**Tips:**
- Use curriculum learning: start with conservative/moderate, progress to aggressive
- Consider increasing learning rate or buffer size to handle variance
- Monitor training closely - be ready to adjust hyperparameters
- Useful for final "hardening" of an already-working policy
- Not recommended as first attempt at domain randomization

---

### 4. Wet Surface Conditions
**Good for: Training on slippery surfaces, adverse weather simulation**

```python
from config.domain_randomization import wet_surface_conditions
config = wet_surface_conditions()
```

**What it does:**
Wet surface conditions simulates driving in rain or on slippery surfaces by reducing grip levels while keeping other parameters constant. This is a **scenario-specific** preset rather than a general robustness training tool.

**Parameter Variations:**
- Lateral grip: 60-80% of normal (0.57-0.76 vs base 0.95) - *Reduced cornering grip*
- Longitudinal grip: 70-90% of normal (0.95-1.22 vs base 1.35) - *Reduced braking*
- Surface friction: 60-80% of normal - *Wet track surface*
- Tire stiffness (B): 85-95% of normal - *Softer, progressive tire response*

**When to use:**
- **Adverse weather training**: Train policies specifically for wet conditions
- **Safety-critical applications**: Ensure safe driving in low-grip scenarios
- **Specialized scenarios**: Racing in rain, winter conditions, oil spills
- **Complementary training**: Combine with other presets for full coverage

**Expected behavior:**
- Training speed: Similar to moderate randomization
- Driving style: Much more cautious, earlier braking, gentler steering
- Success rate: Lower initial success, but policy adapts to low-grip driving
- Performance: Top speeds and cornering speeds notably reduced

**What you'll notice:**
- Car slides much more easily
- Braking distances significantly increased
- Aggressive steering inputs cause spinouts
- Policy learns smooth, progressive control inputs
- Lower overall lap times but more consistent

**Differences from other presets:**
- **Not general robustness**: Specifically targets low-grip scenarios
- **Reduced grip only**: Doesn't vary mass, power, or other parameters
- **Narrower distribution**: All conditions are "wet" - no dry conditions
- **Specialized skill**: Teaches wet-weather driving, not general adaptation

**Training strategies:**

**Option 1: Wet-only training**
Train exclusively on wet conditions to create a wet-weather specialist:
```python
train_env = CarRacing(domain_randomization_config=wet_surface_conditions())
```
- Result: Expert wet-weather driver, may struggle on dry surfaces

**Option 2: Mixed training (recommended)**
Combine wet conditions with general randomization for all-weather capability:
```python
# Alternate between wet and dry randomly
import random
def get_mixed_config():
    if random.random() < 0.3:  # 30% wet conditions
        return wet_surface_conditions()
    else:
        return moderate_randomization()
```
- Result: Robust all-weather driver

**Option 3: Curriculum learning**
Start with dry, progress to wet:
```python
# Episodes 0-500: dry conditions
# Episodes 500-1000: mixed conditions
# Episodes 1000+: includes wet conditions
```
- Result: Gradual adaptation to difficult conditions

**Tips:**
- Use wet conditions to test policy robustness under stress
- Combine with moderate/aggressive for comprehensive training
- Good for evaluating how well policies handle adversity
- Consider as final training phase after mastering dry conditions
- Useful for safety validation before deployment

---

## Choosing the Right Preset

**Decision flowchart:**

```
Are you new to domain randomization?
├─ Yes → Start with CONSERVATIVE
└─ No ↓

Do you have a working policy already?
├─ Yes → Use MODERATE or AGGRESSIVE to add robustness
└─ No → Start with CONSERVATIVE or MODERATE

Is maximum robustness critical?
├─ Yes → Use AGGRESSIVE (after validating with moderate)
└─ No → Use MODERATE

Do you need wet-weather capability?
├─ Yes → Use WET (alone or mixed with others)
└─ No → Use MODERATE or AGGRESSIVE

Is training time a concern?
├─ Yes → Use CONSERVATIVE or MODERATE
└─ No → Use AGGRESSIVE for best robustness
```

**Typical progression:**
1. **Week 1**: Conservative (validate setup)
2. **Week 2-3**: Moderate (build robust baseline)
3. **Week 4**: Aggressive (final hardening)
4. **Week 5**: Mixed with wet conditions (all-weather capability)

**Quick reference:**

| Preset | Difficulty | Training Time | Robustness | Best For |
|--------|-----------|---------------|------------|----------|
| Conservative | Low | Fast | Low | Beginners, debugging |
| Moderate | Medium | Medium | Good | **Most users, production** |
| Aggressive | High | Slow | Excellent | Advanced, max robustness |
| Wet | Medium | Medium | Specialized | Wet conditions, safety |

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
