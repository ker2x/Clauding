# CarRacing Environment - Custom Physics Engine

This directory contains a custom implementation of the CarRacing-v3 environment with a clean 2D physics engine replacing Box2D.

## Overview

The environment provides a realistic car racing simulation with:
- **Custom 2D Physics**: Clean, interpretable physics without Box2D dependency
- **Pacejka Magic Formula Tires**: Industry-standard tire model
- **Realistic Load Transfer**: Physics-based weight distribution during cornering/braking
- **Configurable Suspension**: Independent per-wheel spring-damper system
- **Multiple State Modes**: Vector (71D) and visual (96x96 RGB) observation modes

## Architecture

```
env/
├── __init__.py           # Package exports (CarRacing, Car)
├── car_dynamics.py       # Core physics: Car class and tire model
├── car_racing.py         # Gymnasium environment wrapper
└── suspension_config.py  # Suspension parameters and validation
```

## Core Components

### 1. car_dynamics.py

**Purpose**: Implements 2D top-down car physics simulation

**Key Classes**:

#### `PacejkaTire`
Pacejka Magic Formula tire model for realistic tire forces.

```python
PacejkaTire(
    B_lat=8.5,   # Lateral stiffness
    C_lat=1.9,   # Lateral shape factor
    D_lat=0.95,  # Lateral peak friction
    E_lat=0.97,  # Lateral curvature
    B_lon=12.0,  # Longitudinal stiffness
    C_lon=1.9,   # Longitudinal shape factor
    D_lon=1.35,  # Longitudinal peak friction
    E_lon=0.97   # Longitudinal curvature
)
```

**Methods**:
- `lateral_force(slip_angle, normal_force, max_friction)`: Calculate cornering force
- `longitudinal_force(slip_ratio, normal_force, max_friction)`: Calculate traction/braking force

**Magic Formula**:
```
F = D × sin(C × arctan(B×α - E×(B×α - arctan(B×α))))
```
Where `α` is slip angle (lateral) or slip ratio (longitudinal).

#### `Car`
2D top-down car with Pacejka tires and realistic dynamics.

**Vehicle Parameters** (Based on 2022 Mazda MX-5 Sport):
```python
MASS = 1062.0 kg          # Vehicle mass
LENGTH = 2.310 m          # Wheelbase
WIDTH = 1.50 m            # Track width
LF = LENGTH * 0.5         # Distance CG to front axle (50/50 distribution)
LR = LENGTH * 0.5         # Distance CG to rear axle
TIRE_RADIUS = 0.309 m     # Wheel radius (195/50R16)
CG_HEIGHT = 0.46 m        # Center of gravity height
```

**State Vector** (car_dynamics.py:88-98):
- Position: `(x, y)` in world frame
- Velocity: `(vx, vy)` in vehicle frame
- Heading: `yaw` angle (rad)
- Yaw rate: `r = d(yaw)/dt` (rad/s)
- Wheel speeds: `[fl, fr, rl, rr]` angular velocities (rad/s)
- Steering angle: `delta` (rad, front wheels only)

**Key Methods**:
- `step(dt)`: Integrate dynamics forward by dt seconds
- `gas(throttle)`: Apply throttle [0, 1]
- `brake(brake_force)`: Apply brake [0, 1]
- `steer(steer_input)`: Apply steering [-1, 1]

**Physics Pipeline** (car_dynamics.py:406-449):
```
1. Update steering angle (smooth ramping)
2. Update wheel angular velocities (engine/brake torque)
3. Compute tire forces (Pacejka model with load transfer)
4. Integrate state (position, velocity, yaw)
5. Update hull for rendering
```

**Drivetrain** (car_dynamics.py:215-250):
- **Engine**: 135 kW (181 hp), 2.0L Skyactiv-G
- **Drive**: Rear-wheel drive (RWD)
- **Torque**: 400 Nm per rear wheel (realistic delivery)
- **Power Transition**: 168.75 rad/s (~162 km/h)

**Braking System** (car_dynamics.py:252-260):
- **Front**: 60 Nm max brake torque per wheel
- **Rear**: 40 Nm max brake torque per wheel
- **Bias**: 60/40 front/rear (prevents rear lockup)
- **Performance**: ~1.1-1.3g total braking

**Aerodynamics** (car_dynamics.py:265-272):
```python
RHO_AIR = 1.225 kg/m³           # Air density
FRONTAL_AREA = 1.8 m²           # Frontal area
CD_CAR = 0.33                   # Drag coefficient (ND MX-5 RF)
C_ROLL_RESISTANCE = 0.015       # Rolling resistance
```

**Load Transfer** (car_dynamics.py:583-634):

The car uses a **rigid-body load transfer model** with filtered accelerations:

```python
# Static load: 1062 kg / 4 = 265.5 kg per wheel = 2604.6 N

# Longitudinal transfer (pitch):
lon_transfer = -ax × mass × cg_height / wheelbase
normal_forces[front] += lon_transfer / 2
normal_forces[rear] -= lon_transfer / 2

# Lateral transfer (roll):
lat_transfer = -ay × mass × cg_height / track_width
normal_forces[left] -= lat_transfer / 2
normal_forces[right] += lat_transfer / 2
```

**Acceleration Filtering** (car_dynamics.py:854-861):
```python
# Low-pass filter to prevent oscillations
filter_alpha = 0.15
ax_filtered = ax_filtered × 0.85 + ax × 0.15
ay_filtered = ay_filtered × 0.85 + ay × 0.15
```

This prevents rapid frame-to-frame changes in load transfer that could cause instability.

**Tire Force Feedback** (car_dynamics.py:451-557):

The wheel dynamics include tire force feedback to prevent unrealistic wheel spin:

```python
# Physics: I × α = T_applied - F_tire × r
# Where F_tire is the longitudinal force from previous timestep

tire_force_torque = prev_tire_forces_filtered[i] × tire_radius × coupling
net_torque = engine_torque - tire_force_torque
accel = net_torque / inertia
```

**Tire forces are filtered** (car_dynamics.py:430-436) with the same alpha=0.15 as accelerations to prevent oscillations.

### 2. car_racing.py

**Purpose**: Gymnasium environment wrapper for reinforcement learning

**Key Classes**:

#### `FrictionDetector`
Detects wheel-track collisions using polygon-based geometry (car_racing.py:82-271).

**Methods**:
- `_point_in_polygon(px, py, vertices)`: Ray casting algorithm
- `_distance_to_polygon_edge(px, py, vertices)`: Minimum distance to polygon edge
- `update_contacts(car, road_tiles)`: Update wheel-tile contacts with spatial partitioning

**Performance Optimization** (car_racing.py:158-206):
```python
# Spatial partitioning: Only check nearby tiles
SPATIAL_CHECK_RANGE = 30  # ±30 tiles around car
NEAR_TRACK_THRESHOLD = 0.3  # Allow 0.3 units outside polygon

# Two-stage search:
# 1. Coarse: Check every 10th tile
# 2. Fine: Refine around closest tile
# Result: ~61 tiles checked instead of 300
```

#### `CarRacing`
Main Gymnasium environment.

**Initialization** (car_racing.py:373-420):
```python
CarRacing(
    render_mode: str | None = None,
    verbose: bool = False,
    lap_complete_percent: float = 0.95,
    domain_randomize: bool = False,
    continuous: bool = True,
    terminate_stationary: bool = True,
    stationary_patience: int = 50,
    stationary_min_steps: int = 50,
    state_mode: str = "vector",          # "vector" or "visual"
    max_episode_steps: int | None = 2500,
    reward_shaping: bool = True,
    min_episode_steps: int = 150,
    short_episode_penalty: float = -50.0,
    num_cars: int = 1,                   # Multi-car support
    suspension_config: dict | None = None
)
```

**Action Space** (car_racing.py:289-300):
```python
# Continuous (default):
action_space = Box([-1, -1], [+1, +1])  # [steering, acceleration]
# steering ∈ [-1, 1]: -1 = full left, +1 = full right
# acceleration ∈ [-1, 1]: -1 = full brake, +1 = full gas

# Discrete (alternative):
action_space = Discrete(5)  # [nothing, right, left, gas, brake]
```

**Observation Space** (car_racing.py:493-502):

**Vector Mode (71D)** - RECOMMENDED for training:
```python
state = [
    # Car state (11D)
    hull.angle,                    # Heading angle
    hull.angularVelocity,          # Yaw rate
    vel_x, vel_y,                  # Velocity (car frame)
    pos_x, pos_y,                  # Position (world frame)
    *wheel_contacts,               # 4 wheels on track [0/1]
    progress,                      # Lap progress [0, 1]

    # Track segment (5D)
    distance_to_center,            # Lateral offset from centerline
    track_angle,                   # Track heading
    track_curvature,               # Turn sharpness
    segment_start, segment_end,    # Segment bounds

    # Lookahead waypoints (40D = 20 waypoints × 2 coords)
    *waypoints_in_car_frame,       # Future path (x, y) in car coordinates

    # Vertical forces (4D)
    *normal_forces,                # Tire loads [FL, FR, RL, RR]

    # Additional features (11D)
    *opponent_info,                # Nearest opponent data
    ...
]
```

**Visual Mode (96×96×3)** - For visualization only:
```python
observation_space = Box(0, 255, (96, 96, 3), dtype=uint8)
```

**Reward Structure** (car_racing.py:311-361):

All parameters configurable via `constants.py`:

```python
# Dense rewards (main objective)
+progress_delta × PROGRESS_REWARD_SCALE  # Default: 2000 (full lap = +2000)
+LAP_COMPLETION_REWARD                    # Default: +1000 for completing lap
+ONTRACK_REWARD                           # Default: +0.5 per frame (all wheels on)
+FORWARD_SPEED_REWARD_SCALE × speed       # Default: +0.1 per m/s

# Dense penalties (constraints)
-STEP_PENALTY                             # Default: -0.5 per frame (time pressure)
-STATIONARY_PENALTY                       # Default: -1.0 when speed < 0.5 m/s
-OFFTRACK_PENALTY × wheels_off            # Default: -5.0 per wheel off track

# Terminal penalty
-OFFTRACK_TERMINATION_PENALTY × (1.0 - progress)  # Default: -100 × progress_multiplier
```

**Episode Termination** (car_racing.py:352-361):
1. All tiles visited (lap complete)
2. All 4 wheels off track
3. `max_episode_steps` reached (default: 2500)
4. Stationary for `stationary_patience` frames (default: 50)

### 3. suspension_config.py

**Purpose**: Suspension configuration and validation

**Functions**:

#### `get_suspension_config()`
Returns default MX-5-based suspension parameters:

```python
{
    # Spring and damper (per wheel)
    'spring_rate': 45000.0,        # N/m (balanced street/track)
    'damping': 1900.0,             # N·s/m (damping ratio ~0.70)

    # Geometry
    'ride_height': 0.15,           # m (static height)
    'max_compression': 0.08,       # m (bump travel)
    'max_extension': 0.12,         # m (droop travel)

    # Bump stops
    'bump_stop_stiffness': 150000.0,  # N/m (prevent bottoming)

    # Physical parameters
    'unsprung_mass': 17.0,         # kg per wheel
    'track_width': 1.50,           # m
    'wheelbase': 2.310,            # m
}
```

#### `compute_derived_params(config)`
Computes natural frequency and damping ratio:

```python
natural_frequency = sqrt(k/m) / (2π)  # Hz
damping_ratio = c / (2 × sqrt(k×m))    # dimensionless
```

#### `validate_config(config)`
Validates parameters and warns about over/under-damping.

## Physics Details

### Pacejka Magic Formula

The tire model uses separate parameters for lateral (cornering) and longitudinal (traction/braking) forces:

**Current Calibration** (2022 Mazda MX-5 Sport with 195/50R16 street tires):

| Parameter | Lateral | Longitudinal | Description |
|-----------|---------|--------------|-------------|
| B (Stiffness) | 8.5 | 12.0 | Initial slope of force curve |
| C (Shape) | 1.9 | 1.9 | Curve peakiness |
| D (Peak) | 0.95 | 1.35 | Peak friction multiplier |
| E (Curvature) | 0.97 | 0.97 | Falloff after peak |

**Expected Performance**:
- **Lateral**: 0.95g cornering (9.32 m/s²) - matches real MX-5 skidpad data
- **Braking**: 1.15g deceleration (11.28 m/s²) - matches 60-0 mph in 115 ft
- **Acceleration**: 0.57g (5.64 m/s²) on rear wheels only (RWD)

See `../TIRE_PARAMETERS.md` for detailed calibration and validation.

### Slip Calculations

**Slip Angle** (car_dynamics.py:707-713):
```python
# Angle between tire heading and velocity direction
wheel_vx = vx - yaw_rate × y_pos
wheel_vy = vy + yaw_rate × dist_cg
slip_angle = arctan2(wheel_vy, wheel_vx) - steering_angle
```

**Slip Ratio** (car_dynamics.py:715-727):
```python
# Difference between wheel speed and ground speed
wheel_linear_vel = omega × tire_radius
denom = max(|wheel_vx|, |wheel_linear_vel|, 0.1)
slip_ratio = (wheel_linear_vel - wheel_vx) / denom
# Clipped to [-1, 1]
# -1 = full lockup, 0 = perfect grip, +1 = full spin
```

### Coordinate Frames

**World Frame**: Fixed to track
- X-axis: Arbitrary world direction
- Y-axis: Perpendicular to X
- Origin: Track start

**Vehicle Frame**: Fixed to car
- X-axis: Forward (along car heading)
- Y-axis: Left (perpendicular to heading)
- Origin: Center of gravity

**Transformations** (car_dynamics.py:844-849):
```python
# Body to world:
vx_world = vx × cos(yaw) - vy × sin(yaw)
vy_world = vx × sin(yaw) + vy × cos(yaw)

# World to body (inverse):
vx = vx_world × cos(yaw) + vy_world × sin(yaw)
vy = -vx_world × sin(yaw) + vy_world × cos(yaw)
```

### Centripetal Force Handling

The velocity update accounts for rotating reference frame (car_dynamics.py:827-832):

```python
# Correct centripetal term
vx += (ax + vy × yaw_rate) × dt  # Forward
vy += (ay - vx × yaw_rate) × dt  # Lateral
```

Without the `- vx × yaw_rate` term, turning would incorrectly accelerate the car inward (anti-centrifugal force).

## Usage Examples

### Basic Environment Creation

```python
from env import CarRacing

# Vector mode (RECOMMENDED for training)
env = CarRacing(
    state_mode="vector",
    continuous=True,
    max_episode_steps=2500
)

# Visual mode (for watching)
env = CarRacing(
    render_mode="human",
    state_mode="visual"
)
```

### Custom Suspension Configuration

```python
from env import CarRacing
from env.suspension_config import get_suspension_config

# Get default config
config = get_suspension_config()

# Modify for stiffer suspension
config['spring_rate'] = 60000.0  # Stiffer springs
config['damping'] = 2400.0       # More damping

# Create environment with custom config
env = CarRacing(suspension_config=config)
```

### Multi-Car Racing

```python
# Create environment with 2 cars (ghost racing)
env = CarRacing(num_cars=2, state_mode="vector")

obs, info = env.reset()
# obs shape: (2, 71) - stacked observations for both cars
```

### Direct Car Control

```python
from env import Car

# Create standalone car
car = Car(
    world=None,
    init_angle=0.0,
    init_x=10.0,
    init_y=10.0
)

# Apply controls
car.gas(0.8)      # 80% throttle
car.steer(-0.3)   # Slight left
car.brake(0.0)    # No brake

# Step physics forward
dt = 1.0 / 50.0  # 50 FPS
results = car.step(dt)

# Access state
print(f"Position: ({car.x:.2f}, {car.y:.2f})")
print(f"Velocity: ({car.vx:.2f}, {car.vy:.2f}) m/s")
print(f"Heading: {car.yaw:.2f} rad")
```

### Accessing Tire Forces

```python
# After car.step(dt)
forces = car.last_tire_forces

for i in range(4):  # FL, FR, RL, RR
    wheel_name = ['FL', 'FR', 'RL', 'RR'][i]
    f = forces[i]
    print(f"{wheel_name}:")
    print(f"  Fx: {f['fx']:.1f} N (longitudinal)")
    print(f"  Fy: {f['fy']:.1f} N (lateral)")
    print(f"  Slip angle: {np.degrees(f['slip_angle']):.1f}°")
    print(f"  Slip ratio: {f['slip_ratio']:.3f}")
    print(f"  Normal force: {f['normal_force']:.1f} N")
```

## Performance Considerations

### Vector vs Visual Mode

**Vector Mode** (RECOMMENDED):
- **Speed**: 3-5× faster than visual mode
- **Training**: Efficient state representation (71D)
- **Rendering**: No rendering overhead
- **Information**: Full track geometry and lookahead

**Visual Mode**:
- **Speed**: Slower due to rendering
- **Use Case**: Watching trained agents, human play
- **Resolution**: 96×96 RGB (optimized for small networks)

### Computational Complexity

**Per Step**:
- Wheel dynamics: 4 wheels × O(1) = O(1)
- Tire forces: 4 wheels × O(1) Pacejka evaluation = O(1)
- Load transfer: O(1) rigid-body calculation
- Contact detection: ~61 tiles × 4 wheels × O(n) polygon checks ≈ O(n) where n = polygon vertices
- Total: O(n), very fast in practice (~2ms per step on modern CPU)

### Spatial Partitioning

The FrictionDetector uses **spatial partitioning** to reduce contact checks:
- Without: 300 tiles × 4 wheels = 1200 polygon checks
- With: 61 tiles × 4 wheels = 244 polygon checks (80% reduction)

## Known Limitations

1. **2D Only**: No jumps, elevation changes, or aerodynamic effects from ride height
2. **Simplified Suspension**: Load transfer uses filtered rigid-body model (not full spring-damper dynamics)
3. **No Tire Temperature**: Pacejka parameters are constant (no thermal degradation)
4. **No Fuel Consumption**: Car mass is constant throughout race
5. **Ghost Cars**: Multi-car mode has no collision between cars

## Debugging Tools

### Telemetry Logging

```python
# Enable telemetry in play_human_gui.py
python play_human_gui.py --log-telemetry --log-file my_session.csv
```

### Tire Parameter Visualization

```python
# Interactive tire force curve viewer
python magic_formula_visualizer.py
```

### Text-based Analysis

```python
# Analyze recorded telemetry
python analyze_telemetry.py telemetry_20250113_123456.csv
```

## References

1. **Pacejka, H. B. (2012)**. *Tire and Vehicle Dynamics*. 3rd Edition.
   - Standard reference for Magic Formula tire model

2. **Car and Driver** - Mazda MX-5 Miata Testing Data
   - Real-world performance validation (0.90g lateral)

3. **Motor Trend** - Mazda MX-5 Instrumented Testing
   - Braking distance validation (60-0 mph in 115 ft)

4. **Gymnasium Documentation**
   - Environment API and best practices

## Version History

- **2025-01-13**: Simplified to rigid-body load transfer model
- **2025-01-12**: Added suspension system with physical spring-dampers
- **2025-01-08**: Updated Pacejka parameters to match real MX-5
- **2024-12**: Initial custom 2D physics implementation

---

**Last Updated**: 2025-01-15
**Environment Version**: CarRacing-v3 (Custom Physics)
**Vehicle Model**: 2022 Mazda MX-5 Sport (ND)
