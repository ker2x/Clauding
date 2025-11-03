# CarRacing-v3 Environment

This directory contains a local copy of the CarRacing-v3 environment from Gymnasium, allowing for customization and experimentation with the environment's behavior.

## Overview

**CarRacing-v3** is a top-down racing game where an agent learns to drive a car around a randomly generated track. It's one of the classic reinforcement learning environments for learning control from pixels.

## Files

### `car_racing.py`
The main environment implementation using Box2D physics engine and pygame for rendering.

**Key Components:**
- **Track Generation**: Randomly generates a race track on each reset
- **Observation**: 96x96 RGB top-down view of the car and track
- **Reward System**:
  - +1000/N points for visiting each track tile (N = total tiles)
  - -0.1 penalty per frame
  - -100 penalty for going completely off-track (all 4 wheels off track tiles)
- **Physics Simulation**: Uses Box2D for realistic car physics
- **Rendering**: pygame-based visualization with speed indicators, ABS sensors, steering wheel, and gyroscope

**Important Parameters:**
- `continuous` (bool): Continuous (3D) vs discrete (5 actions) action space
- `lap_complete_percent` (float): % of track tiles required to complete lap (default: 0.95)
- `domain_randomize` (bool): Randomize track/background colors for generalization
- `render_mode` (str): "human", "rgb_array", or "state_pixels"

**Action Space:**
- **Continuous**: [steering (-1 to +1), gas (0 to 1), brake (0 to 1)]
- **Discrete**: 5 actions (do nothing, steer right, steer left, gas, brake)

### `car_dynamics.py`
Implements the car physics simulation using Box2D.

**Key Features:**
- 4-wheel physics model with individual wheel dynamics
- Realistic friction and tire slip simulation
- Engine power and fuel consumption tracking
- Wheel rotation and steering mechanics

**Car Parameters:**
- `ENGINE_POWER`: Acceleration force
- `FRICTION_LIMIT`: Maximum tire grip
- `WHEEL_MOMENT_OF_INERTIA`: Wheel rotation resistance
- Hull and wheel shapes defined as polygons

### `__init__.py`
Package initialization that exports `CarRacing` and `Car` classes using relative imports.

## How It Works

1. **Reset**: Generates a random track using parametric curves with smooth transitions
2. **Track Tiles**: Track is divided into tiles; visiting new tiles gives rewards
3. **Car Spawn**: Car starts at rest at the beginning of the track
4. **Physics Step**: Box2D simulates car dynamics at 50 FPS
5. **Rendering**: Camera follows car with zoom, displays telemetry at bottom
6. **Off-Track Detection**: Each wheel tracks which track tiles it's touching via Box2D collision detection
7. **Termination**: Episode ends when:
   - Lap is complete (visited required % of track tiles)
   - All 4 wheels go off-track (not touching any track tiles) - **NEW**

## Modifications from Original

This is a local copy with the following changes:

### 1. Relative Imports
- Uses `from .car_dynamics import Car` instead of `from gymnasium.envs.box2d.car_dynamics`
- Allows direct modification without affecting the installed gymnasium package

### 2. Stricter Off-Track Termination (NEW)
- **Original behavior**: Car could drive on grass indefinitely, only terminated when going 333+ units from origin (2x track radius away)
- **New behavior**: Episode immediately terminates when all 4 wheels leave track tiles (gives -100 reward)
- **Implementation**: Added check `all(len(wheel.tiles) == 0 for wheel in self.car.wheels)` in `step()` method (car_racing.py:580-586)
- **Removed**: Out-of-bounds check (PLAYFIELD boundary) is no longer needed since off-track always triggers first
- **Rationale**: Prevents agent from learning to drive off-track and makes training more efficient

## TODO: Potential Improvements

### Reward Shaping
- [ ] Add progress-based rewards (distance along track centerline)
- [ ] Penalize excessive steering oscillation
- [ ] Reward smooth driving (minimize jerk)
- [ ] Add checkpoint-based intermediate rewards
- [ ] Implement speed-dependent rewards (encourage faster driving on straights)

### Track Generation
- [ ] Add difficulty levels (more/fewer curves, tighter turns)
- [ ] Implement fixed track seeds for reproducible testing
- [ ] Add track obstacles (oil slicks, barriers)
- [ ] Generate tracks with elevation changes
- [ ] Add track width variations

### Physics
- [ ] Tune car physics parameters for more realistic handling
- [ ] Add weather conditions (rain = reduced friction)
- [ ] Implement damage system (collisions affect performance)
- [ ] Add tire wear over time
- [ ] Model drift mechanics more accurately

### Observations
- [ ] Add first-person camera view option
- [ ] Include velocity vector in observation
- [ ] Add ray-casting sensors (LiDAR-like track distance measurements)
- [ ] Provide track curvature information ahead
- [ ] Add option for grayscale observations

### Training Improvements
- [ ] Add curriculum learning support (start with easier tracks)
- [ ] Implement demonstration recording/replay
- [ ] Add visual indicators for off-track detection
- [ ] Create checkpoints along track for partial episode rollouts
- [ ] Add option to respawn at last checkpoint instead of episode end

### Performance
- [ ] Optimize rendering for faster training
- [ ] Add batch environment support
- [ ] Cache track generation for repeated evaluation
- [ ] Reduce Box2D simulation overhead

### Debugging/Visualization
- [ ] Add heatmap of visited track tiles
- [ ] Visualize reward components separately
- [ ] Show car trajectory overlay
- [ ] Display friction/grip levels visually
- [ ] Add telemetry graphs for speed, steering, acceleration

### Alternative Tasks
- [ ] Time trial mode (complete lap as fast as possible)
- [ ] Multi-car racing (competitive RL)
- [ ] Drift challenge (maximize controlled oversteer)
- [ ] Efficiency challenge (maximize distance per fuel unit)

## Usage Example

```python
from env.car_racing import CarRacing

# Create environment
env = CarRacing(
    render_mode='rgb_array',
    continuous=True,
    lap_complete_percent=0.95,
    domain_randomize=False
)

# Reset and run
obs, info = env.reset()
for _ in range(1000):
    action = env.action_space.sample()  # Replace with agent
    obs, reward, terminated, truncated, info = env.step(action)
    if terminated or truncated:
        break
```

## Credits
- Original environment by Oleg Klimov
- Car dynamics tutorial by Chris Campbell (http://www.iforce2d.net/b2dtut/top-down-car)
- Packaged in Gymnasium by Andrea PIERRÉ
