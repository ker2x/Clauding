# Multi-Car Competitive Training Specification (v005)

## OBJECTIVE
Modify v005 CarRacing environment to support N cars racing simultaneously on the SAME track with evolutionary selection. Cars are ghost cars (no collision with each other, no awareness of each other). Selection of best performer is the fitness function.

## CORE REQUIREMENTS
1. **N cars race on SAME track** (generated once per episode)
2. **Cars do NOT collide** with each other (pass through like ghosts)
3. **Cars are NOT aware** of each other (state stays 67D, no other car positions)
4. **Selection is reward** - pick best car based on objective metrics (lap time, checkpoints, total reward)
5. **Same starting position** for all cars (fair competition)
6. **Parallel execution** - all cars step simultaneously per environment step

## CURRENT ARCHITECTURE (v005)

### File: `005/env/car_racing.py`
- **Class**: `CarRacing(gym.Env)`
- **Current state**: Single car instance at `self.car`
- **Observation space**: `Box(shape=(67,))` - vector state
- **Action space**: `Box(shape=(2,))` - [steering, acceleration]
- **Physics**: Custom 2D physics (no Box2D)
- **Key methods**:
  - `reset()`: Creates track, spawns single car, returns state
  - `step(action)`: Applies action to single car, returns (state, reward, terminated, truncated, info)
  - `_create_track()`: Generates random track (stored in `self.track` and `self.road`)
  - `_create_vector_state()`: Returns 67D numpy array
  - `_render_car()`: Renders single car

### File: `005/train.py`
- **Current flow**:
  ```
  env = make_carracing_env()  # Single environment
  for episode:
      state = env.reset()  # shape: (67,)
      while not done:
          action = agent.select_action(state)  # shape: (2,)
          next_state, reward, done = env.step(action)  # scalars
          replay_buffer.push(state, action, reward, next_state, done)
          agent.update(replay_buffer, batch_size)
  ```

### File: `005/env/car_dynamics.py`
- **Class**: `Car`
- **Independent instance** - can create multiple without interference
- **Key attributes**: `hull.position`, `hull.angle`, `wheels`, `vx`, `vy`
- **Key methods**: `step(dt)`, `steer()`, `gas()`, `brake()`

### File: `005/sac_agent.py`
- **Class**: `SACAgent`
- **Expects**: state shape (67,), action shape (2,)
- **No modification needed** - works with vectorized data

## REQUIRED CHANGES

### 1. MODIFY: `005/env/car_racing.py`

#### A. Change `__init__()` to accept `num_cars` parameter
```python
def __init__(
    self,
    render_mode: str | None = None,
    verbose: bool = False,
    num_cars: int = 1,  # NEW PARAMETER
    # ... existing parameters ...
):
    self.num_cars = num_cars
    self.cars = []  # List instead of single car
    # ... rest unchanged ...
```

#### B. Modify observation space to be vectorized
```python
# In __init__(), after line 446:
if self.state_mode == "vector":
    if self.num_cars > 1:
        # Multi-car: return stacked observations
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf,
            shape=(self.num_cars, 67),
            dtype=np.float32
        )
    else:
        # Single car (backward compatible)
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf,
            shape=(67,),
            dtype=np.float32
        )
```

#### C. Modify `reset()` to spawn N cars
```python
# Around line 755, replace:
# self.car = Car(self.world, init_yaw, init_x, init_y)

# With:
self.cars = []
for car_idx in range(self.num_cars):
    car = Car(self.world, init_yaw, init_x, init_y)
    car.car_id = car_idx  # Track which car this is
    # Assign different colors for rendering
    car.hull.color = self._get_car_color(car_idx)
    self.cars.append(car)

# Maintain backward compatibility
self.car = self.cars[0] if self.num_cars > 0 else None

# Initialize per-car tracking
self.car_rewards = [0.0] * self.num_cars
self.car_tile_visited_counts = [0] * self.num_cars
self.car_last_checkpoints = [-1] * self.num_cars
self.car_frames_since_progress = [0] * self.num_cars

# Return vectorized observations
if self.num_cars > 1:
    return self._get_all_observations(), {}
else:
    return self.step(None)[0], {}
```

#### D. Add helper method for car colors
```python
def _get_car_color(self, car_idx):
    """Return distinct color for each car (for rendering)."""
    colors = [
        (0.8, 0.0, 0.0),  # Red
        (0.0, 0.0, 0.8),  # Blue
        (0.0, 0.8, 0.0),  # Green
        (0.8, 0.8, 0.0),  # Yellow
        (0.8, 0.0, 0.8),  # Magenta
        (0.0, 0.8, 0.8),  # Cyan
        (0.8, 0.4, 0.0),  # Orange
        (0.4, 0.0, 0.8),  # Purple
    ]
    return colors[car_idx % len(colors)]
```

#### E. Add method to get all observations
```python
def _get_all_observations(self):
    """Get observations for all cars."""
    if self.num_cars == 1:
        return self._create_vector_state()
    else:
        observations = []
        for car in self.cars:
            # Temporarily set self.car for _create_vector_state()
            original_car = self.car
            self.car = car
            obs = self._create_vector_state()
            self.car = original_car
            observations.append(obs)
        return np.array(observations, dtype=np.float32)
```

#### F. Modify `step()` to handle N cars
```python
# Around line 761, modify step() signature and implementation:
def step(self, action: np.ndarray | int | None):
    """
    Step environment with N cars.

    Args:
        action:
            - Single car: shape (2,) or int
            - Multi car: shape (num_cars, 2) or array of ints

    Returns:
        observations: shape (num_cars, 67) if multi-car, else (67,)
        rewards: shape (num_cars,) if multi-car, else scalar
        terminated: shape (num_cars,) if multi-car, else bool
        truncated: shape (num_cars,) if multi-car, else bool
        infos: list of dicts if multi-car, else dict
    """

    if self.num_cars == 1:
        # Single car mode (backward compatible)
        return self._step_single_car(action)
    else:
        # Multi-car mode
        return self._step_multi_car(action)
```

#### G. Add `_step_single_car()` method
```python
def _step_single_car(self, action):
    """Original single-car step logic (lines 761-965 from current implementation)."""
    # Copy existing step() implementation here for single car
    # This maintains backward compatibility
    # ... (existing step code) ...
```

#### H. Add `_step_multi_car()` method
```python
def _step_multi_car(self, action):
    """
    Step all cars in parallel on the same track.

    Args:
        action: shape (num_cars, 2) - actions for all cars

    Returns:
        observations: (num_cars, 67)
        rewards: (num_cars,)
        terminated: (num_cars,)
        truncated: (num_cars,)
        infos: list[dict]
    """
    # Validate action shape
    if action is not None:
        if self.continuous:
            assert action.shape == (self.num_cars, 2), \
                f"Expected action shape ({self.num_cars}, 2), got {action.shape}"
        else:
            assert len(action) == self.num_cars

    step_rewards = np.zeros(self.num_cars, dtype=np.float32)
    terminated = np.zeros(self.num_cars, dtype=bool)
    truncated = np.zeros(self.num_cars, dtype=bool)
    infos = [{} for _ in range(self.num_cars)]

    # Step each car independently
    for car_idx, car in enumerate(self.cars):
        if action is not None:
            car_action = action[car_idx]

            # Apply action to this car
            if self.continuous:
                steer_action = -car_action[0]
                accel = np.clip(car_action[1], -1.0, 1.0)
                gas = accel if accel > 0 else 0.0
                brake = -accel if accel < 0 else 0.0
                car.steer(steer_action)
                car.gas(gas)
                car.brake(brake)
            else:
                # Discrete action logic
                steer_action = -0.6 * (car_action == 1) + 0.6 * (car_action == 2)
                gas = 0.2 * (car_action == 3)
                brake = 0.8 * (car_action == 4)
                car.steer(steer_action)
                car.gas(gas)
                car.brake(brake)

            # Step physics for this car
            car.step(1.0 / FPS)

            # Update contacts for this car ONLY (no car-car collision)
            self.friction_detector.update_contacts(car, self.road)

            # Calculate reward for this car
            # Note: friction_detector needs to track per-car state
            # We need to modify friction_detector to be car-aware

            # Apply per-car rewards/penalties (similar to single-car logic)
            self.car_rewards[car_idx] -= STEP_PENALTY

            # Forward velocity reward
            car_forward_x = np.cos(car.hull.angle)
            car_forward_y = np.sin(car.hull.angle)
            forward_velocity = (
                car.hull.linearVelocity[0] * car_forward_x +
                car.hull.linearVelocity[1] * car_forward_y
            )
            self.car_rewards[car_idx] += FORWARD_VEL_REWARD * max(0, forward_velocity)

            # Off-track penalty
            wheels_off = sum(1 for wheel in car.wheels if len(wheel.tiles) == 0)
            if wheels_off > OFFTRACK_THRESHOLD:
                self.car_rewards[car_idx] -= OFFTRACK_PENALTY * wheels_off

            # Check termination conditions for this car
            all_wheels_off = all(len(wheel.tiles) == 0 for wheel in car.wheels)
            if all_wheels_off:
                terminated[car_idx] = True
                infos[car_idx]["off_track"] = True

            # Check lap completion for this car
            # (Would need per-car tile_visited_count tracking)

            # Step reward for this car
            step_rewards[car_idx] = self.car_rewards[car_idx] - (
                self.car_prev_rewards[car_idx] if hasattr(self, 'car_prev_rewards') else 0.0
            )

    # Update time
    self.t += 1.0 / FPS
    self.episode_steps += 1

    # Check max episode steps (applies to all cars)
    if self.max_episode_steps is not None and self.episode_steps >= self.max_episode_steps:
        truncated[:] = True
        for info in infos:
            info['TimeLimit.truncated'] = True

    # Get observations for all cars
    observations = self._get_all_observations()

    # Render if needed
    if self.render_mode == "human":
        self.render()

    return observations, step_rewards, terminated, truncated, infos
```

#### I. Modify `FrictionDetector` to be car-aware
```python
# Around line 74-262, need to modify FrictionDetector to track per-car state
# Specifically: tile_visited_count, last_checkpoint_reached should be per-car

class FrictionDetector:
    def __init__(self, env, lap_complete_percent):
        self.env = env
        self.lap_complete_percent = lap_complete_percent

    def update_contacts(self, car, road_tiles):
        """
        Update contacts for ONE specific car.
        Track per-car progress using car.car_id.
        """
        # Get car-specific tracking indices
        car_idx = car.car_id if hasattr(car, 'car_id') else 0

        # ... existing collision detection code ...

        # When tile is visited, track per-car:
        if not tile.road_visited:
            tile.road_visited = True
            self.env.car_tile_visited_counts[car_idx] += 1

            # Checkpoint logic per car
            current_checkpoint = tile.idx // self.env.checkpoint_size
            expected_next_checkpoint = self.env.car_last_checkpoints[car_idx] + 1

            # ... existing checkpoint reward logic ...

            if current_checkpoint == expected_next_checkpoint and is_moving_forward:
                self.env.car_last_checkpoints[car_idx] = current_checkpoint
                self.env.car_rewards[car_idx] += self.env.checkpoint_reward
```

#### J. Modify rendering to show all cars
```python
# Modify _render_car() around line 1417:
def _render_car(self, zoom, translation, angle, draw_particles=True):
    """Render all cars (multi-car mode) or single car."""
    if self.num_cars == 1:
        self._render_single_car(self.car, zoom, translation, angle, draw_particles)
    else:
        for car in self.cars:
            self._render_single_car(car, zoom, translation, angle, draw_particles)

def _render_single_car(self, car, zoom, translation, angle, draw_particles=True):
    """Render a single car (extracted from original _render_car)."""
    # ... existing car rendering code (lines 1424-1525) ...
    # Replace self.car with car parameter
```

### 2. MODIFY: `005/preprocessing.py`

```python
# Modify make_carracing_env() to accept and pass num_cars parameter:
def make_carracing_env(
    stack_size=4,
    terminate_stationary=True,
    stationary_patience=50,
    render_mode=None,
    state_mode="vector",
    reward_shaping=True,
    min_episode_steps=100,
    short_episode_penalty=-50.0,
    max_episode_steps=1500,
    verbose=False,
    num_cars=1,  # NEW PARAMETER
):
    """
    Create CarRacing environment with preprocessing.

    Args:
        num_cars: Number of cars racing simultaneously (default: 1)
    """
    env = gym.make(
        'CarRacing-v3',
        render_mode=render_mode,
        lap_complete_percent=0.95,
        domain_randomize=False,
        continuous=True,
        terminate_stationary=terminate_stationary,
        stationary_patience=stationary_patience,
        state_mode=state_mode,
        max_episode_steps=max_episode_steps,
        reward_shaping=reward_shaping,
        min_episode_steps=min_episode_steps,
        short_episode_penalty=short_episode_penalty,
        verbose=verbose,
        num_cars=num_cars,  # PASS TO ENV
    )

    # Rest unchanged...
```

### 3. MODIFY: `005/train.py`

```python
# Add num_cars parameter to parse_args() around line 67:
parser.add_argument('--num-cars', type=int, default=1,
                    help='Number of cars racing simultaneously (default: 1)')

# Modify environment creation around line 434:
env = make_carracing_env(
    # ... existing parameters ...
    num_cars=args.num_cars,  # NEW
)

# Modify training loop around line 528:
for episode in range(start_episode, start_episode + args.episodes):
    state, _ = env.reset()

    # Handle vectorized state
    if args.num_cars > 1:
        # state shape: (num_cars, 67)
        episode_reward = np.zeros(args.num_cars)
    else:
        # state shape: (67,)
        episode_reward = 0

    done = False
    steps = 0

    while not done:
        # Select action(s)
        if args.num_cars > 1:
            # Get action for each car
            actions = np.array([
                agent.select_action(state[i], evaluate=False)
                for i in range(args.num_cars)
            ])  # shape: (num_cars, 2)
        else:
            action = agent.select_action(state, evaluate=False)
            actions = action

        # Step environment
        next_state, reward, terminated, truncated, info = env.step(actions)

        # Store experiences
        if args.num_cars > 1:
            # Store each car's experience independently
            for i in range(args.num_cars):
                if not (terminated[i] or truncated[i]):
                    replay_buffer.push(
                        state[i],
                        actions[i],
                        reward[i],
                        next_state[i],
                        float(terminated[i] or truncated[i])
                    )
                episode_reward[i] += reward[i]

            # Episode done when ALL cars are done
            done = all(terminated) or all(truncated)
        else:
            replay_buffer.push(state, actions, reward, next_state, float(terminated or truncated))
            episode_reward += reward
            done = terminated or truncated

        # Train agent
        if total_steps >= args.learning_starts and len(replay_buffer) >= args.batch_size:
            metrics = agent.update(replay_buffer, args.batch_size)
            # ... metrics tracking ...

        state = next_state
        total_steps += 1
        steps += 1

    # After episode: select best car if multi-car mode
    if args.num_cars > 1:
        best_car_idx = np.argmax(episode_reward)
        best_reward = episode_reward[best_car_idx]
        episode_rewards.append(best_reward)

        print(f"Episode {episode + 1}: Car rewards = {episode_reward}, Best = Car {best_car_idx} ({best_reward:.2f})")
    else:
        episode_rewards.append(episode_reward)
        print(f"Episode {episode + 1}: Reward = {episode_reward:.2f}")
```

### 4. NEW FILE: `005/train_competitive.py`

```python
"""
Competitive multi-car training with evolutionary selection.

Usage:
    python train_competitive.py --num-cars 8 --episodes 1000
"""

# Implement evolutionary training loop:
# 1. Initialize N cars with random policies
# 2. Race on same track
# 3. Select top K performers
# 4. Clone/mutate policies
# 5. Repeat

# This is an EXTENSION - implement after basic multi-car works
```

## IMPLEMENTATION ORDER

1. **Phase 1**: Modify `car_racing.py` to support `num_cars=1` (backward compatible)
   - Add `num_cars` parameter
   - Modify `__init__`, keep single-car logic intact
   - Test: `python test_setup.py` should pass

2. **Phase 2**: Implement multi-car reset and observation
   - Modify `reset()` to spawn N cars
   - Add `_get_all_observations()`
   - Test: Reset with `num_cars=4`, check observation shape is (4, 67)

3. **Phase 3**: Implement multi-car step
   - Add `_step_multi_car()`
   - Modify `FrictionDetector` for per-car tracking
   - Test: Step with 4 cars, check all get independent rewards

4. **Phase 4**: Implement multi-car rendering
   - Modify `_render_car()` to show all cars
   - Test: Visual check that all cars render correctly

5. **Phase 5**: Modify training loop
   - Update `train.py` to handle vectorized observations
   - Add selection logic
   - Test: Train with `--num-cars 4` for 100 episodes

6. **Phase 6**: Optimization and testing
   - Profile performance
   - Fix bugs
   - Add proper logging

## TESTING CHECKLIST

- [ ] Backward compatibility: `num_cars=1` works exactly as before
- [ ] Multi-car reset: Returns shape (num_cars, 67)
- [ ] Multi-car step: All cars move independently
- [ ] No car-car collision: Cars pass through each other
- [ ] Same track: All cars race on same track geometry
- [ ] Same start: All cars start at same position
- [ ] Per-car rewards: Each car gets independent reward tracking
- [ ] Per-car checkpoints: Checkpoint progress tracked per car
- [ ] Rendering: All cars visible with different colors
- [ ] Selection: Best car identified correctly by reward/lap time
- [ ] Training: Replay buffer accumulates experiences from all cars

## PERFORMANCE NOTES

- **Memory**: N cars = N× observations (67D × N) - should be fine for N ≤ 16
- **Computation**: Physics step is O(N) - still fast with custom engine
- **Rendering**: Drawing N cars is O(N) - negligible compared to physics
- **Training speed**: N× more data per step = faster learning (same wall-clock time)

## BACKWARD COMPATIBILITY

CRITICAL: When `num_cars=1`, environment must behave EXACTLY as current v005:
- Observation shape: (67,) not (1, 67)
- Reward: scalar not array
- Terminated/truncated: bool not array
- Info: dict not list

All existing code (watch_agent.py, test_*.py) must work without modification.
