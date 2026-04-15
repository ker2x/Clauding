# Toribash 2D

Turn-based 2D ragdoll fighting game inspired by [Toribash](https://en.wikipedia.org/wiki/Toribash). Two fighters set joint states (contract, extend, hold, relax) then physics simulates each turn. Built for human play and RL training.

## Commands

```bash
cd toribash

# Train PPO agent vs hold opponent (100k timesteps)
../.venv/bin/python scripts/train.py --timesteps 100000 --opponent hold

# Train PPO agent with self-play (AI vs AI)
../.venv/bin/python scripts/train.py --opponent selfplay --timesteps 500000

# Watch trained agent
../.venv/bin/python scripts/train.py --watch toribash_ppo.zip --episodes 5

# Watch two AI models fight each other
../.venv/bin/python scripts/train.py --watch toribash_selfplay.zip --opponent selfplay --opponent-model toribash_selfplay_best.zip --episodes 5

# Run all tests
../.venv/bin/python tests/test_ragdoll.py
../.venv/bin/python tests/test_physics.py
../.venv/bin/python tests/test_match.py
../.venv/bin/python tests/test_env.py
```

## Architecture Overview

```
toribash/
├── config/          # Dataclass configs (no logic)
│   ├── body_config.py   # SegmentDef, JointDef, BodyConfig, JointState enum, DEFAULT_BODY
│   ├── constants.py     # Physics/game constants (gravity, timestep, arena dimensions)
│   └── env_config.py    # EnvConfig: reward weights, opponent type, turn settings
├── physics/         # pymunk simulation (no game logic, no rendering)
│   ├── ragdoll.py       # Ragdoll class: creates segments+joints, motor control
│   ├── world.py         # PhysicsWorld: space, ground, two ragdolls, step/simulate_turn
│   └── collision.py     # CollisionHandler: impulse tracking, ground contact tracking
├── game/            # Game rules (no physics internals, no rendering)
│   ├── match.py         # Match: turn orchestration, scoring accumulation, win detection
│   └── scoring.py       # TurnResult, compute_turn_result, compute_reward
├── env/             # gymnasium.Env interface (bridges game → RL)
│   ├── toribash_env.py  # ToribashEnv: step/reset/render, opponent action generation
│   └── observation.py   # build_observation: 239D normalized vector from match state
├── rendering/       # Pygame drawing (pure visualization, no game logic)
│   └── pygame_renderer.py  # PygameRenderer: ragdolls, ground, joint panel, UI
├── scripts/         # Entry points
│   └── train.py         # PPO training + watching script
└── tests/           # Standalone assert-based tests (no pytest)
    ├── test_ragdoll.py  # 5 tests: creation, joint states, angles, positions, dismemberment
    ├── test_physics.py  # 6 tests: world creation, gravity, stability, ground contacts
    ├── test_match.py    # 4 tests: creation, turn flow, full match, relaxed vs hold
    └── test_env.py      # 6 tests: spaces, reset, step, rollout, opponent types, determinism
```

### Dependency direction (strict)

```
config ← physics ← game ← env
                  ↖ rendering ← scripts
```

`config` depends on nothing. `physics` depends only on `config`. `game` depends on `physics` + `config`. `env` depends on `game`. `rendering` depends on `physics` + `game` + `config` (for drawing) but NOT on `env`. `scripts` tie everything together.

## Physics Engine Details

### pymunk 7.x API

We use **pymunk 7.2.0** which has a different API from older tutorials:
- Collision handlers: `space.on_collision(collision_type_a=, collision_type_b=, post_solve=)` — NOT the old `space.add_collision_handler()` which doesn't exist in v7
- Arbiter: `arbiter.total_impulse` returns a `Vec2d`, use `.length` for magnitude
- Constraints: `PivotJoint`, `RotaryLimitJoint`, `SimpleMotor` — unchanged from older versions

### Coordinate system

- **Units**: centimeters (a ragdoll is ~176cm tall)
- **Y-axis**: up is positive (pymunk default), ground at y=50
- **Arena**: 600cm wide × 400cm tall, fighters spawn at x=200 and x=400
- **Timestep**: 1/60s, 20 solver iterations per step, 30 steps per turn (~0.5s real time)

### Ragdoll structure

15 segments connected by 14 joints:

```
                  [head]
                     |  (neck)
                  [chest]
           (shoulder_l) | (shoulder_r)
      [upper_arm_l]  (spine)  [upper_arm_r]
           |        [stomach]       |
     (elbow_l)    (hip_l) (hip_r)  (elbow_r)
      [lower_arm_l] [upper_leg_l] [upper_leg_r] [lower_arm_r]
           |             |              |              |
     (wrist_l)     (knee_l)      (knee_r)       (wrist_r)
       [hand_l]  [lower_leg_l]  [lower_leg_r]    [hand_r]
                       |              |
                  (ankle_l)      (ankle_r)
                   [foot_l]       [foot_r]
```

Joint definitions are in `config/body_config.py` lines 74-145. Each joint has:
- `angle_min`/`angle_max` (radians) — asymmetric to model human anatomy
- `motor_rate` (rad/s) — how fast contract/extend drives the joint
- `motor_max_force` — torque limit (40K–240K depending on joint; high values needed for standing)

### Self-collision

Same-ragdoll segments collide with each other (except directly connected segments, which have `collide_bodies = False` on their PivotJoint). This prevents limbs from passing through the torso and folding into a ball. Only cross-ragdoll collisions trigger damage tracking.

### Mirroring (facing=-1)

Player B faces left (facing=-1). The ragdoll mirrors by:
1. **Anchor x-coordinates negated** in `ragdoll.py:_create_joints` (lines 110-114)
2. **Angle limits negated and swapped**: `angle_min = -jdef.angle_max, angle_max = -jdef.angle_min` (lines 121-126)
3. **Motor rate sign flipped** in `set_joint_state` (line 151): `rate_sign = 1 if facing == 1 else -1`
4. **Segment positions mirrored** via `hip_offset = 8 * f, shoulder_offset = 12 * f` (lines 66-67)

This means CONTRACT on player A's left shoulder moves it in the same anatomical direction as CONTRACT on player B's left shoulder, even though the physics angles are opposite.

### Joint motor states

Defined as `JointState(IntEnum)` with values 0-3:

| State | motor.rate | motor.max_force | Behavior |
|-------|-----------|-----------------|----------|
| CONTRACT (0) | +motor_rate × facing | motor_max_force | Actively closes joint toward angle_min |
| EXTEND (1) | -motor_rate × facing | motor_max_force | Actively opens joint toward angle_max |
| HOLD (2) | 0.0 | motor_max_force | Resists movement (stiff) |
| RELAX (3) | 0.0 | 0.0 | No resistance (limp, gravity takes over) |

### Collision tracking

`CollisionHandler` uses pymunk's `post_solve` callbacks (fires every physics step during contact):
- **Fighter vs fighter**: records `(impulse_magnitude, seg_a_name, seg_b_name, vel_a, vel_b)` into `turn_impulses`. Velocities are used for directional damage attribution — the faster-moving body is the "striker" and deals proportionally more damage.
- **Fighter vs ground**: records `(collision_type, segment_name)` into `ground_contacts`
- Both are cleared at the start of each `simulate_turn()` via `clear_turn()`

Collision types: ground=0, player_a=1, player_b=2 (defined in `physics/world.py`)

## Gym Environment

### Spaces

- **Action**: `MultiDiscrete([4] * 14)` — one of {CONTRACT, EXTEND, HOLD, RELAX} per joint
- **Observation**: `Box(-2, 2, shape=(253,))` — flat float32 vector

### Observation layout (253 floats)

The observation is ego-centric (own data first, positions relative to own torso):

```
Per ragdoll (×2 = own + opponent):
  [0:14]   Joint angles (normalized to [-1,1] by angle limits)
  [14:28]  Joint angular velocities (normalized by 2× motor rate)
  [28:42]  Joint states (0.0, 0.33, 0.67, 1.0 for the 4 states)
  [42:72]  Segment positions relative to ref torso (x,y × 15 segments, /ARENA_WIDTH,HEIGHT)
  [72:102] Segment velocities (vx,vy × 15 segments, /500)
  [102:117] Segment rotations (sin of absolute angle)

Global:
  [234:236] Relative torso dx, dy (2 values)
  [236] Turn progress (turn / max_turns, 0→1)
  [237] Own score / 100
  [238] Opponent score / 100
  [239:253] Previous actions (14 floats, normalized joint states from last turn)
```

Built by `env/observation.py:build_observation()`. For player 1, own/opponent are swapped so a single policy can play both sides.

### Step semantics

Each `env.step(action)`:
1. Sets agent's (player 0) joint states from action array
2. Generates opponent (player 1) action based on `config.opponent_type`
3. Calls `match.simulate_turn()` → runs 30 physics steps, tracks collisions
4. Computes reward from turn result
5. Returns `(obs, reward, terminated, truncated=False, info)`

Episode terminates when `turn >= max_turns`. Info dict contains `turn`, `scores`, `winner`.

### Opponent types

Set via `EnvConfig.opponent_type`:
- `"hold"` — all joints HOLD (default, easiest)
- `"random"` — random joint states each turn
- `"mirror"` — copies agent's action

### Reward function

Computed per turn in `game/scoring.py:compute_reward()`:

| Component | Weight | Source |
|-----------|--------|--------|
| Damage dealt | +1.0 | Velocity-weighted impulses above threshold from fighter-fighter collisions |
| Damage taken | -0.5 | Same impulses, attributed proportionally by body velocity |
| Own non-exempt segment on ground | -0.2 per segment | Ground contact tracking (feet and hands exempt) |
| Opponent non-exempt on ground | +0.1 per segment | Ground contact tracking |
| Opponent dismemberment | +5.0 per joint | (not yet triggered — see Known Issues) |
| KO | +10.0 | (not implemented yet) |
| Win | +20.0 | Higher score at match end |

All weights are configurable via `EnvConfig` dataclass fields.

### Performance

~950 env steps/sec headless (28K physics frames/sec) on M4 Mac Mini. Each episode is 20 steps (turns), so ~47 episodes/sec.

## RL Training

### PPO Training

```bash
# Basic training vs "hold" opponent
../.venv/bin/python scripts/train.py --timesteps 100000 --opponent hold

# Training vs random opponent
../.venv/bin/python scripts/train.py --timesteps 500000 --opponent random

# Watch trained agent
../.venv/bin/python scripts/train.py --watch toribash_ppo.zip --episodes 5
```

### Training Options

- `--timesteps`: Total training steps (default: 100,000)
- `--opponent`: Opponent type - "hold", "random", "mirror", or "selfplay"
- `--turns`: Max turns per episode (default: 20)
- `--save`: Model save path (default: toribash_ppo or toribash_selfplay)
- `--eval-freq`: Evaluation frequency (default: 10,000)
- `--update-opponent`: How often to update opponent in selfplay (default: 10,000)

### Self-play Training

Self-play trains the agent against copies of itself:
```bash
../.venv/bin/python scripts/train.py --opponent selfplay --timesteps 500000
```

- Every `--update-opponent` steps, the opponent is updated to a copy of the current agent
- This creates an "arms race" where both sides improve together
- Auto-resumes from saved model if `toribash_selfplay.zip` exists
- Best model saved to `toribash_selfplay_best.zip`

### TensorBoard

Monitor training in real-time:
```bash
../.venv/bin/tensorboard --logdir=toribash --port 6006
```

Then open http://localhost:6006 in your browser.

Key metrics to watch:
- `rollout/ep_rew_mean` - episode rewards (should increase)
- `train/loss` - training loss (should be stable, < 100)
- `train/value_loss` - value function loss (should stay small)
- `train/explained_variance` - how well value function predicts returns (closer to 1 is better)

### Training Stability

The environment uses `VecNormalize` for stable training:
- Rewards are normalized to prevent value function explosion
- Observations are normalized for consistent learning
- Training loss should stay below 100 (was 2M+ before normalization)

If training is unstable:
- Lower learning rate (try 1e-4 instead of 3e-4)
- Reduce clip_range (try 0.1 instead of 0.2)
- Increase n_steps for more stable updates

### Curriculum Suggestions

1. **Phase 1**: vs `"hold"` opponent, reduced spawn distance — learn to stay upright and approach
2. **Phase 2**: vs `"random"` — learn to exploit unstable opponents
3. **Phase 3**: self-play — agent plays both sides alternately (auto-implemented)

### Observation Improvements for RL

- Add **previous turn's joint states** (gives the agent memory of what it did last)
- Add **distance to opponent per segment** (helps with targeting)
- Add **contact flags** (binary: is each segment touching ground this turn?)
- Consider **frame stacking** (2-3 turns of obs concatenated, like 006/ does)

## Known Issues and Incomplete Features

### 1. Fighters rarely deal damage (scores stay 0.0)

The default spawn distance (100cm apart) may still be too far for effective engagement. Consider reducing SPAWN_OFFSET_X further (e.g., 30-40) or adding an approach reward to incentivize closing distance.

### 2. Body parts clip into ground or opponent

Segments sometimes get stuck inside the ground plane or embedded in opponent's body. Likely needs higher solver iterations, collision margin tuning, or position correction.

## Dependencies

- **pymunk 7.2.0** — Chipmunk2D physics (collision, rigid bodies, joints, motors)
- **pygame 2.6.1** — rendering only, not needed for headless RL
- **gymnasium 1.2.3** — env interface (spaces, step/reset API)
- **stable-baselines3 2.8.0** — PPO implementation
- **numpy** — observation vectors

All in the shared venv at `../.venv/`.
