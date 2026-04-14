# Toribash 2D

Turn-based ragdoll fighting game inspired by Toribash. Two fighters control joint states (contract, extend, hold, relax) then physics simulates each turn.

## Architecture

- `config/` — Dataclass configs: body definition (segments, joints), env settings, constants
- `physics/` — pymunk-based ragdoll simulation, collision tracking
- `game/` — Turn-based match orchestration, scoring
- `env/` — gymnasium.Env for RL training (MultiDiscrete action space, ~239D observation)
- `rendering/` — Pygame renderer, decoupled from game logic
- `scripts/` — Human play and random agent visualization

## Commands

```bash
# Play (human vs human, use TAB to switch players)
../.venv/bin/python scripts/play_human.py

# Watch random agents fight
../.venv/bin/python scripts/watch_random.py

# Run tests
../.venv/bin/python tests/test_ragdoll.py
../.venv/bin/python tests/test_physics.py
../.venv/bin/python tests/test_match.py
../.venv/bin/python tests/test_env.py
```

## Human Play Controls

- **Click** joint labels in bottom panel to cycle states (left=forward, right=backward)
- **1/2/3/4** — Set all joints to contract/extend/hold/relax
- **TAB** — Switch active player
- **SPACE** — Simulate turn (animated)
- **R** — Reset match
- **Q/ESC** — Quit

## Gym Environment

```python
from env.toribash_env import ToribashEnv
from config.env_config import EnvConfig

env = ToribashEnv(EnvConfig(max_turns=20, opponent_type="random"))
obs, info = env.reset()
action = env.action_space.sample()  # [4] * 14 joints
obs, reward, done, truncated, info = env.step(action)
```

## Dependencies

- pymunk (Chipmunk2D physics)
- pygame (rendering only, not needed for headless RL)
- gymnasium (env interface)
- numpy
