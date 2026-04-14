"""Gymnasium environment for Toribash 2D."""

import numpy as np
import gymnasium as gym
from gymnasium import spaces

from config.body_config import DEFAULT_BODY, JointState
from config.env_config import EnvConfig
from game.match import Match
from game.scoring import compute_reward
from .observation import build_observation, compute_obs_dim


class ToribashEnv(gym.Env):
    """Turn-based ragdoll fighting environment.

    Each step = one Toribash turn:
    1. Agent sets joint states for player 0
    2. Opponent sets joint states for player 1 (based on opponent_type)
    3. Physics simulates for steps_per_turn frames
    4. Observation, reward, done returned
    """

    metadata = {"render_modes": ["human", "rgb_array"]}

    def __init__(self, config: EnvConfig | None = None, render_mode: str | None = None):
        super().__init__()
        self.config = config or EnvConfig()
        self.render_mode = render_mode
        self._renderer = None

        n_joints = self.config.body_config.num_joints
        n_segments = len(self.config.body_config.segments)

        # Action: one discrete choice per joint (CONTRACT=0, EXTEND=1, HOLD=2, RELAX=3)
        self.action_space = spaces.MultiDiscrete([4] * n_joints)

        # Observation: flat vector
        obs_dim = compute_obs_dim(n_joints, n_segments)
        self.observation_space = spaces.Box(
            low=-2.0, high=2.0, shape=(obs_dim,), dtype=np.float32
        )

        self.match: Match | None = None

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.match = Match(self.config)
        obs = build_observation(self.match, player=0)
        return obs, {}

    def step(self, action):
        assert self.match is not None, "Must call reset() before step()"

        # Player 0 (agent)
        joint_states = [JointState(int(a)) for a in action]
        self.match.set_actions(0, joint_states)

        # Player 1 (opponent)
        opp_action = self._get_opponent_action()
        self.match.set_actions(1, opp_action)

        # Simulate turn
        result = self.match.simulate_turn()

        # Build observation
        obs = build_observation(self.match, player=0)

        # Compute reward
        done = self.match.is_done()
        won = done and self.match.get_winner() == 0
        reward = compute_reward(result, player=0, config=self.config, won=won)

        return obs, reward, done, False, {
            "turn": self.match.turn,
            "scores": list(self.match.scores),
            "winner": self.match.get_winner() if done else None,
        }

    def render(self):
        if self.render_mode is None:
            return None

        if self._renderer is None:
            from rendering.pygame_renderer import PygameRenderer
            self._renderer = PygameRenderer(self.match, mode=self.render_mode)

        return self._renderer.render(self.match)

    def close(self):
        if self._renderer is not None:
            self._renderer.close()
            self._renderer = None

    def _get_opponent_action(self) -> list[JointState]:
        """Generate opponent actions based on config."""
        n = self.config.body_config.num_joints
        if self.config.opponent_type == "hold":
            return [JointState.HOLD] * n
        elif self.config.opponent_type == "random":
            return [JointState(self.np_random.integers(0, 4)) for _ in range(n)]
        elif self.config.opponent_type == "mirror":
            # Copy agent's last action (already set on ragdoll_a)
            return [
                self.match.world.ragdoll_a.joint_states[jdef.name]
                for jdef in self.config.body_config.joints
            ]
        return [JointState.HOLD] * n
