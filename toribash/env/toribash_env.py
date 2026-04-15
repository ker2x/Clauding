"""Gymnasium environment for Toribash 2D RL training.

This module implements the ToribashEnv class, a gymnasium.Env compatible
environment for reinforcement learning. It provides:
- turn-based action semantics (one step = one game turn)
- Proper action/observation spaces for PPO training
- Opponent AI (hold/random/mirror) for single-agent training
- Rendering support (human view or RGB array)

Environment Semantics:
    Each env.step():
    1. Set agent's (player 0) joint states from action
    2. Generate opponent (player 1) action based on opponent_type
    3. Simulate one turn (30 physics steps)
    4. Return observation, reward, done status

Usage:
    >>> import gymnasium as gym
    >>> from env.toribash_env import ToribashEnv
    >>> env = gym.make("Toribash-v0")  # If registered
    >>> # Or directly:
    >>> env = ToribashEnv()
    >>> obs, info = env.reset(seed=42)
    >>> action = env.action_space.sample()
    >>> obs, reward, terminated, truncated, info = env.step(action)
"""

import numpy as np
import gymnasium as gym
from gymnasium import spaces

from config.body_config import DEFAULT_BODY, JointState
from config.env_config import EnvConfig
from game.match import Match
from game.scoring import compute_reward
from .observation import build_observation, compute_obs_dim


class ToribashEnv(gym.Env):
    """Turn-based ragdoll fighting environment for RL training.
    
    This environment implements the gymnasium.Env interface for training
    reinforcement learning agents on the Toribash game.
    
    Game Loop (per step):
        1. Agent sets joint states for player 0 (action)
        2. Opponent sets joint states for player 1 (based on config)
        3. Physics simulates for steps_per_turn frames
        4. Observation, reward, done returned
    
    Action Space:
        MultiDiscrete([4] * 14) - one of {CONTRACT, EXTEND, HOLD, RELAX} per joint
    
    Observation Space:
        Box(-2, 2, shape=(239,)) - flat float32 vector with normalized values
    
    Attributes:
        action_space: MultiDiscrete([4] * 14)
        observation_space: Box(-2, 2, shape=(obs_dim,))
        config: Environment configuration
        render_mode: Rendering mode ("human", "rgb_array", or None)
        match: Current Match instance (None until reset())
    
    Note:
        The environment is episodic: done=True when turn >= max_turns.
        The info dict contains turn number, scores, and winner at termination.
    """
    
    # Gymnasium metadata
    metadata = {"render_modes": ["human", "rgb_array"]}

    def __init__(self, config: EnvConfig | None = None, render_mode: str | None = None):
        """Initialize the Toribash environment.
        
        Args:
            config: Environment configuration (uses defaults if None).
            render_mode: Rendering mode:
                - None: No rendering
                - "human": Display pygame window
                - "rgb_array": Return RGB array from render()
        """
        super().__init__()
        self.config = config or EnvConfig()
        self.render_mode = render_mode
        self._renderer = None  # Lazy initialization for rendering

        # Get body dimensions for space definitions
        n_joints = self.config.body_config.num_joints
        n_segments = len(self.config.body_config.segments)

        # Action space: one discrete choice per joint
        # 0=CONTRACT, 1=EXTEND, 2=HOLD, 3=RELAX
        self.action_space = spaces.MultiDiscrete([4] * n_joints)

        # Observation space: flat normalized vector
        obs_dim = compute_obs_dim(n_joints, n_segments)
        self.observation_space = spaces.Box(
            low=-2.0, high=2.0, shape=(obs_dim,), dtype=np.float32
        )

        # Match state (initialized on reset)
        self.match: Match | None = None
        
        # Track previous actions for temporal memory in observations
        n = self.config.body_config.num_joints
        self._prev_actions: list[JointState] = [JointState.HOLD] * n
        self._prev_opp_actions: list[JointState] = [JointState.HOLD] * n

    def reset(self, seed=None, options=None) -> tuple[np.ndarray, dict]:
        """Reset the environment to initial state.
        
        Args:
            seed: Random seed for reproducibility.
            options: Additional options (currently unused).
        
        Returns:
            tuple: (initial_observation, info_dict)
                - observation: First observation after reset
                - info: Empty dict (gymnasium convention)
        """
        super().reset(seed=seed)
        
        # Create fresh match
        self.match = Match(self.config)
        
        # Reset action memory
        n = self.config.body_config.num_joints
        self._prev_actions = [JointState.HOLD] * n
        self._prev_opp_actions = [JointState.HOLD] * n
        
        # Build initial observation
        obs = build_observation(self.match, player=0, prev_actions=self._prev_actions)
        return obs, {}

    def step(self, action) -> tuple[np.ndarray, float, bool, bool, dict]:
        """Execute one environment step (one game turn).
        
        Args:
            action: Array-like of shape (14,) with values 0-3.
                Each value is the JointState for one joint:
                0=CONTRACT, 1=EXTEND, 2=HOLD, 3=RELAX
        
        Returns:
            tuple: (observation, reward, terminated, truncated, info)
                - observation: Current observation after the turn
                - reward: Scalar reward for player 0
                - terminated: True if episode is done (max turns reached)
                - truncated: False (no truncation implemented)
                - info: Dict with turn, scores, winner
        """
        assert self.match is not None, "Must call reset() before step()"

        # Save previous actions for observation (before overwrite)
        prev_for_obs = self._prev_actions

        # Convert action to JointState list
        joint_states = [JointState(int(a)) for a in action]
        
        # Set player 0 (agent) actions
        self.match.set_actions(0, joint_states)

        # Set player 1 (opponent) actions based on opponent_type
        opp_action = self._get_opponent_action()
        self.match.set_actions(1, opp_action)

        # Simulate the turn (runs physics and returns result)
        result = self.match.simulate_turn()

        # Build observation with true previous actions (for temporal memory)
        obs = build_observation(self.match, player=0, prev_actions=prev_for_obs)

        # Update action memory for next turn
        self._prev_actions = joint_states
        self._prev_opp_actions = opp_action

        # Compute reward for player 0
        done = self.match.is_done()
        won = done and self.match.get_winner() == 0
        reward = compute_reward(result, player=0, config=self.config, won=won)

        # Info dict with game state
        info = {
            "turn": self.match.turn,
            "scores": list(self.match.scores),
            "winner": self.match.get_winner() if done else None,
        }

        return obs, reward, done, False, info

    def render(self) -> np.ndarray | None:
        """Render the current state.
        
        Args:
            mode: Ignored (uses self.render_mode from __init__)
        
        Returns:
            - None if render_mode is None or "human"
            - RGB array (H, W, 3) if render_mode is "rgb_array"
        """
        if self.render_mode is None:
            return None

        # Lazy initialize renderer
        if self._renderer is None:
            from rendering.pygame_renderer import PygameRenderer
            self._renderer = PygameRenderer(self.match, mode=self.render_mode)

        return self._renderer.render(self.match)

    def close(self) -> None:
        """Clean up rendering resources."""
        if self._renderer is not None:
            self._renderer.close()
            self._renderer = None

    def _get_opponent_action(self) -> list[JointState]:
        """Generate opponent (player 1) actions based on config.
        
        Returns:
            List of JointState values for each joint.
        
        Opponent Types:
            "hold": All joints HOLD (easiest opponent)
            "random": Random joint states each turn
            "mirror": Copy the agent's last action
        """
        n = self.config.body_config.num_joints
        
        if self.config.opponent_type == "hold":
            return [JointState.HOLD] * n
        
        elif self.config.opponent_type == "random":
            # Use seeded random for reproducibility when seeded
            return [JointState(self.np_random.integers(0, 4)) for _ in range(n)]
        
        elif self.config.opponent_type == "mirror":
            # Copy agent's last action (already set on ragdoll_a)
            return [
                self.match.world.ragdoll_a.joint_states[jdef.name]
                for jdef in self.config.body_config.joints
            ]
        
        # Default: hold
        return [JointState.HOLD] * n
