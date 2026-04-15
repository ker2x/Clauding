"""Tests for gymnasium environment: spaces, step, reset, rollout.

This module contains unit tests for the ToribashEnv class, verifying:
- Environment creation with correct spaces
- Reset returns valid observations
- Step with valid actions returns correct types
- Full episode rollout
- Different opponent types
- Determinism (same seed = same trajectory)
- Previous actions in observation

Run with: python tests/test_env.py
"""

import sys
sys.path.insert(0, sys.path[0] + '/..')

import numpy as np
from config.body_config import DEFAULT_BODY, JointState
from config.env_config import EnvConfig
from env.toribash_env import ToribashEnv
from env.observation import compute_obs_dim


def test_env_creation():
    """Test environment creates with correct action/observation spaces."""
    env = ToribashEnv()
    n_joints = DEFAULT_BODY.num_joints

    assert env.action_space.shape == (n_joints,)
    print(f"  Action space: MultiDiscrete({env.action_space.nvec.tolist()[:3]}...)")
    print(f"  Observation space: Box{env.observation_space.shape}")


def test_reset():
    """Test reset returns valid observation."""
    env = ToribashEnv()
    obs, info = env.reset(seed=42)

    assert obs.shape == env.observation_space.shape
    assert obs.dtype == np.float32
    assert env.observation_space.contains(obs), f"Obs out of bounds: min={obs.min():.2f}, max={obs.max():.2f}"
    print(f"  Reset obs shape: {obs.shape}, range: [{obs.min():.3f}, {obs.max():.3f}]")


def test_step():
    """Test step with valid action returns correct types."""
    env = ToribashEnv()
    obs, _ = env.reset(seed=42)

    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)

    assert obs.shape == env.observation_space.shape
    assert isinstance(reward, float)
    assert isinstance(terminated, bool)
    assert isinstance(truncated, bool)
    assert "turn" in info
    assert "scores" in info
    print(f"  Step: reward={reward:.3f}, turn={info['turn']}, scores={info['scores']}")


def test_full_rollout():
    """Test running a complete episode with random actions."""
    env = ToribashEnv(EnvConfig(max_turns=10))
    obs, _ = env.reset(seed=42)

    total_reward = 0.0
    steps = 0
    while True:
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        steps += 1
        if terminated or truncated:
            break

    assert steps <= 10  # max_turns or KO
    print(f"  Episode: {steps} steps, total_reward={total_reward:.3f}")
    print(f"  Final scores: {info['scores']}, winner: {info['winner']}")


def test_opponent_types():
    """Test different opponent types work correctly."""
    for opp_type in ["hold", "random"]:
        config = EnvConfig(max_turns=3, opponent_type=opp_type)
        env = ToribashEnv(config)
        obs, _ = env.reset(seed=42)

        for _ in range(3):
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)

        print(f"  Opponent '{opp_type}': completed 3 turns, score={info['scores']}")


def test_determinism():
    """Test same seed + same actions = same trajectory."""
    config = EnvConfig(max_turns=5)

    results = []
    for _ in range(2):
        env = ToribashEnv(config)
        obs, _ = env.reset(seed=123)
        trajectory = [obs.copy()]
        rng = np.random.default_rng(456)

        for _ in range(5):
            action = rng.integers(0, 4, size=config.body_config.num_joints)
            obs, reward, _, _, _ = env.step(action)
            trajectory.append(obs.copy())

        results.append(trajectory)

    for i, (a, b) in enumerate(zip(*results)):
        assert np.allclose(a, b, atol=1e-5), f"Step {i} differs!"

    print("  Deterministic: two identical runs match perfectly")


def test_prev_actions():
    """Test previous actions in observation lag by one turn."""
    config = EnvConfig(max_turns=5)
    env = ToribashEnv(config)
    n_joints = config.body_config.num_joints

    obs, _ = env.reset(seed=42)
    # Obs dim layout: 2 * per_ragdoll + global (which ends with 14 prev_actions)
    obs_dim = len(obs)
    prev_act_slice = slice(obs_dim - n_joints, obs_dim)

    # After reset, previous actions should be all HOLD (2/3 = 0.667)
    prev_acts_0 = obs[prev_act_slice]
    expected_hold = float(JointState.HOLD.value) / 3.0
    assert np.allclose(prev_acts_0, expected_hold), \
        f"After reset, prev_actions should be HOLD ({expected_hold}), got {prev_acts_0[:3]}"

    # Step with all CONTRACT (value 0)
    action_1 = np.zeros(n_joints, dtype=int)  # all CONTRACT=0
    obs_1, _, _, _, _ = env.step(action_1)
    # Prev actions should STILL be HOLD (from before step 1)
    prev_acts_1 = obs_1[prev_act_slice]
    assert np.allclose(prev_acts_1, expected_hold), \
        f"After step 1, prev_actions should be HOLD, got {prev_acts_1[:3]}"

    # Step with all EXTEND (value 1)
    action_2 = np.ones(n_joints, dtype=int)  # all EXTEND=1
    obs_2, _, _, _, _ = env.step(action_2)
    # Prev actions should now be CONTRACT (0/3 = 0.0) from step 1
    prev_acts_2 = obs_2[prev_act_slice]
    expected_contract = float(JointState.CONTRACT.value) / 3.0
    assert np.allclose(prev_acts_2, expected_contract), \
        f"After step 2, prev_actions should be CONTRACT ({expected_contract}), got {prev_acts_2[:3]}"

    print(f"  Previous actions correctly lag by one turn")


if __name__ == "__main__":
    tests = [
        test_env_creation,
        test_reset,
        test_step,
        test_full_rollout,
        test_opponent_types,
        test_determinism,
        test_prev_actions,
    ]
    for test in tests:
        print(f"Running {test.__name__}...")
        test()
        print(f"  PASSED")
    print(f"\nAll {len(tests)} tests passed!")
