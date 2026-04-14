"""Tests for gymnasium environment: spaces, step, reset, random rollout."""

import sys
sys.path.insert(0, sys.path[0] + '/..')

import numpy as np
from config.body_config import DEFAULT_BODY
from config.env_config import EnvConfig
from env.toribash_env import ToribashEnv


def test_env_creation():
    """Environment creates with correct spaces."""
    env = ToribashEnv()
    n_joints = DEFAULT_BODY.num_joints

    assert env.action_space.shape == (n_joints,)
    print(f"  Action space: MultiDiscrete({env.action_space.nvec.tolist()[:3]}...)")
    print(f"  Observation space: Box{env.observation_space.shape}")


def test_reset():
    """Reset returns valid observation."""
    env = ToribashEnv()
    obs, info = env.reset(seed=42)

    assert obs.shape == env.observation_space.shape
    assert obs.dtype == np.float32
    assert env.observation_space.contains(obs), f"Obs out of bounds: min={obs.min():.2f}, max={obs.max():.2f}"
    print(f"  Reset obs shape: {obs.shape}, range: [{obs.min():.3f}, {obs.max():.3f}]")


def test_step():
    """Step with valid action returns correct types."""
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
    """Run a complete episode with random actions."""
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

    assert steps == 10  # max_turns
    print(f"  Episode: {steps} steps, total_reward={total_reward:.3f}")
    print(f"  Final scores: {info['scores']}, winner: {info['winner']}")


def test_opponent_types():
    """Test different opponent types."""
    for opp_type in ["hold", "random"]:
        config = EnvConfig(max_turns=3, opponent_type=opp_type)
        env = ToribashEnv(config)
        obs, _ = env.reset(seed=42)

        for _ in range(3):
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)

        print(f"  Opponent '{opp_type}': completed 3 turns, score={info['scores']}")


def test_determinism():
    """Same seed + same actions = same trajectory."""
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


if __name__ == "__main__":
    tests = [
        test_env_creation,
        test_reset,
        test_step,
        test_full_rollout,
        test_opponent_types,
        test_determinism,
    ]
    for test in tests:
        print(f"Running {test.__name__}...")
        test()
        print(f"  PASSED")
    print(f"\nAll {len(tests)} tests passed!")
