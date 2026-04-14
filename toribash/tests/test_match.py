"""Tests for match flow: turns, scoring, termination."""

import sys
sys.path.insert(0, sys.path[0] + '/..')

from config.body_config import JointState, DEFAULT_BODY
from config.env_config import EnvConfig
from game.match import Match


def test_match_creation():
    """Match initializes with correct state."""
    match = Match()
    assert match.turn == 0
    assert match.scores == [0.0, 0.0]
    assert not match.is_done()
    print("  Match created, turn 0, not done")


def test_match_turn_flow():
    """Set actions and simulate a turn."""
    match = Match()
    n_joints = DEFAULT_BODY.num_joints

    # Player A: contract everything, Player B: hold everything
    match.set_actions(0, [JointState.CONTRACT] * n_joints)
    match.set_actions(1, [JointState.HOLD] * n_joints)

    result = match.simulate_turn()
    assert match.turn == 1
    print(f"  Turn 1 complete: damage A->B={result.damage_a_to_b:.2f}, B->A={result.damage_b_to_a:.2f}")
    print(f"  Ground contacts A: {result.ground_segments_a}")
    print(f"  Ground contacts B: {result.ground_segments_b}")


def test_full_match():
    """Run a complete match to termination."""
    config = EnvConfig(max_turns=10)
    match = Match(config)
    n_joints = DEFAULT_BODY.num_joints

    for turn in range(config.max_turns):
        # Alternate strategies
        if turn % 2 == 0:
            match.set_actions(0, [JointState.CONTRACT] * n_joints)
            match.set_actions(1, [JointState.EXTEND] * n_joints)
        else:
            match.set_actions(0, [JointState.EXTEND] * n_joints)
            match.set_actions(1, [JointState.CONTRACT] * n_joints)

        match.simulate_turn()

    assert match.is_done()
    winner = match.get_winner()
    print(f"  Match complete after {match.turn} turns")
    print(f"  Scores: A={match.scores[0]:.2f}, B={match.scores[1]:.2f}")
    print(f"  Winner: {'A' if winner == 0 else 'B' if winner == 1 else 'Draw'}")


def test_relaxed_vs_hold():
    """Relaxed fighter should collapse; held fighter's torso stays higher."""
    config = EnvConfig(max_turns=5)
    match = Match(config)
    n_joints = DEFAULT_BODY.num_joints

    for _ in range(config.max_turns):
        match.set_actions(0, [JointState.RELAX] * n_joints)
        match.set_actions(1, [JointState.HOLD] * n_joints)
        match.simulate_turn()

    # Held player's torso should be higher than relaxed player's torso
    chest_a_y = match.world.ragdoll_a.get_torso_position().y
    chest_b_y = match.world.ragdoll_b.get_torso_position().y
    print(f"  Relaxed A chest y={chest_a_y:.1f}, Held B chest y={chest_b_y:.1f}")
    assert chest_b_y > chest_a_y, "Held player's torso should be higher than relaxed"


if __name__ == "__main__":
    tests = [
        test_match_creation,
        test_match_turn_flow,
        test_full_match,
        test_relaxed_vs_hold,
    ]
    for test in tests:
        print(f"Running {test.__name__}...")
        test()
        print(f"  PASSED")
    print(f"\nAll {len(tests)} tests passed!")
