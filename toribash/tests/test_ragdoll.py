"""Tests for ragdoll creation and joint control."""

import sys
sys.path.insert(0, sys.path[0] + '/..')

import pymunk
from config.body_config import DEFAULT_BODY, JointState
from physics.ragdoll import Ragdoll


def test_ragdoll_creation():
    """Ragdoll creates all segments and joints."""
    space = pymunk.Space()
    space.gravity = (0, -900)
    rag = Ragdoll(space, position=(300, 50), facing=1)

    assert len(rag.segments) == len(DEFAULT_BODY.segments), \
        f"Expected {len(DEFAULT_BODY.segments)} segments, got {len(rag.segments)}"
    assert len(rag.joints) == len(DEFAULT_BODY.joints), \
        f"Expected {len(DEFAULT_BODY.joints)} joints, got {len(rag.joints)}"
    print(f"  Created {len(rag.segments)} segments, {len(rag.joints)} joints")


def test_joint_states():
    """Joint states change motor behavior correctly."""
    space = pymunk.Space()
    space.gravity = (0, -900)
    rag = Ragdoll(space, position=(300, 50), facing=1)

    # Default state should be HOLD
    for name, state in rag.joint_states.items():
        assert state == JointState.HOLD, f"Joint {name} should default to HOLD"

    # Set all to CONTRACT
    states = [JointState.CONTRACT] * DEFAULT_BODY.num_joints
    rag.set_all_joint_states(states)
    for name, state in rag.joint_states.items():
        assert state == JointState.CONTRACT

    # Set all to RELAX - motor force should be 0
    states = [JointState.RELAX] * DEFAULT_BODY.num_joints
    rag.set_all_joint_states(states)
    for name in rag.joints:
        _, _, motor = rag.joints[name]
        assert motor.max_force == 0.0, f"Relaxed joint {name} should have 0 force"

    print("  Joint state transitions work correctly")


def test_joint_angles():
    """Joint angles are readable and change under simulation."""
    space = pymunk.Space()
    space.gravity = (0, -900)
    space.iterations = 20
    rag = Ragdoll(space, position=(300, 50), facing=1)

    angles_before = rag.get_joint_angles()
    assert len(angles_before) == DEFAULT_BODY.num_joints

    # Set all joints to RELAX and simulate
    rag.set_all_joint_states([JointState.RELAX] * DEFAULT_BODY.num_joints)
    for _ in range(60):
        space.step(1/60)

    angles_after = rag.get_joint_angles()
    # With gravity and relaxed joints, angles should change
    changed = sum(1 for a, b in zip(angles_before, angles_after) if abs(a - b) > 0.01)
    assert changed > 0, "Some joint angles should change when relaxed under gravity"
    print(f"  {changed}/{DEFAULT_BODY.num_joints} joints moved under gravity when relaxed")


def test_segment_positions():
    """Segment positions are readable."""
    space = pymunk.Space()
    space.gravity = (0, -900)
    rag = Ragdoll(space, position=(300, 50), facing=1)

    positions = rag.get_segment_positions()
    assert len(positions) == len(DEFAULT_BODY.segments)

    # Head should be above chest
    head_y = rag.segments["head"][0].position.y
    chest_y = rag.segments["chest"][0].position.y
    assert head_y > chest_y, "Head should be above chest"
    print(f"  Head at y={head_y:.1f}, chest at y={chest_y:.1f}")


def test_dismemberment():
    """Dismembering a joint removes it from the space."""
    space = pymunk.Space()
    space.gravity = (0, -900)
    rag = Ragdoll(space, position=(300, 50), facing=1)

    constraint_count_before = len(space.constraints)
    rag.dismember_joint("elbow_l")
    constraint_count_after = len(space.constraints)

    assert "elbow_l" in rag.dismembered
    # Each joint adds 3 constraints (pivot, rotary limit, motor)
    assert constraint_count_after == constraint_count_before - 3
    print(f"  Dismembered elbow_l: {constraint_count_before} -> {constraint_count_after} constraints")


if __name__ == "__main__":
    tests = [
        test_ragdoll_creation,
        test_joint_states,
        test_joint_angles,
        test_segment_positions,
        test_dismemberment,
    ]
    for test in tests:
        print(f"Running {test.__name__}...")
        test()
        print(f"  PASSED")
    print(f"\nAll {len(tests)} tests passed!")
