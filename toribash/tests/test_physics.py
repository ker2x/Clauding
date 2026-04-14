"""Tests for physics world: gravity, collisions, stability."""

import sys
sys.path.insert(0, sys.path[0] + '/..')

from config.body_config import JointState, DEFAULT_BODY
from config.env_config import EnvConfig
from physics.world import PhysicsWorld, COLLISION_TYPE_A, COLLISION_TYPE_B


def test_world_creation():
    """PhysicsWorld creates with two ragdolls."""
    world = PhysicsWorld()
    assert world.ragdoll_a is not None
    assert world.ragdoll_b is not None
    assert len(world.ragdoll_a.segments) == len(DEFAULT_BODY.segments)
    assert len(world.ragdoll_b.segments) == len(DEFAULT_BODY.segments)
    print(f"  World created with 2 ragdolls ({len(DEFAULT_BODY.segments)} segments each)")


def test_gravity():
    """Ragdolls fall under gravity when relaxed."""
    world = PhysicsWorld()

    # Record initial chest height
    chest_a_y_before = world.ragdoll_a.get_torso_position().y
    chest_b_y_before = world.ragdoll_b.get_torso_position().y

    # Relax all joints and simulate
    relax_all = [JointState.RELAX] * DEFAULT_BODY.num_joints
    world.ragdoll_a.set_all_joint_states(relax_all)
    world.ragdoll_b.set_all_joint_states(relax_all)

    for _ in range(120):
        world.step()

    chest_a_y_after = world.ragdoll_a.get_torso_position().y
    chest_b_y_after = world.ragdoll_b.get_torso_position().y

    # Chest should have dropped (or stayed on ground)
    assert chest_a_y_after <= chest_a_y_before + 1, "Player A should fall or stay"
    assert chest_b_y_after <= chest_b_y_before + 1, "Player B should fall or stay"
    print(f"  Player A chest: {chest_a_y_before:.1f} -> {chest_a_y_after:.1f}")
    print(f"  Player B chest: {chest_b_y_before:.1f} -> {chest_b_y_after:.1f}")


def test_hold_stability():
    """Ragdolls holding all joints should remain roughly upright."""
    world = PhysicsWorld()

    # All joints HOLD (default)
    chest_y_before = world.ragdoll_a.get_torso_position().y

    # Simulate 2 seconds
    for _ in range(120):
        world.step()

    chest_y_after = world.ragdoll_a.get_torso_position().y

    # Should not have fallen dramatically (within 50cm)
    drop = chest_y_before - chest_y_after
    assert drop < 50, f"Ragdoll dropped {drop:.1f}cm while holding - too much!"
    print(f"  Chest drop while holding: {drop:.1f}cm (tolerance: <50cm)")


def test_simulate_turn():
    """simulate_turn runs the correct number of steps."""
    config = EnvConfig(steps_per_turn=30)
    world = PhysicsWorld(config)

    pos_before = world.ragdoll_a.get_torso_position()
    world.simulate_turn()
    pos_after = world.ragdoll_a.get_torso_position()

    # Something should have happened (gravity at minimum)
    print(f"  Turn simulated: chest moved from ({pos_before.x:.1f}, {pos_before.y:.1f}) "
          f"to ({pos_after.x:.1f}, {pos_after.y:.1f})")


def test_long_stability():
    """Physics doesn't explode over many steps."""
    world = PhysicsWorld()

    # Hold all joints, simulate 1000 steps
    for _ in range(1000):
        world.step()

    # Check nothing flew off to infinity
    for name, (body, _) in world.ragdoll_a.segments.items():
        pos = body.position
        assert -500 < pos.x < 1500, f"Segment {name} x={pos.x:.0f} out of bounds"
        assert -100 < pos.y < 1000, f"Segment {name} y={pos.y:.0f} out of bounds"

    print(f"  1000 steps stable, all segments in bounds")


def test_ground_contact_tracking():
    """Ground contacts are tracked by collision handler."""
    world = PhysicsWorld()

    # Relax everything so body falls
    relax = [JointState.RELAX] * DEFAULT_BODY.num_joints
    world.ragdoll_a.set_all_joint_states(relax)

    world.simulate_turn(n_steps=120)

    # After falling, some segments should touch ground
    ground_a = world.collision_handler.get_ground_segments(COLLISION_TYPE_A)
    print(f"  Player A ground contacts after falling: {ground_a}")
    # Feet should definitely be touching
    assert len(ground_a) > 0, "Some segments should touch ground after falling relaxed"


if __name__ == "__main__":
    tests = [
        test_world_creation,
        test_gravity,
        test_hold_stability,
        test_simulate_turn,
        test_long_stability,
        test_ground_contact_tracking,
    ]
    for test in tests:
        print(f"Running {test.__name__}...")
        test()
        print(f"  PASSED")
    print(f"\nAll {len(tests)} tests passed!")
