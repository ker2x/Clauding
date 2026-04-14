"""Ragdoll body definition: segments, joints, dimensions."""

from dataclasses import dataclass, field
from enum import IntEnum


class JointState(IntEnum):
    CONTRACT = 0
    EXTEND = 1
    HOLD = 2
    RELAX = 3


@dataclass
class SegmentDef:
    name: str
    width: float   # cm
    height: float  # cm
    mass: float    # kg
    color: tuple[int, int, int] = (200, 200, 200)


@dataclass
class JointDef:
    name: str
    parent: str  # parent segment name
    child: str   # child segment name
    anchor_parent: tuple[float, float]  # relative to parent center
    anchor_child: tuple[float, float]   # relative to child center
    angle_min: float = -1.5  # radians
    angle_max: float = 1.5   # radians
    motor_rate: float = 15.0  # rad/s
    motor_max_force: float = 50000.0


@dataclass
class BodyConfig:
    segments: list[SegmentDef] = field(default_factory=list)
    joints: list[JointDef] = field(default_factory=list)

    @property
    def num_joints(self) -> int:
        return len(self.joints)


def _make_default_body() -> BodyConfig:
    """Create the default humanoid ragdoll body."""
    # Segment dimensions (width x height in cm)
    # The ragdoll is roughly 170cm tall
    segments = [
        SegmentDef("head",          16, 16,  5.0, (255, 220, 180)),
        SegmentDef("chest",         24, 28, 20.0, (100, 140, 200)),
        SegmentDef("stomach",       22, 20, 12.0, (100, 140, 200)),
        # Left arm
        SegmentDef("upper_arm_l",   8,  26,  4.0, (200, 160, 140)),
        SegmentDef("lower_arm_l",   7,  24,  3.0, (200, 160, 140)),
        SegmentDef("hand_l",        6,  10,  1.0, (255, 220, 180)),
        # Right arm
        SegmentDef("upper_arm_r",   8,  26,  4.0, (200, 160, 140)),
        SegmentDef("lower_arm_r",   7,  24,  3.0, (200, 160, 140)),
        SegmentDef("hand_r",        6,  10,  1.0, (255, 220, 180)),
        # Left leg
        SegmentDef("upper_leg_l",  10,  34,  8.0, (140, 100, 80)),
        SegmentDef("lower_leg_l",   9,  32,  5.0, (140, 100, 80)),
        SegmentDef("foot_l",       14,   6,  2.0, (80,  80,  80)),
        # Right leg
        SegmentDef("upper_leg_r",  10,  34,  8.0, (140, 100, 80)),
        SegmentDef("lower_leg_r",   9,  32,  5.0, (140, 100, 80)),
        SegmentDef("foot_r",       14,   6,  2.0, (80,  80,  80)),
    ]

    # Joint definitions: anchor offsets are relative to segment center
    joints = [
        # Neck: head to chest
        JointDef("neck", "chest", "head",
                 (0, 14), (0, -8),
                 angle_min=-0.5, angle_max=0.5,
                 motor_rate=10.0, motor_max_force=30000.0),
        # Chest-stomach
        JointDef("spine", "chest", "stomach",
                 (0, -14), (0, 10),
                 angle_min=-0.4, angle_max=0.4,
                 motor_rate=8.0, motor_max_force=60000.0),
        # Left shoulder
        JointDef("shoulder_l", "chest", "upper_arm_l",
                 (-12, 10), (0, 13),
                 angle_min=-3.0, angle_max=1.0,
                 motor_rate=15.0, motor_max_force=40000.0),
        # Left elbow
        JointDef("elbow_l", "upper_arm_l", "lower_arm_l",
                 (0, -13), (0, 12),
                 angle_min=-2.5, angle_max=0.1,
                 motor_rate=18.0, motor_max_force=30000.0),
        # Left wrist
        JointDef("wrist_l", "lower_arm_l", "hand_l",
                 (0, -12), (0, 5),
                 angle_min=-0.8, angle_max=0.8,
                 motor_rate=12.0, motor_max_force=10000.0),
        # Right shoulder
        JointDef("shoulder_r", "chest", "upper_arm_r",
                 (12, 10), (0, 13),
                 angle_min=-1.0, angle_max=3.0,
                 motor_rate=15.0, motor_max_force=40000.0),
        # Right elbow
        JointDef("elbow_r", "upper_arm_r", "lower_arm_r",
                 (0, -13), (0, 12),
                 angle_min=-0.1, angle_max=2.5,
                 motor_rate=18.0, motor_max_force=30000.0),
        # Right wrist
        JointDef("wrist_r", "lower_arm_r", "hand_r",
                 (0, -12), (0, 5),
                 angle_min=-0.8, angle_max=0.8,
                 motor_rate=12.0, motor_max_force=10000.0),
        # Left hip
        JointDef("hip_l", "stomach", "upper_leg_l",
                 (-8, -10), (0, 17),
                 angle_min=-1.5, angle_max=2.0,
                 motor_rate=12.0, motor_max_force=60000.0),
        # Left knee
        JointDef("knee_l", "upper_leg_l", "lower_leg_l",
                 (0, -17), (0, 16),
                 angle_min=-0.1, angle_max=2.5,
                 motor_rate=15.0, motor_max_force=50000.0),
        # Left ankle
        JointDef("ankle_l", "lower_leg_l", "foot_l",
                 (0, -16), (-3, 0),
                 angle_min=-0.8, angle_max=0.8,
                 motor_rate=10.0, motor_max_force=20000.0),
        # Right hip
        JointDef("hip_r", "stomach", "upper_leg_r",
                 (8, -10), (0, 17),
                 angle_min=-2.0, angle_max=1.5,
                 motor_rate=12.0, motor_max_force=60000.0),
        # Right knee
        JointDef("knee_r", "upper_leg_r", "lower_leg_r",
                 (0, -17), (0, 16),
                 angle_min=-2.5, angle_max=0.1,
                 motor_rate=15.0, motor_max_force=50000.0),
        # Right ankle
        JointDef("ankle_r", "lower_leg_r", "foot_r",
                 (0, -16), (3, 0),
                 angle_min=-0.8, angle_max=0.8,
                 motor_rate=10.0, motor_max_force=20000.0),
    ]

    return BodyConfig(segments=segments, joints=joints)


DEFAULT_BODY = _make_default_body()
