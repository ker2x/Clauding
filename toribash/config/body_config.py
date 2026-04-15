"""Ragdoll body configuration: segments, joints, and joint states.

This module defines the humanoid body structure used in Toribash 2D.
Each ragdoll consists of:
    - 15 rigid body segments (boxes)
    - 14 joints connecting segments
    - Each joint has CONTRACT/EXTEND/HOLD/RELAX motor states

The default body is ~170cm tall, proportionally modeled on a human.

Joint Hierarchy:
    Head
      └── Neck
          └── Chest
              ├── Shoulder_L → Elbow_L → Wrist_L → Hand_L
              ├── Shoulder_R → Elbow_R → Wrist_R → Hand_R
              └── Spine
                  ├── Hip_L → Knee_L → Ankle_L → Foot_L
                  └── Hip_R → Knee_R → Ankle_R → Foot_R

Usage:
    >>> from config.body_config import DEFAULT_BODY, JointState
    >>> body = DEFAULT_BODY
    >>> print(f"Ragdoll has {body.num_joints} joints")
    Ragdoll has 14 joints
"""

from dataclasses import dataclass, field
from functools import cached_property
from enum import IntEnum


class JointState(IntEnum):
    """Motor states for controlling a joint.
    
    Each state sets the motor's rate and max_force to achieve
    different behaviors:
    
    - CONTRACT: Actively closes the joint (decreases angle)
    - EXTEND: Actively opens the joint (increases angle)  
    - HOLD: Resists any movement (stiff, zero rate)
    - RELAX: No resistance to movement (limp, gravity takes over)
    
    The actual rate direction depends on the ragdoll's facing direction.
    """
    CONTRACT = 0
    EXTEND = 1
    HOLD = 2
    RELAX = 3


@dataclass
class SegmentDef:
    """Definition of a rigid body segment (body part).
    
    Attributes:
        name: Unique identifier for this segment (e.g., "head", "chest").
        width: Width in centimeters (X dimension).
        height: Height in centimeters (Y dimension).
        mass: Mass in kilograms.
        color: RGB tuple for rendering (0-255 per channel).
    """
    name: str
    width: float   # cm
    height: float  # cm
    mass: float    # kg
    color: tuple[int, int, int] = (200, 200, 200)


@dataclass
class JointDef:
    """Definition of a joint connecting two segments.
    
    A joint uses a pivot constraint with angle limits and a motor.
    The motor drives the joint between min/max angles at a given rate.
    
    Attributes:
        name: Unique identifier (e.g., "elbow_l", "knee_r").
        parent: Name of the parent (more central) segment.
        child: Name of the child (limb) segment.
        anchor_parent: Anchor point relative to parent center (cm).
        anchor_child: Anchor point relative to child center (cm).
        angle_min: Minimum relative angle (radians).
        angle_max: Maximum relative angle (radians).
        motor_rate: Angular velocity for CONTRACT/EXTEND (rad/s).
        motor_max_force: Maximum torque the motor can apply.
    """
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
    """Complete body configuration for a ragdoll.
    
    Contains all segment and joint definitions that make up a fighter.
    
    Attributes:
        segments: List of all segment definitions.
        joints: List of all joint definitions.
    
    Properties:
        num_joints: Number of joints (equals len(joints)).
    """
    segments: list[SegmentDef] = field(default_factory=list)
    joints: list[JointDef] = field(default_factory=list)

    @property
    def num_joints(self) -> int:
        """Return the number of joints in this body configuration."""
        return len(self.joints)

    @cached_property
    def segment_to_joints(self) -> dict[str, list[str]]:
        """Map each segment name to the joint names that connect to it."""
        mapping: dict[str, list[str]] = {}
        for jdef in self.joints:
            mapping.setdefault(jdef.parent, []).append(jdef.name)
            mapping.setdefault(jdef.child, []).append(jdef.name)
        return mapping


def _make_default_body() -> BodyConfig:
    """Create the default humanoid ragdoll body configuration.
    
    The ragdoll stands upright with arms at sides. Total height is
    approximately 176cm (foot to head). Mass distribution favors the
    torso (core strength).
    
    Segment Layout (from feet up):
        - 2 feet (14×12cm, 2kg each)
        - 2 lower legs (9×32cm, 5kg each)
        - 2 upper legs (10×34cm, 8kg each)
        - 1 stomach (22×20cm, 12kg)
        - 1 chest (24×28cm, 20kg) - heaviest segment
        - 2 upper arms (8×26cm, 4kg each)
        - 2 lower arms (7×24cm, 3kg each)
        - 2 hands (6×10cm, 1kg each)
        - 1 head (16×16cm, 5kg)
    
    Returns:
        BodyConfig with all segments and joints defined.
    """
    # Segment dimensions (width × height in cm)
    # Mass distribution: torso heaviest (20+12=32kg), limbs lighter
    segments = [
        # Core body (center of mass)
        SegmentDef("head",          16, 16,  5.0, (255, 220, 180)),
        SegmentDef("chest",         24, 28, 20.0, (100, 140, 200)),
        SegmentDef("stomach",       22, 20, 12.0, (100, 140, 200)),
        
        # Left arm (from shoulder to hand)
        SegmentDef("upper_arm_l",   8,  26,  4.0, (200, 160, 140)),
        SegmentDef("lower_arm_l",   7,  24,  3.0, (200, 160, 140)),
        SegmentDef("hand_l",        6,  10,  1.0, (255, 220, 180)),
        
        # Right arm (from shoulder to hand)
        SegmentDef("upper_arm_r",   8,  26,  4.0, (200, 160, 140)),
        SegmentDef("lower_arm_r",   7,  24,  3.0, (200, 160, 140)),
        SegmentDef("hand_r",        6,  10,  1.0, (255, 220, 180)),
        
        # Left leg (from hip to foot)
        SegmentDef("upper_leg_l",  10,  34,  8.0, (140, 100, 80)),
        SegmentDef("lower_leg_l",   9,  32,  5.0, (140, 100, 80)),
        SegmentDef("foot_l",       14,  12,  2.0, (80,  80,  80)),
        
        # Right leg (from hip to foot)
        SegmentDef("upper_leg_r",  10,  34,  8.0, (140, 100, 80)),
        SegmentDef("lower_leg_r",   9,  32,  5.0, (140, 100, 80)),
        SegmentDef("foot_r",       14,  12,  2.0, (80,  80,  80)),
    ]

    # Joint definitions: anchor offsets are relative to segment center
    # Motors use 1M force for strong standing ability
    joints = [
        # Neck: head to chest (limited rotation, strong motor)
        JointDef("neck", "chest", "head",
                 (0, 14), (0, -8),
                 angle_min=-0.5, angle_max=0.5,
                 motor_rate=10.0, motor_max_force=1000000.0),
        
        # Spine: chest to stomach (core stability)
        JointDef("spine", "chest", "stomach",
                 (0, -14), (0, 10),
                 angle_min=-0.4, angle_max=0.4,
                 motor_rate=8.0, motor_max_force=1000000.0),
        
        # Left shoulder: wide range of motion for striking
        JointDef("shoulder_l", "chest", "upper_arm_l",
                 (-12, 10), (0, 13),
                 angle_min=-3.0, angle_max=1.0,
                 motor_rate=15.0, motor_max_force=1000000.0),
        
        # Left elbow: mostly bends one way
        JointDef("elbow_l", "upper_arm_l", "lower_arm_l",
                 (0, -13), (0, 12),
                 angle_min=-2.5, angle_max=0.1,
                 motor_rate=18.0, motor_max_force=1000000.0),
        
        # Left wrist: small rotation
        JointDef("wrist_l", "lower_arm_l", "hand_l",
                 (0, -12), (0, 5),
                 angle_min=-0.8, angle_max=0.8,
                 motor_rate=12.0, motor_max_force=1000000.0),
        
        # Right shoulder: mirrored from left
        JointDef("shoulder_r", "chest", "upper_arm_r",
                 (12, 10), (0, 13),
                 angle_min=-1.0, angle_max=3.0,
                 motor_rate=15.0, motor_max_force=1000000.0),
        
        # Right elbow: mirrored from left
        JointDef("elbow_r", "upper_arm_r", "lower_arm_r",
                 (0, -13), (0, 12),
                 angle_min=-0.1, angle_max=2.5,
                 motor_rate=18.0, motor_max_force=1000000.0),
        
        # Right wrist: mirrored from left
        JointDef("wrist_r", "lower_arm_r", "hand_r",
                 (0, -12), (0, 5),
                 angle_min=-0.8, angle_max=0.8,
                 motor_rate=12.0, motor_max_force=1000000.0),
        
        # Left hip: leg lifting motion
        JointDef("hip_l", "stomach", "upper_leg_l",
                 (-8, -10), (0, 17),
                 angle_min=-1.5, angle_max=2.0,
                 motor_rate=12.0, motor_max_force=1000000.0),
        
        # Left knee: bends backward
        JointDef("knee_l", "upper_leg_l", "lower_leg_l",
                 (0, -17), (0, 16),
                 angle_min=-0.1, angle_max=2.5,
                 motor_rate=15.0, motor_max_force=1000000.0),
        
        # Left ankle: foot rotation
        JointDef("ankle_l", "lower_leg_l", "foot_l",
                 (0, -16), (-3, 3),
                 angle_min=-0.8, angle_max=0.8,
                 motor_rate=10.0, motor_max_force=1000000.0),
        
        # Right hip: mirrored from left
        JointDef("hip_r", "stomach", "upper_leg_r",
                 (8, -10), (0, 17),
                 angle_min=-2.0, angle_max=1.5,
                 motor_rate=12.0, motor_max_force=1000000.0),
        
        # Right knee: mirrored from left
        JointDef("knee_r", "upper_leg_r", "lower_leg_r",
                 (0, -17), (0, 16),
                 angle_min=-2.5, angle_max=0.1,
                 motor_rate=15.0, motor_max_force=1000000.0),
        
        # Right ankle: mirrored from left
        JointDef("ankle_r", "lower_leg_r", "foot_r",
                 (0, -16), (3, 3),
                 angle_min=-0.8, angle_max=0.8,
                 motor_rate=10.0, motor_max_force=1000000.0),
    ]

    return BodyConfig(segments=segments, joints=joints)


# Default body configuration used throughout the game.
# This creates a standing humanoid ragdoll ~176cm tall.
DEFAULT_BODY: BodyConfig = _make_default_body()
