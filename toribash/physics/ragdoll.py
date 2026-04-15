"""Ragdoll: creates body segments and joints in a pymunk Space.

This module implements the Ragdoll class, which creates a humanoid figure
as a collection of rigid bodies connected by constrained joints. Each ragdoll:
- Contains 15 rigid body segments (boxes)
- Has 14 joints with motors for control
- Supports dismemberment (removing joints during play)

The ragdoll uses pymunk's constraint system:
- PivotJoint: Connects two bodies at a point (no rotation constraint)
- RotaryLimitJoint: Limits relative rotation angle
- SimpleMotor: Drives rotation at a constant rate

Usage:
    >>> import pymunk
    >>> from physics.ragdoll import Ragdoll
    >>> from config.body_config import JointState
    >>> space = pymunk.Space()
    >>> ragdoll = Ragdoll(space, position=(300, 50), facing=1)
    >>> ragdoll.set_joint_state("elbow_l", JointState.CONTRACT)
    >>> for _ in range(30):
    ...     space.step(1/60)
"""

import math
import pymunk
from config.body_config import BodyConfig, JointDef, JointState, DEFAULT_BODY
from config.constants import (
    DEFAULT_MOTOR_RATE, DEFAULT_MOTOR_MAX_FORCE, RELAX_MAX_FORCE,
)


class Ragdoll:
    """A ragdoll fighter composed of pymunk bodies connected by constrained joints.
    
    The ragdoll is a hierarchical kinematic chain representing a humanoid figure.
    Each segment is a rigid box body, connected to its parent by a pivot joint
    with angle limits and a motor.
    
    Attributes:
        space: The pymunk Space this ragdoll belongs to.
        body_config: Definition of segments and joints.
        facing: 1 for right-facing, -1 for left-facing (affects mirroring).
        collision_type: Collision type for all segments (for collision filtering).
        segments: Dict mapping segment names to (body, shape) tuples.
        joints: Dict mapping joint names to (pivot, rot_limit, motor) tuples.
        joint_states: Dict mapping joint names to current JointState.
        dismembered: Set of joint names that have been removed.
    
    Note:
        Player B mirrors Player A's structure. The mirroring affects:
        - Anchor x-coordinates (negated)
        - Angle limits (negated and swapped)
        - Motor rate sign (flipped)
        - Segment positions (horizontal offset reversed)
    """
    
    def __init__(
        self,
        space: pymunk.Space,
        body_config: BodyConfig = DEFAULT_BODY,
        position: tuple[float, float] = (0, 0),
        facing: int = 1,  # 1 = right, -1 = left
        collision_type: int = 1,
    ):
        """Create a new ragdoll in the physics space.
        
        Args:
            space: The pymunk Space to add bodies/constraints to.
            body_config: Definition of the ragdoll's body structure.
            position: Base position (at feet level) for spawning.
            facing: Direction the ragdoll faces (1=right, -1=left).
            collision_type: Collision type for all segments.
        """
        self.space = space
        self.body_config = body_config
        self.facing = facing
        self.collision_type = collision_type

        # Maps: segment_name -> (pymunk.Body, pymunk.Shape)
        self.segments: dict[str, tuple[pymunk.Body, pymunk.Shape]] = {}
        
        # Maps: joint_name -> (PivotJoint, RotaryLimitJoint, SimpleMotor)
        self.joints: dict[str, tuple[pymunk.PivotJoint, pymunk.RotaryLimitJoint, pymunk.SimpleMotor]] = {}
        
        # Current joint states (CONTRACT/EXTEND/HOLD/RELAX)
        self.joint_states: dict[str, JointState] = {}
        
        # Track dismembered joints (removed from space)
        self.dismembered: set[str] = set()

        self._create_segments(position)
        self._create_joints()

    def _create_segments(self, base_pos: tuple[float, float]) -> None:
        """Create all body segments positioned to form a standing humanoid.
        
        Calculates segment centers from feet up, ensuring proper standing pose.
        Each segment is created as a box-shaped rigid body with:
        - Mass from body_config
        - Moment calculated for a box shape
        - Friction and elasticity for realistic collisions
        - Collision type and segment name for callbacks
        
        Args:
            base_pos: (x, y) position at the feet level.
        """
        bx, by = base_pos
        f = self.facing

        # Build position map for looking up segment dimensions
        seg_map = {s.name: s for s in self.body_config.segments}

        # Calculate standing positions from feet up
        # Heights are used to stack segments properly
        foot_h = seg_map["foot_l"].height
        lower_leg_h = seg_map["lower_leg_l"].height
        upper_leg_h = seg_map["upper_leg_l"].height
        stomach_h = seg_map["stomach"].height
        chest_h = seg_map["chest"].height
        head_h = seg_map["head"].height

        # Stack segments from ground up
        ground_to_ankle = foot_h
        ankle_y = by + ground_to_ankle
        knee_y = ankle_y + lower_leg_h
        hip_y = knee_y + upper_leg_h
        stomach_center_y = hip_y + stomach_h / 2
        chest_center_y = stomach_center_y + stomach_h / 2 + chest_h / 2
        head_center_y = chest_center_y + chest_h / 2 + head_h / 2

        # Horizontal offsets for limbs (mirrored for Player B)
        hip_offset = 8 * f
        shoulder_offset = 12 * f

        # Position each segment's center
        positions = {
            "head":         (bx, head_center_y),
            "chest":        (bx, chest_center_y),
            "stomach":      (bx, stomach_center_y),
            "upper_arm_l":  (bx - shoulder_offset, chest_center_y - 3),
            "lower_arm_l":  (bx - shoulder_offset, chest_center_y - 3 - seg_map["upper_arm_l"].height),
            "hand_l":       (bx - shoulder_offset, chest_center_y - 3 - seg_map["upper_arm_l"].height - seg_map["lower_arm_l"].height),
            "upper_arm_r":  (bx + shoulder_offset, chest_center_y - 3),
            "lower_arm_r":  (bx + shoulder_offset, chest_center_y - 3 - seg_map["upper_arm_r"].height),
            "hand_r":       (bx + shoulder_offset, chest_center_y - 3 - seg_map["upper_arm_r"].height - seg_map["lower_arm_r"].height),
            "upper_leg_l":  (bx - hip_offset, hip_y - upper_leg_h / 2),
            "lower_leg_l":  (bx - hip_offset, ankle_y + lower_leg_h / 2),
            "foot_l":       (bx - hip_offset, by + foot_h / 2),
            "upper_leg_r":  (bx + hip_offset, hip_y - upper_leg_h / 2),
            "lower_leg_r":  (bx + hip_offset, ankle_y + lower_leg_h / 2),
            "foot_r":       (bx + hip_offset, by + foot_h / 2),
        }

        # Create each segment as a rigid body with box shape
        for seg_def in self.body_config.segments:
            pos = positions[seg_def.name]
            
            # Create body with mass and moment for box
            body = pymunk.Body(
                seg_def.mass, 
                pymunk.moment_for_box(seg_def.mass, (seg_def.width, seg_def.height))
            )
            body.position = pos

            # Create box shape attached to body
            shape = pymunk.Poly.create_box(body, (seg_def.width, seg_def.height))
            shape.friction = 0.8       # High friction for standing
            shape.elasticity = 0.1      # Slight bounce
            shape.collision_type = self.collision_type
            
            # Store segment name on shape for collision callbacks
            # This allows us to identify which body parts collided
            shape.segment_name = seg_def.name

            self.space.add(body, shape)
            self.segments[seg_def.name] = (body, shape)

    def _create_joints(self) -> None:
        """Create pivot joints with rotary limits and motors for all joints.
        
        Each joint in the body becomes a pymunk constraint chain:
        1. PivotJoint: Pins the two bodies together at the anchor point.
           Has collide_bodies=False so connected segments don't collide.
        2. RotaryLimitJoint: Constrains relative rotation to angle limits.
        3. SimpleMotor: Drives rotation at a constant rate (for CONTRACT/EXTEND).
        
        For mirrored ragdolls (Player B), anchor x-coordinates and angle
        limits are negated to create anatomically correct mirroring.
        """
        f = self.facing
        
        for jdef in self.body_config.joints:
            parent_body = self.segments[jdef.parent][0]
            child_body = self.segments[jdef.child][0]

            # Mirror anchor x-coordinates when facing left
            # This ensures the joint is at the correct anatomical position
            ax, ay = jdef.anchor_parent
            bx, by = jdef.anchor_child
            if f == -1:
                ax = -ax
                bx = -bx

            # Pivot joint: connects the two bodies at the anchor point
            # collide_bodies=False prevents connected segments from colliding
            pivot = pymunk.PivotJoint(parent_body, child_body, (ax, ay), (bx, by))
            pivot.collide_bodies = False

            # Mirror angle limits when facing left
            # Negating and swapping min/max preserves anatomical meaning
            if f == -1:
                angle_min = -jdef.angle_max
                angle_max = -jdef.angle_min
            else:
                angle_min = jdef.angle_min
                angle_max = jdef.angle_max

            # Rotary limit: constrains the relative angle between bodies
            rot_limit = pymunk.RotaryLimitJoint(parent_body, child_body, angle_min, angle_max)
            rot_limit.max_force = jdef.motor_max_force

            # Motor: drives the joint at a constant angular rate
            # Rate is set dynamically based on joint state (CONTRACT/EXTEND/etc)
            motor = pymunk.SimpleMotor(parent_body, child_body, 0.0)
            motor.max_force = jdef.motor_max_force

            self.space.add(pivot, rot_limit, motor)
            self.joints[jdef.name] = (pivot, rot_limit, motor)
            self.joint_states[jdef.name] = JointState.HOLD

    def set_joint_state(self, joint_name: str, state: JointState) -> None:
        """Set the state of a single joint, controlling its motor behavior.
        
        State effects:
        - CONTRACT: Motor rate drives toward angle_min (closing the joint)
        - EXTEND: Motor rate drives toward angle_max (opening the joint)
        - HOLD: Zero rate, max force resists movement (stiff)
        - RELAX: Zero rate, zero force (limb is limp)
        
        Args:
            joint_name: Name of the joint to control.
            state: The desired JointState (CONTRACT/EXTEND/HOLD/RELAX).
        
        Note:
            Does nothing if the joint has been dismembered.
        """
        if joint_name in self.dismembered:
            return
        if joint_name not in self.joints:
            return

        self.joint_states[joint_name] = state
        jdef = self._get_joint_def(joint_name)
        _, _, motor = self.joints[joint_name]

        # Motor rate sign depends on facing direction
        # This ensures CONTRACT has the same anatomical effect for both players
        rate_sign = 1 if self.facing == 1 else -1
        
        if state == JointState.CONTRACT:
            motor.rate = jdef.motor_rate * rate_sign
            motor.max_force = jdef.motor_max_force
        elif state == JointState.EXTEND:
            motor.rate = -jdef.motor_rate * rate_sign
            motor.max_force = jdef.motor_max_force
        elif state == JointState.HOLD:
            motor.rate = 0.0
            motor.max_force = jdef.motor_max_force
        elif state == JointState.RELAX:
            motor.rate = 0.0
            motor.max_force = RELAX_MAX_FORCE

    def set_all_joint_states(self, states: list[JointState]) -> None:
        """Set all joint states from a list ordered by body_config.joints.
        
        Args:
            states: List of JointState values, one per joint in order.
        
        Raises:
            IndexError: If states list length doesn't match joint count.
        """
        for jdef, state in zip(self.body_config.joints, states):
            self.set_joint_state(jdef.name, state)

    def get_joint_angles(self) -> list[float]:
        """Get the current relative angle for each joint.
        
        Returns:
            List of angles in radians, one per joint.
            Returns 0.0 for dismembered joints.
        """
        angles = []
        for jdef in self.body_config.joints:
            if jdef.name in self.dismembered:
                angles.append(0.0)
                continue
            parent_body = self.segments[jdef.parent][0]
            child_body = self.segments[jdef.child][0]
            rel_angle = child_body.angle - parent_body.angle
            angles.append(rel_angle)
        return angles

    def get_joint_angular_velocities(self) -> list[float]:
        """Get the current relative angular velocity for each joint.
        
        Returns:
            List of angular velocities in rad/s, one per joint.
            Returns 0.0 for dismembered joints.
        """
        velocities = []
        for jdef in self.body_config.joints:
            if jdef.name in self.dismembered:
                velocities.append(0.0)
                continue
            parent_body = self.segments[jdef.parent][0]
            child_body = self.segments[jdef.child][0]
            rel_vel = child_body.angular_velocity - parent_body.angular_velocity
            velocities.append(rel_vel)
        return velocities

    def get_segment_positions(self) -> list[tuple[float, float]]:
        """Get the world position of each segment's center.
        
        Returns:
            List of (x, y) tuples, one per segment in body_config order.
        """
        return [self.segments[s.name][0].position for s in self.body_config.segments]

    def get_segment_velocities(self) -> list[tuple[float, float]]:
        """Get the linear velocity of each segment.
        
        Returns:
            List of (vx, vy) velocity tuples, one per segment.
        """
        return [self.segments[s.name][0].velocity for s in self.body_config.segments]

    def get_segment_angles(self) -> list[float]:
        """Get the absolute rotation angle of each segment.
        
        Returns:
            List of angles in radians, one per segment.
        """
        return [self.segments[s.name][0].angle for s in self.body_config.segments]

    def get_torso_position(self) -> tuple[float, float]:
        """Get the chest center position, used as the ragdoll's reference point.
        
        The chest segment serves as the ragdoll's "torso" for:
        - Relative positioning in observations
        - Spawn position calculations
        - Mirroring logic
        
        Returns:
            (x, y) position of the chest segment center.
        """
        return self.segments["chest"][0].position

    def dismember_joint(self, joint_name: str) -> None:
        """Remove a joint from the simulation, simulating limb detachment.
        
        Removes all three constraint objects (pivot, limit, motor) from
        the space. The child segment becomes free-floating, able to
        collide with other bodies including its former parent.
        
        Args:
            joint_name: Name of the joint to dismember.
        
        Note:
            Safe to call multiple times - subsequent calls are no-ops.
            The joint_name is added to self.dismembered set.
        """
        if joint_name in self.dismembered:
            return
        self.dismembered.add(joint_name)
        pivot, rot_limit, motor = self.joints[joint_name]
        self.space.remove(pivot, rot_limit, motor)

    def _get_joint_def(self, joint_name: str) -> JointDef:
        """Look up a joint definition by name.
        
        Args:
            joint_name: Name of the joint to find.
        
        Returns:
            The JointDef for the requested joint.
        
        Raises:
            KeyError: If joint_name is not found in body_config.
        """
        for jdef in self.body_config.joints:
            if jdef.name == joint_name:
                return jdef
        raise KeyError(f"Joint {joint_name} not found")
