"""Ragdoll: creates body segments and joints in a pymunk Space."""

import math
import pymunk
from config.body_config import BodyConfig, JointDef, JointState, DEFAULT_BODY
from config.constants import (
    DEFAULT_MOTOR_RATE, DEFAULT_MOTOR_MAX_FORCE, RELAX_MAX_FORCE,
)


class Ragdoll:
    """A ragdoll fighter composed of pymunk bodies connected by constrained joints."""

    def __init__(
        self,
        space: pymunk.Space,
        body_config: BodyConfig = DEFAULT_BODY,
        position: tuple[float, float] = (0, 0),
        facing: int = 1,  # 1 = right, -1 = left
        collision_type: int = 1,
    ):
        self.space = space
        self.body_config = body_config
        self.facing = facing
        self.collision_type = collision_type

        # Maps: segment_name -> (pymunk.Body, pymunk.Shape)
        self.segments: dict[str, tuple[pymunk.Body, pymunk.Shape]] = {}
        # Maps: joint_name -> (PivotJoint, RotaryLimitJoint, SimpleMotor)
        self.joints: dict[str, tuple[pymunk.PivotJoint, pymunk.RotaryLimitJoint, pymunk.SimpleMotor]] = {}
        # Current joint states
        self.joint_states: dict[str, JointState] = {}
        # Track dismembered joints
        self.dismembered: set[str] = set()

        self._create_segments(position)
        self._create_joints()

    def _create_segments(self, base_pos: tuple[float, float]):
        """Create all body segments positioned to form a standing humanoid."""
        bx, by = base_pos
        f = self.facing

        # Build position map for a standing pose
        # Positions are center of each segment, relative to base_pos
        # base_pos is at the feet level
        seg_map = {s.name: s for s in self.body_config.segments}

        # Calculate standing positions from feet up
        foot_h = seg_map["foot_l"].height
        lower_leg_h = seg_map["lower_leg_l"].height
        upper_leg_h = seg_map["upper_leg_l"].height
        stomach_h = seg_map["stomach"].height
        chest_h = seg_map["chest"].height
        head_h = seg_map["head"].height

        ground_to_ankle = foot_h
        ankle_y = by + ground_to_ankle
        knee_y = ankle_y + lower_leg_h
        hip_y = knee_y + upper_leg_h
        stomach_center_y = hip_y + stomach_h / 2
        chest_center_y = stomach_center_y + stomach_h / 2 + chest_h / 2
        head_center_y = chest_center_y + chest_h / 2 + head_h / 2

        # Horizontal offsets for limbs
        hip_offset = 8 * f
        shoulder_offset = 12 * f

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

        for seg_def in self.body_config.segments:
            pos = positions[seg_def.name]
            body = pymunk.Body(seg_def.mass, pymunk.moment_for_box(seg_def.mass, (seg_def.width, seg_def.height)))
            body.position = pos

            shape = pymunk.Poly.create_box(body, (seg_def.width, seg_def.height))
            shape.friction = 0.8
            shape.elasticity = 0.1
            shape.collision_type = self.collision_type
            # Store segment name on shape for collision callbacks
            shape.segment_name = seg_def.name

            # Prevent self-collision: same-group segments don't collide
            # Group = collision_type so each ragdoll's segments ignore each other
            shape.filter = pymunk.ShapeFilter(group=self.collision_type)

            self.space.add(body, shape)
            self.segments[seg_def.name] = (body, shape)

    def _create_joints(self):
        """Create pivot joints with rotary limits and motors."""
        f = self.facing
        for jdef in self.body_config.joints:
            parent_body = self.segments[jdef.parent][0]
            child_body = self.segments[jdef.child][0]

            # Mirror anchor x-coordinates when facing left
            ax, ay = jdef.anchor_parent
            bx, by = jdef.anchor_child
            if f == -1:
                ax = -ax
                bx = -bx

            # Pivot joint: connects the two bodies at the anchor point
            pivot = pymunk.PivotJoint(parent_body, child_body, (ax, ay), (bx, by))
            pivot.collide_bodies = False

            # Mirror angle limits when facing left
            if f == -1:
                angle_min = -jdef.angle_max
                angle_max = -jdef.angle_min
            else:
                angle_min = jdef.angle_min
                angle_max = jdef.angle_max

            # Rotary limit: constrains the relative angle
            rot_limit = pymunk.RotaryLimitJoint(parent_body, child_body, angle_min, angle_max)
            rot_limit.max_force = jdef.motor_max_force

            # Motor: drives the joint
            motor = pymunk.SimpleMotor(parent_body, child_body, 0.0)
            motor.max_force = jdef.motor_max_force

            self.space.add(pivot, rot_limit, motor)
            self.joints[jdef.name] = (pivot, rot_limit, motor)
            self.joint_states[jdef.name] = JointState.HOLD

    def set_joint_state(self, joint_name: str, state: JointState):
        """Set the state of a single joint."""
        if joint_name in self.dismembered:
            return
        if joint_name not in self.joints:
            return

        self.joint_states[joint_name] = state
        jdef = self._get_joint_def(joint_name)
        _, _, motor = self.joints[joint_name]

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

    def set_all_joint_states(self, states: list[JointState]):
        """Set all joint states from a list (ordered by body_config.joints)."""
        for jdef, state in zip(self.body_config.joints, states):
            self.set_joint_state(jdef.name, state)

    def get_joint_angles(self) -> list[float]:
        """Get relative angle for each joint (parent-child angle difference)."""
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
        """Get relative angular velocity for each joint."""
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
        """Get position of each segment's center."""
        return [self.segments[s.name][0].position for s in self.body_config.segments]

    def get_segment_velocities(self) -> list[tuple[float, float]]:
        """Get velocity of each segment."""
        return [self.segments[s.name][0].velocity for s in self.body_config.segments]

    def get_segment_angles(self) -> list[float]:
        """Get absolute angle of each segment."""
        return [self.segments[s.name][0].angle for s in self.body_config.segments]

    def get_torso_position(self) -> tuple[float, float]:
        """Get the chest center position (used as reference point)."""
        return self.segments["chest"][0].position

    def dismember_joint(self, joint_name: str):
        """Remove a joint (simulating limb detachment)."""
        if joint_name in self.dismembered:
            return
        self.dismembered.add(joint_name)
        pivot, rot_limit, motor = self.joints[joint_name]
        self.space.remove(pivot, rot_limit, motor)

    def _get_joint_def(self, joint_name: str) -> JointDef:
        for jdef in self.body_config.joints:
            if jdef.name == joint_name:
                return jdef
        raise KeyError(f"Joint {joint_name} not found")
