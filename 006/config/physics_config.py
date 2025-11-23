"""
Physics configuration for car dynamics simulation.

This module centralizes all physics-related constants used in the Car class
and provides a single source of truth for vehicle parameters, tire models,
and drivetrain characteristics.

All values are based on the 2022 Mazda MX-5 Sport (ND) with 195/50R16 tires.
"""

from dataclasses import dataclass, field


@dataclass
class VehicleParams:
    """
    Vehicle physical parameters (Based on 2022 Mazda MX-5 Sport).
    """
    # Mass and dimensions
    MASS: float = 1062.0  # Vehicle mass (kg)
    LENGTH: float = 2.310  # Wheelbase (m)
    WIDTH: float = 1.50  # Track width (m)

    # Weight distribution (50/50)
    LF: float = LENGTH * 0.5  # Distance from CG to front axle (m)
    LR: float = LENGTH * 0.5  # Distance from CG to rear axle (m)

    # Center of Gravity
    CG_HEIGHT: float = 0.55  # Center of Gravity height (m) - raised for stiffer suspension feel


@dataclass
class TireParams:
    """
    Tire physical parameters for 195/50R16 wheels.
    """
    TIRE_RADIUS: float = 0.309  # Wheel radius (m)
    TIRE_WIDTH: float = 0.205  # Tire width (m)

    # Estimated wheel + tire inertia
    # 16" wheel + tire = ~17kg. I = 0.8 * m * r^2 = 0.8 * 17 * 0.3^2 = ~1.2
    INERTIA: float = 1.2  # Wheel inertia (kg*m^2)


@dataclass
class PacejkaParams:
    """
    Pacejka Magic Formula tire model parameters - FULLY UNIFIED.

    Calibrated for MX-5 Sport with street tires (195/50R16).

    Expected Performance:
    - Unified traction circle: 0.95g max grip (lateral, longitudinal, or combined)
    - Same tire coefficients for all directions (realistic physics)
    - Differences only in slip inputs (angle vs ratio)

    Magic Formula: F = D × sin(C × arctan(B×α - E×(B×α - arctan(B×α))))

    UNIFIED Parameters (same for lateral AND longitudinal):
    - B: Stiffness factor (initial slope of force curve)
      Higher B = more force across entire slip range
      Typical range: 8-15 for street tires

    - C: Shape factor (affects curve peakiness)
      Standard value for passenger car tires
      Typical range: 1.3-2.5

    - D: Peak friction multiplier (traction circle radius)
      Combined with normal force to determine max grip
      D=0.95 means peak friction = 0.95g (realistic street tire)
      Traction circle constraint ensures Fx² + Fy² ≤ (D × Fz)²

    - E: Curvature factor (shape near/after peak)
      Controls how sharply grip falls off past optimal slip
      Typical range: 0.9-1.0

    Peak slip characteristics:
    - Lateral: Peak at ~8-10° slip angle (street tire typical)
    - Longitudinal: Peak at ~12-15% slip ratio (street tire typical)
    """
    # FULLY UNIFIED tire model parameters
    B: float = 10   # Stiffness factor (same for lat/lon)
    C: float = 1.9   # Shape factor (same for lat/lon)
    D: float = 1.0  # Peak friction (traction circle radius)
    E: float = 0.97  # Curvature factor (same for lat/lon)


@dataclass
class DrivetrainParams:
    """
    Drivetrain parameters for 2.0L Skyactiv-G engine.

    Realistic torque delivery accounting for:
    - Drivetrain loss: 15% (clutch, gearbox, differential friction)
    - Progressive throttle: Real pedal isn't instant 100% torque
    - Clutch slip: Engagement isn't perfect lockup
    - Weight transfer: Front wheels lift slightly on hard accel
    """
    # Engine power
    ENGINE_POWER: float = 135000.0  # Power (Watts) (181 hp * 745.7)

    # Torque delivery (per wheel, rear-wheel drive)
    # Theoretical max (1st gear): ~1496 Nm per wheel at crank
    # Practical max: 400 Nm per wheel (accounts for losses and throttle mapping)
    MAX_TORQUE_PER_WHEEL: float = 400.0  # Conservative for good traction

    # Power curve transition
    # Transition from constant torque to constant power
    # P = τ * ω  =>  ω = P / τ
    # (135000 W / 2 wheels) / 400 Nm = 168.75 rad/s (~162 km/h)
    POWER_TRANSITION_OMEGA: float = 168.75  # rad/s

    # Startup acceleration
    # Alpha = Torque / Inertia = 400 / 1.2 = 333 rad/s^2
    STARTUP_ACCEL: float = 333.0  # Angular acceleration (rad/s^2)

    # Brake torque (per wheel)
    # Reduced from theoretical max to prevent instant wheel locking
    # Allows smooth brake modulation
    # Front: 60 N·m → 50 rad/s² → 1.58g per wheel
    # Rear: 40 N·m → 33 rad/s² → 1.05g per wheel
    # Combined: ~1.1-1.3g total braking with load transfer
    MAX_BRAKE_TORQUE_FRONT: float = 60.0  # N·m
    MAX_BRAKE_TORQUE_REAR: float = 40.0   # N·m


@dataclass
class AerodynamicsParams:
    """
    Aerodynamic parameters for ND MX-5 RF.
    """
    RHO_AIR: float = 1.225  # Air density (kg/m^3)
    FRONTAL_AREA: float = 1.8  # Frontal area (m^2)
    CD_CAR: float = 0.33  # Drag coefficient
    C_ROLL_RESISTANCE: float = 0.015  # Rolling resistance (tires on asphalt)


@dataclass
class SteeringParams:
    """
    Steering system parameters.
    """
    MAX_STEER_ANGLE: float = 0.4  # Max steering angle (rad) (~23 degrees)
    STEER_RATE: float = 1.5  # Steering response rate (rad/s) - more realistic


@dataclass
class FrictionParams:
    """
    Surface friction coefficients.

    Note: Pacejka D parameter already handles peak friction.
    These values modify the effective friction for different surfaces.
    """
    BASE_FRICTION: float = 1.0  # Asphalt
    GRASS_FRICTION: float = 0.5  # Grass


@dataclass
class NormalizationParams:
    """
    Normalization constants for better training stability.

    These constants are used to normalize state observation values to similar scales,
    which improves neural network training stability and convergence.

    All values are based on typical maximum values observed during racing:
    - Velocity: ~25-30 m/s top speed
    - Angular velocity: ~3-4 rad/s typical max
    - Acceleration: ~30-40 m/s^2 peak acceleration
    - Curvature: Sharp turn radius
    - Slip ratio: Dimensionless, clipped at extreme values
    - Vertical force: ~1000kg car * 1.5g per wheel + safety margin
    """
    MAX_VELOCITY: float = 30.0  # m/s (typical max speed ~25-30 m/s)
    MAX_ANGULAR_VEL: float = 5.0  # rad/s (typical max ~3-4 rad/s)
    MAX_ACCELERATION: float = 50.0  # m/s^2 (typical max ~30-40 m/s^2)
    MAX_CURVATURE: float = 1.0  # 1/m (typical sharp turn)
    MAX_SLIP_RATIO: float = 2.0  # Dimensionless (clip extreme values)
    MAX_VERTICAL_FORCE: float = 5000.0  # N (approx 1000kg car * 1.5g per wheel + safety margin)


@dataclass
class ObservationParams:
    """
    Observation space configuration for vector state representation.

    These parameters control what information is included in the state observation
    and how waypoint lookahead is structured.

    Waypoint Configuration:
    - NUM_LOOKAHEAD: Number of waypoints to include in observation
    - WAYPOINT_STRIDE: Skip factor between waypoints (1=consecutive, 2=every other, 3=every 3rd, etc.)

    Examples:
    - NUM_LOOKAHEAD=20, STRIDE=1: 20 consecutive waypoints (70m @ 3.5m spacing) - DEFAULT
    - NUM_LOOKAHEAD=20, STRIDE=2: Every 2nd waypoint (140m @ 7m spacing) - 2× horizon
    - NUM_LOOKAHEAD=20, STRIDE=3: Every 3rd waypoint (210m @ 10.5m spacing) - 3× horizon
    - NUM_LOOKAHEAD=10, STRIDE=2: 10 waypoints at 7m spacing (70m horizon, 20D observation)

    Frame Stacking:
    - FRAME_STACK: Number of consecutive frames to stack (1=no stacking, 2+=stacking enabled)
    - When FRAME_STACK > 1, observations from multiple timesteps are concatenated
    - Provides temporal information (derivatives via finite differences)
    - Example: FRAME_STACK=4 gives access to velocity, acceleration, jerk through differences

    Note: Changing these values changes the observation space dimension and requires retraining.
    Base dimension = 33 + (NUM_LOOKAHEAD × 2) where 33 = car state + track + speed + accel + slip + forces + steering
    Final dimension = (33 + NUM_LOOKAHEAD × 2) × FRAME_STACK
    """
    NUM_LOOKAHEAD: int = 10  # Number of waypoints to include in observation
    WAYPOINT_STRIDE: int = 2  # Spacing between waypoints (1=consecutive, 2=every other, etc.)
    FRAME_STACK: int = 3  # Number of frames to stack (1=no stacking, 2+=enabled)


@dataclass
class PhysicsConfig:
    """
    Complete physics configuration combining all parameter groups.

    Usage:
        config = PhysicsConfig()
        print(config.vehicle.MASS)  # 1062.0
        print(config.pacejka.D_LAT)  # 0.95
        print(config.normalization.MAX_VELOCITY)  # 30.0
        print(config.observation.WAYPOINT_STRIDE)  # 1
    """
    vehicle: VehicleParams = field(default_factory=VehicleParams)
    tire: TireParams = field(default_factory=TireParams)
    pacejka: PacejkaParams = field(default_factory=PacejkaParams)
    drivetrain: DrivetrainParams = field(default_factory=DrivetrainParams)
    aerodynamics: AerodynamicsParams = field(default_factory=AerodynamicsParams)
    steering: SteeringParams = field(default_factory=SteeringParams)
    friction: FrictionParams = field(default_factory=FrictionParams)
    normalization: NormalizationParams = field(default_factory=NormalizationParams)
    observation: ObservationParams = field(default_factory=ObservationParams)


def get_base_observation_dim(num_lookahead: int) -> int:
    """
    Calculate base (single-frame) observation dimension.

    Components (35 fixed + num_lookahead * 2):
        - Car state: 11D (position, velocity, angle, contacts, progress)
        - Track info: 5D (distance to center, angle, curvature, etc.)
        - Speed: 1D
        - Accelerations: 2D (longitudinal, lateral)
        - Tire slip angles: 4D
        - Tire slip ratios: 4D
        - Tire vertical forces: 4D
        - Steering state: 2D (angle, rate)
        - Previous action: 2D (steering, acceleration)
        - Lookahead waypoints: num_lookahead * 2D (x, y per waypoint)

    Args:
        num_lookahead: Number of lookahead waypoints

    Returns:
        Single-frame observation dimension
    """
    return 35 + (num_lookahead * 2)


def get_stacked_observation_dim(num_lookahead: int, frame_stack: int) -> int:
    """
    Calculate stacked observation dimension with frame stacking.

    Args:
        num_lookahead: Number of lookahead waypoints
        frame_stack: Number of frames to stack

    Returns:
        Stacked observation dimension (base_dim * frame_stack)
    """
    base_dim = get_base_observation_dim(num_lookahead)
    return base_dim * frame_stack
