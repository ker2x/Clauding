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
    CG_HEIGHT: float = 0.46  # Center of Gravity height (m)


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
    Pacejka Magic Formula tire model parameters.

    Calibrated for MX-5 Sport with street tires (195/50R16).

    Expected Performance:
    - Lateral grip: 0.95g cornering (matches real MX-5 skidpad)
    - Braking: 1.15g max deceleration (60-0 mph in 115 ft)
    - Acceleration: 0.57g on rear wheels (RWD)

    Magic Formula: F = D × sin(C × arctan(B×α - E×(B×α - arctan(B×α))))

    Parameters:
    - B: Stiffness factor (initial slope of force curve)
      Higher B = more force across entire slip range
      Typical range: 8-15 for street tires

    - C: Shape factor (affects curve peakiness)
      Standard value for passenger car tires
      Typical range: 1.3-2.5

    - D: Peak friction multiplier
      Combined with normal force to determine max grip
      D=1.0 means peak friction = surface friction
      D>1.0 means tire can exceed surface friction at optimal slip

    - E: Curvature factor (shape near/after peak)
      Controls how sharply grip falls off past optimal slip
      Typical range: 0.9-1.0

    Peak slip characteristics:
    - Lateral: Peak at ~8-10° slip angle (street tire typical)
    - Longitudinal: Peak at ~12-15% slip ratio (street tire typical)
    """
    # Lateral (cornering) parameters
    B_LAT: float = 8.5   # Softer for street tires
    C_LAT: float = 1.9   # Standard shape
    D_LAT: float = 0.95  # Realistic for street tires (~0.95g lateral)
    E_LAT: float = 0.97  # Gradual falloff

    # Longitudinal (accel/brake) parameters
    B_LON: float = 12.0  # Increased for better low-speed grip
    C_LON: float = 1.9   # Standard shape
    D_LON: float = 1.35  # Increased for better traction (~1.35g braking)
    E_LON: float = 0.97  # Gradual falloff


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
    STEER_RATE: float = 3.0  # Steering response rate (rad/s)


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
class PhysicsConfig:
    """
    Complete physics configuration combining all parameter groups.

    Usage:
        config = PhysicsConfig()
        print(config.vehicle.MASS)  # 1062.0
        print(config.pacejka.D_LAT)  # 0.95
    """
    vehicle: VehicleParams = field(default_factory=VehicleParams)
    tire: TireParams = field(default_factory=TireParams)
    pacejka: PacejkaParams = field(default_factory=PacejkaParams)
    drivetrain: DrivetrainParams = field(default_factory=DrivetrainParams)
    aerodynamics: AerodynamicsParams = field(default_factory=AerodynamicsParams)
    steering: SteeringParams = field(default_factory=SteeringParams)
    friction: FrictionParams = field(default_factory=FrictionParams)


# Default physics configuration instance
DEFAULT_PHYSICS_CONFIG = PhysicsConfig()


def get_physics_config():
    """
    Get the default physics configuration.

    Returns:
        PhysicsConfig: Default physics configuration
    """
    return DEFAULT_PHYSICS_CONFIG
