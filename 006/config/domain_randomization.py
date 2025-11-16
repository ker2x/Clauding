"""
Domain Randomization Configuration for CarRacing Environment.

This module provides configurable domain randomization for the car racing
environment to improve policy robustness and generalization.

Domain randomization varies physical parameters, visual properties, and
track characteristics across episodes to prevent overfitting to specific
conditions.

Usage:
    # Create with default settings (disabled)
    config = DomainRandomizationConfig()

    # Enable randomization
    config = DomainRandomizationConfig(enabled=True)

    # Custom randomization ranges
    config = DomainRandomizationConfig(
        enabled=True,
        mass_range=(0.9, 1.1),  # ±10% mass variation
        friction_range=(0.85, 1.15),  # ±15% friction variation
    )
"""

from dataclasses import dataclass, field
from typing import Tuple, Optional
import numpy as np


@dataclass
class VehicleRandomization:
    """
    Vehicle parameter randomization ranges.

    All ranges are specified as (min_multiplier, max_multiplier) applied to base values.
    For example, mass_range=(0.9, 1.1) means mass will vary between 90% and 110% of base.
    """
    # Mass and inertia (affects acceleration, braking, handling)
    mass_range: Tuple[float, float] = (1.0, 1.0)  # Default: no randomization

    # Dimensions (affects load transfer and handling)
    # Note: Changing these significantly may require track adjustments
    wheelbase_range: Tuple[float, float] = (1.0, 1.0)
    track_width_range: Tuple[float, float] = (1.0, 1.0)
    cg_height_range: Tuple[float, float] = (1.0, 1.0)

    # Weight distribution (affects balance)
    # weight_distribution is the fraction of weight on front axle (0.5 = 50/50)
    weight_distribution_range: Tuple[float, float] = (0.5, 0.5)  # Default: 50/50


@dataclass
class TireRandomization:
    """
    Tire parameter randomization ranges.

    These affect grip levels, tire characteristics, and overall vehicle behavior.
    """
    # Pacejka Magic Formula parameters
    # B: Stiffness factor (affects initial slope of force curve)
    pacejka_b_lat_range: Tuple[float, float] = (1.0, 1.0)
    pacejka_b_lon_range: Tuple[float, float] = (1.0, 1.0)

    # D: Peak friction multiplier (affects max grip)
    # This is the main parameter for grip variation
    pacejka_d_lat_range: Tuple[float, float] = (1.0, 1.0)
    pacejka_d_lon_range: Tuple[float, float] = (1.0, 1.0)

    # C: Shape factor (affects curve peakiness)
    pacejka_c_lat_range: Tuple[float, float] = (1.0, 1.0)
    pacejka_c_lon_range: Tuple[float, float] = (1.0, 1.0)

    # E: Curvature factor (affects falloff after peak)
    pacejka_e_lat_range: Tuple[float, float] = (1.0, 1.0)
    pacejka_e_lon_range: Tuple[float, float] = (1.0, 1.0)

    # Rolling resistance
    rolling_resistance_range: Tuple[float, float] = (1.0, 1.0)


@dataclass
class DrivetrainRandomization:
    """
    Drivetrain parameter randomization ranges.

    Affects power delivery and braking performance.
    """
    # Engine power and torque
    engine_power_range: Tuple[float, float] = (1.0, 1.0)
    max_torque_range: Tuple[float, float] = (1.0, 1.0)

    # Braking
    brake_torque_front_range: Tuple[float, float] = (1.0, 1.0)
    brake_torque_rear_range: Tuple[float, float] = (1.0, 1.0)


@dataclass
class AerodynamicsRandomization:
    """
    Aerodynamic parameter randomization ranges.

    Affects drag and top speed.
    """
    # Drag coefficient (affects top speed and high-speed stability)
    drag_coefficient_range: Tuple[float, float] = (1.0, 1.0)

    # Frontal area
    frontal_area_range: Tuple[float, float] = (1.0, 1.0)


@dataclass
class TrackRandomization:
    """
    Track parameter randomization ranges.

    Affects track surface properties and dimensions.
    """
    # Surface friction (affects all tires equally)
    # Note: This multiplies with tire friction parameters
    surface_friction_range: Tuple[float, float] = (1.0, 1.0)

    # Track width (affects difficulty)
    # Note: Changing this may make some tracks impossible
    track_width_range: Tuple[float, float] = (1.0, 1.0)


@dataclass
class VisualRandomization:
    """
    Visual parameter randomization ranges.

    These don't affect physics but help with visual generalization.
    Useful if training with vision-based observations.
    """
    # Track colors (RGB values will be perturbed)
    randomize_track_color: bool = False
    track_color_noise_std: float = 0.0  # Standard deviation for RGB noise

    # Background colors
    randomize_bg_color: bool = False
    bg_color_noise_std: float = 0.0

    # Car color
    randomize_car_color: bool = False


@dataclass
class DomainRandomizationConfig:
    """
    Complete domain randomization configuration.

    Controls which parameters are randomized and by how much.

    Usage Examples:

    1. Conservative randomization (good for initial training):
        config = DomainRandomizationConfig(
            enabled=True,
            vehicle=VehicleRandomization(
                mass_range=(0.95, 1.05),  # ±5% mass
            ),
            tire=TireRandomization(
                pacejka_d_lat_range=(0.95, 1.05),  # ±5% lateral grip
                pacejka_d_lon_range=(0.95, 1.05),  # ±5% longitudinal grip
            ),
        )

    2. Aggressive randomization (for robust policies):
        config = DomainRandomizationConfig(
            enabled=True,
            vehicle=VehicleRandomization(
                mass_range=(0.85, 1.15),  # ±15% mass
                cg_height_range=(0.9, 1.1),  # ±10% CG height
            ),
            tire=TireRandomization(
                pacejka_d_lat_range=(0.8, 1.2),  # ±20% lateral grip
                pacejka_d_lon_range=(0.8, 1.2),  # ±20% longitudinal grip
                pacejka_b_lat_range=(0.9, 1.1),  # ±10% stiffness
                pacejka_b_lon_range=(0.9, 1.1),
            ),
            track=TrackRandomization(
                surface_friction_range=(0.85, 1.15),  # ±15% surface friction
            ),
        )

    3. Realistic weather/surface conditions:
        # Dry conditions
        dry_config = DomainRandomizationConfig(enabled=True)

        # Wet/slippery conditions
        wet_config = DomainRandomizationConfig(
            enabled=True,
            tire=TireRandomization(
                pacejka_d_lat_range=(0.6, 0.8),  # Reduced lateral grip
                pacejka_d_lon_range=(0.7, 0.9),  # Reduced braking grip
            ),
            track=TrackRandomization(
                surface_friction_range=(0.6, 0.8),  # Wet surface
            ),
        )
    """
    # Master enable/disable flag
    enabled: bool = False

    # Randomization seed (None = random seed each time)
    seed: Optional[int] = None

    # Parameter groups
    vehicle: VehicleRandomization = field(default_factory=VehicleRandomization)
    tire: TireRandomization = field(default_factory=TireRandomization)
    drivetrain: DrivetrainRandomization = field(default_factory=DrivetrainRandomization)
    aerodynamics: AerodynamicsRandomization = field(default_factory=AerodynamicsRandomization)
    track: TrackRandomization = field(default_factory=TrackRandomization)
    visual: VisualRandomization = field(default_factory=VisualRandomization)

    # Randomization frequency
    # 'episode': Randomize once per episode (recommended)
    # 'reset': Randomize on every reset
    # 'never': Use fixed parameters
    randomization_frequency: str = 'episode'

    def __post_init__(self):
        """Validate configuration."""
        if self.randomization_frequency not in ['episode', 'reset', 'never']:
            raise ValueError(
                f"Invalid randomization_frequency: {self.randomization_frequency}. "
                "Must be 'episode', 'reset', or 'never'"
            )


# ============================================================================
# Preset Configurations
# ============================================================================

def conservative_randomization() -> DomainRandomizationConfig:
    """
    Conservative randomization for initial training.

    Small variations (±5-10%) to improve generalization without making
    the task too difficult.

    Returns:
        DomainRandomizationConfig with conservative settings
    """
    return DomainRandomizationConfig(
        enabled=True,
        vehicle=VehicleRandomization(
            mass_range=(0.95, 1.05),  # ±5%
            cg_height_range=(0.95, 1.05),  # ±5%
        ),
        tire=TireRandomization(
            pacejka_d_lat_range=(0.95, 1.05),  # ±5% lateral grip
            pacejka_d_lon_range=(0.95, 1.05),  # ±5% longitudinal grip
        ),
        track=TrackRandomization(
            surface_friction_range=(0.95, 1.05),  # ±5%
        ),
    )


def moderate_randomization() -> DomainRandomizationConfig:
    """
    Moderate randomization for intermediate training.

    Medium variations (±10-15%) for good robustness.

    Returns:
        DomainRandomizationConfig with moderate settings
    """
    return DomainRandomizationConfig(
        enabled=True,
        vehicle=VehicleRandomization(
            mass_range=(0.90, 1.10),  # ±10%
            cg_height_range=(0.90, 1.10),  # ±10%
            weight_distribution_range=(0.45, 0.55),  # 45/55 to 55/45
        ),
        tire=TireRandomization(
            pacejka_d_lat_range=(0.90, 1.10),  # ±10% lateral grip
            pacejka_d_lon_range=(0.90, 1.10),  # ±10% longitudinal grip
            pacejka_b_lat_range=(0.95, 1.05),  # ±5% stiffness
            pacejka_b_lon_range=(0.95, 1.05),
        ),
        track=TrackRandomization(
            surface_friction_range=(0.90, 1.10),  # ±10%
        ),
        drivetrain=DrivetrainRandomization(
            engine_power_range=(0.90, 1.10),  # ±10%
            max_torque_range=(0.90, 1.10),  # ±10%
        ),
        aerodynamics=AerodynamicsRandomization(
            drag_coefficient_range=(0.95, 1.05),  # ±5%
        ),
    )


def aggressive_randomization() -> DomainRandomizationConfig:
    """
    Aggressive randomization for maximum robustness.

    Large variations (±15-25%) to train highly robust policies.
    May slow down initial learning.

    Returns:
        DomainRandomizationConfig with aggressive settings
    """
    return DomainRandomizationConfig(
        enabled=True,
        vehicle=VehicleRandomization(
            mass_range=(0.85, 1.15),  # ±15%
            cg_height_range=(0.85, 1.15),  # ±15%
            weight_distribution_range=(0.40, 0.60),  # 40/60 to 60/40
            wheelbase_range=(0.95, 1.05),  # ±5%
            track_width_range=(0.95, 1.05),  # ±5%
        ),
        tire=TireRandomization(
            pacejka_d_lat_range=(0.80, 1.20),  # ±20% lateral grip
            pacejka_d_lon_range=(0.80, 1.20),  # ±20% longitudinal grip
            pacejka_b_lat_range=(0.90, 1.10),  # ±10% stiffness
            pacejka_b_lon_range=(0.90, 1.10),
            pacejka_c_lat_range=(0.95, 1.05),  # ±5% shape
            pacejka_c_lon_range=(0.95, 1.05),
            rolling_resistance_range=(0.90, 1.10),  # ±10%
        ),
        track=TrackRandomization(
            surface_friction_range=(0.85, 1.15),  # ±15%
        ),
        drivetrain=DrivetrainRandomization(
            engine_power_range=(0.85, 1.15),  # ±15%
            max_torque_range=(0.85, 1.15),  # ±15%
            brake_torque_front_range=(0.90, 1.10),  # ±10%
            brake_torque_rear_range=(0.90, 1.10),  # ±10%
        ),
        aerodynamics=AerodynamicsRandomization(
            drag_coefficient_range=(0.90, 1.10),  # ±10%
            frontal_area_range=(0.95, 1.05),  # ±5%
        ),
    )


def wet_surface_conditions() -> DomainRandomizationConfig:
    """
    Simulate wet/slippery surface conditions.

    Reduced grip levels to simulate rain or wet track.
    Useful for training robust policies that work in all conditions.

    Returns:
        DomainRandomizationConfig simulating wet conditions
    """
    return DomainRandomizationConfig(
        enabled=True,
        tire=TireRandomization(
            pacejka_d_lat_range=(0.60, 0.80),  # 60-80% lateral grip (wet tires)
            pacejka_d_lon_range=(0.70, 0.90),  # 70-90% longitudinal grip
            pacejka_b_lat_range=(0.85, 0.95),  # Softer tire response
            pacejka_b_lon_range=(0.85, 0.95),
        ),
        track=TrackRandomization(
            surface_friction_range=(0.60, 0.80),  # Wet surface
        ),
    )


def visual_randomization() -> DomainRandomizationConfig:
    """
    Enable visual randomization for vision-based training.

    Randomizes colors and visual appearance while keeping physics constant.
    Useful for training vision-based policies.

    Returns:
        DomainRandomizationConfig with visual randomization only
    """
    return DomainRandomizationConfig(
        enabled=True,
        visual=VisualRandomization(
            randomize_track_color=True,
            track_color_noise_std=20.0,  # RGB noise std dev
            randomize_bg_color=True,
            bg_color_noise_std=15.0,
            randomize_car_color=True,
        ),
    )
