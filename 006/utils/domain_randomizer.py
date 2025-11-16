"""
Domain Randomizer Utility for CarRacing Environment.

This module handles the actual randomization of environment parameters
based on DomainRandomizationConfig specifications.

The DomainRandomizer class:
1. Stores base/default parameters
2. Samples random variations based on config
3. Applies randomized parameters to environment components

Usage:
    from config.domain_randomization import conservative_randomization
    from utils.domain_randomizer import DomainRandomizer

    # Create randomizer with config
    config = conservative_randomization()
    randomizer = DomainRandomizer(config)

    # Randomize environment on reset
    env = CarRacing()
    randomized_params = randomizer.randomize()

    # Apply to environment (see integrate_with_env() for integration pattern)
"""

from typing import Dict, Any, Optional
import numpy as np
from config.domain_randomization import DomainRandomizationConfig
from config.physics_config import PhysicsConfig


class DomainRandomizer:
    """
    Handles domain randomization for the CarRacing environment.

    This class samples random parameter variations based on the
    DomainRandomizationConfig and provides methods to apply them
    to environment components.
    """

    def __init__(
        self,
        config: DomainRandomizationConfig,
        base_physics: Optional[PhysicsConfig] = None,
    ):
        """
        Initialize domain randomizer.

        Args:
            config: Domain randomization configuration
            base_physics: Base physics parameters (default: create new PhysicsConfig)
        """
        self.config = config
        self.base_physics = base_physics or PhysicsConfig()

        # Random number generator (for reproducibility if seed is set)
        self.rng = np.random.RandomState(config.seed)

        # Track randomization state
        self.current_params: Optional[Dict[str, Any]] = None
        self.randomization_count = 0

    def randomize(self) -> Dict[str, Any]:
        """
        Sample new random parameters based on configuration.

        Returns:
            Dictionary of randomized parameters ready to apply to environment
        """
        if not self.config.enabled:
            # Return base parameters (no randomization)
            return self._get_base_params()

        # Sample random multipliers for each parameter group
        params = {}

        # Vehicle parameters
        params['vehicle'] = self._randomize_vehicle()

        # Tire parameters
        params['tire'] = self._randomize_tire()

        # Drivetrain parameters
        params['drivetrain'] = self._randomize_drivetrain()

        # Aerodynamics parameters
        params['aerodynamics'] = self._randomize_aerodynamics()

        # Track parameters
        params['track'] = self._randomize_track()

        self.current_params = params
        self.randomization_count += 1

        return params

    def _randomize_vehicle(self) -> Dict[str, float]:
        """Randomize vehicle parameters."""
        cfg = self.config.vehicle
        base = self.base_physics.vehicle

        # Sample mass
        mass = base.MASS * self._sample_range(cfg.mass_range)

        # Sample dimensions
        length = base.LENGTH * self._sample_range(cfg.wheelbase_range)
        width = base.WIDTH * self._sample_range(cfg.track_width_range)
        cg_height = base.CG_HEIGHT * self._sample_range(cfg.cg_height_range)

        # Sample weight distribution (fraction on front axle)
        weight_dist = self._sample_range(cfg.weight_distribution_range)

        # Calculate LF and LR from weight distribution
        # weight_dist = LR / (LF + LR)
        # LF + LR = length
        # => LR = weight_dist * length
        # => LF = (1 - weight_dist) * length
        lf = (1.0 - weight_dist) * length
        lr = weight_dist * length

        return {
            'MASS': mass,
            'LENGTH': length,
            'WIDTH': width,
            'LF': lf,
            'LR': lr,
            'CG_HEIGHT': cg_height,
        }

    def _randomize_tire(self) -> Dict[str, float]:
        """Randomize tire parameters."""
        cfg = self.config.tire
        base_pacejka = self.base_physics.pacejka
        base_tire = self.base_physics.tire

        return {
            # Pacejka parameters
            'PACEJKA_B_LAT': base_pacejka.B_LAT * self._sample_range(cfg.pacejka_b_lat_range),
            'PACEJKA_B_LON': base_pacejka.B_LON * self._sample_range(cfg.pacejka_b_lon_range),
            'PACEJKA_C_LAT': base_pacejka.C_LAT * self._sample_range(cfg.pacejka_c_lat_range),
            'PACEJKA_C_LON': base_pacejka.C_LON * self._sample_range(cfg.pacejka_c_lon_range),
            'PACEJKA_D_LAT': base_pacejka.D_LAT * self._sample_range(cfg.pacejka_d_lat_range),
            'PACEJKA_D_LON': base_pacejka.D_LON * self._sample_range(cfg.pacejka_d_lon_range),
            'PACEJKA_E_LAT': base_pacejka.E_LAT * self._sample_range(cfg.pacejka_e_lat_range),
            'PACEJKA_E_LON': base_pacejka.E_LON * self._sample_range(cfg.pacejka_e_lon_range),
            # Rolling resistance (from aerodynamics in base config)
            'C_ROLL_RESISTANCE': self.base_physics.aerodynamics.C_ROLL_RESISTANCE * self._sample_range(
                cfg.rolling_resistance_range
            ),
        }

    def _randomize_drivetrain(self) -> Dict[str, float]:
        """Randomize drivetrain parameters."""
        cfg = self.config.drivetrain
        base = self.base_physics.drivetrain

        return {
            'ENGINE_POWER': base.ENGINE_POWER * self._sample_range(cfg.engine_power_range),
            'MAX_TORQUE_PER_WHEEL': base.MAX_TORQUE_PER_WHEEL * self._sample_range(cfg.max_torque_range),
            'MAX_BRAKE_TORQUE_FRONT': base.MAX_BRAKE_TORQUE_FRONT * self._sample_range(
                cfg.brake_torque_front_range
            ),
            'MAX_BRAKE_TORQUE_REAR': base.MAX_BRAKE_TORQUE_REAR * self._sample_range(
                cfg.brake_torque_rear_range
            ),
            # Power transition omega scales with torque
            'POWER_TRANSITION_OMEGA': base.POWER_TRANSITION_OMEGA * self._sample_range(
                cfg.max_torque_range
            ),
        }

    def _randomize_aerodynamics(self) -> Dict[str, float]:
        """Randomize aerodynamic parameters."""
        cfg = self.config.aerodynamics
        base = self.base_physics.aerodynamics

        return {
            'CD_CAR': base.CD_CAR * self._sample_range(cfg.drag_coefficient_range),
            'FRONTAL_AREA': base.FRONTAL_AREA * self._sample_range(cfg.frontal_area_range),
        }

    def _randomize_track(self) -> Dict[str, float]:
        """Randomize track parameters."""
        cfg = self.config.track

        return {
            'surface_friction_multiplier': self._sample_range(cfg.surface_friction_range),
            'track_width_multiplier': self._sample_range(cfg.track_width_range),
        }

    def _sample_range(self, range_tuple: tuple) -> float:
        """
        Sample a value uniformly from a range.

        Args:
            range_tuple: (min, max) range

        Returns:
            Sampled value in range
        """
        min_val, max_val = range_tuple
        return self.rng.uniform(min_val, max_val)

    def _get_base_params(self) -> Dict[str, Any]:
        """Get base parameters (no randomization)."""
        base = self.base_physics

        return {
            'vehicle': {
                'MASS': base.vehicle.MASS,
                'LENGTH': base.vehicle.LENGTH,
                'WIDTH': base.vehicle.WIDTH,
                'LF': base.vehicle.LF,
                'LR': base.vehicle.LR,
                'CG_HEIGHT': base.vehicle.CG_HEIGHT,
            },
            'tire': {
                'PACEJKA_B_LAT': base.pacejka.B_LAT,
                'PACEJKA_B_LON': base.pacejka.B_LON,
                'PACEJKA_C_LAT': base.pacejka.C_LAT,
                'PACEJKA_C_LON': base.pacejka.C_LON,
                'PACEJKA_D_LAT': base.pacejka.D_LAT,
                'PACEJKA_D_LON': base.pacejka.D_LON,
                'PACEJKA_E_LAT': base.pacejka.E_LAT,
                'PACEJKA_E_LON': base.pacejka.E_LON,
                'C_ROLL_RESISTANCE': base.aerodynamics.C_ROLL_RESISTANCE,
            },
            'drivetrain': {
                'ENGINE_POWER': base.drivetrain.ENGINE_POWER,
                'MAX_TORQUE_PER_WHEEL': base.drivetrain.MAX_TORQUE_PER_WHEEL,
                'MAX_BRAKE_TORQUE_FRONT': base.drivetrain.MAX_BRAKE_TORQUE_FRONT,
                'MAX_BRAKE_TORQUE_REAR': base.drivetrain.MAX_BRAKE_TORQUE_REAR,
                'POWER_TRANSITION_OMEGA': base.drivetrain.POWER_TRANSITION_OMEGA,
            },
            'aerodynamics': {
                'CD_CAR': base.aerodynamics.CD_CAR,
                'FRONTAL_AREA': base.aerodynamics.FRONTAL_AREA,
            },
            'track': {
                'surface_friction_multiplier': 1.0,
                'track_width_multiplier': 1.0,
            },
        }

    def apply_to_car(self, car, params: Dict[str, Any]) -> None:
        """
        Apply randomized parameters to a Car instance.

        Args:
            car: Car instance to modify
            params: Randomized parameters from randomize()
        """
        # Apply vehicle parameters
        for key, value in params['vehicle'].items():
            if hasattr(car, key):
                setattr(car, key, value)

        # Apply tire parameters (Pacejka)
        for key, value in params['tire'].items():
            if hasattr(car, key):
                setattr(car, key, value)

        # Update tire model with new Pacejka parameters
        if hasattr(car, 'tire'):
            car.tire.B_lat = params['tire']['PACEJKA_B_LAT']
            car.tire.B_lon = params['tire']['PACEJKA_B_LON']
            car.tire.C_lat = params['tire']['PACEJKA_C_LAT']
            car.tire.C_lon = params['tire']['PACEJKA_C_LON']
            car.tire.D_lat = params['tire']['PACEJKA_D_LAT']
            car.tire.D_lon = params['tire']['PACEJKA_D_LON']
            car.tire.E_lat = params['tire']['PACEJKA_E_LAT']
            car.tire.E_lon = params['tire']['PACEJKA_E_LON']

        # Apply drivetrain parameters
        for key, value in params['drivetrain'].items():
            if hasattr(car, key):
                setattr(car, key, value)

        # Apply aerodynamics parameters
        for key, value in params['aerodynamics'].items():
            if hasattr(car, key):
                setattr(car, key, value)

        # Apply rolling resistance
        if 'C_ROLL_RESISTANCE' in params['tire']:
            car.C_ROLL_RESISTANCE = params['tire']['C_ROLL_RESISTANCE']

    def apply_to_track(self, env, params: Dict[str, Any]) -> None:
        """
        Apply randomized track parameters to environment.

        Args:
            env: CarRacing environment instance
            params: Randomized parameters from randomize()
        """
        track_params = params['track']

        # Apply surface friction multiplier to all road tiles
        if hasattr(env, 'road') and env.road:
            friction_mult = track_params['surface_friction_multiplier']
            for tile in env.road:
                if hasattr(tile, 'road_friction'):
                    # Multiply base friction by randomized multiplier
                    tile.road_friction *= friction_mult

        # Note: Track width randomization would require regenerating the track
        # or scaling existing geometry. This is more complex and may be added later.

    def get_info_dict(self) -> Dict[str, Any]:
        """
        Get information about current randomization state.

        Returns:
            Dictionary with randomization info (useful for logging)
        """
        if self.current_params is None:
            return {'domain_randomization_enabled': self.config.enabled}

        # Extract key parameters for logging
        info = {
            'domain_randomization_enabled': self.config.enabled,
            'randomization_count': self.randomization_count,
        }

        if self.config.enabled:
            # Add key physics parameters
            info.update({
                'mass': self.current_params['vehicle']['MASS'],
                'pacejka_d_lat': self.current_params['tire']['PACEJKA_D_LAT'],
                'pacejka_d_lon': self.current_params['tire']['PACEJKA_D_LON'],
                'surface_friction': self.current_params['track']['surface_friction_multiplier'],
                'engine_power': self.current_params['drivetrain']['ENGINE_POWER'],
            })

        return info

    def reset_seed(self, seed: Optional[int] = None) -> None:
        """
        Reset the random seed.

        Args:
            seed: New seed (None for random)
        """
        self.rng = np.random.RandomState(seed)
        self.config.seed = seed
