"""
Suspension configuration system for car dynamics.

Provides multiple suspension models:
- Virtual: Original smoothed acceleration approach (backward compatible)
- Quarter-Car: Per-wheel spring-damper model
- Full: Quarter-car + anti-roll bars

Author: Claude
Date: 2025-01-13
"""

import numpy as np
from typing import Dict, Any


class SuspensionMode:
    """Suspension simulation mode constants."""
    VIRTUAL = 'virtual'          # Original smoothed acceleration
    QUARTER_CAR = 'quarter_car'  # Per-wheel spring-damper
    FULL = 'full'                # Quarter-car + anti-roll bars


class SuspensionConfig:
    """
    Configuration for suspension dynamics.

    Provides preset configurations for different suspension models
    and allows custom parameter tuning.
    """

    @staticmethod
    def get_virtual() -> Dict[str, Any]:
        """
        Original virtual suspension (smoothed acceleration).

        This is the default mode for backward compatibility.
        Parameters are physically interpreted but behavior matches original.

        Returns:
            Configuration dict with virtual suspension parameters
        """
        # Physical interpretation of original parameters:
        # - Natural frequency ~1.4 Hz (typical soft street suspension)
        # - Damping ratio ~0.6 (slightly underdamped)
        # - Roll stiffness factor represents effective anti-roll bar effect

        return {
            'mode': SuspensionMode.VIRTUAL,

            # Suspension response characteristics
            'natural_frequency': 1.5,    # Hz (suspension natural frequency)
            'damping_ratio': 0.6,         # Critical damping ratio

            # Load transfer factors
            'lateral_factor': 0.3,        # Lateral load transfer multiplier
            'longitudinal_factor': None,  # Auto-calculated from CG height

            # Virtual parameters (computed from above)
            'lerp_factor': None,          # Auto-calculated from natural_freq

            # CG height for load transfer calculation
            'cg_height': 0.45,            # meters (MX-5 typical)
        }

    @staticmethod
    def get_quarter_car(preset: str = 'stock') -> Dict[str, Any]:
        """
        Quarter-car spring-damper model.

        Physical suspension with per-wheel springs and dampers.
        Provides realistic suspension travel and load transfer.

        Args:
            preset: 'stock', 'sport', or 'track'

        Returns:
            Configuration dict with quarter-car parameters
        """
        presets = {
            'stock': {
                'spring_rate': 40000.0,      # N/m per wheel (realistic stock)
                'damping': 2500.0,            # N·s/m per wheel
                'description': 'Stock MX-5 suspension (comfortable street)'
            },
            'sport': {
                'spring_rate': 50000.0,      # N/m per wheel (firmer)
                'damping': 3000.0,            # N·s/m per wheel
                'description': 'Sport suspension (balanced street/track)'
            },
            'track': {
                'spring_rate': 65000.0,      # N/m per wheel (stiff)
                'damping': 3500.0,            # N·s/m per wheel
                'description': 'Track suspension (maximum performance)'
            }
        }

        if preset not in presets:
            raise ValueError(f"Unknown preset '{preset}'. Choose from: {list(presets.keys())}")

        params = presets[preset]

        return {
            'mode': SuspensionMode.QUARTER_CAR,

            # Spring and damper parameters
            'spring_rate': params['spring_rate'],    # N/m per wheel
            'damping': params['damping'],             # N·s/m per wheel

            # Geometry
            'ride_height': 0.15,                      # m (static height)
            'max_compression': 0.08,                  # m (bump travel)
            'max_extension': 0.12,                    # m (droop travel)

            # Bump stops (progressive spring at travel limits)
            'bump_stop_stiffness': 100000.0,          # N/m (very stiff)
            'bump_stop_engagement': 0.02,             # m (distance before full travel)

            # Unsprung mass
            'unsprung_mass': 17.0,                    # kg per wheel

            # CG height for load transfer
            'cg_height': 0.45,                        # m

            # Metadata
            'preset': preset,
            'description': params['description']
        }

    @staticmethod
    def get_full(preset: str = 'stock') -> Dict[str, Any]:
        """
        Full suspension with anti-roll bars.

        Extends quarter-car model with anti-roll bars (sway bars)
        for realistic roll stiffness and handling balance tuning.

        Args:
            preset: 'stock', 'sport', 'track', or 'drift'

        Returns:
            Configuration dict with full suspension parameters
        """
        presets = {
            'stock': {
                'spring_rate': 40000.0,
                'damping': 2500.0,
                'arb_front': 25000.0,        # N·m/rad (moderate)
                'arb_rear': 20000.0,         # N·m/rad (softer rear for RWD)
                'description': 'Stock MX-5 (balanced, slight understeer bias)'
            },
            'sport': {
                'spring_rate': 50000.0,
                'damping': 3000.0,
                'arb_front': 32000.0,        # N·m/rad (stiffer)
                'arb_rear': 28000.0,         # N·m/rad
                'description': 'Sport setup (neutral handling)'
            },
            'track': {
                'spring_rate': 65000.0,
                'damping': 3500.0,
                'arb_front': 40000.0,        # N·m/rad (very stiff)
                'arb_rear': 35000.0,         # N·m/rad
                'description': 'Track setup (maximum grip, slight understeer)'
            },
            'drift': {
                'spring_rate': 55000.0,
                'damping': 3200.0,
                'arb_front': 28000.0,        # N·m/rad (stiffer front)
                'arb_rear': 35000.0,         # N·m/rad (stiffer rear = oversteer)
                'description': 'Drift setup (oversteer bias, rear-limited)'
            }
        }

        if preset not in presets:
            raise ValueError(f"Unknown preset '{preset}'. Choose from: {list(presets.keys())}")

        params = presets[preset]

        # Start with quarter-car config
        config = SuspensionConfig.get_quarter_car('stock')  # Get base config

        # Override with full parameters
        config.update({
            'mode': SuspensionMode.FULL,

            # Update spring/damper from preset
            'spring_rate': params['spring_rate'],
            'damping': params['damping'],

            # Anti-roll bar parameters
            'arb_front': params['arb_front'],         # N·m/rad
            'arb_rear': params['arb_rear'],           # N·m/rad

            # Body roll dynamics (optional, for visualization)
            'track_body_roll': True,                  # Enable body roll state

            # Metadata
            'preset': preset,
            'description': params['description']
        })

        return config

    @staticmethod
    def validate_config(config: Dict[str, Any]) -> None:
        """
        Validate suspension configuration.

        Args:
            config: Configuration dict to validate

        Raises:
            ValueError: If configuration is invalid
        """
        mode = config.get('mode')

        if mode not in [SuspensionMode.VIRTUAL, SuspensionMode.QUARTER_CAR, SuspensionMode.FULL]:
            raise ValueError(f"Invalid suspension mode: {mode}")

        if mode == SuspensionMode.VIRTUAL:
            # Check virtual mode parameters
            if 'natural_frequency' not in config or config['natural_frequency'] <= 0:
                raise ValueError("natural_frequency must be positive")

        elif mode in [SuspensionMode.QUARTER_CAR, SuspensionMode.FULL]:
            # Check physical parameters
            required = ['spring_rate', 'damping', 'ride_height', 'max_compression', 'max_extension']
            for param in required:
                if param not in config or config[param] <= 0:
                    raise ValueError(f"{param} must be positive")

            # Check damping is not too high (overdamped)
            k = config['spring_rate']
            c = config['damping']
            m = config.get('unsprung_mass', 17.0)
            damping_ratio = c / (2 * np.sqrt(k * m))
            if damping_ratio > 2.0:
                print(f"Warning: Very high damping ratio ({damping_ratio:.2f}). Suspension may be overdamped.")

        if mode == SuspensionMode.FULL:
            # Check ARB parameters
            if 'arb_front' not in config or config['arb_front'] <= 0:
                raise ValueError("arb_front must be positive")
            if 'arb_rear' not in config or config['arb_rear'] <= 0:
                raise ValueError("arb_rear must be positive")

    @staticmethod
    def compute_derived_params(config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Compute derived parameters from primary configuration.

        For example:
        - Compute lerp_factor from natural_frequency (virtual mode)
        - Compute longitudinal_factor from cg_height
        - Compute damping_ratio from spring_rate and damping

        Args:
            config: Configuration dict

        Returns:
            Updated config with derived parameters
        """
        config = config.copy()  # Don't modify original

        if config['mode'] == SuspensionMode.VIRTUAL:
            # Compute lerp factor from natural frequency
            # For first-order system: lerp = 1 - exp(-2π * freq * dt)
            # Using dt = 0.02 (50 FPS)
            dt = 0.02
            freq = config['natural_frequency']
            config['lerp_factor'] = 1.0 - np.exp(-2 * np.pi * freq * dt)

        # Compute longitudinal factor from CG height and wheelbase
        # (applies to all modes)
        cg_height = config.get('cg_height', 0.45)
        wheelbase = 2.310  # MX-5 wheelbase (should be passed in, but hardcoded for now)
        config['longitudinal_factor'] = cg_height / wheelbase

        if config['mode'] in [SuspensionMode.QUARTER_CAR, SuspensionMode.FULL]:
            # Compute damping ratio
            k = config['spring_rate']
            c = config['damping']
            m = config.get('unsprung_mass', 17.0)
            config['damping_ratio'] = c / (2 * np.sqrt(k * m))

            # Compute natural frequency
            config['natural_frequency'] = np.sqrt(k / m) / (2 * np.pi)  # Hz

        return config


class SuspensionPresets:
    """
    Convenience class for accessing common suspension presets.
    """

    # Virtual (original model)
    VIRTUAL = SuspensionConfig.get_virtual()

    # Quarter-car presets
    QUARTER_CAR_STOCK = SuspensionConfig.get_quarter_car('stock')
    QUARTER_CAR_SPORT = SuspensionConfig.get_quarter_car('sport')
    QUARTER_CAR_TRACK = SuspensionConfig.get_quarter_car('track')

    # Full suspension presets
    FULL_STOCK = SuspensionConfig.get_full('stock')
    FULL_SPORT = SuspensionConfig.get_full('sport')
    FULL_TRACK = SuspensionConfig.get_full('track')
    FULL_DRIFT = SuspensionConfig.get_full('drift')


# Example usage and testing
if __name__ == '__main__':
    print("Suspension Configuration Examples")
    print("=" * 60)

    # Virtual suspension (default)
    virtual = SuspensionConfig.get_virtual()
    virtual = SuspensionConfig.compute_derived_params(virtual)
    print(f"\n1. VIRTUAL (Original Model)")
    print(f"   Mode: {virtual['mode']}")
    print(f"   Natural frequency: {virtual['natural_frequency']:.2f} Hz")
    print(f"   Lerp factor: {virtual['lerp_factor']:.4f}")
    print(f"   Lateral factor: {virtual['lateral_factor']:.2f}")

    # Quarter-car presets
    for preset in ['stock', 'sport', 'track']:
        qc = SuspensionConfig.get_quarter_car(preset)
        qc = SuspensionConfig.compute_derived_params(qc)
        print(f"\n2. QUARTER-CAR ({preset.upper()})")
        print(f"   {qc['description']}")
        print(f"   Spring rate: {qc['spring_rate']:.0f} N/m")
        print(f"   Damping: {qc['damping']:.0f} N·s/m")
        print(f"   Natural frequency: {qc['natural_frequency']:.2f} Hz")
        print(f"   Damping ratio: {qc['damping_ratio']:.2f}")

    # Full suspension presets
    for preset in ['stock', 'sport', 'track', 'drift']:
        full = SuspensionConfig.get_full(preset)
        full = SuspensionConfig.compute_derived_params(full)
        print(f"\n3. FULL SUSPENSION ({preset.upper()})")
        print(f"   {full['description']}")
        print(f"   Spring rate: {full['spring_rate']:.0f} N/m")
        print(f"   ARB Front: {full['arb_front']:.0f} N·m/rad")
        print(f"   ARB Rear: {full['arb_rear']:.0f} N·m/rad")
        print(f"   Front/Rear ARB ratio: {full['arb_front']/full['arb_rear']:.2f}")
