"""
Simplified suspension configuration for car dynamics.

Single spring-damper configuration for realistic racing simulation.
No body dynamics - load transfer comes directly from suspension forces.

Author: Claude
Date: 2025-01-13
"""

import numpy as np
from typing import Dict, Any


def get_suspension_config() -> Dict[str, Any]:
    """
    Get default suspension configuration.

    Single balanced setup suitable for street/track driving.
    Based on MX-5 Sport suspension characteristics.

    Returns:
        Configuration dict with suspension parameters
    """
    return {
        # Spring and damper parameters (per wheel)
        'spring_rate': 45000.0,      # N/m per wheel (balanced street/track)
        'damping': 1900.0,            # N·s/m per wheel (damping ratio ~0.70)

        # Geometry
        'ride_height': 0.15,          # m (static height above ground)
        'max_compression': 0.08,      # m (bump travel limit)
        'max_extension': 0.12,        # m (droop travel limit)

        # Bump stops (progressive spring at travel limits)
        'bump_stop_stiffness': 150000.0,  # N/m (very stiff, prevent bottoming)

        # Physical parameters
        'unsprung_mass': 17.0,        # kg per wheel (wheel + tire + suspension)
        'track_width': 1.50,          # m (MX-5 front track)
        'wheelbase': 2.310,           # m (MX-5 wheelbase)

        # Derived parameters (computed)
        'natural_frequency': None,    # Hz (computed from spring/mass)
        'damping_ratio': None,        # (computed from damping/spring/mass)
    }


def compute_derived_params(config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Compute derived parameters from primary configuration.

    Calculates:
    - Natural frequency from spring rate and mass
    - Damping ratio from damping coefficient

    Args:
        config: Configuration dict

    Returns:
        Updated config with derived parameters
    """
    config = config.copy()

    # Get parameters
    k = config['spring_rate']
    c = config['damping']
    m = config['unsprung_mass']

    # Natural frequency: f_n = sqrt(k/m) / (2π)
    config['natural_frequency'] = np.sqrt(k / m) / (2 * np.pi)  # Hz

    # Damping ratio: ζ = c / (2 * sqrt(k*m))
    config['damping_ratio'] = c / (2 * np.sqrt(k * m))

    return config


def validate_config(config: Dict[str, Any]) -> None:
    """
    Validate suspension configuration.

    Args:
        config: Configuration dict to validate

    Raises:
        ValueError: If configuration is invalid
    """
    required = ['spring_rate', 'damping', 'ride_height', 'max_compression', 'max_extension']

    for param in required:
        if param not in config:
            raise ValueError(f"Missing required parameter: {param}")
        if config[param] <= 0:
            raise ValueError(f"{param} must be positive, got {config[param]}")

    # Check for reasonable damping
    k = config['spring_rate']
    c = config['damping']
    m = config.get('unsprung_mass', 17.0)
    damping_ratio = c / (2 * np.sqrt(k * m))

    if damping_ratio > 1.5:
        print(f"Warning: High damping ratio ({damping_ratio:.2f}). Suspension may be overdamped.")
    elif damping_ratio < 0.3:
        print(f"Warning: Low damping ratio ({damping_ratio:.2f}). Suspension may oscillate.")


# Example usage
if __name__ == '__main__':
    print("Suspension Configuration")
    print("=" * 60)

    config = get_suspension_config()
    config = compute_derived_params(config)
    validate_config(config)

    print(f"\nSpring rate: {config['spring_rate']:.0f} N/m")
    print(f"Damping: {config['damping']:.0f} N·s/m")
    print(f"Natural frequency: {config['natural_frequency']:.2f} Hz")
    print(f"Damping ratio: {config['damping_ratio']:.2f}")
    print(f"Ride height: {config['ride_height']:.3f} m")
    print(f"Max travel: +{config['max_compression']:.3f}/-{config['max_extension']:.3f} m")
    print(f"\nConfiguration is valid and ready to use.")
