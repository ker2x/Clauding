"""
Display utilities for CarRacing environment.

This module provides shared utility functions for formatting and displaying
information about the car's state and actions.
"""

import numpy as np


def format_action(action):
    """
    Format continuous action for display.

    Args:
        action: Continuous action [steering, acceleration] or [steering, gas, brake] (legacy)

    Returns:
        Human-readable action description
    """
    # Handle both 2-element and 3-element action formats
    if len(action) == 2:
        steering, accel = action

        # Describe steering
        if steering < -0.3:
            steer_desc = f"LEFT({steering:.2f})"
        elif steering > 0.3:
            steer_desc = f"RIGHT({steering:.2f})"
        else:
            steer_desc = f"STRAIGHT({steering:.2f})"

        # Describe acceleration
        if accel > 0.1:
            pedal_desc = f"GAS({accel:.2f})"
        elif accel < -0.1:
            pedal_desc = f"BRAKE({-accel:.2f})"
        else:
            pedal_desc = "COAST"
    else:
        # Legacy 3-element format: [steering, gas, brake]
        steering, gas, brake = action

        # Describe steering
        if steering < -0.3:
            steer_desc = f"LEFT({steering:.2f})"
        elif steering > 0.3:
            steer_desc = f"RIGHT({steering:.2f})"
        else:
            steer_desc = f"STRAIGHT({steering:.2f})"

        # Describe pedals
        if brake > 0.1:
            pedal_desc = f"BRAKE({brake:.2f})"
        elif gas > 0.1:
            pedal_desc = f"GAS({gas:.2f})"
        else:
            pedal_desc = "COAST"

    return f"{steer_desc} + {pedal_desc}"


def get_car_speed(env):
    """
    Extract car speed from the environment and convert to km/h.

    Args:
        env: CarRacing environment (may be wrapped)

    Returns:
        Speed in km/h
    """
    speed_kmh = 0.0

    if hasattr(env, 'unwrapped') and hasattr(env.unwrapped, 'car'):
        car = env.unwrapped.car
        if car is not None and hasattr(car, 'vx') and hasattr(car, 'vy'):
            # Calculate speed magnitude from velocity components (m/s)
            speed_ms = np.sqrt(car.vx**2 + car.vy**2)
            # Convert to km/h (1 m/s = 3.6 km/h)
            speed_kmh = speed_ms * 3.6

    return speed_kmh
