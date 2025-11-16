"""
Pacejka Magic Formula tire model for car dynamics simulation.

This module provides the tire force calculations using the industry-standard
Pacejka Magic Formula. The model separates lateral (cornering) and
longitudinal (traction/braking) forces for realistic tire behavior.

References:
- Pacejka, H. B. (2012). Tire and Vehicle Dynamics. 3rd Edition.
"""

import numpy as np


class PacejkaTire:
    """
    Pacejka Magic Formula tire model for slip-based force calculation.

    Uses separate coefficients for lateral (cornering) and longitudinal (traction/braking) forces.

    Magic Formula:
        F = D × sin(C × arctan(B×α - E×(B×α - arctan(B×α))))

    Where:
        α = slip angle (lateral) or slip ratio (longitudinal)
        B = Stiffness factor (initial slope)
        C = Shape factor (curve peakiness)
        D = Peak friction multiplier
        E = Curvature factor (falloff after peak)

    Default parameters calibrated for 2022 Mazda MX-5 Sport with 195/50R16 street tires.
    See ../TIRE_PARAMETERS.md for validation against real-world performance data.
    """

    def __init__(self,
                 B_lat=8.5, C_lat=1.9, D_lat=0.95, E_lat=0.97,
                 B_lon=8.0, C_lon=1.9, D_lon=1.15, E_lon=0.97):
        """
        Args:
            B_lat: Stiffness factor (Lateral)
            C_lat: Shape factor (Lateral)
            D_lat: Peak friction coefficient (Lateral)
            E_lat: Curvature factor (Lateral)
            B_lon: Stiffness factor (Longitudinal)
            C_lon: Shape factor (Longitudinal)
            D_lon: Peak friction coefficient (Longitudinal)
            E_lon: Curvature factor (Longitudinal)
        """
        self.B_lat = B_lat
        self.C_lat = C_lat
        self.D_lat = D_lat
        self.E_lat = E_lat
        self.B_lon = B_lon
        self.C_lon = C_lon
        self.D_lon = D_lon
        self.E_lon = E_lon

    def lateral_force(self, slip_angle, normal_force, max_friction=1.0):
        """
        Calculate lateral (cornering) force using Pacejka formula.

        Args:
            slip_angle: Angle between wheel direction and velocity direction (rad)
                       Range: -π/2 to +π/2
            normal_force: Vertical load on tire (N)
            max_friction: Surface friction coefficient (default: 1.0 for asphalt)

        Returns:
            Lateral force (N)
        """
        # The slip angle is the difference between the direction a wheel is pointing
        # and the direction it's actually traveling.
        # A slip angle of 90 degrees (or -π/2) means the wheel is sliding purely sideways.
        # There is no physical scenario where the lateral (sideways) slip angle can be
        # greater than 90 degrees. An angle of, for example, 100 degrees would imply the
        # wheel is also rolling backward, which is handled by the longitudinal force model.
        sa = np.clip(slip_angle, -np.pi / 2, np.pi / 2)

        # Use LATERAL coefficients
        arg = self.B_lat * sa   # B_lat is stiffness factor (Lateral)

        # Pacejka Magic Formula for lateral force:
        # F is the lateral force
        # sa is the slip angle (the input variable)
        # D_lat is the peak friction coefficient (Lateral)
        #   This is self.D_lat * normal_force * max_friction.
        #   It scales the maximum possible force (the peak of the sine wave).
        # C_lat is the shape factor (Lateral)
        #   This is np.arctan(arg - self.E_lat * (arg - np.arctan(arg)))
        # E_lat is the curvature factor (Lateral)
        #   This is self.E_lat * (arg - np.arctan(arg)).
        #   It controls the curvature of the force curve near its peak.
        F = (self.D_lat * normal_force * max_friction *
             np.sin(self.C_lat * np.arctan(arg - self.E_lat * (arg - np.arctan(arg)))))
        return F

    def longitudinal_force(self, slip_ratio, normal_force, max_friction=1.0):
        """
        Calculate longitudinal (traction) force using Pacejka formula.

        Args:
            slip_ratio: Difference between wheel speed and ground speed
                       -1.0 = full brake lockup (wheel stopped, car moving)
                        0.0 = perfect grip (wheel speed = ground speed)
                       +1.0 = full wheelspin (wheel spinning, car stationary)
                       Range: -1.0 to +1.0
            normal_force: Vertical load on tire (N)
            max_friction: Surface friction coefficient (default: 1.0 for asphalt)

        Returns:
            Longitudinal force (N)
        """
        sr = np.clip(slip_ratio, -1.0, 1.0)

        # Use LONGITUDINAL coefficients
        # This is the same as the LATERAL force formula but uses slip_ratio
        # instead of slip_angle
        arg = self.B_lon * sr
        F = (self.D_lon * normal_force * max_friction *
             np.sin(self.C_lon * np.arctan(arg - self.E_lon * (arg - np.arctan(arg)))))

        return F
