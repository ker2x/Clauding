"""
Pacejka Magic Formula tire model for car dynamics simulation.

This module provides the tire force calculations using the industry-standard
Pacejka Magic Formula. The model separates lateral (cornering) and
longitudinal (traction/braking) forces for realistic tire behavior.

References:
- Pacejka, H. B. (2012). Tire and Vehicle Dynamics. 3rd Edition.
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt


class PacejkaTire:
    """
    Pacejka Magic Formula tire model - FULLY UNIFIED.

    Implements proper traction circle physics where the combined force
    magnitude is limited by the tire's peak friction coefficient:
        sqrt(Fx² + Fy²) ≤ D × Fz

    Magic Formula (same for both lateral and longitudinal):
        F = D × sin(C × arctan(B×α - E×(B×α - arctan(B×α))))

    Where:
        α = slip angle (lateral) or slip ratio (longitudinal)
        B = Stiffness factor (UNIFIED - same for both directions)
        C = Shape factor (UNIFIED - same for both directions)
        D = Peak friction multiplier (UNIFIED - traction circle radius)
        E = Curvature factor (UNIFIED - same for both directions)

    Default parameters calibrated for 2022 Mazda MX-5 Sport with 195/50R16 street tires.
    See ../TIRE_PARAMETERS.md for validation against real-world performance data.
    """

    def __init__(
        self,
        B: float = 8.5,
        C: float = 1.9,
        D: float = 0.95,
        E: float = 0.97,
    ) -> None:
        """
        Args:
            B: Stiffness factor (unified for lateral and longitudinal)
            C: Shape factor (unified for lateral and longitudinal)
            D: Peak friction coefficient (unified - traction circle radius)
            E: Curvature factor (unified for lateral and longitudinal)
        """
        self.B = B
        self.C = C
        self.D = D
        self.E = E

    def lateral_force(
        self, slip_angle: float, normal_force: float, max_friction: float = 1.0
    ) -> float:
        """
        Calculate lateral (cornering) force using Pacejka formula.

        NOTE: This returns the unconstrained lateral force. Use combined_forces()
        to get traction-circle-constrained forces.

        Args:
            slip_angle: Angle between wheel direction and velocity direction (rad)
                       Range: -π/2 to +π/2
            normal_force: Vertical load on tire (N)
            max_friction: Surface friction coefficient (default: 1.0 for asphalt)

        Returns:
            Lateral force (N) - unconstrained by traction circle
        """
        # The slip angle is the difference between the direction a wheel is pointing
        # and the direction it's actually traveling.
        # A slip angle of 90 degrees (or -π/2) means the wheel is sliding purely sideways.
        # There is no physical scenario where the lateral (sideways) slip angle can be
        # greater than 90 degrees. An angle of, for example, 100 degrees would imply the
        # wheel is also rolling backward, which is handled by the longitudinal force model.
        sa = min(np.pi / 2, max(-np.pi / 2, slip_angle))

        # Use UNIFIED coefficients (same as longitudinal)
        arg = self.B * sa

        # Pacejka Magic Formula for lateral force
        F = (self.D * normal_force * max_friction *
             np.sin(self.C * np.arctan(arg - self.E * (arg - np.arctan(arg)))))
        return F

    def longitudinal_force(
        self, slip_ratio: float, normal_force: float, max_friction: float = 1.0
    ) -> float:
        """
        Calculate longitudinal (traction) force using Pacejka formula.

        NOTE: This returns the unconstrained longitudinal force. Use combined_forces()
        to get traction-circle-constrained forces.

        Args:
            slip_ratio: Difference between wheel speed and ground speed
                       -1.0 = full brake lockup (wheel stopped, car moving)
                        0.0 = perfect grip (wheel speed = ground speed)
                       +1.0 = full wheelspin (wheel spinning, car stationary)
                       Range: -1.0 to +1.0
            normal_force: Vertical load on tire (N)
            max_friction: Surface friction coefficient (default: 1.0 for asphalt)

        Returns:
            Longitudinal force (N) - unconstrained by traction circle
        """
        sr = min(1.0, max(-1.0, slip_ratio))

        # Use UNIFIED coefficients (same as lateral)
        arg = self.B * sr
        F = (self.D * normal_force * max_friction *
             np.sin(self.C * np.arctan(arg - self.E * (arg - np.arctan(arg)))))

        return F

    def combined_forces(
        self,
        slip_angle: float,
        slip_ratio: float,
        normal_force: float,
        max_friction: float = 1.0,
    ) -> tuple[float, float]:
        """
        Calculate combined tire forces with traction circle constraint.

        This is the proper way to get realistic tire forces that respect
        the physical limitation: sqrt(Fx² + Fy²) ≤ D × Fz × μ

        Args:
            slip_angle: Lateral slip angle (rad)
            slip_ratio: Longitudinal slip ratio
            normal_force: Vertical load on tire (N)
            max_friction: Surface friction coefficient (default: 1.0)

        Returns:
            Tuple of (fx, fy) - longitudinal and lateral forces (N)
        """
        # Calculate unconstrained forces
        fy_unconstrained = self.lateral_force(slip_angle, normal_force, max_friction)
        fx_unconstrained = self.longitudinal_force(slip_ratio, normal_force, max_friction)

        # Calculate combined force magnitude
        f_combined = np.sqrt(fx_unconstrained**2 + fy_unconstrained**2)

        # Maximum available grip (traction circle radius)
        f_max = self.D * normal_force * max_friction

        # Apply traction circle constraint
        if f_combined > f_max and f_combined > 1e-6:
            # Scale both forces proportionally to fit within circle
            scale = f_max / f_combined
            fx = fx_unconstrained * scale
            fy = fy_unconstrained * scale
        else:
            # Forces are within limit, use unconstrained values
            fx = fx_unconstrained
            fy = fy_unconstrained

        return fx, fy
