"""
Clean 2D top-down car dynamics simulation using Pacejka magic formula tires.

This replaces the Box2D implementation with a more interpretable physics model:
- Bicycle kinematic model for steering
- Pacejka magic formula for tire forces
- Simple numerical integration (RK4)
- No external physics library dependency
"""

import math
import numpy as np


class PacejkaTire:
    """
    Pacejka magic formula tire model for slip-based force calculation.
    Now with separate longitudinal and lateral coefficients.
    """

    def __init__(self,
                 B_lat=12.0, C_lat=1.9, D_lat=1.0, E_lat=0.97,
                 B_lon=10.0, C_lon=1.9, D_lon=1.0, E_lon=0.97):
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
        """
        sa = np.clip(slip_angle, -np.pi / 2, np.pi / 2)

        # Use LATERAL coefficients
        arg = self.B_lat * sa
        F = (self.D_lat * normal_force * max_friction *
             np.sin(self.C_lat * np.arctan(arg - self.E_lat * (arg - np.arctan(arg)))))

        return F

    def longitudinal_force(self, slip_ratio, normal_force, max_friction=1.0):
        """
        Calculate longitudinal (traction) force using Pacejka formula.
        """
        sr = np.clip(slip_ratio, -1.0, 1.0)

        # Use LONGITUDINAL coefficients
        arg = self.B_lon * sr
        F = (self.D_lon * normal_force * max_friction *
             np.sin(self.C_lon * np.arctan(arg - self.E_lon * (arg - np.arctan(arg)))))

        return F

class Car:
    """
    2D top-down bicycle model car with Pacejka tires.

    State vector:
    - Position: (x, y) in world frame
    - Velocity: (vx, vy) in vehicle frame
    - Heading: yaw angle
    - Yaw rate: r = d(yaw)/dt
    - Wheel speeds: [fl, fr, rl, rr] angular velocities
    - Steering angle: delta (front wheels only)
    """

    # Vehicle parameters (Based on 2022 Mazda MX-5 Sport)
    MASS = 1062.0  # Vehicle mass (kg)
    LENGTH = 2.310  # Wheelbase (m)
    WIDTH = 1.50  # Track width (m)

    # 50/50 weight distribution
    LF = LENGTH * 0.5  # Distance from CG to front axle
    LR = LENGTH * 0.5  # Distance from CG to rear axle

    # Tire parameters
    TIRE_RADIUS = 0.30  # Wheel radius (m) (from 195/50R16)
    TIRE_WIDTH = 0.195  # Tire width (m)

    # Pacejka parameters
    # Pacejka parameters
    # We now use separate lateral (cornering) and longitudinal (accel) values
    # Lateral is stiffer (higher B) but with slightly less peak grip (lower D)
    PACEJKA_B_LAT = 12.0  # Lateral stiffness
    PACEJKA_C_LAT = 1.9
    PACEJKA_D_LAT = 1.1  # Lateral peak friction
    PACEJKA_E_LAT = 0.97

    PACEJKA_B_LON = 10.0  # Longitudinal stiffness
    PACEJKA_C_LON = 1.9
    PACEJKA_D_LON = 1.2  # Longitudinal peak friction
    PACEJKA_E_LON = 0.97

    # Drivetrain (2.0L Skyactiv-G)
    ENGINE_POWER = 135000.0  # Power (Watts) (181 hp * 745.7)

    # Derived from ~205 Nm torque in 1st gear (5.09) & final drive (2.87)
    # (205 * 5.09 * 2.87) / 2 wheels = ~1500 Nm torque per wheel
    # Alpha = Torque / Inertia = 1500 / 1.2 = ~1250 rad/s^2
    STARTUP_ACCEL = 1250.0  # Angular acceleration (rad/s^2) for startup

    # Derived from max braking (1.0 G)
    # Alpha = a / r = 9.81 / 0.3 = 32.7 rad/s^2 (at limit)
    # Brakes must be able to lock the wheel, which requires:
    # Max Torque = F_mu * r = (F_z * 1.2) * r = (2600N * 1.2) * 0.3 = ~936 Nm
    # Max Alpha = Torque / Inertia = 936 / 1.2 = ~780 rad/s^2
    BRAKE_ANG_DECEL = 780.0  # Max angular deceleration from brakes (rad/s^2)

    # Estimated: 16" wheel + tire = ~17kg. I = 0.8 * m * r^2 = 0.8 * 17 * 0.3^2 = ~1.2
    INERTIA = 1.2  # Wheel inertia (kg*m^2)

    # Steering
    MAX_STEER_ANGLE = 0.4  # Max steering angle (rad) (~23 degrees)
    STEER_RATE = 3.0  # Steering response rate (rad/s)

    # Friction (surface-dependent, modified per tile)
    BASE_FRICTION = 1.0  # Asphalt (Pacejka D already handles peak friction)
    GRASS_FRICTION = 0.5  # Grass




    def __init__(self, world, init_angle, init_x, init_y):
        """
        Initialize car at position with given heading.

        Args:
            world: Box2D world (kept for compatibility, not used)
            init_angle: Initial heading angle (rad)
            init_x, init_y: Initial position
        """
        # State variables
        self.x = init_x
        self.y = init_y
        self.vx = 0.0  # Longitudinal velocity (body frame)
        self.vy = 0.0  # Lateral velocity (body frame)
        self.yaw = init_angle  # Heading angle
        self.yaw_rate = 0.0  # Angular velocity

        # Wheel states
        self.wheel_omega = np.array([0.0, 0.0, 0.0, 0.0])  # Angular velocity [FL, FR, RL, RR]
        self.steering_angle = 0.0  # Front wheel steering

        # Control inputs
        self._gas = 0.0
        self._brake = 0.0
        self.steer_input = 0.0

        # Wheels contact with track (for friction)
        self.wheels = []
        wheel_pos_local = [
            (self.LF, self.WIDTH / 2),     # FL - (longitudinal, lateral) - left is positive y
            (self.LF, -self.WIDTH / 2),    # FR - right is negative y
            (-self.LR, self.WIDTH / 2),    # RL
            (-self.LR, -self.WIDTH / 2),   # RR
        ]

        cos_yaw = np.cos(init_angle)
        sin_yaw = np.sin(init_angle)

        for i, (dx, dy) in enumerate(wheel_pos_local):
            wheel = type('Wheel', (), {})()
            wheel.tiles = set()
            wheel.omega = 0.0
            wheel.joint = None
            wheel.color = (0, 0, 0)
            wheel.phase = 0.0
            wheel.skid_start = None
            wheel.skid_particle = None

            # Calculate wheel position in world frame
            rx = dx * cos_yaw - dy * sin_yaw
            ry = dx * sin_yaw + dy * cos_yaw
            wheel.position = (init_x + rx, init_y + ry)

            self.wheels.append(wheel)

        # Tire model
        self.tire = PacejkaTire(
            B_lat=self.PACEJKA_B_LAT,
            C_lat=self.PACEJKA_C_LAT,
            D_lat=self.PACEJKA_D_LAT,
            E_lat=self.PACEJKA_E_LAT,
            B_lon=self.PACEJKA_B_LON,
            C_lon=self.PACEJKA_C_LON,
            D_lon=self.PACEJKA_D_LON,
            E_lon=self.PACEJKA_E_LON
        )

        # For rendering
        self.hull = type('Body', (), {})()
        self.hull.position = (init_x, init_y)
        self.hull.angle = init_angle
        self.hull.linearVelocity = (0, 0)
        self.hull.angularVelocity = 0.0
        self.hull.color = (0.8, 0.0, 0.0)

        self.fuel_spent = 0.0
        self.drawlist = self.wheels + [self.hull]
        self.particles = []

    def gas(self, throttle):
        """Apply throttle (0 to 1)."""
        self._gas = np.clip(throttle, 0, 1)

    def brake(self, brake_force):
        """Apply brake (0 to 1)."""
        self._brake = np.clip(brake_force, 0, 1)

    def steer(self, steer_input):
        """
        Apply steering input (-1 to 1).
        Smoothly ramps to target angle.
        """
        self.steer_input = np.clip(steer_input, -1, 1)

    def step(self, dt):
        """
        Integrate dynamics forward by dt seconds using RK4.
        """
        # Smooth steering angle toward input
        target_angle = self.steer_input * self.MAX_STEER_ANGLE
        angle_diff = target_angle - self.steering_angle
        angle_diff = np.clip(angle_diff, -self.STEER_RATE * dt, self.STEER_RATE * dt)
        self.steering_angle += angle_diff

        # Update wheel angular velocities
        self._update_wheel_dynamics(dt)

        # Compute tire forces
        tire_friction = self._get_surface_friction()
        forces = self._compute_tire_forces(tire_friction)

        # Integrate state with RK4
# dead code
#        k1 = self._state_derivative(forces)
#        k2 = self._state_derivative(forces)
#        k3 = self._state_derivative(forces)
#        k4 = self._state_derivative(forces)
        # Simple forward Euler (RK4 is overkill for this model)
        # CAPTURE THE RETURNED DEBUG INFO
        integration_results = self._integrate_state(forces, dt)

        # Update hull for rendering/compatibility
        self._update_hull()

        # RETURN DEBUG INFO
        integration_results['tire_forces'] = forces
        return integration_results

    def _update_wheel_dynamics(self, dt):
        """Update wheel angular velocities based on engine and brake."""
        for i in range(4):
            wheel = self.wheels[i]
            is_rear = (i >= 2)

            # Calculate wheel's longitudinal velocity at the contact patch
            if i < 2:  # Front wheels
                y_pos = self.WIDTH / 2 if i == 0 else -self.WIDTH / 2
            else:  # Rear wheels
                y_pos = self.WIDTH / 2 if i == 2 else -self.WIDTH / 2
            wheel_vx = self.vx - self.yaw_rate * y_pos

            # --- REVISED LOGIC ---
            # Priority: 1. Brakes, 2. Engine, 3. Free-Rolling

            # 1. Apply Brakes (highest priority)
            if self._brake >= 0.9:
                wheel.omega = 0.0
            elif self._brake > 0:
                # Use our brake constant
                brake_alpha = self.BRAKE_ANG_DECEL * self._brake
                sign = -np.sign(wheel.omega) if wheel.omega != 0 else 0

                # Calculate the change in angular velocity for this time step
                delta_omega = brake_alpha * dt

                # Clamp the *change* so it doesn't "overshoot"
                if delta_omega > abs(wheel.omega):
                    wheel.omega = 0.0
                else:
                    wheel.omega += sign * delta_omega

            # 2. Apply Engine (if not braking and is a driven wheel)
            elif is_rear and self._gas > 0.0:
                # Rear wheel drive: constant power model
                if abs(wheel.omega) > 0.1:
                    # Constant power mode: P = τ * ω
                    torque = self.ENGINE_POWER * self._gas / abs(wheel.omega)
                    accel = torque / self.INERTIA
                    wheel.omega += accel * dt
                else:
                    # At low speeds: use a strong startup angular acceleration
                    wheel.omega += (dt * self.STARTUP_ACCEL * self._gas)

            # 3. Free Rolling (if not braking and not engine-driven)
            else:
                # Wheel is free-rolling. Its angular velocity should match the
                # longitudinal velocity of the contact patch.
                # This results in a slip_ratio of 0.
                wheel.omega = wheel_vx / self.TIRE_RADIUS

            # Sync to wheel_omega array for consistency
            self.wheel_omega[i] = wheel.omega

            # Update phase (rotation angle)
            wheel.phase += wheel.omega * dt

    def _get_surface_friction(self):
        """Get average friction coefficient from wheels' contact surfaces."""
        friction = self.BASE_FRICTION
        contact_count = 0

        for wheel in self.wheels:
            if len(wheel.tiles) > 0:
                # Average friction from all tiles this wheel touches
                tile_friction = np.mean([
                    tile.road_friction if hasattr(tile, 'road_friction') else self.BASE_FRICTION
                    for tile in wheel.tiles
                ])
                friction += tile_friction
                contact_count += 1

        if contact_count > 0:
            friction = friction / (contact_count + 1)
        else:
            friction = self.GRASS_FRICTION

        return friction

    def _compute_tire_forces(self, friction):
        """
        Compute tire forces using Pacejka model.

        Returns dict with forces for each wheel.
        """
        # Load distribution (simple: equal weight per wheel)
        weight_per_wheel = (self.MASS * 9.81) / 4.0  # Force in Newtons

        # --- Corrected Lateral Load Transfer ---
        # Calculate centripetal acceleration (a = v * ω)
        # The sign matters here: positive yaw_rate (left turn) = positive accel (rightward force)
        lateral_accel = self.vx * self.yaw_rate

        # Simplified load transfer: dF_z = (a_y * h * m) / t
        # h = CG height (est. 0.5m), t = track width (self.WIDTH)
        # We'll use a simplified, tuned factor
        # A positive lateral_accel (left turn) moves load to the RIGHT wheels (FR, RR)
        load_transfer_factor = 0.5  # Tuned factor
        lateral_load_transfer = load_transfer_factor * self.MASS * lateral_accel
        # ----------------------------------------

        # Compute longitudinal and lateral slip for each wheel
        forces = {}

        for i in range(4):
            wheel = self.wheels[i]

            # Wheel position relative to CG
            if i < 2:  # Front wheels
                dist_cg = self.LF
                y_pos = self.WIDTH / 2 if i == 0 else -self.WIDTH / 2
                steer_ang = self.steering_angle
            else:  # Rear wheels
                dist_cg = -self.LR
                y_pos = self.WIDTH / 2 if i == 2 else -self.WIDTH / 2
                steer_ang = 0.0

            # Velocity at wheel contact point (body frame)
            # Correct rotation: v = v_center + omega × r
            # In 2D: v_x = v_center_x - omega * y, v_y = v_center_y + omega * x
            wheel_vx = self.vx - self.yaw_rate * y_pos
            wheel_vy = self.vy + self.yaw_rate * dist_cg

            # Slip angle (angle between tire heading and velocity)
            v_mag = np.sqrt(wheel_vx ** 2 + wheel_vy ** 2)
            if v_mag > 0.5:
                slip_angle = np.arctan2(wheel_vy, wheel_vx + 1e-6) - steer_ang
            else:
                # When stationary, no slip angle - prevents rotation when not moving
                slip_angle = 0.0

            # Slip ratio (wheel vs ground)
            wheel_linear_vel = wheel.omega * self.TIRE_RADIUS

            # Use a stable denominator: the maximum of the two speeds.
            # This prevents division by zero and numerical instability at low speeds.
            denom = max(abs(wheel_vx), abs(wheel_linear_vel))

            if denom < 0.1:  # If both speeds are near zero, there is no slip
                slip_ratio = 0.0
            else:
                # Standard slip ratio definition
                slip_ratio = (wheel_linear_vel - wheel_vx) / denom

            # === CORRECTED NORMAL FORCE BLOCK ===
            # Start with base weight
            normal_force = weight_per_wheel

            # Apply LATERAL load transfer
            if i == 0 or i == 2:  # Left wheels (FL, RL)
                normal_force -= lateral_load_transfer / 2
            else:  # Right wheels (FR, RR)
                normal_force += lateral_load_transfer / 2

            normal_force = max(50.0, normal_force)  # Prevent negative/zero load
            # ====================================

            # Tire forces
            # Note: Negate lateral force because positive slip angle (velocity left of wheel heading)
            # should produce force to the right (negative) to correct the slip
            fy = -self.tire.lateral_force(slip_angle, normal_force, friction)
            # USE STATIC weight for Fx (simulates differential)
            # USE DYNAMIC normal_force for Fy (correctly models grip)
            fx = self.tire.longitudinal_force(slip_ratio, normal_force, friction)
            #fx = self.tire.longitudinal_force(slip_ratio, weight_per_wheel, friction)

            forces[i] = {
                'fx': fx,
                'fy': fy,
                'steer': steer_ang,
                'slip_angle': slip_angle,
                'slip_ratio': slip_ratio,
                'normal_force': normal_force,
            }

        return forces
#    def _state_derivative(self, forces):
#        """Compute state derivatives (not used in simple Euler, kept for structure)."""
#        return {}

    def _integrate_state(self, forces, dt):
        """
        Integrate vehicle state forward using forces.
        Uses simple Euler integration.
        """
        # Sum forces and torques
        fx_total = 0.0
        fy_total = 0.0
        torque = 0.0

        for i, f in forces.items():
            steer = f['steer']

            # Rotate forces to body frame
            fx = f['fx'] * np.cos(steer) - f['fy'] * np.sin(steer)
            fy = f['fx'] * np.sin(steer) + f['fy'] * np.cos(steer)

            fx_total += fx
            fy_total += fy

            # Get wheel position
            if i < 2:  # Front wheels
                dist_cg = self.LF
                y_pos = self.WIDTH / 2 if i == 0 else -self.WIDTH / 2
            else:  # Rear wheels
                dist_cg = -self.LR
                y_pos = self.WIDTH / 2 if i == 2 else -self.WIDTH / 2

            # Torque from forces (using body frame forces, not tire frame)
            # τ = r × F = (x, y) × (fx, fy) = x*fy - y*fx
            torque += dist_cg * fy - y_pos * fx

        # --- ADD DRAG FORCES HERE ---
        # (After the for loop)
        C_drag = 0.003
        fx_drag = -C_drag * self.vx * abs(self.vx)
        C_roll = 0.05
        fx_roll = -C_roll * self.vx

        fx_total += fx_drag + fx_roll
#        fx_total += fx_drag #+ fx_roll

        # --- END DRAG BLOCK ---

        # Compute accelerations
        ax = fx_total / self.MASS
        ay = fy_total / self.MASS

        # Moment of inertia (rough estimate)
        Iz = self.MASS * (self.LENGTH**2 / 12 + self.WIDTH**2 / 12)
        ang_accel = torque / Iz

        # Update velocity in body frame
        # (Note: this is a simplification - proper handling requires transformation)
#        self.vx += ax * dt
#        self.vy += ay * dt
# CORRECTED integration for rotating reference frame
        self.vx += (ax - self.vy * self.yaw_rate) * dt
        self.vy += (ay + self.vx * self.yaw_rate) * dt

        # Limit velocity (drag effect)
        v_mag = np.sqrt(self.vx**2 + self.vy**2)
        if v_mag > 30.0:
            scale = 30.0 / v_mag
            self.vx *= scale
            self.vy *= scale

        # Update rotation
        self.yaw_rate += ang_accel * dt
        self.yaw += self.yaw_rate * dt

        # Update position (in world frame)
        cos_yaw = np.cos(self.yaw)
        sin_yaw = np.sin(self.yaw)

        self.x += (self.vx * cos_yaw - self.vy * sin_yaw) * dt
        self.y += (self.vx * sin_yaw + self.vy * cos_yaw) * dt

        # ADD THIS RETURN STATEMENT
        return {
            'fx_total': fx_total, 'fy_total': fy_total, 'torque': torque,
            'ax': ax, 'ay': ay, 'ang_accel': ang_accel
        }

    def _update_hull(self):
        """Update hull position/velocity for rendering and compatibility."""
        self.hull.position = (self.x, self.y)
        self.hull.angle = self.yaw

        # Compute world frame velocity
        cos_yaw = np.cos(self.yaw)
        sin_yaw = np.sin(self.yaw)

        vx_world = self.vx * cos_yaw - self.vy * sin_yaw
        vy_world = self.vx * sin_yaw + self.vy * cos_yaw

        self.hull.linearVelocity = (vx_world, vy_world)
        self.hull.angularVelocity = self.yaw_rate

        # Update wheel states for rendering
        wheel_pos_local = [
            (self.LF, self.WIDTH / 2),  # FL
            (self.LF, -self.WIDTH / 2),  # FR
            (-self.LR, self.WIDTH / 2),  # RL
            (-self.LR, -self.WIDTH / 2),  # RR
        ]

        # !! DEFINE cos_yaw and sin_yaw HERE !!
        # This was the missing piece.
        cos_yaw = np.cos(self.yaw)
        sin_yaw = np.sin(self.yaw)

        for i, (wheel, (dx, dy)) in enumerate(zip(self.wheels, wheel_pos_local)):
            # Rotate position to world frame
            rx = dx * cos_yaw - dy * sin_yaw
            ry = dx * sin_yaw + dy * cos_yaw

            wheel.position = (self.x + rx, self.y + ry)
            wheel.omega = self.wheel_omega[i]
            wheel.phase += wheel.omega * 0.02  # Approximate dt

    def draw(self, surface, zoom, translation, angle, draw_particles=True):
        """
        Draw car and particles (compatible with original interface).
        This is a simplified version - full drawing would need pygame.
        """
        # For now, just a placeholder to maintain interface compatibility
        # The actual rendering happens in car_racing.py using pygame
        pass

    def destroy(self):
        """Clean up (no resources to clean up in this implementation)."""
        pass
