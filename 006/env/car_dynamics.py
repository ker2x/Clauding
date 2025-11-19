"""
Clean 2D top-down car dynamics simulation using Pacejka magic formula tires.

This module replaces the Box2D implementation with a clean, interpretable physics model:
- Pacejka Magic Formula for tire forces (industry standard)
- Rigid-body load transfer model with filtered accelerations
- Realistic drivetrain (RWD) and braking (60/40 bias)
- No external physics library dependency

Vehicle Model: 2022 Mazda MX-5 Sport (ND)
- Mass: 1062 kg, 50/50 weight distribution
- Tires: 195/50R16 street tires (Bridgestone Potenza RE050A equivalent)
- Expected Performance:
  * Lateral: 0.95g cornering (matches real MX-5 skidpad data)
  * Braking: 1.15g deceleration (matches 60-0 mph in 115 ft)
  * Acceleration: 0.57g on rear wheels (RWD, realistic)

Load Transfer Model:
- Uses rigid-body approximation with filtered accelerations
- Longitudinal: Front/rear weight transfer during braking/acceleration
- Lateral: Left/right weight transfer during cornering
- Low-pass filter (alpha=0.15) prevents oscillations

Classes:
- Car: Full vehicle dynamics with 4 wheels, drivetrain, and braking

See Also:
- tire_model.py: Pacejka Magic Formula tire model (extracted)
- ../TIRE_PARAMETERS.md: Detailed calibration and validation
- car_racing.py: Gymnasium environment wrapper

References:
- Pacejka, H. B. (2012). Tire and Vehicle Dynamics. 3rd Edition.
"""

from __future__ import annotations

from typing import Any
import numpy as np
import numpy.typing as npt
from env.tire_model import PacejkaTire
from config.physics_config import PhysicsConfig
from env.mx5_powertrain import MX5Powertrain


class Car:
    """
    2D top-down model car with Pacejka tires.

    State vector:
    - Position: (x, y) in world frame
    - Velocity: (vx, vy) in vehicle frame
    - Heading: yaw angle
    - Yaw rate: r = d(yaw)/dt
    - Wheel speeds: [fl, fr, rl, rr] angular velocities
    - Steering angle: delta (front wheels only)

    NOTE: All physics parameters are now loaded from config.physics_config.PhysicsConfig
    instead of being hardcoded as class constants. This enables:
    - Domain randomization
    - Easy parameter tuning
    - Single source of truth for all physics values

    See config/physics_config.py for parameter documentation and default values.
    """

    def __init__(
        self,
        world: Any,
        init_angle: float,
        init_x: float,
        init_y: float,
        physics_config: PhysicsConfig | None = None,
    ) -> None:
        """
        Initialize car at position with given heading.

        Args:
            world: Box2D world (kept for compatibility, not used)
            init_angle: Initial heading angle (rad)
            init_x, init_y: Initial position
            physics_config: PhysicsConfig instance (defaults to PhysicsConfig())
        """
        # Load physics configuration
        if physics_config is None:
            physics_config = PhysicsConfig()

        # Store configuration for later use
        self._physics_config = physics_config

        # Load all physics parameters from config
        # Vehicle parameters
        self.MASS = physics_config.vehicle.MASS
        self.LENGTH = physics_config.vehicle.LENGTH
        self.WIDTH = physics_config.vehicle.WIDTH
        self.LF = physics_config.vehicle.LF
        self.LR = physics_config.vehicle.LR
        self.CG_HEIGHT = physics_config.vehicle.CG_HEIGHT

        # Tire parameters
        self.TIRE_RADIUS = physics_config.tire.TIRE_RADIUS
        self.TIRE_WIDTH = physics_config.tire.TIRE_WIDTH
        self.INERTIA = physics_config.tire.INERTIA

        # Pacejka parameters
        self.PACEJKA_B_LAT = physics_config.pacejka.B_LAT
        self.PACEJKA_C_LAT = physics_config.pacejka.C_LAT
        self.PACEJKA_D_LAT = physics_config.pacejka.D_LAT
        self.PACEJKA_E_LAT = physics_config.pacejka.E_LAT
        self.PACEJKA_B_LON = physics_config.pacejka.B_LON
        self.PACEJKA_C_LON = physics_config.pacejka.C_LON
        self.PACEJKA_D_LON = physics_config.pacejka.D_LON
        self.PACEJKA_E_LON = physics_config.pacejka.E_LON

        # Drivetrain parameters
        self.ENGINE_POWER = physics_config.drivetrain.ENGINE_POWER
        self.MAX_TORQUE_PER_WHEEL = physics_config.drivetrain.MAX_TORQUE_PER_WHEEL
        self.POWER_TRANSITION_OMEGA = physics_config.drivetrain.POWER_TRANSITION_OMEGA
        self.STARTUP_ACCEL = physics_config.drivetrain.STARTUP_ACCEL
        self.MAX_BRAKE_TORQUE_FRONT = physics_config.drivetrain.MAX_BRAKE_TORQUE_FRONT
        self.MAX_BRAKE_TORQUE_REAR = physics_config.drivetrain.MAX_BRAKE_TORQUE_REAR

        # Aerodynamics parameters
        self.RHO_AIR = physics_config.aerodynamics.RHO_AIR
        self.FRONTAL_AREA = physics_config.aerodynamics.FRONTAL_AREA
        self.CD_CAR = physics_config.aerodynamics.CD_CAR
        self.C_ROLL_RESISTANCE = physics_config.aerodynamics.C_ROLL_RESISTANCE

        # Steering parameters
        self.MAX_STEER_ANGLE = physics_config.steering.MAX_STEER_ANGLE
        self.STEER_RATE = physics_config.steering.STEER_RATE

        # Friction parameters
        self.BASE_FRICTION = physics_config.friction.BASE_FRICTION
        self.GRASS_FRICTION = physics_config.friction.GRASS_FRICTION

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
        self.wheels: list[Any] = []
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

        # MX5 Powertrain (replaces placeholder engine)
        self.powertrain = MX5Powertrain()
        # Start in 2nd gear (1st is too short for racing)
        self.powertrain.shift_to(2)

        # Automatic shifting thresholds with hysteresis
        self.SHIFT_UP_RPM = 6500  # Shift up before redline
        self.SHIFT_DOWN_RPM = 2500  # Shift down to avoid lugging
        self.SHIFT_DOWN_RPM_HYSTERESIS = 1800  # Lower threshold after upshift (prevents hunting)

        # Shift cooldown to prevent rapid shifting
        self.shift_cooldown = 0.0  # Time since last shift (seconds)
        self.MIN_SHIFT_INTERVAL = 0.5  # Minimum time between shifts (seconds)
        self.last_shift_was_upshift = False  # Track shift direction for hysteresis

        # For rendering
        self.hull: Any = type('Body', (), {})()
        self.hull.position = (init_x, init_y)
        self.hull.angle = init_angle
        self.hull.linearVelocity = (0, 0)
        self.hull.angularVelocity = 0.0
        self.hull.color = (0.8, 0.0, 0.0)

        # Store previous tire forces for wheel dynamics feedback
        # This prevents unrealistic wheel spin/lock by applying tire force torque
        self.prev_tire_forces = np.zeros(4)  # Longitudinal force per wheel [FL, FR, RL, RR]

        # Filtered tire forces for smooth wheel dynamics (prevents oscillations)
        self.prev_tire_forces_filtered = np.zeros(4)

        # Store last computed tire forces for GUI/debugging (avoids recomputation)
        self.last_tire_forces = None

        # Store last computed accelerations for load transfer calculation
        self.ax = 0.0  # Longitudinal acceleration (body frame)
        self.ay = 0.0  # Lateral acceleration (body frame)

        # Filtered accelerations for smooth load transfer (prevents oscillations)
        self.ax_filtered = 0.0
        self.ay_filtered = 0.0

        self.drawlist = self.wheels + [self.hull]
        self.particles = []

    def gas(self, throttle: float) -> None:
        """Apply throttle (0 to 1)."""
        self._gas = min(1, max(0, throttle))

    def brake(self, brake_force: float) -> None:
        """Apply brake (0 to 1)."""
        self._brake = min(1, max(0, brake_force))

    def steer(self, steer_input: float) -> None:
        """
        Apply steering input (-1 to 1).
        Smoothly ramps to target angle.
        """
        self.steer_input = min(1, max(-1, steer_input))

    def step(self, dt: float) -> dict[str, Any]:
        """
        Integrate dynamics forward by dt seconds.
        """
        # Smooth steering angle toward input
        target_angle = self.steer_input * self.MAX_STEER_ANGLE
        angle_diff = target_angle - self.steering_angle
        angle_diff = min(self.STEER_RATE * dt, max(-self.STEER_RATE * dt, angle_diff))
        self.steering_angle += angle_diff

        # Update wheel angular velocities (uses previous tire forces)
        self._update_wheel_dynamics(dt)

        # Update MX5 powertrain state (engine RPM, gearbox)
        # Use average rear wheel RPM (convert from rad/s to RPM)
        rear_wheel_rpm_avg = (self.wheel_omega[2] + self.wheel_omega[3]) / 2.0 * 9.5493
        self.powertrain.update(dt, rear_wheel_rpm_avg)

        # Update shift cooldown timer
        self.shift_cooldown += dt

        # Automatic gear shifting with hysteresis to prevent hunting
        current_gear = self.powertrain.gearbox.current_gear
        engine_rpm = self.powertrain.engine.rpm

        # Only allow shifting if cooldown has expired
        if self.shift_cooldown >= self.MIN_SHIFT_INTERVAL:
            # Upshift logic
            if engine_rpm > self.SHIFT_UP_RPM and current_gear < 6:
                self.powertrain.shift_up()
                self.shift_cooldown = 0.0  # Reset cooldown
                self.last_shift_was_upshift = True  # Track for hysteresis

            # Downshift logic with hysteresis
            # Use lower threshold if we just upshifted (prevents hunting)
            elif current_gear > 2:
                downshift_threshold = (self.SHIFT_DOWN_RPM_HYSTERESIS if self.last_shift_was_upshift
                                      else self.SHIFT_DOWN_RPM)

                if engine_rpm < downshift_threshold:
                    # Don't shift below 2nd gear (1st is too short for racing)
                    self.powertrain.shift_down()
                    self.shift_cooldown = 0.0  # Reset cooldown
                    self.last_shift_was_upshift = False  # Track for hysteresis

        # Compute tire forces
        tire_friction = self._get_surface_friction()
        forces = self._compute_tire_forces(tire_friction)

        # Store tire forces for next timestep's wheel dynamics
        for i in range(4):
            self.prev_tire_forces[i] = forces[i]['fx']

            # Apply low-pass filter to tire forces to prevent oscillations
            # Same filter strength as acceleration filtering (alpha = 0.15)
            filter_alpha = 0.15
            self.prev_tire_forces_filtered[i] = (
                self.prev_tire_forces_filtered[i] * (1.0 - filter_alpha) +
                self.prev_tire_forces[i] * filter_alpha
            )

        # Store forces for GUI/debugging (prevents double computation)
        self.last_tire_forces = forces

        # Integrate state with forward Euler
        integration_results = self._integrate_state(forces, dt)

        # Update hull for rendering/compatibility
        self._update_hull()

        # Return debug info
        integration_results['tire_forces'] = forces
        return integration_results

    def _update_wheel_dynamics(self, dt: float) -> None:
        """
        Update wheel angular velocities based on engine, brake, and tire forces.

        Uses tire force feedback to prevent unrealistic wheel spin/lock:
        I × α = T_applied - F_x × r

        Where F_x is the longitudinal tire force from the previous timestep.
        """
        for i in range(4):
            wheel = self.wheels[i]
            is_rear = (i >= 2)

            # Calculate wheel's longitudinal velocity at the contact patch
            if i < 2:  # Front wheels
                y_pos = self.WIDTH / 2 if i == 0 else -self.WIDTH / 2
            else:  # Rear wheels
                y_pos = self.WIDTH / 2 if i == 2 else -self.WIDTH / 2
            wheel_vx = self.vx - self.yaw_rate * y_pos

            # Get tire force feedback from previous timestep
            # Use FILTERED forces to prevent oscillations from frame-to-frame variations
            # Reduced coupling to prevent oscillations and runaway wheel spin
            # Lower values = more stable but slightly less accurate physics
            # Higher values = more accurate but can cause oscillations
            feedback_coupling = 0.5  # Conservative coupling for stability and grip

            # Physics: I × α = T_applied - (F_x × r)
            # F_x is tire force on car, ground reaction is opposite (Newton's 3rd law)
            # We SUBTRACT the torque because ground reaction opposes wheel motion
            tire_force_torque = self.prev_tire_forces_filtered[i] * self.TIRE_RADIUS * feedback_coupling

            # Simple logic: Apply brakes, engine, or free-roll
            # Note: Environment ensures gas and brake are mutually exclusive
            # (only one can be non-zero at a time)

            # 1. Apply Brakes
            if self._brake > 0:
                # Apply brake torque with proper front/rear bias (60/40)
                if i < 2:  # Front wheels (higher braking force)
                    brake_torque = -self.MAX_BRAKE_TORQUE_FRONT * self._brake
                else:  # Rear wheels (lower braking force to prevent lockup)
                    brake_torque = -self.MAX_BRAKE_TORQUE_REAR * self._brake

                # During braking: ONLY apply brake torque (no tire feedback)
                # Tire feedback uses delayed forces from previous timestep, which
                # creates oscillation during braking (ABS-like pulsing)
                # The brake is the primary control - let it work cleanly
                net_torque = brake_torque
                accel = net_torque / self.INERTIA
                new_omega = wheel.omega + accel * dt

                # Car has NO REVERSE GEAR - wheels cannot spin backward
                # Brakes can only slow wheels to zero, not reverse them
                wheel.omega = max(0.0, new_omega)

            # 2. Apply Engine (rear-wheel drive with MX5 powertrain)
            elif is_rear:
                # Calculate wheel RPM for powertrain (rad/s to RPM)
                wheel_rpm = abs(wheel.omega) * 9.5493  # 60 / (2π)

                # Get wheel torque from MX5 powertrain (includes engine braking when throttle closed)
                # The powertrain returns TOTAL wheel torque, divide by 2 for each rear wheel
                total_wheel_torque = self.powertrain.get_wheel_torque(self._gas, wheel_rpm)
                engine_torque = total_wheel_torque / 2.0  # Divide by 2 for each rear wheel

                # CORRECT PHYSICS WITH TIRE FORCE FEEDBACK
                # ==========================================
                # Newton's 3rd law: Tire pushes car forward → Ground pushes tire backward
                # This backward force creates torque opposing wheel acceleration
                #
                # I × α = T_engine - F_tire × r
                #
                # Without this feedback, wheels spin freely (unrealistic)
                # With feedback, wheels grip properly and only slip when force exceeds tire limit
                #
                # The tire force feedback is now FILTERED (alpha=0.15) to prevent oscillations
                # This provides smooth, stable traction control without additional damping
                #
                # NOTE: Engine braking is now realistic (from MX5 powertrain)
                # When throttle = 0, engine_torque will be negative (engine braking)
                net_torque = engine_torque - tire_force_torque

                # Note: Speed-dependent damping removed - filtered tire forces provide
                # sufficient resistance to prevent runaway wheel spin

                accel = net_torque / self.INERTIA
                new_omega = wheel.omega + accel * dt

                # Ensure wheels only spin forward (no reverse gear)
                wheel.omega = max(0.0, new_omega)

            # 3. Free Rolling (coasting)
            else:
                # Smoothly approach ground speed using damping
                # This simulates tire compliance and rolling resistance
                target_omega = wheel_vx / self.TIRE_RADIUS

                # Apply damping toward target speed
                # This is more stable than force-based feedback for coasting
                damping = 0.9  # Damping coefficient (reduce it for stability)
                omega_change = (target_omega - wheel.omega) * damping

                wheel.omega += omega_change

                # Car has NO REVERSE GEAR - wheels cannot spin backward
                # Even when coasting, clamp to non-negative rotation
                wheel.omega = max(0.0, wheel.omega)

            # Sync to wheel_omega array for consistency
            self.wheel_omega[i] = wheel.omega

            # Update phase (rotation angle)
            wheel.phase += wheel.omega * dt

    def _get_surface_friction(self) -> float:
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

    def _compute_normal_forces(self) -> npt.NDArray[np.float64]:
        """
        Compute per-wheel normal forces using a rigid-body load transfer model.

        This replaces the complex spring/damper simulation with a direct
        calculation of load transfer based on longitudinal (ax) and
        lateral (ay) accelerations from the previous physics step.

        Returns:
            np.ndarray: Normal force for each wheel [FL, FR, RL, RR] in Newtons
        """
        # Get accelerations from the previous physics step
        # Use FILTERED accelerations to prevent oscillations
        # (raw accelerations can change rapidly frame-to-frame)
        ax = self.ax_filtered  # Longitudinal acceleration (body frame)
        ay = self.ay_filtered  # Lateral acceleration (body frame)

        # 1. Static weight distribution (50/50)
        static_load_per_wheel = (self.MASS * 9.81) / 4.0
        normal_forces = np.full(4, static_load_per_wheel)

        # 2. Longitudinal Load Transfer (Pitch)
        # F_transfer = ax * mass * h_cg / wheelbase
        # This force is *subtracted* from rear axle and *added* to front axle
        # (Positive ax = acceleration, load moves to rear)
        # Note: Our ax is positive for forward accel, so a positive ax
        # should *decrease* front load and *increase* rear load.
        lon_transfer_force = -ax * self.MASS * self.CG_HEIGHT / self.LENGTH

        normal_forces[0] += lon_transfer_force / 2.0  # FL
        normal_forces[1] += lon_transfer_force / 2.0  # FR
        normal_forces[2] -= lon_transfer_force / 2.0  # RL
        normal_forces[3] -= lon_transfer_force / 2.0  # RR

        # 3. Lateral Load Transfer (Roll)
        # During cornering, centripetal force causes body roll toward outside
        # Outside wheels compress, inside wheels extend
        lat_transfer_force = ay * self.MASS * self.CG_HEIGHT / self.WIDTH

        normal_forces[0] -= lat_transfer_force / 2.0  # FL
        normal_forces[1] += lat_transfer_force / 2.0  # FR
        normal_forces[2] -= lat_transfer_force / 2.0  # RL
        normal_forces[3] += lat_transfer_force / 2.0  # RR

        # Prevent negative forces (wheel liftoff)
        # Minimum force keeps tire model stable
        # Inline max() is 3.88x faster than np.clip for 4-element array
        normal_forces[0] = max(50.0, normal_forces[0])
        normal_forces[1] = max(50.0, normal_forces[1])
        normal_forces[2] = max(50.0, normal_forces[2])
        normal_forces[3] = max(50.0, normal_forces[3])

        return normal_forces

    def _compute_tire_forces(self, friction: float) -> dict[int, dict[str, float]]:
        """
        Compute tire forces using Pacejka model with load transfer.

        Load transfer comes from rigid-body dynamics - weight shifts
        during acceleration, braking, and cornering.

        Returns dict with forces for each wheel.
        """
        # Get normal forces from rigid-body load transfer model
        normal_forces = self._compute_normal_forces()

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

            # Standard slip ratio calculation
            # Use maximum of speeds as denominator (prevents division by zero)
            denom = max(abs(wheel_vx), abs(wheel_linear_vel), 0.1)
            slip_ratio = (wheel_linear_vel - wheel_vx) / denom

            # Clip to valid range [-1, 1]
            # -1 = full lockup (wheel stopped, car moving)
            #  0 = perfect grip (wheel speed = car speed)
            # +1 = full spin (wheel spinning, car stationary)
            slip_ratio = min(1.0, max(-1.0, slip_ratio))

            # Get normal force from load transfer model
            normal_force = normal_forces[i]

            # Tire forces
            # Note: Negate lateral force because positive slip angle (velocity left of wheel heading)
            # should produce force to the right (negative) to correct the slip
            fy = -self.tire.lateral_force(slip_angle, normal_force, friction)
            fx = self.tire.longitudinal_force(slip_ratio, normal_force, friction)



            forces[i] = {
                'fx': fx,
                'fy': fy,
                'steer': steer_ang,
                'slip_angle': slip_angle,
                'slip_ratio': slip_ratio,
                'normal_force': normal_force,
            }

        return forces

    def _integrate_state(
        self, forces: dict[int, dict[str, float]], dt: float
    ) -> dict[str, float]:
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

        # === REAL PHYSICS FIX ===
        # Add physically-based drag and rolling resistance
        # 1. Aerodynamic Drag (opposes forward velocity)
        fx_drag = -0.5 * self.RHO_AIR * self.FRONTAL_AREA * self.CD_CAR * self.vx * abs(self.vx)

        # 2. Rolling Resistance (constant force opposing motion)
        # Use total static normal force
        total_normal_force = self.MASS * 9.81
        fx_roll = -self.C_ROLL_RESISTANCE * total_normal_force * np.sign(self.vx)

        fx_total += fx_drag + fx_roll
        # === END FIX ===

        # Compute accelerations
        ax = fx_total / self.MASS
        ay = fy_total / self.MASS

        # Moment of inertia (rough estimate)
        # good for generalization, but let's stick to mx5 value instead
        # Iz = self.MASS * (self.LENGTH**2 / 12 + self.WIDTH**2 / 12)

        # Realistic yaw inertia for an MX-5 (not a uniform plate)
        Iz = 1700.0

        ang_accel = torque / Iz

        # Update velocity in body frame (corrected for rotating reference frame)
        # - vx is forward velocity.
        # - vy is sideways (lateral) velocity.
        # - ax is forward acceleration (from tires, drag).
        # - ay is sideways acceleration (from tires).
        # If the car wasn't turning (yaw_rate = 0), the update would be simple:
        # vx += ax * dt
        # vy += ay * dt
        # The Centripetal Term: self.vx * self.yaw_rate
        # Use temporary variables to ensure simultaneous update (symplectic Euler)
        # Otherwise, updating vx then using new vx to update vy introduces bias
        old_vx = self.vx
        old_vy = self.vy
        
        self.vx += (ax + old_vy * self.yaw_rate) * dt  # Forward
        self.vy += (ay - old_vx * self.yaw_rate) * dt  # Lateral
        # If the operator was a "+" This meant turning left would magically
        # pull the car into the turn (a positive ay),
        # which is the opposite of reality.
        # It's like having anti-centrifugal force.

        # === REAL PHYSICS FIX ===
        # Removed the artificial velocity clamp that was here
        # (if v_mag > 30.0: ...)
        # Top speed is now an emergent property of P_engine vs F_drag
        # === END FIX ===

        # Update rotation
        self.yaw_rate += ang_accel * dt
        self.yaw += self.yaw_rate * dt

        # Update position (in world frame)
        cos_yaw = np.cos(self.yaw)
        sin_yaw = np.sin(self.yaw)

        self.x += (self.vx * cos_yaw - self.vy * sin_yaw) * dt
        self.y += (self.vx * sin_yaw + self.vy * cos_yaw) * dt

        # Store accelerations for next step's load transfer calculation
        self.ax = ax
        self.ay = ay

        # Apply low-pass filter to accelerations for smooth load transfer
        # This prevents oscillations caused by frame-to-frame acceleration changes
        # Higher alpha = more responsive, lower alpha = more damped
        # alpha = 0.15 gives strong damping (85% old, 15% new) to eliminate oscillations
        filter_alpha = 0.15
        self.ax_filtered = self.ax_filtered * (1.0 - filter_alpha) + ax * filter_alpha
        self.ay_filtered = self.ay_filtered * (1.0 - filter_alpha) + ay * filter_alpha

        return {
            'fx_total': fx_total, 'fy_total': fy_total, 'torque': torque,
            'ax': ax, 'ay': ay, 'ang_accel': ang_accel
        }

    def _update_hull(self) -> None:
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

        for i, (wheel, (dx, dy)) in enumerate(zip(self.wheels, wheel_pos_local)):
            # Rotate position to world frame
            rx = dx * cos_yaw - dy * sin_yaw
            ry = dx * sin_yaw + dy * cos_yaw

            wheel.position = (self.x + rx, self.y + ry)
            wheel.omega = self.wheel_omega[i]
            wheel.phase += wheel.omega * 0.02  # Approximate dt

    def draw(
        self,
        surface: Any,
        zoom: float,
        translation: tuple[float, float],
        angle: float,
        draw_particles: bool = True,
    ) -> None:
        """
        Draw car and particles (compatible with original interface).
        This is a simplified version - full drawing would need pygame.
        """
        # For now, just a placeholder to maintain interface compatibility
        # The actual rendering happens in car_racing.py using pygame
        pass

    def destroy(self) -> None:
        """Clean up (no resources to clean up in this implementation)."""
        pass
