"""
Clean 2D top-down car dynamics simulation using Pacejka magic formula tires.

This replaces the Box2D implementation with a more interpretable physics model:
- Pacejka magic formula for tire forces
- No external physics library dependency

The curves output tire forces (lateral/longitudinal) and moments (Mz for aligning moment for example) based on just a few inputs. These inputs are:

slip angle; the difference between the direction the tire is facing, and its velocity. 0 means the tire is going straight ahead (no slip). Typical peak slip angles for tire are around 3-6 degrees.
slip ratio; the spin velocity divided by its actual world velocity. A slip ratio of -1 means full braking lock; a ratio of 0 means the tire is spinning at the exact same rate as the road is disappearing below it. A slip ratio of 1 means it's spinning.
camber; the angle of the tire with respect to the surface
load; the amount of force pressing down on the tire. Typically each wheel carries around 1/4th of the car's weight.


"""

import numpy as np

class PacejkaTire:
    """
    Pacejka magic formula tire model for slip-based force calculation.
    Now with separate longitudinal and lateral coefficients.
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
        """
        # The slip angle is the difference between the direction a wheel is pointing and the direction it's actually traveling.
        # A slip angle of 90 degrees (or -np.pi / 2) means the wheel is sliding purely sideways.
        # There is no physical scenario where the lateral (sideways) slip angle can be greater than 90 degrees.
        # An angle of, for example, 100 degrees would imply the wheel is also rolling backward,
        # which is handled by the longitudinal force model, not the lateral one.
        sa = np.clip(slip_angle, -np.pi / 2, np.pi / 2)

        # Use LATERAL coefficients
        arg = self.B_lat * sa   # B_Lat is stiffness factor (Lateral)

        # This is the Pacejka Magic Formula.
        # F is the lateral force
        # sa is the slip angle (the x in the original formula)
        # D_lat is the peak friction coefficient (Lateral) This is self.D_lat * normal_force * max_friction. It scales the maximum possible force (the peak of the sine wave).
        # C_lat is the shape factor (Lateral) This is np.arctan(arg - self.E_lat * (arg - np.arctan(arg)))
        # E_lat is the curvature factor (Lateral) This is self.E_lat * (arg - np.arctan(arg)). It controls the curvature of the force curve near its peak. (the sine wave)
        F = (self.D_lat * normal_force * max_friction * np.sin(self.C_lat * np.arctan(arg - self.E_lat * (arg - np.arctan(arg)))))
        return F

    def longitudinal_force(self, slip_ratio, normal_force, max_friction=1.0):
        """
        Calculate longitudinal (traction) force using Pacejka formula.
        """
        sr = np.clip(slip_ratio, -1.0, 1.0)

        # Use LONGITUDINAL coefficients
        # this is pretty much the as as the LATERAL force (but longitudinal) except it uses slip_ratio instead of slip_angle
        arg = self.B_lon * sr
        F = (self.D_lon * normal_force * max_friction * np.sin(self.C_lon * np.arctan(arg - self.E_lon * (arg - np.arctan(arg)))))

        return F

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
    """

    # Vehicle parameters (Based on 2022 Mazda MX-5 Sport)
    MASS = 1062.0  # Vehicle mass (kg)
    LENGTH = 2.310  # Wheelbase (m)
    WIDTH = 1.50  # Track width (m)

    # 50/50 weight distribution
    LF = LENGTH * 0.5  # Distance from CG to front axle
    LR = LENGTH * 0.5  # Distance from CG to rear axle

    # Tire parameters
    TIRE_RADIUS = 0.309  # Wheel radius (m) (from 195/50R16)
    TIRE_WIDTH = 0.205  # Tire width (m)

    # Pacejka parameters
    # ------------------
    # See this guide as well https://www.edy.es/dev/docs/pacejka-94-parameters-explained-a-comprehensive-guide/
    #
    # For road cars, typical peak slip angles are around 10-15 degrees, while F1 tires may peak around 6 degrees Racer.
    # We now use separate lateral (cornering) and longitudinal (accel) values
    #
    # ----------------------------------------------
    # Pacejka Magic Formula Parameters (B, C, D, E):
    # These are empirical curve-fitting coefficients with no direct physical meaning.
    # They shape the tire force vs slip curve to match real tire behavior.
    #
    # B (Stiffness): Controls initial slope - how quickly force builds with slip.
    #   Higher B = stiffer, more responsive tire. Typical range: 8-15
    #   B=12.0 is a good starting point for MX-5 tires.
    #
    # C (Shape): Controls overall curve shape. Typically 1.3-2.5
    #   Affects the "peakiness" of the force curve.
    #   C=1.9 is a good starting point for MX-5 tires.
    #
    # D (Peak): Peak force multiplier. Combined with normal force and friction,
    #   this determines maximum grip. D=1.0 means peak friction = surface friction.
    #   D>1.0 means tire can exceed surface friction at optimal slip.
    #   D=1.1 is a good starting point for MX-5 tires.
    #
    # E (Curvature): Controls curve shape near and after peak.
    #   Affects how sharply grip falls off past optimal slip. Typically 0.9-1.0
    #   E=0.97 is a good starting point for MX-5 tires.
    #
    # Note: Peak slip angles for road cars are typically 10-15 degrees lateral.
    # These values are tuned for MX-5 performance street tires.

    # Pacejka Magic Formula Parameters - Tuned for MX-5 on street tires (195/50R16)
    # These values are calibrated to match real MX-5 performance:
    # - Lateral grip: ~0.90g in corners (typical for good street tires)
    # - Braking: ~1.10g max deceleration
    # - Acceleration: ~0.60g (RWD, 2 wheels only)

    # B: Stiffness factor (initial slope of force curve)
    # Street tires are softer than race tires (8-9 vs 10-12)
    PACEJKA_B_LAT = 8.5   # Lateral stiffness - softer for street tires
    PACEJKA_B_LON = 8.0   # Longitudinal stiffness - softer for street tires

    # C: Shape factor (affects curve peakiness)
    # Standard value for passenger car tires
    PACEJKA_C_LAT = 1.9   # Lateral shape
    PACEJKA_C_LON = 1.9   # Longitudinal shape

    # D: Peak friction multiplier
    # Calibrated to real MX-5 grip levels on street tires:
    # - D_lat=0.95 × 2605N × 4 wheels = 9919N = 0.95g lateral
    # - D_lon=1.15 × 2605N × 4 wheels = 11983N = 1.15g braking
    # - D_lon=1.15 × 2605N × 2 wheels = 5992N = 0.57g acceleration (RWD)
    PACEJKA_D_LAT = 0.95  # Lateral peak - realistic for street tires
    PACEJKA_D_LON = 1.15  # Longitudinal peak - realistic for street tire braking

    # E: Curvature factor (shape near/after peak)
    # Controls how gradually grip falls off after peak slip
    # Street tires typically have smoother falloff than race tires
    PACEJKA_E_LAT = 0.97  # Lateral curvature - gradual falloff
    PACEJKA_E_LON = 0.97  # Longitudinal curvature - gradual falloff

    # ============================================================
    # EXPECTED PERFORMANCE WITH THESE TIRE PARAMETERS
    # ============================================================
    # Vehicle mass: 1062 kg
    # Weight per wheel: 2604.6 N
    #
    # LATERAL (Cornering):
    #   Peak force per wheel: 2474.4 N (D_lat=0.95 × 2604.6N)
    #   Total lateral force: 9897.5 N (all 4 wheels)
    #   Max lateral accel: 9.32 m/s² (0.95g)
    #   Target: 0.85-0.95g ✓
    #   Real MX-5: ~0.90g skidpad (Car and Driver)
    #
    # LONGITUDINAL - Acceleration (RWD, 2 wheels):
    #   Peak force per wheel: 2995.3 N (D_lon=1.15 × 2604.6N)
    #   Total driving force: 5990.6 N (rear wheels only)
    #   Max acceleration: 5.64 m/s² (0.57g)
    #   Target: 0.50-0.70g ✓
    #   Real MX-5: Power-limited, not traction-limited
    #
    # LONGITUDINAL - Braking (all 4 wheels):
    #   Total braking force: 11981.2 N (all 4 wheels)
    #   Max deceleration: 11.28 m/s² (1.15g)
    #   Target: 1.00-1.20g ✓
    #   Real MX-5: 60-0 mph in ~115 ft @ 1.10g (Motor Trend)
    #
    # Peak slip characteristics:
    #   Lateral: Peak at ~8-10° slip angle (street tire typical)
    #   Longitudinal: Peak at ~12-15% slip ratio (street tire typical)
    #
    # These values match real MX-5 Sport with stock Bridgestone
    # Potenza RE050A or similar performance street tires.
    # ============================================================

    # Drivetrain (2.0L Skyactiv-G)
    ENGINE_POWER = 135000.0  # Power (Watts) (181 hp * 745.7)

    # Derived from ~205 Nm torque in 1st gear (5.09) & final drive (2.87)
    # (205 * 5.09 * 2.87) / 2 wheels = ~1500 Nm torque per wheel
    MAX_TORQUE_PER_WHEEL = 1500.0

    # We transition from "constant torque" to "constant power"
    # P = τ * ω  =>  ω = P / τ
    # (135000 W / 2 wheels) / 1500 Nm = 45 rad/s
    POWER_TRANSITION_OMEGA = 45.0  # Speed at which torque starts to drop

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

        # ADDED: State for virtual suspension (smoothed load transfer)
        self.smoothed_lateral_accel = 0.0

        # Store previous tire forces for wheel dynamics feedback
        # This prevents unrealistic wheel spin/lock by applying tire force torque
        self.prev_tire_forces = np.zeros(4)  # Longitudinal force per wheel [FL, FR, RL, RR]

        # Store last computed tire forces for GUI/debugging (avoids recomputation)
        self.last_tire_forces = None

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

        # Update wheel angular velocities (uses previous tire forces)
        self._update_wheel_dynamics(dt)

        # Compute tire forces
        tire_friction = self._get_surface_friction()
        forces = self._compute_tire_forces(tire_friction)

        # Store tire forces for next timestep's wheel dynamics
        for i in range(4):
            self.prev_tire_forces[i] = forces[i]['fx']

        # Store forces for GUI/debugging (prevents double computation)
        self.last_tire_forces = forces

        # Integrate state with forward Euler
        integration_results = self._integrate_state(forces, dt)

        # Update hull for rendering/compatibility
        self._update_hull()

        # Return debug info
        integration_results['tire_forces'] = forces
        return integration_results

    def _update_wheel_dynamics(self, dt):
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
            # This creates opposing torque that prevents unrealistic slip
            # Reduced coupling to prevent oscillations in RL environment
            # (track collisions + feedback can cause instability at full coupling)
            feedback_coupling = 0.5  # Balanced coupling for stability
            tire_force_torque = self.prev_tire_forces[i] * self.TIRE_RADIUS * feedback_coupling

            # Simple logic: Apply brakes, engine, or free-roll
            # Note: Environment ensures gas and brake are mutually exclusive
            # (only one can be non-zero at a time)

            # 1. Apply Brakes
            if self._brake > 0:
                # Brake torque (negative)
                brake_torque = -self.BRAKE_ANG_DECEL * self.INERTIA * self._brake

                # Net torque includes tire force feedback
                # During braking, tire force opposes wheel (helps slow it down)
                net_torque = brake_torque - tire_force_torque

                accel = net_torque / self.INERTIA
                new_omega = wheel.omega + accel * dt

                # CRITICAL: Car has NO REVERSE GEAR - wheels cannot spin backward
                # Brakes can only slow wheels to zero, not reverse them
                # Clamp wheel speed to non-negative (forward only)
                wheel.omega = max(0.0, new_omega)

            # 2. Apply Engine (rear-wheel drive)
            elif is_rear and self._gas > 0:
                # Engine torque (positive)
                if abs(wheel.omega) < self.POWER_TRANSITION_OMEGA:
                    engine_torque = self.MAX_TORQUE_PER_WHEEL * self._gas
                else:
                    engine_torque = (self.ENGINE_POWER / 2) * self._gas / abs(wheel.omega)

                # Net torque includes tire force feedback
                # During acceleration, tire force opposes wheel spin
                net_torque = engine_torque - tire_force_torque

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
                damping = 0.3  # Damping coefficient (reduced for stability)
                omega_change = (target_omega - wheel.omega) * damping

                wheel.omega += omega_change

                # Car has NO REVERSE GEAR - wheels cannot spin backward
                # Even when coasting, clamp to non-negative rotation
                wheel.omega = max(0.0, wheel.omega)

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
        Compute tire forces using Pacejka model with load transfer.

        Returns dict with forces for each wheel.
        """
        # Base load distribution (equal weight per wheel)
        weight_per_wheel = (self.MASS * 9.81) / 4.0  # Force in Newtons

        # --- Load Transfer Calculation ---
        # Simulate weight shift during acceleration, braking, and cornering

        # LATERAL Load Transfer (cornering)
        # When turning, centrifugal effect shifts weight to outside wheels
        target_lateral_accel = self.vx * self.yaw_rate

        # Smooth the lateral acceleration (prevents abrupt load changes)
        lerp_factor = 0.15  # Suspension response rate
        self.smoothed_lateral_accel += (target_lateral_accel - self.smoothed_lateral_accel) * lerp_factor

        # Calculate lateral load transfer
        # CG height / track width ratio affects transfer magnitude
        # Lower factor = less transfer (stiffer anti-roll bars)
        lateral_factor = 0.3  # Reduced from 0.5 to reduce oscillations
        lateral_load_transfer = lateral_factor * self.MASS * self.smoothed_lateral_accel

        # LONGITUDINAL Load Transfer (acceleration/braking)
        # Calculate longitudinal acceleration from previous timestep tire forces
        # Sum of all longitudinal tire forces divided by mass
        total_fx = sum(self.prev_tire_forces)
        longitudinal_accel = total_fx / self.MASS

        # Calculate longitudinal load transfer
        # CG height / wheelbase ratio affects transfer magnitude
        # Positive accel = weight shifts back (more load on rear)
        # Negative accel (braking) = weight shifts forward (more load on front)
        cg_height = 0.45  # Estimated CG height for MX-5 (meters)
        longitudinal_factor = cg_height / self.LENGTH
        longitudinal_load_transfer = longitudinal_factor * self.MASS * 9.81 * longitudinal_accel / 9.81
        # Clamp to prevent excessive transfer
        longitudinal_load_transfer = np.clip(longitudinal_load_transfer,
                                             -weight_per_wheel * 1.5,
                                             weight_per_wheel * 1.5)

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

            # Calculate normal force with lateral AND longitudinal load transfer
            normal_force = weight_per_wheel

            # Apply LATERAL load transfer (left/right weight shift during cornering)
            if i == 0 or i == 2:  # Left wheels (FL, RL)
                normal_force -= lateral_load_transfer / 2
            else:  # Right wheels (FR, RR)
                normal_force += lateral_load_transfer / 2

            # Apply LONGITUDINAL load transfer (front/back weight shift during accel/brake)
            if i < 2:  # Front wheels (FL, FR)
                # Positive longitudinal_load_transfer = accelerating = weight shifts BACK = LESS load on front
                normal_force -= longitudinal_load_transfer / 2
            else:  # Rear wheels (RL, RR)
                # Positive longitudinal_load_transfer = accelerating = weight shifts BACK = MORE load on rear
                normal_force += longitudinal_load_transfer / 2

            # Prevent negative/zero load (wheels lifting off ground)
            normal_force = max(50.0, normal_force)

            # Tire forces
            # Note: Negate lateral force because positive slip angle (velocity left of wheel heading)
            # should produce force to the right (negative) to correct the slip
            fy = -self.tire.lateral_force(slip_angle, normal_force, friction)
            fx = self.tire.longitudinal_force(slip_ratio, normal_force, friction)

            # BACKWARD MOTION PENALTY (for passive rolling backward)
            # Note: Active backward driving is impossible (wheels clamped to >= 0)
            # This penalty handles passive backward rolling (e.g., down a hill)
            # Real tires moving backward have reduced lateral grip (cornering is unstable)
            # Longitudinal grip (braking) is less affected
            if wheel_vx < -0.5:  # Significant backward rolling velocity
                # Reduce lateral force significantly (unstable cornering when rolling backward)
                # Keep longitudinal force mostly intact (can still brake to stop)
                lateral_penalty = 0.3  # 70% reduction in lateral grip
                fy *= lateral_penalty

            forces[i] = {
                'fx': fx,
                'fy': fy,
                'steer': steer_ang,
                'slip_angle': slip_angle,
                'slip_ratio': slip_ratio,
                'normal_force': normal_force,
            }

        return forces

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

        # Add drag and rolling resistance
        C_drag = 0.003
        fx_drag = -C_drag * self.vx * abs(self.vx)
        C_roll = 0.05
        fx_roll = -C_roll * self.vx
        fx_total += fx_drag + fx_roll

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
        self.vx += (ax + self.vy * self.yaw_rate) * dt  # Forward
        self.vy += (ay - self.vx * self.yaw_rate) * dt  # Lateral
                                                        # If the operator was a "+" This meant turning left would magically
                                                        # pull the car into the turn (a positive ay),
                                                        # which is the opposite of reality.
                                                        # It's like having anti-centrifugal force.

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
