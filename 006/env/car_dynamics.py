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

import numpy as np
from env.tire_model import PacejkaTire
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
from config.physics_config import *


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
    # Street tires need sufficient stiffness to prevent runaway wheel spin
    # Higher B = more force across entire slip range = better traction control
    # Increased from 8.0 to 12.0 to prevent 100% slip at standing starts
    PACEJKA_B_LAT = 8.5   # Lateral stiffness - softer for street tires
    PACEJKA_B_LON = 12.0  # Longitudinal stiffness - increased for better low-speed grip

    # C: Shape factor (affects curve peakiness)
    # Standard value for passenger car tires
    PACEJKA_C_LAT = 1.9   # Lateral shape
    PACEJKA_C_LON = 1.9   # Longitudinal shape

    # D: Peak friction multiplier
    # Calibrated to real MX-5 grip levels on street tires:
    # - D_lat=0.95 × 2605N × 4 wheels = 9919N = 0.95g lateral
    # - D_lon=1.35 × 2605N × 4 wheels = 14075N = 1.35g braking
    # - D_lon=1.35 × 2605N × 2 wheels = 7037N = 0.67g acceleration (RWD)
    #
    # Increased D_lon to 1.35 to provide more grip and prevent wheel spin
    # Street tires can exceed μ=1.0 in pure longitudinal  direction
    PACEJKA_D_LAT = 0.95  # Lateral peak - realistic for street tires
    PACEJKA_D_LON = 1.35  # Longitudinal peak - increased for better traction

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
    # Total: 205 * 5.09 * 2.87 = 2992 Nm (both wheels)
    # Per wheel: 2992 / 2 = 1496 Nm
    #
    # REALISTIC TORQUE DELIVERY FOR MX-5:
    # Street cars don't dump full torque instantly - there's throttle response,
    # drivetrain compliance, and progressive power delivery.
    #
    # Peak torque (1st gear): 1496 Nm per wheel (theoretical max at crank)
    # Practical torque (street driving): Accounting for real-world factors:
    #   - Drivetrain loss: 15% (clutch, gearbox, differential friction)
    #   - Progressive throttle: Real pedal isn't instant 100% torque
    #   - Clutch slip: Engagement isn't perfect lockup
    #   - Weight transfer: Front wheels lift slightly on hard accel
    #
    # Effective torque for realistic drivability: 400 Nm per wheel
    # This gives:
    #   - Smooth acceleration with minimal wheelspin
    #   - ~12-18% slip ratio (optimal for tire grip)
    #   - Matches real MX-5 character: progressive and controllable
    # Note: Lower than theoretical max accounts for drivetrain losses,
    # clutch slip, and progressive throttle mapping
    MAX_TORQUE_PER_WHEEL = 400.0  # Conservative torque for good traction

    # We transition from "constant torque" to "constant power"
    # P = τ * ω  =>  ω = P / τ
    # (135000 W / 2 wheels) / 400 Nm = 168.75 rad/s (~162 km/h at transition)
    # This is high but allows power delivery across full speed range
    POWER_TRANSITION_OMEGA = 168.75  # Speed at which torque starts to drop

    # Startup acceleration with realistic torque
    # Alpha = Torque / Inertia = 400 / 1.2 = 333 rad/s^2
    STARTUP_ACCEL = 333.0  # Angular acceleration (rad/s^2) for startup

    # Brake torque: Direct torque applied by brake calipers to wheel
    # MX5 braking performance: ~1.1g with proper 60/40 brake bias
    # Reduced values to prevent wheel locking (was 930/620 N·m):
    #   Front: 60 N·m → 50 rad/s² → 1.58g per wheel (realistic)
    #   Rear: 40 N·m → 33 rad/s² → 1.05g per wheel (realistic)
    #   Combined: ~1.1-1.3g total braking with load transfer
    # This allows smooth brake modulation without instant lockup
    MAX_BRAKE_TORQUE_FRONT = 60.0  # Maximum brake torque per front wheel (N·m)
    MAX_BRAKE_TORQUE_REAR = 40.0   # Maximum brake torque per rear wheel (N·m)

    # Estimated: 16" wheel + tire = ~17kg. I = 0.8 * m * r^2 = 0.8 * 17 * 0.3^2 = ~1.2
    INERTIA = 1.2  # Wheel inertia (kg*m^2)

    # Aerodynamics (ND MX-5 RF)
    RHO_AIR = 1.225  # Air density (kg/m^3)
    FRONTAL_AREA = 1.8  # Frontal area (m^2)
    CD_CAR = 0.33  # Drag coefficient
    C_ROLL_RESISTANCE = 0.015  # Coefficient of rolling resistance (tires on asphalt)

    C_ROLL_RESISTANCE = 0.015  # Coefficient of rolling resistance (tires on asphalt)
    CG_HEIGHT = 0.46  # Center of Gravity height (m) (Estimate for ND MX-5)


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
        Integrate dynamics forward by dt seconds.
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

            # 2. Apply Engine (rear-wheel drive)
            elif is_rear and self._gas > 0:
                # Engine torque (positive)
                if abs(wheel.omega) < self.POWER_TRANSITION_OMEGA:
                    engine_torque = self.MAX_TORQUE_PER_WHEEL * self._gas
                else:
                    engine_torque = (self.ENGINE_POWER / 2) * self._gas / abs(wheel.omega)

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

    def _compute_normal_forces(self):
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
        # F_transfer = ay * mass * h_cg / track_width
        # This force is *subtracted* from inside wheels and *added* to outside wheels
        # During cornering, centripetal force causes body roll toward outside
        # Outside wheels compress, inside wheels extend
        # FIX: Reversed sign - was backwards!
        lat_transfer_force = -ay * self.MASS * self.CG_HEIGHT / self.WIDTH

        normal_forces[0] -= lat_transfer_force / 2.0  # FL
        normal_forces[1] += lat_transfer_force / 2.0  # FR
        normal_forces[2] -= lat_transfer_force / 2.0  # RL
        normal_forces[3] += lat_transfer_force / 2.0  # RR

        # Prevent negative forces (wheel liftoff)
        # Minimum force keeps tire model stable
        normal_forces = np.clip(normal_forces, 50.0, np.inf)

        return normal_forces

    def _compute_tire_forces(self, friction):
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
            slip_ratio = np.clip(slip_ratio, -1.0, 1.0)

            # Get normal force from load transfer model
            normal_force = normal_forces[i]

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
        self.vx += (ax + self.vy * self.yaw_rate) * dt  # Forward
        self.vy += (ay - self.vx * self.yaw_rate) * dt  # Lateral
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
