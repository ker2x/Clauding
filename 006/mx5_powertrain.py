"""
Mazda MX-5 ND (2016+) Powertrain Simulation
===========================================

A realistic simulation of the 2.0L SKYACTIV-G engine and 6-speed manual transmission
found in the Mazda MX-5 ND (2016-present). This module provides accurate engine torque
curves, RPM dynamics, gear ratios, and transmission behavior.

Engine Specifications (2.0L SKYACTIV-G):
- Displacement: 1998 cc
- Configuration: Inline-4, naturally aspirated
- Peak Power: 181 hp (135 kW) @ 7000 RPM
- Peak Torque: 205 Nm (151 lb-ft) @ 4000 RPM
- Redline: 7500 RPM
- Idle Speed: 800 RPM
- Compression Ratio: 13.0:1

Transmission Specifications (6-Speed Manual):
- Gear Ratios: 3.760, 2.269, 1.645, 1.257, 1.000, 0.830
- Final Drive: 3.909
- Clutch: Single plate dry clutch

Usage Example:
    engine = MX5Engine()
    gearbox = MX5Gearbox()

    # Simulate one timestep
    throttle = 0.8  # 80% throttle
    clutch = 0.0    # Clutch fully engaged

    engine_torque = engine.get_torque(throttle)
    wheel_torque = gearbox.get_wheel_torque(engine_torque, clutch)

    # Update engine RPM based on wheel speed
    wheel_rpm = 500  # From wheel angular velocity
    engine.update(dt=0.02, wheel_rpm=wheel_rpm, gear_ratio=gearbox.get_current_ratio())

References:
- Mazda MX-5 ND Technical Specifications
- SKYACTIV-G Engine Technical Data
"""

import numpy as np
import math


class MX5Engine:
    """
    Realistic simulation of the Mazda MX-5 2.0L SKYACTIV-G engine.

    Provides accurate torque curves, RPM dynamics, rev limiter, and engine braking.
    The torque curve is based on real dyno data from the MX-5 ND.

    Attributes:
        rpm (float): Current engine RPM
        temperature (float): Engine coolant temperature (°C)
        oil_pressure (float): Engine oil pressure (bar)
    """

    # Engine specifications
    DISPLACEMENT = 1.998  # Liters
    CYLINDERS = 4

    # Power characteristics
    PEAK_POWER_HP = 181  # hp @ 7000 RPM
    PEAK_POWER_W = PEAK_POWER_HP * 745.7  # Convert to watts
    PEAK_POWER_RPM = 7000

    PEAK_TORQUE_NM = 205  # Nm @ 4000 RPM
    PEAK_TORQUE_RPM = 4000

    # RPM limits
    IDLE_RPM = 800
    REDLINE_RPM = 7500
    FUEL_CUT_RPM = 7500
    MAX_RPM = 8000  # Absolute maximum (over-rev)

    # Engine inertia (kg⋅m²) - typical for 2.0L I4
    ENGINE_INERTIA = 0.25

    # Engine friction and pumping losses
    FRICTION_COEFFICIENT = 0.015  # Nm/(RPM²)
    PUMPING_LOSS = 5.0  # Nm (constant)

    def __init__(self):
        """Initialize the engine at idle."""
        self.rpm = self.IDLE_RPM
        self.temperature = 90.0  # Operating temperature (°C)
        self.oil_pressure = 0.0  # Will be calculated based on RPM

        # Rev limiter state
        self._fuel_cut_active = False
        self._fuel_cut_cooldown = 0.0

        # Build torque curve lookup table
        self._build_torque_curve()

    def _build_torque_curve(self):
        """
        Build a realistic torque curve based on MX-5 ND dyno data.

        The SKYACTIV-G engine has a relatively flat torque curve with peak
        torque around 4000 RPM and good power delivery across the rev range.
        """
        # RPM points for torque curve (based on real dyno data)
        rpm_points = np.array([
            1000, 1500, 2000, 2500, 3000, 3500, 4000, 4500,
            5000, 5500, 6000, 6500, 7000, 7500, 8000
        ])

        # Torque in Nm (approximate MX-5 ND curve)
        # Peak torque ~205 Nm @ 4000 RPM
        torque_points = np.array([
            120,  # 1000 RPM - low torque at idle
            145,  # 1500 RPM
            165,  # 2000 RPM
            185,  # 2500 RPM
            200,  # 3000 RPM
            205,  # 3500 RPM
            205,  # 4000 RPM - PEAK TORQUE
            202,  # 4500 RPM
            198,  # 5000 RPM
            192,  # 5500 RPM
            186,  # 6000 RPM
            180,  # 6500 RPM
            175,  # 7000 RPM (peak power here)
            150,  # 7500 RPM - redline
            100,  # 8000 RPM - over-rev
        ])

        self._torque_rpm = rpm_points
        self._torque_nm = torque_points

    def get_torque(self, throttle):
        """
        Get engine torque output at current RPM and throttle position.

        Args:
            throttle (float): Throttle position [0.0 = closed, 1.0 = wide open]

        Returns:
            float: Engine torque in Nm
        """
        throttle = np.clip(throttle, 0.0, 1.0)

        # Fuel cut (rev limiter)
        if self.rpm >= self.FUEL_CUT_RPM:
            self._fuel_cut_active = True
            self._fuel_cut_cooldown = 0.1  # 100ms fuel cut
            return 0.0

        # Fuel cut cooldown (prevents immediate recovery)
        if self._fuel_cut_active:
            if self.rpm < self.FUEL_CUT_RPM - 200:  # Resume at -200 RPM
                self._fuel_cut_active = False
            else:
                return 0.0

        # Interpolate torque curve at current RPM
        max_torque = np.interp(self.rpm, self._torque_rpm, self._torque_nm)

        # Apply throttle (assume linear throttle response for simplicity)
        # Real engines have non-linear throttle maps, but this is close enough
        engine_torque = max_torque * throttle

        # Engine braking when throttle is closed
        if throttle < 0.01:
            # Pumping losses and friction create engine braking
            braking_torque = self._calculate_engine_braking()
            engine_torque = braking_torque  # Negative torque

        return engine_torque

    def _calculate_engine_braking(self):
        """
        Calculate engine braking torque (negative torque when throttle closed).

        Engine braking comes from:
        1. Pumping losses (throttle plate closed creates vacuum)
        2. Friction losses (pistons, bearings, valvetrain)
        3. Compression losses

        Returns:
            float: Negative torque in Nm
        """
        # Pumping losses increase with RPM (vacuum is higher)
        pumping_loss = -self.PUMPING_LOSS * (self.rpm / 1000.0)

        # Friction losses increase with RPM²
        friction_loss = -self.FRICTION_COEFFICIENT * (self.rpm / 1000.0) ** 2

        total_braking = pumping_loss + friction_loss

        # Clamp to reasonable values
        return max(total_braking, -80.0)  # Max ~80 Nm engine braking

    def update(self, dt, wheel_rpm, gear_ratio, clutch_engaged=True):
        """
        Update engine state based on drivetrain connection.

        Args:
            dt (float): Time step in seconds
            wheel_rpm (float): Wheel angular velocity in RPM
            gear_ratio (float): Current overall gear ratio (gear × final drive)
            clutch_engaged (bool): True if clutch is engaged (default: True)
        """
        # Calculate target RPM from wheels (if clutch engaged)
        if clutch_engaged and gear_ratio > 0:
            target_rpm = wheel_rpm * gear_ratio
        else:
            # Clutch disengaged - engine returns to idle
            target_rpm = self.IDLE_RPM

        # Smooth RPM transition (engine has inertia)
        rpm_diff = target_rpm - self.rpm

        # Damping coefficient (lower = more inertia, slower RPM change)
        damping = 0.3 if clutch_engaged else 0.1

        self.rpm += rpm_diff * damping

        # Clamp RPM to valid range
        self.rpm = np.clip(self.rpm, self.IDLE_RPM, self.MAX_RPM)

        # Update oil pressure (simple model: proportional to RPM)
        # Typical values: 1-2 bar @ idle, 4-6 bar @ high RPM
        self.oil_pressure = 1.5 + (self.rpm / self.REDLINE_RPM) * 4.5

        # Update fuel cut cooldown
        if self._fuel_cut_cooldown > 0:
            self._fuel_cut_cooldown -= dt

    def get_power_kw(self):
        """
        Get current power output in kilowatts.

        Power (W) = Torque (Nm) × Angular velocity (rad/s)
        P = T × (2π × RPM / 60)

        Returns:
            float: Power in kW
        """
        torque = self.get_torque(throttle=1.0)  # Max torque at current RPM
        angular_velocity = (2 * math.pi * self.rpm) / 60.0
        power_w = torque * angular_velocity
        return power_w / 1000.0  # Convert to kW

    def get_power_hp(self):
        """Get current power output in horsepower."""
        return self.get_power_kw() * 1.341  # 1 kW = 1.341 hp


class MX5Gearbox:
    """
    Realistic simulation of the Mazda MX-5 6-speed manual transmission.

    Models gear ratios, shift timing, clutch engagement, and gear selection.
    Includes realistic shift delays and clutch slip during gear changes.

    Attributes:
        current_gear (int): Current gear [0=neutral, 1-6=gears]
        clutch_position (float): Clutch pedal position [0=engaged, 1=disengaged]
    """

    # Gear ratios (MX-5 ND 6-speed manual)
    GEAR_RATIOS = {
        0: 0.0,      # Neutral
        1: 3.760,    # 1st gear
        2: 2.269,    # 2nd gear
        3: 1.645,    # 3rd gear
        4: 1.257,    # 4th gear
        5: 1.000,    # 5th gear
        6: 0.830,    # 6th gear
    }

    # Final drive ratio
    FINAL_DRIVE = 3.909

    # Shift timing (seconds)
    SHIFT_TIME = 0.15  # Time for a gear change (realistic for manual)
    CLUTCH_ENGAGEMENT_TIME = 0.2  # Time for clutch to fully engage

    # Synchromesh limits (RPM difference for clean shifts)
    MAX_SYNCHRO_RPM_DIFF = 500  # Modern synchros can handle large differences

    def __init__(self):
        """Initialize gearbox in neutral."""
        self.current_gear = 1  # Start in 1st gear (car is ready to go)
        self.clutch_position = 0.0  # 0 = engaged, 1 = disengaged

        # Shift state
        self._shifting = False
        self._shift_timer = 0.0
        self._target_gear = 1

        # Clutch state
        self._clutch_engaged = True

    def shift_up(self):
        """Shift to the next higher gear."""
        if not self._shifting and self.current_gear < 6:
            self._start_shift(self.current_gear + 1)

    def shift_down(self):
        """Shift to the next lower gear."""
        if not self._shifting and self.current_gear > 1:
            self._start_shift(self.current_gear - 1)

    def shift_to(self, gear):
        """
        Shift to a specific gear.

        Args:
            gear (int): Target gear [0=neutral, 1-6=gears]
        """
        if not self._shifting and 0 <= gear <= 6:
            self._start_shift(gear)

    def _start_shift(self, target_gear):
        """Begin a gear change."""
        self._shifting = True
        self._target_gear = target_gear
        self._shift_timer = self.SHIFT_TIME
        self.clutch_position = 1.0  # Disengage clutch during shift
        self._clutch_engaged = False

    def update(self, dt):
        """
        Update gearbox state (handles shift timing).

        Args:
            dt (float): Time step in seconds
        """
        if self._shifting:
            self._shift_timer -= dt

            if self._shift_timer <= 0:
                # Shift complete
                self.current_gear = self._target_gear
                self._shifting = False

                # Begin clutch engagement
                # (in reality, driver controls this, but we auto-engage)
                self.clutch_position = 0.8  # Start re-engaging

        # Smooth clutch engagement
        if not self._shifting and self.clutch_position > 0:
            engagement_speed = 1.0 / self.CLUTCH_ENGAGEMENT_TIME
            self.clutch_position -= engagement_speed * dt
            self.clutch_position = max(0.0, self.clutch_position)

        # Update clutch state
        self._clutch_engaged = (self.clutch_position < 0.1)

    def get_current_ratio(self):
        """
        Get the current overall gear ratio (gear ratio × final drive).

        Returns:
            float: Overall ratio [0.0 if in neutral or clutch disengaged]
        """
        if self.current_gear == 0 or self.clutch_position > 0.9:
            return 0.0

        gear_ratio = self.GEAR_RATIOS[self.current_gear]
        overall_ratio = gear_ratio * self.FINAL_DRIVE

        # Clutch slip reduces effective ratio
        clutch_engagement = 1.0 - self.clutch_position
        return overall_ratio * clutch_engagement

    def get_wheel_torque(self, engine_torque, clutch_override=None):
        """
        Convert engine torque to wheel torque through the transmission.

        Args:
            engine_torque (float): Engine torque in Nm
            clutch_override (float, optional): Manual clutch position [0-1]

        Returns:
            float: Torque at the wheels in Nm
        """
        # Use manual clutch override if provided
        if clutch_override is not None:
            clutch_engagement = 1.0 - clutch_override
        else:
            clutch_engagement = 1.0 - self.clutch_position

        # No torque if in neutral or clutch disengaged
        if self.current_gear == 0 or clutch_engagement < 0.01:
            return 0.0

        # Calculate wheel torque: T_wheel = T_engine × ratio × clutch × efficiency
        gear_ratio = self.GEAR_RATIOS[self.current_gear]
        overall_ratio = gear_ratio * self.FINAL_DRIVE

        # Transmission efficiency (typical for manual gearbox: 90-95%)
        efficiency = 0.92

        wheel_torque = engine_torque * overall_ratio * clutch_engagement * efficiency

        return wheel_torque

    def is_clutch_engaged(self):
        """Check if clutch is fully engaged."""
        return self._clutch_engaged

    def is_shifting(self):
        """Check if currently shifting gears."""
        return self._shifting

    def get_gear_ratio(self, gear=None):
        """
        Get gear ratio for a specific gear (without final drive).

        Args:
            gear (int, optional): Gear number. If None, uses current gear.

        Returns:
            float: Gear ratio
        """
        if gear is None:
            gear = self.current_gear
        return self.GEAR_RATIOS.get(gear, 0.0)


class MX5Powertrain:
    """
    Complete MX-5 powertrain system (engine + gearbox).

    This class combines the engine and gearbox into a single interface
    for easy integration with vehicle dynamics simulations.

    Example:
        powertrain = MX5Powertrain()

        # Each timestep:
        wheel_torque = powertrain.get_wheel_torque(throttle=0.8, wheel_rpm=500)
        powertrain.update(dt=0.02)

        # Shift gears:
        powertrain.shift_up()
    """

    def __init__(self):
        """Initialize powertrain with engine and gearbox."""
        self.engine = MX5Engine()
        self.gearbox = MX5Gearbox()

    def get_wheel_torque(self, throttle, wheel_rpm, clutch=0.0):
        """
        Get torque at the wheels based on throttle and wheel speed.

        Args:
            throttle (float): Throttle position [0.0 - 1.0]
            wheel_rpm (float): Wheel angular velocity in RPM
            clutch (float, optional): Manual clutch override [0=engaged, 1=disengaged]

        Returns:
            float: Torque at the wheels in Nm
        """
        # Get engine torque
        engine_torque = self.engine.get_torque(throttle)

        # Convert to wheel torque through gearbox
        wheel_torque = self.gearbox.get_wheel_torque(engine_torque, clutch)

        return wheel_torque

    def update(self, dt, wheel_rpm):
        """
        Update powertrain state.

        Args:
            dt (float): Time step in seconds
            wheel_rpm (float): Wheel angular velocity in RPM
        """
        # Update gearbox (handle shifting)
        self.gearbox.update(dt)

        # Update engine RPM based on wheel speed and gear ratio
        gear_ratio = self.gearbox.get_current_ratio()
        clutch_engaged = self.gearbox.is_clutch_engaged()
        self.engine.update(dt, wheel_rpm, gear_ratio, clutch_engaged)

    def shift_up(self):
        """Shift to next higher gear."""
        self.gearbox.shift_up()

    def shift_down(self):
        """Shift to next lower gear."""
        self.gearbox.shift_down()

    def shift_to(self, gear):
        """Shift to specific gear."""
        self.gearbox.shift_to(gear)

    def get_state(self):
        """
        Get complete powertrain state for telemetry.

        Returns:
            dict: Dictionary with all telemetry values
        """
        return {
            # Engine
            'engine_rpm': self.engine.rpm,
            'engine_torque_nm': self.engine.get_torque(throttle=1.0),
            'engine_power_kw': self.engine.get_power_kw(),
            'engine_power_hp': self.engine.get_power_hp(),
            'engine_temp_c': self.engine.temperature,
            'oil_pressure_bar': self.engine.oil_pressure,
            'fuel_cut_active': self.engine._fuel_cut_active,

            # Gearbox
            'current_gear': self.gearbox.current_gear,
            'clutch_position': self.gearbox.clutch_position,
            'is_shifting': self.gearbox.is_shifting(),
            'gear_ratio': self.gearbox.get_gear_ratio(),
            'overall_ratio': self.gearbox.get_current_ratio(),
        }


if __name__ == '__main__':
    """
    Test and demonstration of the MX5 powertrain.
    """
    print("="*70)
    print("Mazda MX-5 ND Powertrain Simulation Test")
    print("="*70)

    # Create powertrain
    powertrain = MX5Powertrain()

    print("\n1. Engine Torque Curve:")
    print("-" * 70)
    print(f"{'RPM':>6} | {'Torque (Nm)':>12} | {'Power (hp)':>12}")
    print("-" * 70)

    for rpm in range(1000, 8001, 500):
        powertrain.engine.rpm = rpm
        torque = powertrain.engine.get_torque(throttle=1.0)
        power = powertrain.engine.get_power_hp()
        print(f"{rpm:6d} | {torque:12.1f} | {power:12.1f}")

    print("\n2. Gear Ratios:")
    print("-" * 70)
    print(f"{'Gear':>6} | {'Ratio':>10} | {'Overall':>10} | {'60mph RPM':>12}")
    print("-" * 70)

    # 60 mph = 96.56 km/h, typical tire radius = 0.305m
    # wheel_rpm at 60mph ≈ 1000 RPM (approximate)
    wheel_rpm_60mph = 1000

    for gear in range(1, 7):
        gear_ratio = powertrain.gearbox.get_gear_ratio(gear)
        overall = gear_ratio * powertrain.gearbox.FINAL_DRIVE
        engine_rpm_60 = wheel_rpm_60mph * overall
        print(f"{gear:6d} | {gear_ratio:10.3f} | {overall:10.3f} | {engine_rpm_60:12.0f}")

    print("\n3. Acceleration Simulation (1st gear, full throttle):")
    print("-" * 70)
    print(f"{'Time (s)':>8} | {'RPM':>8} | {'Torque':>10} | {'Wheel Torque':>12}")
    print("-" * 70)

    # Reset to 1st gear
    powertrain.gearbox.current_gear = 1
    powertrain.engine.rpm = 2000

    dt = 0.1
    time = 0.0

    for _ in range(30):
        # Simulate wheel RPM increasing (simple model)
        wheel_rpm = powertrain.engine.rpm / powertrain.gearbox.get_current_ratio()

        # Get torque
        wheel_torque = powertrain.get_wheel_torque(throttle=1.0, wheel_rpm=wheel_rpm)
        engine_torque = powertrain.engine.get_torque(throttle=1.0)

        # Update
        powertrain.update(dt, wheel_rpm)

        # Simulate engine speed increasing (simplified)
        powertrain.engine.rpm += 200

        if time % 0.5 < 0.1:  # Print every 0.5s
            print(f"{time:8.1f} | {powertrain.engine.rpm:8.0f} | "
                  f"{engine_torque:10.1f} | {wheel_torque:12.1f}")

        time += dt

        if powertrain.engine.rpm > 7000:
            break

    print("\n" + "="*70)
    print("Test complete!")
    print("="*70)
