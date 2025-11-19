# MX5 Interactive Telemetry Controls

## Overview

The MX5 telemetry system now includes **interactive controls** allowing you to manually drive the vehicle and control the engine/gearbox in real-time while viewing professional telemetry data.

## Running the Interactive Simulator

```bash
python mx5_telemetry.py
```

This launches the telemetry display with full manual control.

## Keyboard Controls (AZERTY Layout)

### Driving Controls

| Key | Action | Description |
|-----|--------|-------------|
| **Z** | Accelerate | Hold to increase throttle (0-100%) |
| **S** | Brake | Hold to apply brakes (0-100%) |
| **Space** | Release | Instantly release throttle and brake |

### Gearbox Controls

| Key | Action | Description |
|-----|--------|-------------|
| **E** | Shift Up | Shift to next higher gear |
| **A** | Shift Down | Shift to next lower gear |
| **1-6** | Direct Gear | Select specific gear (1st through 6th) |
| **N** | Neutral | Shift to neutral (0) |

### System Controls

| Key | Action | Description |
|-----|--------|-------------|
| **ESC** | Quit | Exit the simulator |

## How It Works

### Input Behavior

- **Throttle/Brake**:
  - Holding **Z** gradually increases throttle (5% per frame)
  - Holding **S** gradually increases brake (5% per frame)
  - Releasing keys causes smooth decay (95% per frame)
  - Throttle and brake are mutually exclusive (applying one releases the other)

- **Gear Shifting**:
  - Shifts take **150ms** to complete (realistic manual transmission timing)
  - Clutch automatically disengages during shifts
  - Clutch re-engages after **200ms** (smooth engagement)
  - Cannot shift to gears that don't exist (e.g., can't shift up from 6th)

### Physics Simulation

The simulator includes:
- **Realistic engine torque curve** (2.0L SKYACTIV-G, 181 hp, 205 Nm)
- **6-speed manual gearbox** with accurate ratios
- **Wheel torque calculation** based on current gear
- **Drag forces** (aerodynamic resistance)
- **Brake forces** (8000N maximum braking force)
- **Vehicle mass** (1062 kg - MX-5 ND curb weight)

### Telemetry Display

While driving, you can monitor:

**Gauges:**
- RPM tachometer (with redline warning at 7500 RPM)
- Speed (km/h)
- Current gear (with shift indicator)
- Engine temperature
- Oil pressure

**Input Bars:**
- Throttle position (green)
- Brake position (red)
- Oil pressure (gold)

**Live Charts:**
- RPM history (10-second window)
- Speed history (10-second window)
- Throttle/brake overlay

**Info Panel:**
- Current power (hp)
- Current torque (Nm)
- Gear ratio
- Fuel cut status

## Tips for Driving

### Starting from Stop

1. Vehicle starts in **1st gear**
2. Press and hold **Z** to accelerate
3. Watch the RPM gauge - shift at ~7000 RPM for maximum acceleration
4. Press **E** to shift up when ready

### Optimal Shifting

The SKYACTIV-G engine produces:
- Peak torque: **205 Nm @ 4000 RPM**
- Peak power: **181 hp @ 7000 RPM**
- Redline: **7500 RPM** (fuel cut activates)

For best acceleration:
- Shift at **7000-7200 RPM** (just before redline)
- After shifting, RPM will drop based on gear ratios

### Gear Ratios

| Gear | Ratio | Overall (×3.909) | 60 mph RPM |
|------|-------|------------------|------------|
| 1st  | 3.760 | 14.698 | ~14,700 |
| 2nd  | 2.269 | 8.870 | ~8,900 |
| 3rd  | 1.645 | 6.430 | ~6,400 |
| 4th  | 1.257 | 4.914 | ~4,900 |
| 5th  | 1.000 | 3.909 | ~3,900 |
| 6th  | 0.830 | 3.244 | ~3,200 |

### Top Speed

The simulation includes realistic drag forces, so top speed is limited by:
- Engine power vs. aerodynamic drag
- Current gear ratio
- In 6th gear, expect ~200+ km/h top speed

### Braking

- Press **S** to brake
- Maximum braking force is 8000N
- Vehicle cannot reverse (no reverse gear)
- Speed can only decrease to 0 km/h

## Example Session

```
1. Launch: python mx5_telemetry.py
2. Wait for GUI to appear (starts in 1st gear at idle)
3. Hold Z to accelerate
4. Watch RPM climb to 7000
5. Press E to shift to 2nd
6. Continue accelerating
7. Experiment with different gears
8. Press S to brake
9. Press ESC to quit
```

## Troubleshooting

### Keys not responding
- Make sure the telemetry window has focus (click on it)
- Check keyboard layout is AZERTY

### Gear won't shift
- Wait for current shift to complete (150ms delay)
- Cannot shift beyond 6th or below 1st

### RPM stuck at redline
- Fuel cut is active at 7500 RPM
- Shift up or release throttle

### Speed not increasing
- Check current gear (might be in neutral)
- Check throttle position (hold Z)
- Ensure clutch is engaged (wait after shift)

## Advanced Features

### Session Statistics

When you quit (ESC), the simulator displays:
- Total session time
- Maximum speed achieved
- Telemetry history is preserved in the charts

### Manual Clutch Control

The clutch is automatically controlled:
- Engages when not shifting
- Disengages during gear changes
- Gradual engagement over 200ms

## Integration

You can also use the `InteractiveDriveSimulator` class in your own code:

```python
from mx5_telemetry import InteractiveDriveSimulator

simulator = InteractiveDriveSimulator()
simulator.run()
```

Or integrate with existing telemetry:

```python
from mx5_telemetry import MX5TelemetryDisplay
from mx5_powertrain import MX5Powertrain

telemetry = MX5TelemetryDisplay()
powertrain = MX5Powertrain()

# Your control loop
while running:
    state = powertrain.get_state()
    state['throttle'] = your_throttle
    state['brake'] = your_brake
    state['speed_kmh'] = your_speed

    telemetry.update(state)
```

## Technical Details

### Update Rate
- Physics: 50 Hz (0.02s timestep)
- Display: 50 Hz (synchronized with physics)

### Key Event Handling
- Uses matplotlib key press/release events
- Maintains set of currently pressed keys
- Supports key hold detection for smooth throttle/brake control

### AZERTY vs QWERTY
If you're using QWERTY keyboard, the physical keys are:
- Z (AZERTY) = W (QWERTY physical position)
- S (AZERTY) = S (QWERTY physical position)
- A (AZERTY) = Q (QWERTY physical position)
- E (AZERTY) = E (QWERTY physical position)

The code uses key names (not positions), so on QWERTY you press W/S instead of Z/S.
