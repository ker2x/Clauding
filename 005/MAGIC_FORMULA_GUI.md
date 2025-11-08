# Magic Formula Parameter GUI Tools

This directory contains two interactive tools for experimenting with Pacejka Magic Formula tire parameters:

## 1. Magic Formula Visualizer (Standalone)

A standalone tool for visualizing and experimenting with tire force curves without running the game.

### Usage:
```bash
python magic_formula_visualizer.py
```

### Features:
- **Interactive Sliders**: Adjust all 8 Pacejka parameters (B, C, D, E for lateral and longitudinal)
- **Real-time Graphs**: See tire force curves update instantly
- **Peak Indicators**: Automatic peak force and slip angle/ratio markers
- **Parameter Info**: Tooltips and descriptions for each parameter

### Controls:
- **Mouse**: Drag sliders to adjust parameters
- **R**: Reset to default values
- **ESC**: Quit

### Perfect for:
- Learning how Magic Formula parameters affect tire behavior
- Tuning parameters before testing in-game
- Understanding tire physics concepts

---

## 2. Play Human GUI (In-Game)

Play CarRacing-v3 with live parameter adjustment and visualization.

### Usage:
```bash
python play_human_gui.py [--episodes N] [--fps FPS]
```

### Features:
- **Full Gameplay**: Drive with keyboard controls
- **Live Parameter Adjustment**: Change tire parameters while driving
- **Real-time Visualization**: See force curves update as you adjust
- **Immediate Effect**: Parameters update the car physics instantly
- **Wheel Slip Display**: Real-time slip angle and slip ratio for all 4 wheels (FL, FR, RL, RR)

### Controls:

#### Driving (AZERTY Keyboard):
- **Steering**: Q/D or Arrow Keys
- **Gas**: Z or Up Arrow
- **Brake**: S or Down Arrow
- **Reset Episode**: R
- **Quit**: ESC

#### Parameter Adjustment:
- **Mouse**: Drag sliders on the right panel
- Changes take effect immediately

### Command-line Options:
```bash
--episodes N      Number of episodes to play (default: 5)
--fps FPS         Display frame rate (default: 50)
--no-render       Run without visualization (for testing)
```

---

## Pacejka Magic Formula Parameters

The Pacejka Magic Formula models tire force as a function of slip:

```
F = D × sin(C × arctan(B×α - E×(B×α - arctan(B×α))))
```

Where `α` is slip angle (lateral) or slip ratio (longitudinal).

### Parameters Explained:

#### **B - Stiffness Factor**
- Controls the initial slope of the force curve
- Higher B = stiffer, more responsive tire
- **Typical range**: 8-15 for road cars
- **Current defaults**: 10.0 (lateral), 9.0 (longitudinal)

#### **C - Shape Factor**
- Controls overall curve shape and "peakiness"
- Affects how sharp the peak is
- **Typical range**: 1.3-2.5
- **Current defaults**: 1.9 (both)

#### **D - Peak Friction Coefficient**
- Peak force multiplier
- D=1.0 means peak friction equals surface friction
- D>1.0 means tire can exceed surface friction at optimal slip
- **Typical range**: 0.9-1.3 for road cars
- **Current defaults**: 1.1 (lateral), 1.4 (longitudinal)

#### **E - Curvature Factor**
- Controls curve shape near and after peak
- Affects how sharply grip falls off past optimal slip
- **Typical range**: 0.9-1.0
- **Current defaults**: 0.95 (both)

### Lateral vs Longitudinal:

**Lateral (Cornering)**:
- Uses slip angle (difference between tire direction and travel direction)
- Peak slip angles typically 10-15° for road cars, 6° for race cars
- Controls cornering grip and understeer/oversteer behavior

**Longitudinal (Acceleration/Braking)**:
- Uses slip ratio (wheel speed vs ground speed)
- Slip ratio of 0 = no slip, 1 = full wheelspin, -1 = locked wheel
- Controls acceleration and braking performance

---

## Tips for Parameter Tuning:

### For Better Grip:
- Increase **D** (peak friction) - more overall grip
- Keep B high (10-15) - responsive initial grip

### For More Forgiving Handling:
- Lower **B** slightly (7-9) - more progressive breakaway
- Increase **E** (0.95-1.1) - gentler falloff after peak

### For Racing Setup:
- High **B** (12-15) - sharp response
- High **D_lat** (1.2-1.4) - cornering grip
- High **D_lon** (1.3-1.5) - acceleration/braking grip

### For Drifting Setup:
- Lower **D_lat** (0.9-1.0) - easier to break rear loose
- Higher **D_lon** (1.2-1.4) - maintain acceleration in drift
- Lower **B_lat** (7-9) - more progressive slide

---

## Dependencies:

Both tools require:
- `pygame` - GUI and game rendering
- `matplotlib` - Graph visualization
- `numpy` - Numerical computations
- Custom environment files from this directory

Install with:
```bash
pip install pygame matplotlib numpy
```

---

## Visual Guide:

### Standalone Visualizer Layout:
```
┌─────────────────────┬───────────────────────────┐
│  Parameter Sliders  │   Lateral Force Curve     │
│                     │   (vs Slip Angle)         │
│  B_lat ========○    │                           │
│  C_lat ======○      │   ╱╲  Peak markers        │
│  D_lat =======○     │  ╱  ╲  shown              │
│  E_lat ========○    ├───────────────────────────┤
│                     │   Longitudinal Force      │
│  B_lon =======○     │   (vs Slip Ratio)         │
│  C_lon ======○      │                           │
│  D_lon ========○    │   ╱╲  Peak markers        │
│  E_lon =======○     │  ╱  ╲  shown              │
└─────────────────────┴───────────────────────────┘
```

### In-Game GUI Layout (Horizontal, Optimized for Wide Screens):
```
┌────────────────────┬─────────────┬──────────────┐
│  Game Info         │   Param     │  Lateral     │
├────────────────────┤  Sliders    │  Force Graph │
│                    │  B_lat ===○ │   ╱╲         │
│  Car Racing View   │  C_lat ==○  │  ╱  ╲        │
│  (Gameplay)        │  D_lat ===○ ├──────────────┤
│                    │  E_lat ===○ │ Longitudinal │
│                    │  B_lon ===○ │ Force Graph  │
│                    │  C_lon ==○  │   ╱╲         │
│                    │  D_lon ===○ │  ╱  ╲        │
│                    │  E_lon ===○ │              │
│                    ├─────────────┤              │
│                    │ Wheel Slip  │              │
│                    │ FL SA:|█|SR │              │
│                    │ FR SA:|█|SR │              │
│                    │ RL SA:|█|SR │              │
│                    │ RR SA:|█|SR │              │
└────────────────────┴─────────────┴──────────────┘
```

**Horizontal Compact Design:**
- Game view on the left
- Parameter sliders in middle column (compact spacing)
- Wheel slip display below sliders
- Tire force graphs on the right (stacked vertically)
- Total width ~1400px, height ~600px
- Optimized for wide 1080p/1440p screens

**Wheel Slip Display:**
- **SA (Slip Angle)**: Shows the angle between tire direction and travel direction (-25° to +25°)
  - Blue bars = positive slip (left turn)
  - Orange bars = negative slip (right turn)
- **SR (Slip Ratio)**: Shows wheel spin vs ground speed (-1 to +1)
  - Green bars = acceleration (positive slip)
  - Red bars = braking (negative slip)

---

## Example Workflow:

1. **Start with the visualizer** to experiment:
   ```bash
   python magic_formula_visualizer.py
   ```

2. **Adjust parameters** and observe curve changes:
   - Try increasing D_lat to see peak grip increase
   - Lower B_lat to see more gradual initial response
   - Adjust E to change falloff characteristics

3. **Test in-game**:
   ```bash
   python play_human_gui.py --episodes 3
   ```

4. **Fine-tune while driving**:
   - Adjust sliders during gameplay
   - Feel the immediate handling changes
   - Find the setup that matches your driving style

---

## Understanding Wheel Slip:

The real-time wheel slip display shows exactly what's happening at each tire contact patch:

### Slip Angle (SA):
- **What it is**: The angle between where the tire is pointed and where it's actually going
- **Range**: -25° to +25° (typical peak grip occurs around 10-15° for road tires)
- **When you see it**:
  - Cornering: Front wheels show slip angle as you steer
  - Understeer: Front wheels have higher slip angles than rear
  - Oversteer: Rear wheels have higher slip angles than front
- **Color coding**:
  - Blue = Left slip (tire sliding to the left of its heading)
  - Orange = Right slip (tire sliding to the right)

### Slip Ratio (SR):
- **What it is**: The difference between wheel speed and ground speed
- **Range**: -1 (locked wheel) to +1 (full wheelspin)
- **Optimal**: Peak grip typically occurs around 0.1-0.2 slip ratio
- **When you see it**:
  - Acceleration: Rear wheels show positive slip (green)
  - Braking: All wheels show negative slip (red)
  - Wheelspin: Large positive values on rear wheels
  - Wheel lock: Values near -1.0 during hard braking
- **Color coding**:
  - Green = Positive slip (wheel spinning faster than ground)
  - Red = Negative slip (wheel spinning slower, braking)

### Tips for using slip data:
1. **Smooth driving**: Keep slip values in the optimal range (SA: 5-12°, SR: 0.1-0.2)
2. **Detect understeer**: Front SA much higher than rear during cornering
3. **Detect oversteer**: Rear SA suddenly spikes during turn
4. **Optimize braking**: Watch for wheels approaching -1.0 SR (lock-up)
5. **Traction control**: Monitor rear SR during acceleration, avoid exceeding 0.3

---

## Understanding the Graphs:

### Lateral Force Graph:
- **X-axis**: Slip angle in degrees (-25° to +25°)
- **Y-axis**: Lateral force in Newtons
- **Peak marker**: Shows maximum cornering force and optimal slip angle
- **Shape**: Symmetric curve peaking around 10-15° for road tires

### Longitudinal Force Graph:
- **X-axis**: Slip ratio (-1 to +1)
  - Negative = braking
  - Positive = acceleration
  - 0 = no slip (rolling)
- **Y-axis**: Longitudinal force in Newtons
- **Peak marker**: Shows maximum accel/brake force and optimal slip
- **Shape**: Symmetric curve peaking around 0.1-0.2 slip ratio

---

## Technical Notes:

- Normal force used in graphs: 2600N (≈ weight per wheel for MX-5)
- Surface friction coefficient: 1.0 (asphalt)
- Actual peak force = D × normal_force × surface_friction
- Parameters affect the **shape** of curves, not just magnitude

---

## Troubleshooting:

### Graphs not updating?
- Make sure you're dragging the slider, not just clicking
- Check that matplotlib is properly installed

### Game feels different but looks the same?
- Parameters only affect physics, not visuals
- Try extreme values (like D_lat=0.5) to feel the difference clearly

### Performance issues?
- Lower FPS: `python play_human_gui.py --fps 30`
- Reduce episodes: `python play_human_gui.py --episodes 1`

---

Enjoy experimenting with tire physics! 🏎️
