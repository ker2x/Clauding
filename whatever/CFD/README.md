# Virtual Wind Tunnel - 2D CFD Simulation

Real-time computational fluid dynamics (CFD) simulation using the Lattice Boltzmann Method (LBM) on Apple Silicon GPU.

## Features

- **Real CFD Physics**: Solves incompressible Navier-Stokes equations using D2Q9 Lattice Boltzmann Method
- **GPU Accelerated**: All computation runs on Metal GPU for real-time performance
- **Interactive**: Draw obstacles with mouse, adjust wind speed in real-time
- **Layered Visualization**: Velocity magnitude (color) + vorticity contours (overlay)
- **Configurable**: JSON configuration file for simulation parameters

## Quick Start

### Option 1: Command Line (Simplest!)

```bash
cd CFD
./run.sh
```

That's it! The script will build if needed and launch the app.

Or manually:
```bash
cd CFD
./build.sh           # Build the app bundle
open WindTunnel.app  # Launch it
```

### Option 2: Xcode (For debugging)

1. Open Xcode
2. Create new project: **File > New > Project > macOS > App**
3. Name it "WindTunnel", select SwiftUI interface
4. Delete the default `ContentView.swift` and `WindTunnelApp.swift`
5. Add files to project:
   - Drag `WindTunnel.swift` into the project
   - Drag `Shaders.metal` into the project
   - Drag `config.json` into the project (ensure "Copy items if needed" is checked)
6. **Build and Run** (⌘R)

**Note**: Metal shaders will be compiled from source at runtime (fast enough). Pre-compilation requires the Metal toolchain component, which is optional.

## Usage

### Controls

**Mode Buttons:**
- **✏️ Draw** - Click and drag to draw obstacles (pauses simulation)
- **🧹 Erase** - Click and drag to erase obstacles
- **▶️ Run** - Run the simulation

**Sliders:**
- **Wind Speed** - Adjust inlet velocity (0 to max_wind_speed from config)
- **Brush Size** - Change drawing brush radius

**Action Buttons:**
- **Clear Obstacles** - Remove all drawn obstacles
- **Reset Simulation** - Clear everything and reinitialize

### Tips

1. **Start Simple**: Run simulation first to see baseline flow
2. **Draw a Circle**: Switch to Draw mode, draw a circle obstacle
3. **Watch Vortices**: Switch back to Run mode and observe vortex streets forming
4. **Experiment**: Try different shapes (airfoil, cylinder, flat plate)
5. **Adjust Speed**: Higher wind speed = more dramatic vortices (but less stable)

## Configuration

Edit `config.json` to customize:

```json
{
  "simulation": {
    "grid_width": 512,           // Grid resolution
    "grid_height": 256,
    "steps_per_frame": 5,        // Higher = more accurate, slower
    "relaxation_time": 0.7,      // tau: viscosity (0.5-1.0)
    "initial_wind_speed": 0.05,  // Starting velocity
    "max_wind_speed": 0.15       // Maximum slider value
  },
  "visualization": {
    "vorticity_threshold": 0.01  // Vorticity overlay sensitivity
  },
  "interaction": {
    "initial_brush_size": 5,
    "min_brush_size": 1,
    "max_brush_size": 20
  }
}
```

### Key Parameters

- **steps_per_frame**: Balance between accuracy and speed
  - 1 = fastest, least accurate
  - 5-10 = good balance (recommended)
  - 20+ = very accurate, slower

- **relaxation_time** (τ): Controls fluid viscosity
  - τ = 0.6: Low viscosity (water-like)
  - τ = 0.7-0.8: Medium (air-like) ✓
  - τ = 1.0: High viscosity (honey-like)
  - Must be > 0.5 for numerical stability

- **wind_speed**: Inlet velocity in lattice units
  - < 0.1: Safe, stable
  - 0.1-0.15: Interesting vortices
  - > 0.2: May become unstable (fun to experiment!)

## Technical Details

### Lattice Boltzmann Method (D2Q9)

This is **real CFD**, not a game approximation:

- Solves Navier-Stokes equations from kinetic theory
- 9 discrete velocity directions (D2Q9 model)
- BGK collision operator for simplicity
- Weakly compressible (Mach number < 0.3)
- Approximates incompressible flow at low velocities

### Boundary Conditions

- **Left (Inlet)**: Fixed velocity (Zou-He equilibrium)
- **Right (Outlet)**: Zero-gradient extrapolation
- **Top/Bottom**: Slip walls (specular reflection)
- **Obstacles**: Bounce-back (no-slip)

### Kernels

1. **lbm_stream_collide**: Core LBM algorithm
2. **lbm_boundary**: Apply boundary conditions
3. **compute_fields**: Calculate velocity and vorticity
4. **render_field**: Visualization with Viridis colormap
5. **update_obstacles**: Mouse drawing
6. **initialize_distributions**: Set initial state

### Performance

On M3 MacBook Air:
- 512x256 grid
- 5 steps/frame
- Expected: 60 FPS easily

## Physics to Observe

- **Von Kármán Vortex Street**: Alternating vortices behind cylinders
- **Flow Separation**: Boundary layer detachment
- **Recirculation Zones**: Low-pressure regions behind obstacles
- **Vorticity Generation**: Shear at obstacle boundaries

## Troubleshooting

**Black screen?**
- Check console for Metal initialization errors
- Verify Metal shaders compiled correctly

**Simulation unstable (explodes)?**
- Reduce wind speed
- Increase relaxation time
- Decrease steps_per_frame

**Performance issues?**
- Reduce grid resolution in config
- Reduce steps_per_frame
- Check Activity Monitor for GPU usage

**Config not loading?**
- Ensure `config.json` is in `CFD/` directory
- Check JSON syntax
- Look for error messages in console

## Theory References

- **LBM**: McNamara & Zanetti (1988)
- **D2Q9 Model**: Qian, d'Humières, Lallemand (1992)
- **BGK Approximation**: Bhatnagar, Gross, Krook (1954)

## Future Enhancements

- Temperature/heat transfer
- Multiple fluid phases
- Turbulence models (LES)
- Particle tracers
- Pressure field visualization
- Export to video
- 3D (if feasible on M3)

---

**Made with real CFD physics on Apple Silicon** 🌪️
