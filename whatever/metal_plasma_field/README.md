# Metal Plasma Field Simulation

**What**: GPU-accelerated charged particle simulation with electromagnetic field physics.

**Why**: Creates mesmerizing plasma-like visualizations through realistic electromagnetic force calculations. Unlike simple particle systems, this simulates the fundamental physics of charged particles in fields.

**How**: 50,000 particles with three charge types (electrons, neutral, ions) interact via Coulomb forces and Lorentz dynamics.

## Libraries and Dependencies

**Zero external dependencies**. Uses only built-in macOS frameworks:
- Metal
- MetalKit
- AppKit
- simd

**100% Custom Implementation** - Does NOT use:
- CoreML / MPSGraph / BNNS
- Any physics engine
- Any particle system library

## Build

```bash
cd metal_plasma_field
clang++ -o PlasmaField main.mm \
    -framework Metal -framework MetalKit -framework AppKit \
    -framework QuartzCore -fobjc-link-runtime -lz
./PlasmaField
```

## Controls

| Key | Action |
|-----|--------|
| Q | Quit |
| R | Reset particles |
| F | Cycle field mode |
| Space | Pause/Resume |

## Physics

### Forces Implemented

1. **Coulomb Force**: `F = k * q₁ * q₂ / r²`
   - Like charges repel, opposite charges attract
   - Creates charge clustering and separation

2. **Lorentz Force**: `F = q(E + v × B)`
   - Electric field accelerates particles
   - Magnetic field causes spiraling (gyromotion)

3. **Thermal Noise**: Brownian motion for realistic plasma temperature

### Field Modes

Press **F** to cycle between:

1. **Uniform E+B Field**: Constant fields throughout space. Electrons spiral, ions drift.

2. **Central Pole**: Radial 1/r² field. Creates orbiting patterns and density waves.

3. **Rotating Vortex**: Time-varying spiral field. Particles trace out mesmerizing spirals.

## Visual Design

- **Electrons (cyan)**: Negative charge, spiral in magnetic fields
- **Neutral (white)**: No charge, drift with turbulence
- **Ions (orange/red)**: Positive charge, move toward electron clusters

## Performance

- 50,000 particles at 60 FPS on Apple Silicon
- O(n²) particle interactions (could optimize with spatial hashing)
- GPU compute shader for all physics
