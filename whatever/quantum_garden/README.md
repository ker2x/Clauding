# Quantum Garden

An interactive quantum mechanics simulation game where you cultivate virtual quantum gardens through wave function manipulation.

## Gameplay

- **Plant Seeds**: The game starts with a quantum wave packet (Gaussian distribution)
- **Shape Reality**: Click and drag to draw potential barriers that bend quantum trajectories
- **Nurture Complexity**: Watch as particles follow Bohmian paths, creating emergent patterns
- **Score Points**: Earn points based on wave function complexity and pattern stability

## Quick Start

```bash
# Build
make

# Run
./SwiftParticles
```

### Controls

| Key   | Action |
|-------|--------|
| Q     | Quit application |
| R     | Reset simulation |
| C     | Clear all potentials |
| Space | Pause/Resume simulation |
| Mouse | Draw potential barriers |
| Option + Mouse | Erase potential barriers |

## Project Structure

```
quantum_garden/
├── Makefile          # Build configuration
├── main.swift        # Application code (Pure Swift)
├── Compute.metal     # GPU shaders (Metal Shading Language)
├── README.md         # This file
├── PLAN.md           # Original game design plan
└── SwiftParticles    # Built executable
```

## Technical Implementation

This game implements quantum mechanics simulation using GPU compute:

- **Wave Function Evolution**: Finite difference solution to Schrödinger equation
- **Bohmian Particle Dynamics**: Particles follow quantum potential gradients
- **Interactive Potentials**: User-drawn barriers affect quantum behavior
- **Real-time Scoring**: Complexity metrics based on wave function properties

### Architecture Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                         Quantum Garden                          │
├─────────────────────────────────────────────────────────────────┤
│  main()           Create window, handle input, manage game      │
│  AppDelegate      Handle application lifecycle                   │
│  InputView        Capture mouse and keyboard input               │
│  Renderer         Manage Metal resources and quantum simulation  │
├─────────────────────────────────────────────────────────────────┤
│                          Metal GPU                               │
├─────────────────────────────────────────────────────────────────┤
│  Wave Kernel      evolve_wave() - quantum time evolution        │
│  Particle Kernel  update_particles() - Bohmian dynamics         │
│  Vertex Shader    vertex_main() - particle rendering            │
│  Fragment Shader  fragment_main() - color based on wave function │
└─────────────────────────────────────────────────────────────────┘
```

## Requirements

- macOS 10.14 or later (Metal support)
- Xcode Command Line Tools (for `swiftc`)

Install command line tools if needed:
```bash
xcode-select --install
```

## Libraries and Dependencies

**Zero external dependencies**. This game uses only built-in macOS frameworks:

- **Metal** - Core GPU framework for compute and graphics
- **MetalKit** - High-level utilities (MTKView, render loop)
- **AppKit** - macOS windowing and event handling
- **QuartzCore** - High-precision timing (CACurrentMediaTime)

**Pure Metal Implementation** - This game does NOT use:
- ❌ Metal Performance Shaders (MPS) for neural networks
- ❌ MPSGraph for computation graphs
- ❌ CoreML for machine learning
- ❌ Accelerate framework
- ❌ Any third-party libraries

All quantum computations are **hand-coded in Metal Shading Language**:
- Wave function time evolution (finite difference Schrödinger equation)
- Bohmian particle dynamics (∇ψ/ψ velocity calculations)
- Interactive potential field manipulation
- Real-time complexity scoring

## How Quantum Simulation Works

This game implements **Bohmian quantum mechanics** where particles follow deterministic trajectories guided by a quantum wave function. The simulation runs entirely on the GPU for real-time performance.

### Frame Loop

Each frame follows this sequence:

1. **Wave Evolution**: GPU runs the `evolve_wave` kernel
   - Finite difference time-stepping of Schrödinger equation
   - User-drawn potentials affect wave propagation
   - Decoherence reduces wave function complexity over time

2. **Particle Update**: GPU runs the `update_particles` kernel
   - Bohmian velocities calculated from ∇ψ/ψ
   - Particles follow quantum potential gradients
   - Positions updated with velocity integration

3. **Render Pass**: GPU draws particles and wave visualization
   - Vertex shader transforms particle positions to screen coordinates
   - Fragment shader colors particles based on local wave function
   - Wave function visualized through particle color mapping

4. **Score Update**: Complexity metrics calculated from wave function
   - Points awarded based on pattern stability and emergence
   - Real-time feedback in window title

5. **Present**: The rendered frame is displayed

### Quantum Physics Implementation

The game uses simplified quantum mechanics:

- **Wave Function**: Complex field ψ(x,y,t) evolving by i∂ψ/∂t = -∇²ψ/2 + V(x,y)ψ
- **Bohmian Trajectories**: Particle velocities v = Im(∇ψ / ψ)
- **Interactive Potentials**: User-drawn V(x,y) barriers
- **Scoring**: Complexity = ∫|ψ|² dA (wave function "energy")

## Customization

### Adjusting Simulation Parameters

Edit `Config` struct in `main.swift`:

```swift
struct Config {
    static let gridWidth = 256      // Wave function resolution
    static let gridHeight = 256     // Higher = more detailed but slower
    static let numParticles = 10000 // More particles = richer visualization
    static let waveDt: Float = 0.1  // Time step for quantum evolution
}
```

### Modifying Quantum Behavior

The quantum physics are defined in `Compute.metal`:

```metal
// Adjust wave evolution speed
float dt = params.wave_dt;

// Modify decoherence rate
float decay = 1.0 - params.decoherence_rate * dt;

// Change Bohmian velocity scaling
vel.x += vx * params.particle_dt * 10.0;
```

## Performance Notes

- **GPU Requirements**: Modern Mac with Metal support (2012+)
- **Typical Performance**: 60 FPS with 10K particles on M1/M2 Macs
- **Memory Usage**: ~50MB for 256×256 wave grid + 10K particles
- **Quantum Accuracy**: Simplified finite difference (not fully physically accurate)
- **Educational Value**: Demonstrates Bohmian interpretation of quantum mechanics

## Comparing Swift and Objective-C++ Versions

This project has a **nearly identical** Objective-C++ version at `../metal_particle_template/`.

### Code Comparison Examples

**Creating a buffer:**

```objc
// Objective-C++
id<MTLBuffer> buffer = [device newBufferWithLength:size
                                           options:MTLResourceStorageModeShared];
```

```swift
// Swift
let buffer = device.makeBuffer(
    length: size,
    options: .storageModeShared
)
```

**Handling errors:**

```objc
// Objective-C++
NSError* error = nil;
NSString* source = [NSString stringWithContentsOfFile:path
                                            encoding:NSUTF8StringEncoding
                                               error:&error];
if (!source) {
    NSLog(@"Error: %@", error);
    return NO;
}
```

```swift
// Swift
do {
    let source = try String(contentsOfFile: path, encoding: .utf8)
} catch {
    print("Error: \(error)")
    return false
}
```

**Creating objects:**

```objc
// Objective-C++
MTKView* view = [[MTKView alloc] initWithFrame:frame];
```

```swift
// Swift
let view = MTKView(frame: frame)
```

### Which Should You Use?

**Use Swift if:**
- ✅ Starting a new project
- ✅ Want cleaner, more readable code
- ✅ Prefer modern language features
- ✅ Building for Apple platforms only

**Use Objective-C++ if:**
- ✅ Integrating with existing C++ code
- ✅ Need maximum portability
- ✅ Working with legacy codebases
- ✅ Prefer explicit memory control

## Performance Comparison

| Metric | Objective-C++ | Swift |
|--------|---------------|-------|
| **Compile time** | ~0.5s | ~2-3s |
| **Binary size** | ~77 KB | ~110 KB |
| **Runtime FPS** | 60 FPS | 60 FPS |
| **GPU performance** | Identical | Identical |

**Conclusion**: Both versions have **identical GPU performance** because they use the same Metal shaders. The Swift version has longer compile times but the same runtime speed.

## Common Issues

### Shader Compilation Errors

If you see "Could not load shader file", make sure `Compute.metal` is in the same directory as the executable:

```bash
ls -la
# Should show both SwiftParticles and Compute.metal
```

### Metal Not Supported

If you see "Metal is not supported on this device", you're on older hardware without Metal support (pre-2012 Macs).

### Swift Version Mismatch

If you see "Module compiled with Swift X.Y cannot be imported", your Swift version might be incompatible. Check your version:

```bash
swiftc --version
```

## Future Extensions

Ideas for expanding Quantum Garden:

1. **Multiple Wave Packets**: Allow planting different colored quantum seeds
2. **Quantum Interference**: Create superposition states with multiple potentials
3. **Time Reversal**: Allow rewinding quantum evolution
4. **Level Progression**: Structured challenges with target patterns
5. **Sound Integration**: Audio generated from wave function frequencies
6. **3D Extension**: Move to 3D quantum simulation with raytracing

## Related Projects

- `../satori_quantum.py` - Bohmian quantum mechanics in Python
- `../metal_particle_template/` - Original particle template
- `../CFD/` - Fluid dynamics simulation (similar GPU compute)
- `../metal_neural_crystal/` - Neural cellular automata

## Documentation Philosophy

This game follows the **"What, Why, How"** framework:

- **What**: Interactive quantum mechanics game in Swift + Metal
- **Why**: Creative exploration of quantum concepts through gaming
- **How**: GPU-accelerated wave function evolution and Bohmian particle dynamics

## License

This project is released into the public domain. Experiment and learn!
