# Metal Neural Aura

Neural particle field simulation with Metal compute shaders. A Metal/ObjC++ port of `aura_sim.py` featuring a 3-layer neural network implemented entirely on the GPU.

## Overview

Metal Neural Aura simulates 131,072 particles driven by a neural network that generates forces based on particle position and local density context. The network's weights are mutable, allowing the simulation to evolve and explore different emergent behaviors through mutation.

### Features

- **GPU-accelerated neural network**: 3-layer MLP (65→128→128→2) running in Metal compute shaders
- **Fourier positional encoding**: 32 frequency pairs for rich spatial features
- **Density-aware forces**: Particles respond to local crowding via a 128×128 density grid
- **Mutable weights**: ~25,000 neural network parameters that can be mutated in real-time
- **Per-particle colors**: RGB values that evolve based on velocity and density
- **High performance**: 60 FPS with 131k particles on Apple Silicon

### Libraries and Dependencies

**Zero external dependencies**. Metal Neural Aura uses only built-in macOS frameworks:

- **Metal** - Core GPU API for compute and rendering
- **MetalKit** - MTKView and render loop management
- **AppKit** - macOS windowing and event handling
- **QuartzCore** - High-precision timing (CACurrentMediaTime)

**No ML frameworks required**: Unlike the Python version (PyTorch + MPS), this implementation uses raw Metal compute shaders. The neural network is implemented manually with custom matrix-vector multiplication and activation functions written in Metal Shading Language.

**100% Custom Neural Network Implementation** - This project does NOT use:
- ❌ CoreML
- ❌ MPSGraph (Metal Performance Shaders Graph)
- ❌ MPSMatrix / MPSMatrixMultiplication
- ❌ MPSNNGraph (MPS Neural Network Graph)
- ❌ BNNS (Basic Neural Network Subroutines)
- ❌ Accelerate framework's neural network functions
- ❌ Any ML/NN library whatsoever

Instead, **everything is hand-coded in Metal Shading Language**:
- ✅ Matrix-vector multiplication (`matvec` function in Compute.metal)
- ✅ Tanh activation (`tanh_approx` using rational function approximation)
- ✅ Forward pass logic manually implemented in `neural_force_kernel`
- ✅ Weight storage as plain `MTLBuffer` float arrays
- ✅ Manual Gaussian noise generation for mutations (Box-Muller transform)

This is **bare-metal GPU programming** - every matrix multiply, every activation, every forward pass is explicitly written out. No abstraction layers, no framework magic, just direct GPU computation.

**Why this approach?**
- Full control over neural network architecture and execution
- Direct weight mutation via CPU-GPU shared memory
- Educational transparency - see exactly how neural networks operate at the GPU level
- Smaller binary size and faster startup
- No Python runtime or package dependencies
- Complete understanding of every operation

## Building and Running

### Prerequisites

- macOS with Metal support
- Xcode Command Line Tools: `xcode-select --install`
- Apple Silicon (M1/M2/M3) recommended for best performance

### Build

```bash
make
```

### Run

```bash
./MetalNeuralAura
```

Or build and run in one command:

```bash
make run
```

### Clean

```bash
make clean
```

## Controls

| Key | Action |
|-----|--------|
| **Space** | Mutate neural network (evolve new behaviors) |
| **R** | Reset particles to random positions |
| **P** | Pause/Resume simulation |
| **Q** | Quit |

## Architecture

### Three-Kernel Compute Pipeline

The simulation uses three sequential Metal compute kernels per frame:

#### 1. Density Map Kernel (`density_map_kernel`)
- Computes a 128×128 histogram of particle positions
- Uses atomic operations for thread-safe accumulation
- Applies log-normalization for better dynamic range

#### 2. Neural Force Kernel (`neural_force_kernel`)
- Runs neural network forward pass for each particle
- **Input**: 64 Fourier features + 1 density context = 65 dimensions
- **Layer 1**: 65 → 128 with Tanh activation
- **Layer 2**: 128 → 128 with Tanh activation
- **Layer 3**: 128 → 2 (linear output for force_x, force_y)
- Uses custom `matvec` function for matrix-vector multiplication
- Fast tanh approximation using rational functions

#### 3. Physics Update Kernel (`physics_update_kernel`)
- Combines neural force with mouse interaction and random jitter
- Adds density-based pressure to prevent excessive clustering
- Updates velocity with friction (0.94)
- Updates position with toroidal boundary wrapping
- Evolves particle color based on speed (red) and density (green)

### Neural Network Details

**Total Parameters**: ~25,000
- Layer 1: 8,320 weights + 128 biases
- Layer 2: 16,384 weights + 128 biases
- Layer 3: 256 weights + 2 biases
- Fourier frequencies: 32 × 2D vectors

**Initialization**: Small random weights (scale=0.1) for stability

**Mutation Strategy**:
- Weights: ±1.5× base mutation rate (0.075)
- Biases: ±0.3× rate (0.015) for stability
- Fourier frequencies: Occasional randomization (33% chance, 2-3 frequencies)
- Uses Box-Muller transform for Gaussian noise

### Memory Layout

All buffers use `MTLResourceStorageModeShared` for CPU-GPU shared access:

- Position buffer: 131k × float2 = 1 MB
- Velocity buffer: 131k × float2 = 1 MB
- Color buffer: 131k × float3 = 1.5 MB
- Density map: 128×128 × float = 64 KB
- Neural weights: 25k × float = 100 KB
- Force buffer: 131k × float2 = 1 MB (intermediate)

**Total GPU memory**: ~4.7 MB

### Coordinate System

- Particle positions: normalized coordinates [-1, 1]
- Toroidal topology: particles wrap at boundaries
- Density grid: [0, 127] integer grid mapped from normalized coords
- Vertex shader flips Y-axis for Metal's coordinate system

## Implementation Notes

### Why Metal Shaders?

This implementation uses raw Metal compute kernels rather than MPS (Metal Performance Shaders) or CoreML for several reasons:

1. **Full control**: Direct access to buffer layout and memory management
2. **Mutation support**: Easy to modify weights via CPU pointer access
3. **Custom operations**: Implements specialized functions like Fourier encoding and toroidal wrapping
4. **Educational clarity**: Shows exactly how neural networks work at the GPU level
5. **Minimal dependencies**: Pure Metal framework, no external ML libraries

### Performance Considerations

- **Thread dispatch**: 256 threads per threadgroup (optimal for Apple Silicon)
- **Fast math**: Enabled via `MTLMathModeFast` for shader compilation
- **Rational tanh**: Avoids expensive exponentials while maintaining accuracy
- **O(N) complexity**: Each particle's network runs independently (no N² interactions)

### Comparison to Python Version

The original `aura_sim.py` uses PyTorch with MPS backend:
- Similar architecture (65→128→128→2 network)
- Same Fourier encoding and density context
- Equivalent mutation mechanism
- Metal version is ~10-20× faster due to tighter GPU integration and compiled shaders

## Customization

Edit `main.mm` Config namespace to adjust:

```cpp
constexpr int NUM_PARTICLES = 131072;    // Particle count (2^17)
constexpr int DENSITY_RES = 128;         // Density grid resolution
constexpr float FRICTION = 0.94f;        // Velocity damping
constexpr float NEURAL_STRENGTH = 0.006f;// Force multiplier
constexpr float MUTATION_RATE = 0.05f;   // Mutation scale
constexpr float POINT_SIZE = 2.0f;       // Particle render size
```

## Technical Details

### Fourier Positional Encoding

Encodes 2D position as 64 sinusoidal features:
```
freq[i] = random(1-8) × π
encoding = [sin(pos·freq[0]), cos(pos·freq[0]), ..., sin(pos·freq[31]), cos(pos·freq[31])]
```

This allows the network to learn patterns at multiple spatial scales.

### Density Context

The density value is transformed before feeding to the network:
```
density_normalized = log(1 + particle_count) / 5.0
density_feature = sin(density_normalized × 4.0)
```

The sine transformation creates periodic density responses.

### Color Evolution

Particle colors evolve with exponential moving average:
```
red += (speed × 5.0 - red) × 0.01    // Faster particles → more red
green += (density × 0.5 - green) × 0.01  // Denser areas → more green
blue = 1.0 - red - green (clamped to [0.2, 1.0])
```

## Credits

- Original concept: `aura_sim.py` (PyTorch implementation)
- Metal port: Based on `metal_particle_template` architecture
- Built with: Metal, MetalKit, AppKit, Objective-C++

## License

Part of the computational simulation suite. See parent repository for license details.
