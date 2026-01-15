# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

Multi-platform computational simulation suite featuring particle systems, cellular automata, fluid dynamics, and quantum mechanics visualizations. Contains 8 Metal GPU projects (ObjC++), 1 Swift CFD simulator, and 8 Python simulations.

**Host**: MacBook Air M3 16GB

**Primary goal**: Creativity and visual exploration. These are artistic/scientific simulations—novel behaviors and beautiful emergent patterns matter most.

For detailed documentation on each project, algorithms, and data structures, see `README.md`.

## Development Guidelines

- **Creativity first**: Prioritize interesting emergent behaviors and visual appeal
- **Performance vs readability**: Optimize for performance, but never sacrifice code clarity or stability
- **New Metal projects**: Use `metal_particle_template/` as the starting point for new Metal simulations. Never edit the template itself
- **Apple APIs**: When writing or commenting on Metal/AppKit/Swift code, explain Apple-specific concepts for developers unfamiliar with the ecosystem
- **Math explanations**: When implementing complex algorithms (FFT, Navier-Stokes, Lattice Boltzmann, etc.), include comments explaining the underlying mathematics
- **Python library choice**: Prefer JAX for GPU acceleration; use PyTorch only for complex neural networks
- **No Xcode IDE**: All Swift/ObjC++ code must build and run from command line only (Xcode CLI tools are fine)

## User Consultation (IMPORTANT)

**Always use AskUserQuestion before starting significant work:**

### New Applications
Before creating a new project, ask about:
- Visual style preferences (colors, aesthetics, mood)
- Interaction model (mouse, keyboard, MIDI, etc.)
- Performance targets (resolution, FPS, particle count)
- Specific algorithms or papers to implement
- Output format (realtime window, video export, screenshots)

### Major Refactoring
Before restructuring existing code, confirm:
- Which parts to preserve vs. rewrite
- Whether to maintain backwards compatibility
- Performance vs. readability trade-offs
- Scope of changes (single file vs. multi-file)

### Algorithm Choices
When multiple valid approaches exist, present options:
- e.g., "Ray marching vs. ray tracing for this scene?"
- e.g., "Euler vs. RK4 integration for particles?"
- e.g., "Grid-based vs. particle-based fluid simulation?"

**Rationale**: These are creative/artistic projects where user vision matters. A quick question upfront avoids hours of rework and ensures the result matches expectations.

## Documentation Requirements

Every project README must include the **"What, Why, How"** framework:

### What (Libraries & Dependencies)
Be **explicit and complete** about library choices:
- ✅ List all frameworks/libraries used (Metal, MetalKit, AppKit, JAX, PyTorch, etc.)
- ✅ **Equally important**: List what's NOT used when it might be expected
  - Example: "No CoreML, no MPSGraph, no BNNS" for custom Metal implementations
  - Example: "No TensorFlow, pure JAX" for Python projects
- ✅ Explain whether implementations are custom or framework-based
  - "Custom neural network in Metal compute shaders" vs "Using MPSGraph"
  - "Hand-coded matrix operations" vs "Using Accelerate framework"

### Why (Rationale)
Justify technical decisions:
- Why raw Metal instead of MPS/CoreML? (control, education, mutation flexibility)
- Why JAX instead of PyTorch? (performance, functional programming)
- Why custom implementation? (learning, flexibility, specific requirements)
- Trade-offs made (binary size, startup time, maintainability)

### How (Implementation Details)
Document the approach:
- Architecture overview (kernel pipeline, buffer layout, etc.)
- Key algorithms and their mathematical basis
- Performance characteristics (FPS, memory usage, complexity)
- Coordinate systems and data flow

### Example Structure
```markdown
### Libraries and Dependencies
**Zero external dependencies**. Uses only built-in frameworks:
- Metal, MetalKit, AppKit, QuartzCore

**100% Custom Implementation** - Does NOT use:
- ❌ CoreML / MPSGraph / BNNS
- ❌ Any ML framework

**Why this approach?**
- Full control over implementation
- Educational transparency
- [other reasons]

**How it works:**
- Custom matrix-vector multiply in Metal
- [technical details]
```

This level of clarity prevents confusion and helps future developers understand both what the code does and why it was built that way.

## Build Commands

### Metal Projects (ObjC++/Metal)
```bash
cd metal_particle_template && make && ./MetalParticles
# Or: make run (builds and runs)
# Or: make clean (removes binaries)
```

### CFD (Swift)
```bash
cd CFD && ./run.sh
# Or: ./build.sh && open WindTunnel.app
```

### Python Simulations
```bash
source ../.venv/bin/activate  # Required: use parent directory venv
python particle_swarm.py
python satori_quantum.py
```

## Architecture

### Metal Projects Structure
All Metal projects follow the same pattern:
```
metal_*/
├── main.mm         # ObjC++ app with Renderer class, MTKView setup
├── Shader.metal    # Compute + render kernels
└── Makefile        # clang++ build with Metal frameworks
```

Key patterns:
- `namespace Config {}` contains tunable simulation parameters
- Double-buffered compute: particles/fluid states alternate between buffers to prevent read/write conflicts
- Compute kernel dispatches 256 threads per threadgroup (optimal for Apple Silicon)
- Shared storage mode (`MTLResourceStorageModeShared`) allows both CPU and GPU to access the same memory
- Runtime shader compilation from source files (no pre-build step)
- Common controls: Q=quit, R=reset, Space=pause

### Python Projects Structure
Standard pygame loop with GPU acceleration via JAX or PyTorch MPS:
```python
# JAX device selection (preferred)
import jax
jax.devices()  # Automatically uses Metal on Apple Silicon

# PyTorch fallback for neural networks
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
```

### CFD (Lattice Boltzmann)
- D2Q9 lattice configuration (9 discrete velocity directions per cell)
- Parameters in `CFD/config.json`: grid size, relaxation time, wind speed
- Swift + Metal implementation with SwiftUI

## Key Projects

| Project | Purpose |
|---------|---------|
| `metal_particle_template/` | Reference template for Metal particle simulations (ObjC++) |
| `swift_particle_template/` | Reference template for Metal particle simulations (Swift) |
| `metal_neural_aura/` | Neural network-driven particle forces (131K particles) |
| `metal_lumen/` | Particle life simulation (20K particles) |
| `metal_neural_lava/` | 3D Navier-Stokes fluid + raytracing (64³ grid) |
| `metal_neural_crystal/` | 3D NCA + raytracing (128³ grid) |
| `metal_physarum/` | 1M agent slime mold simulation |
| `CFD/` | Real LBM wind tunnel simulator |
| `satori_quantum.py` | Bohmian pilot-wave quantum mechanics |
| `xenobots_softbody.py` | Evolutionary soft-body robots |

## Dependencies

**System**: macOS with Metal support, Xcode CLI tools (`xcode-select --install`)

**Python**: Use virtual environment at `../.venv`. Packages: pygame, numpy, jax, jaxlib, torch, numba, moderngl

## Notes

- `timeout` shell command not available on this system (use alternative methods)
- MPS fallback for PyTorch: `os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"`
