# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

Multi-platform computational simulation suite featuring particle systems, cellular automata, fluid dynamics, and quantum mechanics visualizations. Contains 6 Metal GPU projects, 1 Swift CFD simulator, and 17 Python simulations.

**Host**: MacBook Air M3 16GB

**Primary goal**: Creativity and visual exploration. These are artistic/scientific simulations—novel behaviors and beautiful emergent patterns matter most.

For detailed documentation on each project, algorithms, and data structures, see `README.md`.

## Development Guidelines

- **Creativity first**: Prioritize interesting emergent behaviors and visual appeal
- **Performance vs readability**: Optimize for performance, but never sacrifice code clarity or stability
- **Apple APIs**: When writing or commenting on Metal/AppKit/Swift code, explain Apple-specific concepts for developers unfamiliar with the ecosystem
- **Math explanations**: When implementing complex algorithms (FFT, Navier-Stokes, Lattice Boltzmann, etc.), include comments explaining the underlying mathematics
- **Python library choice**: Prefer JAX for GPU acceleration; use PyTorch only for complex neural networks
- **No Xcode IDE**: All Swift/ObjC++ code must build and run from command line only (Xcode CLI tools are fine)

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
| `metal_particle_template/` | Reference template for Metal particle simulations |
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
