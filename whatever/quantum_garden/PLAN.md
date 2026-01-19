# Quantum Garden - Game Plan

## Concept Overview
"Quantum Garden" is an interactive quantum mechanics simulation game where you cultivate a virtual garden of quantum particles. Inspired by Bohmian quantum mechanics (similar to `satori_quantum.py`), particles follow probabilistic trajectories guided by wave functions, but you actively shape their evolution through interactive controls. The garden grows with "blossoms" (stable quantum bound states) that form fractal-like patterns, creating mesmerizing visuals. Your goal is to nurture complex, aesthetically pleasing quantum ecosystems while defending against "quantum decoherence" events that disrupt patterns.

## Core Gameplay Mechanics
- **Planting Seeds**: Start by placing initial wave function "seeds" (Gaussian packets or superpositions) on a 2D grid. Particles emerge and follow quantum paths, leaving glowing trails that visualize probability density.
- **Environmental Shaping**: Use mouse/touch to draw potential barriers (walls, hills) that bend particle trajectories, creating "gardens" (enclosed regions) where particles can form bound states. Add dynamic elements like oscillating potentials or magnetic fields that curve paths.
- **Quantum Interactions**: Introduce "fertilizers" (additional wave packets) that interfere constructively/destructively with existing waves, spawning new patterns. Manage superposition states to create interference fringes that bloom into flowers.
- **Defense Against Decoherence**: Random "quantum storms" introduce noise that collapses wave functions. Counter by stabilizing patterns through strategic barrier placement or harmonic resonance (matching frequencies).
- **Harvesting & Scoring**: "Harvest" blossoms (stable states) by enclosing them in gardens. Score points based on:
  - Pattern complexity (fractal dimension, number of bound states)
  - Aesthetic metrics (symmetry, color vibrancy from phase visualization)
  - Survival time against decoherence
- **Levels & Progression**: Start with simple single-particle gardens, advance to multi-species quantum ecosystems with competing wave functions. Unlock new tools like entangled pairs or relativistic effects as you level up.
- **Creative Mode**: Free-form exploration for artistic expression, with export options for patterns as images or animations.

## Technical Implementation in Swift
- **Framework**: Pure Swift + Metal, building on `swift_particle_template` for the particle system and CFD for fluid-like wave propagation if needed.
- **Architecture**:
  - **Metal Compute Shaders**: Handle wave function evolution (split-step Fourier method for Schrödinger equation) and particle trajectory updates (Bohmian velocity from ∇ψ/ψ).
  - **Rendering Pipeline**: Real-time visualization with additive blending for glowing trails, color-mapped to phase/wave amplitude (inspired by photonic resonance projects).
  - **Data Structures**:
    ```swift
    struct WaveFunction {
        var real: [Float]     // Real part of ψ
        var imag: [Float]     // Imaginary part
        var potential: [Float] // User-drawn barriers
    }
    struct Particle {
        var position: SIMD2<Float>
        var velocity: SIMD2<Float> // Bohmian v = Im(∇ψ/ψ)
        var phase: Float         // For coloring
    }
    ```
  - **Controls**: Mouse for drawing potentials, keyboard for parameter tweaks (Q: decoherence, R: reset, Space: harvest).
  - **Performance**: Target 60 FPS with 10K-50K particles, using Metal's shared buffers for CPU-GPU data sync.
  - **Audio Integration** (Optional): Generate subtle ambient sounds from wave interference frequencies, turning the game into a "quantum symphony."

## Why This Fits & Is Creative
- **Fits Repository Theme**: Builds on quantum simulations (`satori_quantum.py`) but adds game layers—turning passive visualization into active creation, with emergent beauty as the core reward.
- **Creativity**: Combines quantum physics with gardening metaphors for a unique, poetic experience. Unlike typical physics games, it emphasizes artistic growth over combat, encouraging players to "compose" quantum art.
- **Feasibility**: Uses proven patterns from your Swift/Metal projects—particle updates via compute shaders, real-time rendering. Estimated development: 2-3 weeks for core loop, leveraging templates.
- **Visual Appeal**: Expect hypnotic, ever-evolving patterns: swirling probability clouds, branching quantum trees, and crystalline blossoms that shatter into particle rain.

## Next Steps
- Review and approve this plan
- Set up basic Swift/Metal project structure
- Implement core wave function evolution
- Add particle visualization
- Iterate on gameplay mechanics