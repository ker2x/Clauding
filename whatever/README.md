# Project Catalog

A comprehensive summary of all simulation and visualization projects for future AI reference.

---

## Quick Reference

| Project | Tech Stack | Type | Scale | Description |
|---------|-----------|------|-------|-------------|
| `metal_particle_template` | ObjC++/Metal | Particle Life | 20K | **Reference template** for Metal projects (ObjC++) |
| `swift_particle_template` | Swift/Metal | Particle Life | 20K | **Reference template** for Metal projects (Swift) |
| `metal_chromatic_bloom` | ObjC++/Metal | Particle Art | 30K | Bloom effect with color cycling |
| `metal_neural_crystal` | ObjC++/Metal | 3D NCA + RT | 128³ | Neural cellular automata + raytracing |
| `metal_neural_lava` | ObjC++/Metal | 3D Fluid + RT | 64³ | Navier-Stokes fluid sim + raytracing |
| `metal_neural_lenia` | ObjC++/Metal | Lenia CA | 1280×720 | Continuous cellular automata |
| `metal_physarum` | ObjC++/Metal | Physarum | 1M agents | Slime mold multi-species warfare |
| `metal_lumen` | ObjC++/Metal | Particle Life | 20K | Particle life simulation |
| `metal_neural_aura` | ObjC++/Metal | Neural Field | 131K | Neural network particle forces |
| `CFD` | Swift/Metal | Wind Tunnel | 512×256 | LBM fluid dynamics (D2Q9) |
| `reaction_diffusion*.py` | Python | Turing Patterns | Fullscreen | Gray-Scott with JAX acceleration |
| `evolving_creatures_grid.py` | Python | Evolution | 100 agents | Genetic algo + neural networks |
| `photonic_resonance*.py` | Python | Wave Billiards | 1024-2048² | 2D wave equation in cavities |
| `aura_sim.py` | Python/PyTorch | Neural Field | ~1K | Neural network driving particle forces |
| `satori_quantum.py` | Python/PyTorch | Bohmian QM | 16K particles | Quantum pilot-wave simulation |
| `xenobots_softbody.py` | Python/PyTorch | Soft-body Evo | 40 bots | Evolutionary soft robots |

---

## Metal/Native Projects (ObjC++/Swift)

### Shared Architecture Pattern

All Metal projects follow this structure:
```
project_name/
├── main.mm          # ObjC++ app (MTKView + Renderer)
├── *.metal          # Metal shaders (compute + render)
├── Makefile         # Build configuration
└── README.md        # Optional docs
```

**Build and run**: `make && ./BinaryName`

---

### metal_particle_template
> **The canonical reference template for Metal particle simulations (Objective-C++)**

**Path**: `metal_particle_template/`
**Language**: Objective-C++ (.mm) + Metal

#### Data Structures
```cpp
// C++ side (main.mm)
namespace Config {
    constexpr int NUM_PARTICLES = 20000;
    constexpr int NUM_GROUPS = 6;
    constexpr float WORLD_SIZE = 600.0f;
    constexpr float FRICTION = 0.85f;
}

struct Particle {
    simd_float2 position;
    simd_float2 velocity;
    int group;
};

// Interaction matrix: matrix[i][j] = attraction of group i to group j
// Positive = attract, Negative = repel
```

#### Shader Pipeline
```metal
// Compute.metal
kernel void update_particles(
    device Particle* particles [[buffer(0)]],
    device float* interactionMatrix [[buffer(1)]],
    constant Params& params [[buffer(2)]],
    uint gid [[thread_position_in_grid]]
) {
    // For each particle, sum forces from all others
    // F = interaction * (distance-based curve)
    // Apply friction, update velocity and position
}

vertex VertexOut vertex_main(...) { /* Position + color lookup */ }
fragment float4 fragment_main(...) { /* Point rendering */ }
```

#### Controls
- **Q**: Quit
- **R**: Randomize particles & matrix
- **Space**: Pause/Resume

---

### swift_particle_template
> **The Swift equivalent of metal_particle_template**

**Path**: `swift_particle_template/`
**Language**: Pure Swift + Metal

#### Overview
Identical functionality to `metal_particle_template` but using Swift's modern syntax:
- 50% more concise (~500 vs ~650 lines)
- Stronger type safety with Swift's optional system
- Same Metal shaders (language-agnostic)
- Same GPU performance

#### Key Swift Features
```swift
// Swift style
let device = MTLCreateSystemDefaultDevice()
app.setActivationPolicy(.regular)

// vs Objective-C++ style
MTLDevice* device = MTLCreateSystemDefaultDevice();
[NSApp setActivationPolicy:NSApplicationActivationPolicyRegular];
```

**Use this template** if you prefer Swift over Objective-C++ for new projects.

---

### metal_chromatic_bloom
> Chromatic particle art with dynamic bloom effects

**Path**: `metal_chromatic_bloom/`

#### Data Structures
```cpp
struct Particle {
    simd_float2 position;
    simd_float2 velocity;
    simd_float3 color;      // RGB
    float age;
    float hue_phase;        // For color animation
    float speed_amplitude;
};

struct Params {
    uint width, height;
    float time;
    float time_scale;
    float attraction_strength;
    float color_rotation;
    float bloom_intensity;   // Adjustable: 0.0 - 2.0
    float chaos;             // Adjustable: 0.0 - 1.0
    uint num_particles;
};
```

#### Render Pipeline
- Uses **additive blending** for bloom:
  ```objc
  renderDesc.colorAttachments[0].sourceRGBBlendFactor = MTLBlendFactorOne;
  renderDesc.colorAttachments[0].destinationRGBBlendFactor = MTLBlendFactorOne;
  ```
- Renders 6 triangles per particle for glow effect

---

### metal_neural_crystal
> 3D Neural Cellular Automata with raytracing visualization

**Path**: `metal_neural_crystal/`

#### Implementation
```cpp
static const uint32_t GRID_SIZE = 128;  // 128³ = 2M cells

// Each cell stores float4: (density, temperature, state, extra)
id<MTLBuffer> _ncaBufferA;  // Double buffering
id<MTLBuffer> _ncaBufferB;
```

#### Shader Pipeline
1. **NCA Update Kernel**: `nca_update_kernel`
   - 3D threadgroups (8×8×8)
   - Samples 6 face neighbors (Von Neumann)
   - Applies growth rules based on neighbor states
   
2. **Raytracing Kernel**: `raytrace_kernel`
   - Casts rays through 3D volume
   - Accumulates density along ray
   - Outputs to 2D render target texture

3. **Display Pass**: Full-screen quad with texture

#### Controls
- **R**: Reset (seed center point)
- **Q**: Quit
- **Mouse drag**: Rotate view

---

### metal_neural_lava
> 3D incompressible fluid simulation with raytraced rendering

**Path**: `metal_neural_lava/`

#### Core Algorithm (Navier-Stokes)
```
Each frame:
1. Advection       - Move velocity/density fields
2. Divergence      - Compute velocity divergence
3. Pressure        - Jacobi iteration (20 passes)
4. Projection      - Subtract pressure gradient
5. Raytrace        - Volume render to screen
```

#### Data Structures
```cpp
static const uint32_t GRID_SIZE = 64;  // 64³ = 262K cells

// Using 3D textures (RGBA16Float) for efficient sampling
id<MTLTexture> _velocityTex[2];   // Double buffer
id<MTLTexture> _densityTex[2];    // Double buffer
id<MTLTexture> _pressureTex[2];   // Double buffer
id<MTLTexture> _divergenceTex;

struct Uniforms {
    simd_float2 resolution;
    float time, dt;
    uint32_t gridSize;
    simd_float2 mousePos;
    int mouseDown;
    float viscosity;    // 0.1
    float buoyancy;     // 1.0
};
```

#### Shader Kernels
- `advect_kernel`: Semi-Lagrangian advection
- `divergence_kernel`: ∇·v calculation
- `pressure_kernel`: Jacobi solver step
- `project_kernel`: v = v - ∇p
- `raytrace_kernel`: Volume rendering with color ramp

---

### metal_neural_lenia
> Continuous cellular automata (Lenia) with randomized species

**Path**: `metal_neural_lenia/`

#### Lenia Parameters
```cpp
struct Uniforms {
    float mu;    // Growth center (0.12 - 0.16)
    float sigma; // Growth width (0.012 - 0.020)
    float rho;   // Kernel radius (12.0 - 16.0)
    float dt;    // Time step (0.05)
};

// On reset, parameters are randomized:
_mu = 0.12f + (rand() % 40) / 1000.0f;
_sigma = 0.012f + (rand() % 80) / 10000.0f;
_rho = 12.0f + (rand() % 40) / 10.0f;
```

#### Core Algorithm
```metal
kernel void lenia_kernel(...) {
    // 1. Circular convolution with kernel K(r)
    float U = convolve_with_gaussian_ring(state, pos, rho);
    
    // 2. Growth function G(U)
    float growth = gaussian(U, mu, sigma) * 2.0 - 1.0;
    
    // 3. Update: A' = clamp(A + dt * G(U), 0, 1)
    float new_state = state + dt * growth;
}
```

---

### metal_physarum
> Physarum polycephalum (slime mold) simulation with 1M agents

**Path**: `metal_physarum/`

#### Agent-Based Model
```cpp
static const uint NUM_AGENTS = 1000000;

struct Agent {
    simd_float2 position;
    float heading;      // Angle in radians
    int species;        // 0 or 1 (two-species warfare)
};

struct Params {
    float move_speed;      // 1.2
    float sensor_angle;    // 0.45 radians
    float sensor_dist;     // 22.0 pixels
    float turn_speed;      // 0.35
    float decay_rate;      // 0.94
    float combat_strength; // 0.6
};
```

#### Shader Pipeline
1. **update_agents**: Sense pheromones at 3 angles, turn toward highest, move forward
2. **deposit_pheromones**: Write agent position to trail texture
3. **process_trail**: 3×3 diffuse + decay

#### Two-Species Dynamics
- Species 0 deposits to R channel, Species 1 to G channel
- Each species attracted to own pheromone, repelled by other
- Creates battle frontlines at boundaries

---

### metal_lumen
> Metal particle life simulation

**Path**: `metal_lumen/`
**Language**: Objective-C++ (.mm) + Metal

#### Features
- 20K particles with 6 interaction types
- Color palette: Cyan, Magenta, Yellow, Electric Green, Orange, Purple
- FPS counter in window title
- Same particle life algorithm as `metal_particle_template`

---

### metal_neural_aura
> Neural network-driven particle forces with GPU compute

**Path**: `metal_neural_aura/`
**Language**: Objective-C++ (.mm) + Metal

#### Features
- 131K particles driven by a 3-layer MLP (65→128→128→2)
- 100% custom neural network in Metal shaders (no CoreML/MPS)
- Fourier positional encoding (32 frequency pairs)
- Density-aware forces via 128×128 grid
- Real-time weight mutation for evolving behaviors

#### Neural Network
```cpp
// Total ~25,000 parameters in Metal buffers
// Layer 1: 65 → 128 (Fourier features + density)
// Layer 2: 128 → 128
// Layer 3: 128 → 2 (force output)
```

#### Controls
- **Space**: Mutate network weights
- **R**: Reset particles
- **P**: Pause/Resume
- **Q**: Quit

---

### CFD (Wind Tunnel)
> Real CFD simulation using Lattice Boltzmann Method

**Path**: `CFD/`  
**Language**: Swift + Metal

#### Lattice Boltzmann (D2Q9)
```swift
// 9 velocity directions:
// 6 7 8
// 3 4 5  (4 = rest, others = movement)
// 0 1 2

// Distribution function: f_i(x, t)
// Equilibrium: f_eq = w_i * rho * (1 + 3*c_i·u + 4.5*(c_i·u)² - 1.5*u²)
// Collision: f_out = f_in - (f_in - f_eq) / tau
// Streaming: f(x + c_i*dt, t+dt) = f_out(x, t)
```

#### Config (config.json)
```json
{
  "simulation": {
    "grid_width": 512,
    "grid_height": 256,
    "steps_per_frame": 5,
    "relaxation_time": 0.7,    // tau (viscosity)
    "initial_wind_speed": 0.05,
    "max_wind_speed": 0.15
  }
}
```

#### Kernels
- `lbm_stream_collide`: Core LBM step
- `lbm_boundary`: Inlet/outlet/wall conditions
- `compute_fields`: Extract velocity + vorticity
- `render_field`: Viridis colormap + vorticity overlay
- `update_obstacles`: Mouse drawing

---

## Python Projects

### Common Base Pattern
```python
import pygame
import numpy as np

class Simulation:
    def __init__(self):
        pygame.init()
        self.screen = pygame.display.set_mode((WIDTH, HEIGHT))
        self.clock = pygame.time.Clock()
        # Initialize buffers...
        
    def update(self):
        # Physics step
        
    def draw(self):
        # Render to screen
        
    def handle_events(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False
                
    def run(self):
        while self.running:
            self.handle_events()
            self.update()
            self.draw()
            self.clock.tick(FPS)

if __name__ == "__main__":
    sim = Simulation()
    sim.run()
```

---

### particle_swarm.py (PyTorch)
> Particle swarm with group interactions

#### Implementation
```python
NUM_PARTICLES = 3000
NUM_GROUPS = 6

# Interaction matrix: [6x6] attraction/repulsion values
INTERACTION_MATRIX = self._create_interaction_matrix_np()

def _physics_update_torch(self, positions, velocities, groups, matrix, dt):
    # Vectorized: compute all pairwise distances
    diff = positions.unsqueeze(0) - positions.unsqueeze(1)  # [N, N, 2]
    dist = torch.norm(diff, dim=-1)  # [N, N]
    
    # Distance-dependent force curve
    force_mag = torch.where(dist < 30, -25/(dist+0.5), 
                torch.where(dist < 150, 20/(dist+0.5), 5/(dist+0.5)))
    
    # Apply interaction matrix
    g1, g2 = groups.unsqueeze(0), groups.unsqueeze(1)
    interaction = matrix[g1, g2]
    force_mag *= interaction
    
    # Sum forces, update velocity with friction
    forces = (diff / dist.unsqueeze(-1)) * force_mag.unsqueeze(-1)
    velocities += forces.sum(dim=1) * dt
    velocities *= 0.85  # friction
```

### particle_swarm_fast.py (Numba)
> Same algorithm, JIT-compiled for CPU

```python
@nb.njit(parallel=True, cache=True)
def update_particles_jitted(positions, velocities, groups, matrix, dt, w, h, n):
    for i in nb.prange(n):  # Parallel loop
        fx, fy = 0.0, 0.0
        for j in range(n):
            if i != j:
                # Distance and force calculation
                # ... (same logic as PyTorch, but scalar)
        velocities[i] = (velocities[i] + [fx, fy] * dt) * 0.80
```

### particle_swarm_gpu.py (ModernGL)
> OpenGL compute shader version

```python
import moderngl as mgl

# Compute shader (GLSL)
COMPUTE_SHADER = """
#version 430
layout(local_size_x=256) in;
layout(std430, binding=0) buffer Positions { vec2 pos[]; };
layout(std430, binding=1) buffer Velocities { vec2 vel[]; };
// ... physics in GLSL
"""
```

---

### reaction_diffusion.py
> Gray-Scott reaction-diffusion system

#### Algorithm
```python
# Two chemicals: U (substrate) and V (catalyst)
# Reactions: U + 2V → 3V, V → inert

def update(self):
    laplacian_U = self.laplacian(self.U)
    laplacian_V = self.laplacian(self.V)
    
    reaction = self.U * self.V * self.V
    
    # Gray-Scott equations
    dU = Du * laplacian_U - reaction + feed * (1 - self.U)
    dV = Dv * laplacian_V + reaction - (feed + kill) * self.V
    
    self.U += dU * dt
    self.V += dV * dt

def laplacian(self, grid):
    # 9-point stencil with diagonals weighted 0.05
    return (
        np.roll(grid, 1, 0) + np.roll(grid, -1, 0) +
        np.roll(grid, 1, 1) + np.roll(grid, -1, 1)
    ) * 0.2 + (
        # diagonals...
    ) * 0.05 - grid
```

#### Pattern Presets
| Pattern | Feed | Kill |
|---------|------|------|
| SPOTS | 0.055 | 0.062 |
| STRIPES | 0.035 | 0.060 |
| CORAL | 0.0545 | 0.062 |
| WAVES | 0.014 | 0.045 |

### reaction_diffusion_coupled.py (JAX)
> Multiple species with interactions

```python
# 3 species (RGB) that interact
interaction_matrix = jnp.array([
    [0.0, 0.5, -0.3],   # R: neutral self, helps G, hurts B
    [-0.3, 0.0, 0.5],   # G: ...
    [0.5, -0.3, 0.0]    # B: ...
])

@jit
def update_step(U_list, V_list, ...):
    # Each species evolves with Gray-Scott
    # Plus cross-species interaction terms
```

---

### evolving_creatures_grid.py
> Discrete grid version with larger brain

```python
# Grid-based: 60×60 cells
# 6 inputs: food_N, food_E, food_S, food_W, creature_nearby, energy
# 5 outputs: move N/E/S/W/Stay (argmax selection)

GRID_WIDTH, GRID_HEIGHT = 60, 60
UPDATES_PER_FRAME = 10  # Faster evolution
```

---

---

### photonic_resonance_jax.py
> 2D wave equation with JAX

#### Wave Equation
```python
@jit
def update_wave(u, u_prev, boundary, source_mask, source_val):
    # Finite difference Laplacian
    laplacian = (
        jnp.roll(u, 1, 0) + jnp.roll(u, -1, 0) +
        jnp.roll(u, 1, 1) + jnp.roll(u, -1, 1) - 4*u
    )
    
    # Wave equation: u_new = 2u - u_prev + c²dt²∇²u
    c_sq_dt_sq = 0.25  # Wave speed squared × dt squared
    u_new = 2*u - u_prev + c_sq_dt_sq * laplacian
    
    # Apply source (oscillating point)
    u_new = jnp.where(source_mask, source_val, u_new)
    
    # Enforce boundary (Dirichlet: u=0 at walls)
    u_new = jnp.where(boundary, 0.0, u_new)
    
    return u_new
```

#### Domain Shapes
```python
def init_stadium_domain(self):
    """Rectangle with semicircular ends"""
    rect = (abs(x) < 0.3) & (abs(y) < 0.2)
    left_cap = ((x + 0.3)**2 + y**2) < 0.2**2
    right_cap = ((x - 0.3)**2 + y**2) < 0.2**2
    return ~(rect | left_cap | right_cap)  # boundary = outside shape
```

---

### aura_sim.py
> Neural network-driven particle forces

#### Neural Architecture
```python
class NeuralAura(nn.Module):
    def __init__(self):
        # Fourier feature encoding for positions
        self.B = torch.randn(2, 64) * 10  # Random frequencies
        
        self.mlp = nn.Sequential(
            nn.Linear(128 + 1, 64),  # 128 Fourier + 1 density
            nn.GELU(),
            nn.Linear(64, 64),
            nn.GELU(),
            nn.Linear(64, 2)  # Output: force (fx, fy)
        )
        
    def forward(self, pos, density):
        # Fourier features: [sin(B·pos), cos(B·pos)]
        features = torch.cat([
            torch.sin(pos @ self.B),
            torch.cos(pos @ self.B),
            density
        ], dim=-1)
        return self.mlp(features)
```

#### Mutation
```python
def mutate(self):
    """Slightly perturb weights to evolve patterns"""
    for param in self.neural.parameters():
        param.data += torch.randn_like(param) * 0.01
```

---

### satori_quantum.py
> Bohmian quantum mechanics

#### Wave Function Evolution (Split-Step Fourier)
```python
def update_wave(self):
    # 1. Potential step (real space)
    phase = -self.potential * self.dt * 1000
    self.psi *= torch.exp(1j * phase)
    
    # 2. Kinetic step (momentum space)
    psi_k = torch.fft.fft2(self.psi)
    k_phase = -self.K2 * self.dt * 20  # K2 = kx² + ky²
    psi_k *= torch.exp(1j * k_phase)
    self.psi = torch.fft.ifft2(psi_k)
    
    # 3. Normalize
    self.psi /= torch.sqrt(torch.sum(torch.abs(self.psi)**2))
```

#### Bohmian Velocity
```python
def update_particles(self):
    # v = Im(∇ψ / ψ) = (ψ_real * ∇ψ_imag - ψ_imag * ∇ψ_real) / |ψ|²
    grad_real = central_difference(self.psi.real)
    grad_imag = central_difference(self.psi.imag)
    
    vx = (psi_imag * grad_real_x - psi_real * grad_imag_x) / rho
    vy = (psi_imag * grad_real_y - psi_real * grad_imag_y) / rho
    
    self.pos += velocity * dt
```

---

### xenobots_softbody.py
> Soft-body evolutionary robots

#### Physics Model
```python
NODES_PER_BOT = 6
LINKS_PER_BOT = 15  # Fully connected: 6*5/2

# Each link is a spring with oscillating rest length
def physics_step(self, dt):
    # 1. Compute spring forces
    for link in self.links:
        # DNA controls oscillation: freq, phase, amplitude
        target_length = base_length * (1 + amp * sin(freq * t + phase))
        force = k * (current_length - target_length)
        
    # 2. Apply gravity
    forces[:, 1] += GRAVITY
    
    # 3. Ground collision with friction
    if node.y > GROUND_Y:
        node.y = GROUND_Y
        node.vx *= FRICTION
        node.vy = 0
```

#### Evolution
```python
def evolve(self):
    # Fitness = maximum X position reached
    fitness = self.pos[:, :, 0].max(dim=1).values
    
    # Tournament selection: top 6 become parents
    elite_indices = torch.argsort(fitness, descending=True)[:6]
    
    # Offspring = parent DNA + Gaussian noise
    for i in range(6, POP_SIZE):
        parent = elite_dna[random.choice(range(6))]
        child = parent + torch.randn_like(parent) * MUTATION_SCALE
```

---

## Key Implementation Patterns

### Metal Buffer Setup
```objc
// Shared mode: CPU + GPU access
id<MTLBuffer> buffer = [device newBufferWithBytes:data
                                           length:size
                                          options:MTLResourceStorageModeShared];

// Access from CPU
float* ptr = (float*)[buffer contents];
```

### Metal Compute Dispatch
```objc
[encoder dispatchThreads:MTLSizeMake(NUM_PARTICLES, 1, 1)
    threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
```

### JAX JIT Pattern
```python
from jax import jit

@jit
def update(state):
    # Pure function, no side effects
    return new_state

# First call compiles, subsequent calls are fast
state = update(state)
```

### PyTorch MPS Acceleration
```python
import os
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
tensor = tensor.to(device)
```

---

## Dependencies

### Python
```bash
pip install pygame numpy torch jax jaxlib numba moderngl
```

### Metal/Swift
```bash
xcode-select --install  # Command line tools
```

---

*Last updated: 2026-01-14*
