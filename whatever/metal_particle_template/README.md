# Metal Particle Template

A minimal, well-documented template for building GPU-accelerated particle simulations on macOS using Metal and Objective-C. This project can be compiled and run entirely from the command line without Xcode IDE.

## Quick Start

```bash
# Build
make

# Run
./MetalParticles

# Or build and run in one command
make run
```

### Controls

| Key   | Action |
|-------|--------|
| Q     | Quit application |
| R     | Randomize particles and interaction matrix |
| Space | Pause/Resume simulation |

## Project Structure

```
metal_particle_template/
├── Makefile          # Build configuration
├── main.mm           # Application code (Objective-C++)
├── Compute.metal     # GPU shaders (Metal Shading Language)
└── README.md         # This file
```

## Requirements

- macOS 10.14 or later (Metal support)
- Xcode Command Line Tools (for `clang++`)

Install command line tools if needed:
```bash
xcode-select --install
```

## Libraries and Dependencies

**Zero external dependencies**. This template uses only built-in macOS frameworks:

- **Metal** - Core GPU framework for compute and graphics
- **MetalKit** - High-level utilities (MTKView, render loop)
- **AppKit** - macOS windowing and event handling
- **QuartzCore** - High-precision timing (CACurrentMediaTime)

**Pure Metal Implementation** - This template does NOT use:
- ❌ Metal Performance Shaders (MPS) for neural networks
- ❌ MPSGraph for computation graphs
- ❌ CoreML for machine learning
- ❌ Accelerate framework
- ❌ Any third-party libraries

All compute operations (particle physics, force calculations) are **hand-coded in Metal Shading Language**. This approach provides:
- Full control over GPU execution
- Direct buffer management
- Educational clarity - see exactly how GPU programming works
- Minimal binary size (~77 KB)
- No external dependencies to manage

> **Documentation Note**: When creating new projects based on this template, always document your library choices explicitly. List both what you DO use and what you DON'T use (especially when alternatives might be expected). This prevents confusion and helps future developers understand your technical decisions. See `CLAUDE.md` for the "What, Why, How" documentation framework.

## How It Works

This template implements a particle life simulation where thousands of particles interact based on configurable attraction/repulsion rules. The simulation runs entirely on the GPU for high performance.

### Architecture Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                         Application                              │
├─────────────────────────────────────────────────────────────────┤
│  main()           Create window, view, and start event loop     │
│  AppDelegate      Handle application lifecycle                   │
│  InputView        Capture keyboard input                         │
│  Renderer         Manage Metal resources and render loop         │
├─────────────────────────────────────────────────────────────────┤
│                          Metal GPU                               │
├─────────────────────────────────────────────────────────────────┤
│  Compute Kernel   update_particles() - physics simulation       │
│  Vertex Shader    vertex_main() - transform positions           │
│  Fragment Shader  fragment_main() - color pixels                │
└─────────────────────────────────────────────────────────────────┘
```

### Frame Loop

Each frame follows this sequence:

1. **Compute Pass**: GPU runs the `update_particles` kernel
   - Each particle calculates forces from all other particles
   - Velocities and positions are updated

2. **Render Pass**: GPU draws all particles
   - Vertex shader transforms world coordinates to screen coordinates
   - Fragment shader colors each pixel

3. **Present**: The rendered frame is displayed

## Understanding the Code

### Objective-C Basics

If you're new to Objective-C, here are the key syntax elements:

```objc
// Class interface declaration
@interface MyClass : ParentClass
@property (nonatomic, strong) NSString* name;  // Property
- (void)doSomething;                            // Instance method
+ (void)classMethod;                            // Class method
@end

// Class implementation
@implementation MyClass
- (void)doSomething {
    NSLog(@"Hello, %@", self.name);  // NSLog is like printf
}
@end

// Creating objects
MyClass* obj = [[MyClass alloc] init];

// Calling methods
[obj doSomething];           // Call instance method
[MyClass classMethod];       // Call class method

// Memory management (with ARC - Automatic Reference Counting)
// ARC automatically handles retain/release, you just use objects normally
```

### Metal Basics

Metal is Apple's low-level GPU API. Key concepts:

#### Device and Command Queue

```objc
// Get the GPU
id<MTLDevice> device = MTLCreateSystemDefaultDevice();

// Create a command queue (reuse this)
id<MTLCommandQueue> commandQueue = [device newCommandQueue];
```

#### Buffers

Buffers hold data accessible by both CPU and GPU:

```objc
// Create a buffer
id<MTLBuffer> buffer = [device newBufferWithLength:size
                                           options:MTLResourceStorageModeShared];

// Write data from CPU
float* data = (float*)buffer.contents;
data[0] = 1.0f;

// The GPU can now read this data in shaders
```

#### Pipeline States

Pipeline states are compiled shader programs:

```objc
// Load shader function
id<MTLFunction> function = [library newFunctionWithName:@"my_kernel"];

// Create compute pipeline
id<MTLComputePipelineState> pipeline =
    [device newComputePipelineStateWithFunction:function error:&error];
```

#### Command Encoding

Commands are encoded into command buffers, then submitted to the GPU:

```objc
// Create command buffer
id<MTLCommandBuffer> commandBuffer = [commandQueue commandBuffer];

// Encode compute commands
id<MTLComputeCommandEncoder> encoder = [commandBuffer computeCommandEncoder];
[encoder setComputePipelineState:pipeline];
[encoder setBuffer:dataBuffer offset:0 atIndex:0];
[encoder dispatchThreads:gridSize threadsPerThreadgroup:groupSize];
[encoder endEncoding];

// Submit to GPU
[commandBuffer commit];
```

### Metal Shading Language (MSL)

MSL is based on C++14 with GPU-specific features:

#### Function Attributes

```metal
// Compute kernel - runs in parallel on GPU
kernel void my_kernel(
    device float* data [[buffer(0)]],     // Read/write buffer
    constant Params& p [[buffer(1)]],     // Read-only constant
    uint gid [[thread_position_in_grid]]  // Thread ID
) { ... }

// Vertex shader - processes vertices
vertex VertexOut vertex_main(
    device const float2* positions [[buffer(0)]],
    uint vid [[vertex_id]]
) { ... }

// Fragment shader - colors pixels
fragment float4 fragment_main(
    VertexOut in [[stage_in]]
) { ... }
```

#### Buffer Attributes

| Attribute | Description |
|-----------|-------------|
| `device` | Read/write buffer in device memory |
| `constant` | Read-only data, optimized for broadcast to all threads |
| `threadgroup` | Shared memory within a threadgroup |

#### Thread Attributes

| Attribute | Description |
|-----------|-------------|
| `[[thread_position_in_grid]]` | Global thread ID |
| `[[thread_position_in_threadgroup]]` | Local thread ID within group |
| `[[threadgroup_position_in_grid]]` | Which threadgroup this is |
| `[[threads_per_threadgroup]]` | Size of threadgroup |

## Customization Guide

### Changing Particle Count

Edit `Config::NUM_PARTICLES` in `main.mm`:

```cpp
namespace Config {
    constexpr int NUM_PARTICLES = 20000;  // Increase for more particles
}
```

### Changing Colors

Edit the color palette in `Renderer::initColors`:

```objc
simd_float4 palette[] = {
    simd_make_float4(1.0f, 0.0f, 0.0f, 1.0f),  // Red
    simd_make_float4(0.0f, 1.0f, 0.0f, 1.0f),  // Green
    simd_make_float4(0.0f, 0.0f, 1.0f, 1.0f),  // Blue
};
```

### Modifying Physics

The physics behavior is defined in `Compute.metal`:

```metal
// Adjust the force calculation in update_particles
if (normalizedDist < 0.3f) {
    // Short range behavior
    forceMagnitude = normalizedDist / 0.3f - 1.0f;
} else {
    // Medium range behavior
    float t = (normalizedDist - 0.3f) / 0.7f;
    forceMagnitude = attraction * (1.0f - abs(2.0f * t - 1.0f));
}
```

### Making Circular Particles

Uncomment the circle code in `fragment_main`:

```metal
fragment float4 fragment_main(VertexOut in [[stage_in]], float2 pointCoord [[point_coord]]) {
    float2 centered = pointCoord - 0.5f;
    float dist = length(centered);
    if (dist > 0.5f) {
        discard_fragment();
    }
    float alpha = 1.0f - smoothstep(0.4f, 0.5f, dist);
    return float4(in.color.rgb, in.color.a * alpha);
}
```

## Performance Tips

1. **Use shared storage mode** for buffers accessed by both CPU and GPU:
   ```objc
   MTLResourceStorageModeShared
   ```

2. **Minimize CPU-GPU synchronization**: The CPU and GPU run asynchronously. Avoid `waitUntilCompleted` unless necessary.

3. **Use appropriate threadgroup sizes**: Match your hardware. 256 is a good default.

4. **Prefer `half` precision** when full precision isn't needed (especially on Apple Silicon):
   ```metal
   half2 pos = half2(positions[gid]);
   ```

5. **Cache matrix lookups**: Load frequently accessed data into local variables.

## Common Issues

### Shader Compilation Errors

If you see "Could not load shader file", make sure `Compute.metal` is in the same directory as the executable:

```bash
ls -la
# Should show both MetalParticles and Compute.metal
```

### Metal Not Supported

If you see "Metal is not supported on this device", you're on older hardware without Metal support (pre-2012 Macs).

### Window Not Responding

Make sure `[window makeFirstResponder:metalView]` is called so the view receives keyboard events.

## Extending This Template

Ideas for extensions:

1. **Add mouse interaction**: Implement `mouseMoved:` in InputView
2. **Add trails**: Store particle history and draw lines
3. **Add spatial hashing**: Improve performance for large particle counts
4. **Add external forces**: Gravity, wind, attractors
5. **Add different particle types**: Different sizes, behaviors
6. **Save/load configurations**: JSON/plist for interaction matrices

## Resources

### Apple Documentation
- [Metal Programming Guide](https://developer.apple.com/documentation/metal)
- [Metal Shading Language Specification](https://developer.apple.com/metal/Metal-Shading-Language-Specification.pdf)
- [MetalKit Framework](https://developer.apple.com/documentation/metalkit)

### Tutorials
- [Metal by Example](https://metalbyexample.com/)
- [Ray Wenderlich Metal Tutorial](https://www.raywenderlich.com/metal)

### Similar Projects
- [Particle Life](https://github.com/hunar4321/particle-life) - Original inspiration
- [Lenia](https://github.com/Chakazul/Lenia) - Continuous cellular automata

## License

This template is released into the public domain. Use it however you like.
