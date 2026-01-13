// =============================================================================
// METAL SHADING LANGUAGE (MSL) - GPU Code for Beginners
// =============================================================================
// This file contains code that runs ON THE GPU, not the CPU.
// Metal Shading Language is based on C++14 with GPU-specific extensions.
//
// If you know C++, most of this will look familiar!
// Key differences:
// - Special function types: kernel, vertex, fragment
// - Special types: float4, half3, texture2d, etc.
// - Special attributes: [[buffer(0)]], [[position]], etc.
// - Runs on thousands of threads in parallel
//
// =============================================================================

#include <metal_stdlib>
// metal_stdlib = Metal's standard library (like <iostream> in C++)
// Contains: math functions, vector types, texture samplers, etc.

using namespace metal;
// "metal" namespace contains all Metal types
// Now we can use "float4" instead of "metal::float4"

// =============================================================================
// DATA STRUCTURES (Shared with CPU)
// =============================================================================
// These structures MUST match the definitions in main.mm exactly!

struct Uniforms {
    float time;         // Time in seconds since app started
    float deltaTime;    // Time since last frame
    int frameCount;     // Total number of frames rendered
    float _padding;     // Padding for 16-byte alignment
};

// =============================================================================
// METAL TYPES - Vector Math
// =============================================================================
/*
Metal has built-in vector and matrix types:

Vectors (2D, 3D, 4D):
    float2  = { float x, y; }
    float3  = { float x, y, z; }
    float4  = { float x, y, z, w; }
    half2, half3, half4  (16-bit floats - faster on mobile GPUs)
    int2, int3, int4
    uint2, uint3, uint4

Access components:
    float4 color = float4(1.0, 0.5, 0.2, 1.0);
    float r = color.x;     // or color.r (same thing)
    float g = color.y;     // or color.g
    float b = color.z;     // or color.b
    float a = color.w;     // or color.a

    // Swizzling (reordering components):
    float3 rgb = color.rgb;    // Take first 3 components
    float3 bgr = color.bgr;    // Reverse order
    float2 rg = color.rg;      // Take first 2

Matrices:
    float2x2, float3x3, float4x4
    half4x4 (for transforms)

Common Math Functions:
    sin(), cos(), tan()
    pow(), sqrt(), exp(), log()
    dot(a, b)        - dot product
    cross(a, b)      - cross product
    length(v)        - vector magnitude
    normalize(v)     - make unit length
    mix(a, b, t)     - linear interpolation (lerp)
    clamp(x, min, max) - constrain value
    smoothstep(a, b, x) - smooth interpolation
*/

// =============================================================================
// SHADER FUNCTION TYPES
// =============================================================================
/*
Metal has three main shader types:

1. VERTEX SHADER (vertex)
   - Processes each vertex (corner of a triangle)
   - Input: vertex data (position, color, etc.)
   - Output: transformed position + data for fragment shader
   - Runs: Once per vertex

2. FRAGMENT SHADER (fragment)
   - Processes each pixel
   - Input: interpolated data from vertex shader
   - Output: final pixel color
   - Runs: Once per pixel

3. COMPUTE KERNEL (kernel)
   - General-purpose parallel computation
   - Input: buffers, thread ID
   - Output: writes to buffers
   - Runs: Once per thread (you specify how many threads)

This demo uses vertex + fragment shaders to draw a colorful gradient.
*/

// =============================================================================
// ATTRIBUTE SYNTAX EXPLAINED
// =============================================================================
/*
The [[attribute]] syntax tells Metal how to connect data:

VERTEX SHADER:
    [[vertex_id]]           - Which vertex is this? (0, 1, 2, ...)
    [[instance_id]]         - Which instance (for instanced rendering)
    [[buffer(N)]]           - Data from buffer #N

FRAGMENT SHADER:
    [[stage_in]]            - Input from vertex shader (interpolated)
    [[color(N)]]            - Output to color attachment N

BOTH:
    [[position]]            - Clip-space position (required for vertex output)
    [[user(name)]]          - User-defined varying data

COMPUTE KERNEL:
    [[thread_position_in_grid]] - Global thread ID
    [[threadgroup_position_in_grid]] - Which threadgroup
    [[thread_position_in_threadgroup]] - Local thread ID within group
    [[buffer(N)]]           - Buffer binding
    [[texture(N)]]          - Texture binding

Example:
    vertex VertexOut myVertex(uint vid [[vertex_id]])

    This means: "This is a vertex shader that takes one parameter called 'vid',
                 which is the vertex ID provided by Metal automatically"
*/

// =============================================================================
// VERTEX SHADER OUTPUT STRUCTURE
// =============================================================================
// This struct is passed from vertex shader → fragment shader
// Metal automatically interpolates values across the triangle

struct VertexOut {
    // [[position]] is REQUIRED for vertex output
    // It's the final position in "clip space" coordinates
    // Clip space: x,y in range [-1, 1], (0,0) = center, (-1,-1) = bottom-left
    float4 position [[position]];

    // Custom data to pass to fragment shader
    // Metal will interpolate these values across the triangle
    // Each pixel gets a smoothly blended value
    float4 color;
};

// =============================================================================
// VERTEX SHADER - Runs once per vertex
// =============================================================================
// This shader generates a fullscreen triangle procedurally.
// We don't need vertex buffers - we generate positions from the vertex ID!

vertex VertexOut vertex_main(
    uint vertexID [[vertex_id]]
    // uint = unsigned int (32-bit)
    // [[vertex_id]] = Metal provides this automatically (0, 1, 2, ...)
) {
    // We're drawing 3 vertices to make 1 triangle that covers the screen
    // This is a trick to draw a fullscreen quad efficiently

    // FULLSCREEN TRIANGLE TECHNIQUE
    // ------------------------------
    // Instead of drawing 2 triangles (6 vertices) for a quad,
    // we draw 1 big triangle (3 vertices) that covers the screen:
    //
    //     (-1, 3)
    //        ▲
    //        |╲
    //        | ╲
    //        |  ╲
    // (-1,1) +---+ (3, 1)
    //        |╱╱╱|
    //        |╱╱╱|
    // (-1,-1)+---+ (1, -1)  ← Screen
    //
    // The parts outside the screen are automatically clipped by the GPU.
    // This saves a little bit of processing compared to 2 triangles.

    // Generate positions based on vertex ID
    float2 positions[3] = {
        float2(-1.0, -1.0),  // Vertex 0: bottom-left
        float2( 3.0, -1.0),  // Vertex 1: off-screen right
        float2(-1.0,  3.0)   // Vertex 2: off-screen top
    };

    // Generate colors for each vertex (will be interpolated)
    float3 colors[3] = {
        float3(1.0, 0.2, 0.4),  // Red-ish
        float3(0.2, 0.6, 1.0),  // Blue-ish
        float3(0.3, 1.0, 0.5)   // Green-ish
    };

    // Create output struct
    VertexOut out;

    // Set position (float4 = x, y, z, w)
    // z = 0 (depth), w = 1 (no perspective division)
    float2 pos = positions[vertexID];
    out.position = float4(pos.x, pos.y, 0.0, 1.0);

    // Set color (float4 = r, g, b, a)
    float3 col = colors[vertexID];
    out.color = float4(col.r, col.g, col.b, 1.0);

    return out;
    // Metal will now:
    // 1. Rasterize the triangle (find which pixels it covers)
    // 2. Interpolate out.color across the triangle
    // 3. Call fragment_main() for each pixel
}

// =============================================================================
// FRAGMENT SHADER - Runs once per pixel
// =============================================================================
// This shader determines the final color of each pixel.
// It receives interpolated data from the vertex shader.

fragment float4 fragment_main(
    VertexOut in [[stage_in]],
    // [[stage_in]] means "input from previous stage" (the vertex shader)
    // 'in' contains interpolated values for this pixel

    constant Uniforms& uniforms [[buffer(0)]]
    // [[buffer(0)]] = receive the uniform buffer we bound on CPU
    // 'constant' = read-only data, optimized for GPU caches
    // This gives us access to time, deltaTime, and frameCount
) {
    // At this point, in.color has been interpolated across the triangle
    // Pixels near vertex 0 are more red, near vertex 1 more blue, etc.

    // -------------------------------------------------------------------------
    // ANIMATED COLOR EFFECTS
    // -------------------------------------------------------------------------
    // Now we can use uniforms.time to animate the colors!

    // Start with the interpolated color
    float3 color = in.color.rgb;

    // ANIMATION 1: Pulsating brightness
    // sin() oscillates between -1 and 1
    // 0.5 + 0.5 * sin() oscillates between 0 and 1
    float pulse = 0.5 + 0.5 * sin(uniforms.time * 2.0);
    // Multiply by pulse to make it breathe
    color *= (0.5 + 0.5 * pulse); // Range: 0.5 to 1.0 brightness

    // ANIMATION 2: Color cycling (hue rotation)
    // This rotates colors through the spectrum
    float hueShift = uniforms.time * 0.5; // Rotate slowly
    float cosH = cos(hueShift);
    float sinH = sin(hueShift);
    // Simple 2D rotation in RG color space
    float r = color.r * cosH - color.g * sinH;
    float g = color.r * sinH + color.g * cosH;
    color = float3(r, g, color.b);

    // ANIMATION 3: Saturation pulse
    // Calculate grayscale (luminance)
    float gray = dot(color, float3(0.299, 0.587, 0.114));
    // Oscillate between grayscale and full color
    float saturation = 0.5 + 0.5 * sin(uniforms.time * 3.0);
    color = mix(float3(gray), color, saturation);

    return float4(color, 1.0);

    // Try disabling animations by uncommenting this:
    // return in.color;  // No animation, just static gradient

    // Other experiment ideas:
    // - return float4(color * (1.0 + 0.5 * sin(uniforms.time)), 1.0);
    // - float wave = sin(uniforms.time + in.position.x * 0.01);
    // - Use uniforms.frameCount instead of time for frame-based animation
}

// =============================================================================
// HOW THE ANIMATION WORKS - CPU ↔ GPU DATA FLOW
// =============================================================================
/*
To animate on the GPU, we pass time from the CPU each frame:

CPU SIDE (main.mm):
┌─────────────────────────────────────────────────────────────────┐
│ 1. Create buffer:                                               │
│    _uniformBuffer = [device newBufferWithLength:sizeof(Uniforms)│
│                                          options:...Shared];    │
│                                                                 │
│ 2. Each frame, write current time:                             │
│    Uniforms* u = (Uniforms*)_uniformBuffer.contents;           │
│    u->time = CACurrentMediaTime() - _startTime;                │
│    u->frameCount = _frameCount;                                │
│                                                                 │
│ 3. Bind buffer to shader:                                      │
│    [encoder setFragmentBuffer:_uniformBuffer ... atIndex:0];   │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
GPU SIDE (Compute.metal):
┌─────────────────────────────────────────────────────────────────┐
│ fragment float4 fragment_main(                                  │
│     VertexOut in [[stage_in]],                                  │
│     constant Uniforms& uniforms [[buffer(0)]]  // ← Receive it! │
│ ) {                                                             │
│     // Now use uniforms.time for animation                     │
│     float pulse = sin(uniforms.time);                           │
│     return float4(in.color.rgb * pulse, 1.0);                   │
│ }                                                               │
└─────────────────────────────────────────────────────────────────┘

This pattern works for ANY CPU→GPU data:
- Mouse position
- User input state
- Simulation parameters
- Anything that changes each frame!
*/

// =============================================================================
// BONUS: COMPUTE KERNEL EXAMPLE (commented out)
// =============================================================================
/*
Compute kernels are for general-purpose GPU computing.
Here's a simple example that squares numbers in parallel:

kernel void square_numbers(
    device float* input [[buffer(0)]],    // Read from this
    device float* output [[buffer(1)]],   // Write to this
    uint id [[thread_position_in_grid]]   // Which thread am I?
) {
    // Each thread processes one element
    output[id] = input[id] * input[id];
}

To use from CPU:

// Create input buffer
float input[1000] = { 1, 2, 3, ... };
id<MTLBuffer> inputBuf = [device newBufferWithBytes:input ...];

// Create output buffer
id<MTLBuffer> outputBuf = [device newBufferWithLength:sizeof(float)*1000 ...];

// Create compute pipeline
id<MTLFunction> func = [library newFunctionWithName:@"square_numbers"];
id<MTLComputePipelineState> pipeline = [device newComputePipelineStateWithFunction:func ...];

// Encode compute command
id<MTLComputeCommandEncoder> encoder = [commandBuffer computeCommandEncoder];
[encoder setComputePipelineState:pipeline];
[encoder setBuffer:inputBuf offset:0 atIndex:0];
[encoder setBuffer:outputBuf offset:0 atIndex:1];

// Launch 1000 threads (one per number)
MTLSize gridSize = MTLSizeMake(1000, 1, 1);
MTLSize threadgroupSize = MTLSizeMake(64, 1, 1);  // 64 threads per group
[encoder dispatchThreads:gridSize threadsPerThreadgroup:threadgroupSize];

[encoder endEncoding];
[commandBuffer commit];
[commandBuffer waitUntilCompleted];

// Now outputBuf contains the squared numbers!
float* results = (float*)outputBuf.contents;
*/

// =============================================================================
// RENDERING PIPELINE SUMMARY
// =============================================================================
/*
╔═══════════════════════════════════════════════════════════════════════════╗
║                        Graphics Pipeline Flow                             ║
╠═══════════════════════════════════════════════════════════════════════════╣
║                                                                           ║
║  CPU Side (main.mm):                                                      ║
║  ┌────────────────────────────────────────────────────────────────┐      ║
║  │ [encoder drawPrimitives:MTLPrimitiveTypeTriangle               │      ║
║  │                vertexStart:0 vertexCount:3]                     │      ║
║  └────────────────────────────────────────────────────────────────┘      ║
║                              │                                            ║
║                              ▼                                            ║
║  GPU Side (Compute.metal):                                                ║
║  ┌────────────────────────────────────────────────────────────────┐      ║
║  │ vertex_main() called 3 times:                                   │      ║
║  │   - Vertex 0 → position(-1, -1), color(red)                     │      ║
║  │   - Vertex 1 → position( 3, -1), color(blue)                    │      ║
║  │   - Vertex 2 → position(-1,  3), color(green)                   │      ║
║  └────────────────────────────────────────────────────────────────┘      ║
║                              │                                            ║
║                              ▼                                            ║
║  ┌────────────────────────────────────────────────────────────────┐      ║
║  │ Rasterizer: Find which pixels the triangle covers               │      ║
║  │   - Interpolate colors smoothly across the surface              │      ║
║  └────────────────────────────────────────────────────────────────┘      ║
║                              │                                            ║
║                              ▼                                            ║
║  ┌────────────────────────────────────────────────────────────────┐      ║
║  │ fragment_main() called for each pixel (800x600 = 480,000 times!)│      ║
║  │   - Each pixel gets interpolated color                          │      ║
║  │   - Returns final pixel color                                   │      ║
║  └────────────────────────────────────────────────────────────────┘      ║
║                              │                                            ║
║                              ▼                                            ║
║  ┌────────────────────────────────────────────────────────────────┐      ║
║  │ Output Merger: Write pixel colors to framebuffer                │      ║
║  └────────────────────────────────────────────────────────────────┘      ║
║                              │                                            ║
║                              ▼                                            ║
║                         [Screen Display]                                  ║
║                                                                           ║
╚═══════════════════════════════════════════════════════════════════════════╝

All of this happens in ~16ms for 60 FPS!
The GPU runs vertex_main and fragment_main in parallel on thousands of cores.
*/

// =============================================================================
// EXERCISE IDEAS FOR LEARNING
// =============================================================================
/*
1. CHANGE THE COLORS
   - Modify the colors array in vertex_main
   - Try making it monochrome, or use your favorite colors

2. ADD MORE TRIANGLES
   - In main.mm, change vertexCount:3 to vertexCount:6
   - Add 3 more positions/colors to the arrays
   - What happens?

3. MAKE IT MOVE
   - Add a Uniforms buffer with time (see example above)
   - Use sin(time) to animate vertex positions or colors

4. TRY DIFFERENT MATH
   - In fragment_main, modify in.color before returning
   - Try: pow(in.color, 2.2) for gamma correction
   - Try: mix(in.color, float4(1.0), 0.5) for fading

5. GRADIENT PATTERNS
   - Use in.position to create patterns
   - Try: sin(in.position.x * 10) for stripes
   - Try: length(in.position.xy) for radial gradient

6. READ THE TEMPLATE PROJECT
   - Look at metal_particle_template/ for more advanced usage
   - See how to use buffers, multiple shaders, compute kernels

7. CONVERT FROM C++
   - Take a simple C++ program you wrote
   - Move the main loop logic into a compute kernel
   - See how much faster it runs on GPU!
*/
