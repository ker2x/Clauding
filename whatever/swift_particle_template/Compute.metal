// =============================================================================
// Metal Particle Template - Compute.metal
// =============================================================================
// This file contains Metal Shading Language (MSL) code that runs on the GPU.
// Metal shaders are written in a C++14-based language with GPU-specific features.
//
// This file contains:
// 1. Shared data structures (must match CPU definitions)
// 2. Compute kernel for particle physics simulation
// 3. Vertex and fragment shaders for rendering particles
//
// Key Concepts:
// - Compute shaders run massively parallel computations (thousands of threads)
// - Vertex shaders process each vertex (particle position -> screen position)
// - Fragment shaders determine the color of each pixel
// =============================================================================

#include <metal_stdlib>
using namespace metal;

// =============================================================================
// DATA STRUCTURES
// =============================================================================
// These structures MUST match the definitions in main.mm exactly.
// Metal and C++ share the same memory layout for these types.

// Simulation parameters (read-only in shaders)
struct SimParams {
    float width;           // Canvas width in pixels
    float height;          // Canvas height in pixels
    float interaction_radius;  // Maximum distance for particle interaction
    float force_strength;      // Multiplier for interaction forces
    float friction;            // Velocity damping factor (0-1)
    int   num_particles;       // Total number of particles
    int   num_types;           // Number of particle types
    float point_size;          // Particle render size in pixels
};

// =============================================================================
// COMPUTE KERNEL: Particle Physics Simulation
// =============================================================================
// This kernel runs once per particle, updating its velocity and position.
//
// How it works:
// 1. Each thread handles one particle
// 2. The particle checks interactions with all other particles
// 3. Forces are accumulated based on distance and particle types
// 4. Velocity and position are updated using Euler integration
//
// Thread Organization:
// - Grid: One thread per particle (num_particles threads total)
// - Each thread is identified by its global thread ID (gid)
//
// Buffer Attributes:
// - device: Buffer lives in GPU memory, accessible by all threads
// - constant: Read-only data, can be cached efficiently

kernel void update_particles(
    device float2*       positions  [[buffer(0)]],  // Particle positions (read/write)
    device float2*       velocities [[buffer(1)]],  // Particle velocities (read/write)
    device const int*    types      [[buffer(2)]],  // Particle types (read-only)
    device const float*  matrix     [[buffer(3)]],  // Interaction matrix (read-only)
    constant SimParams&  params     [[buffer(4)]],  // Simulation parameters (read-only)
    uint gid [[thread_position_in_grid]]            // This thread's global ID
) {
    // Bounds check: Make sure we don't process non-existent particles
    if (gid >= uint(params.num_particles)) return;

    // Load this particle's data
    float2 pos  = positions[gid];
    float2 vel  = velocities[gid];
    int    type = types[gid];

    // Accumulate forces from all other particles
    float2 totalForce = float2(0.0f, 0.0f);

    // Cache some values for efficiency
    const float rMax  = params.interaction_radius;
    const float rMax2 = rMax * rMax;  // Squared distance (avoid sqrt in inner loop)

    // Loop through all other particles
    for (int j = 0; j < params.num_particles; j++) {
        // Skip self-interaction
        if (uint(j) == gid) continue;

        // Get the other particle's position
        float2 other_pos = positions[j];

        // Calculate displacement vector
        float2 delta = other_pos - pos;

        // Handle wrapping (toroidal world)
        // If a particle is more than half the world away, wrap around
        if (delta.x > params.width * 0.5f)  delta.x -= params.width;
        if (delta.x < -params.width * 0.5f) delta.x += params.width;
        if (delta.y > params.height * 0.5f)  delta.y -= params.height;
        if (delta.y < -params.height * 0.5f) delta.y += params.height;

        // Calculate squared distance (faster than actual distance)
        float dist2 = dot(delta, delta);

        // Skip if too far away
        if (dist2 > rMax2 || dist2 < 0.0001f) continue;

        // Calculate actual distance (only when needed)
        float dist = sqrt(dist2);
        float normalizedDist = dist / rMax;  // 0.0 to 1.0

        // Look up interaction strength from matrix
        // The matrix is indexed as: matrix[my_type * num_types + other_type]
        int other_type = types[j];
        float attraction = matrix[type * params.num_types + other_type];

        // Calculate force magnitude using a simple interaction function
        // This creates interesting emergent behavior:
        // - Very close particles repel (prevents clumping)
        // - Medium distance: attraction/repulsion based on matrix
        // - Far particles: no interaction
        float forceMagnitude = 0.0f;

        if (normalizedDist < 0.3f) {
            // Short range: repulsion (prevents particles from overlapping)
            forceMagnitude = normalizedDist / 0.3f - 1.0f;
        } else {
            // Medium range: attraction or repulsion based on matrix
            // Scales from 0 at the edges to full strength in the middle
            float t = (normalizedDist - 0.3f) / 0.7f;  // 0 to 1
            forceMagnitude = attraction * (1.0f - abs(2.0f * t - 1.0f));
        }

        // Apply force in the direction of the other particle
        float2 direction = delta / dist;  // Normalized direction
        totalForce += direction * forceMagnitude * params.force_strength;
    }

    // Update velocity (apply force, then friction)
    vel += totalForce;
    vel *= params.friction;

    // Update position
    pos += vel;

    // Wrap position to keep particles in bounds (toroidal world)
    if (pos.x < 0.0f)           pos.x += params.width;
    if (pos.x >= params.width)  pos.x -= params.width;
    if (pos.y < 0.0f)           pos.y += params.height;
    if (pos.y >= params.height) pos.y -= params.height;

    // Write back updated values
    positions[gid]  = pos;
    velocities[gid] = vel;
}

// =============================================================================
// VERTEX SHADER OUTPUT STRUCTURE
// =============================================================================
// This structure is passed from the vertex shader to the fragment shader.
// Each field has an attribute specifying its purpose.

struct VertexOut {
    float4 position [[position]];    // Clip-space position (required)
    float4 color    [[flat]];        // Particle color (flat = no interpolation)
    float  pointSize [[point_size]]; // Size of point primitive in pixels
};

// =============================================================================
// VERTEX SHADER: Transform Particle Positions
// =============================================================================
// The vertex shader runs once per vertex (particle).
// It transforms world coordinates to clip coordinates and assigns colors.
//
// Clip coordinates:
// - X: -1 (left) to +1 (right)
// - Y: -1 (bottom) to +1 (top)
// - Z: 0 (near) to 1 (far) - we use 0 for 2D
// - W: 1 (no perspective divide needed for 2D)

vertex VertexOut vertex_main(
    device const float2* positions [[buffer(0)]],   // Particle positions
    device const int*    types     [[buffer(1)]],   // Particle types
    device const float4* colors    [[buffer(2)]],   // Colors per type
    constant SimParams&  params    [[buffer(3)]],   // Simulation parameters
    uint vid [[vertex_id]]                          // Which vertex (particle) is this
) {
    VertexOut out;

    // Get this particle's position
    float2 pos = positions[vid];

    // Transform from pixel coordinates to clip coordinates
    // Pixel: (0,0) top-left, (width,height) bottom-right
    // Clip:  (-1,-1) bottom-left, (1,1) top-right
    float2 clipPos;
    clipPos.x = (pos.x / params.width)  * 2.0f - 1.0f;
    clipPos.y = (pos.y / params.height) * 2.0f - 1.0f;
    clipPos.y = -clipPos.y;  // Flip Y (Metal's Y is up, our Y is down)

    out.position = float4(clipPos, 0.0f, 1.0f);

    // Get color based on particle type
    int type = types[vid];
    out.color = colors[type];

    // Set point size
    out.pointSize = params.point_size;

    return out;
}

// =============================================================================
// FRAGMENT SHADER: Color Each Pixel
// =============================================================================
// The fragment shader runs once per pixel covered by a primitive.
// For point primitives, it colors the pixels within the point's square.
//
// This simple shader just returns the color from the vertex shader.
// More advanced versions could:
// - Create circular particles using point_coord
// - Add glow effects
// - Implement smooth edges

fragment float4 fragment_main(
    VertexOut in [[stage_in]],                    // Interpolated data from vertex shader
    float2 pointCoord [[point_coord]]             // Position within point (0-1)
) {
    // Optional: Make particles circular instead of square
    // Uncomment to enable:
    /*
    float2 centered = pointCoord - 0.5f;
    float dist = length(centered);
    if (dist > 0.5f) {
        discard_fragment();  // Outside circle
    }
    // Smooth edge
    float alpha = 1.0f - smoothstep(0.4f, 0.5f, dist);
    return float4(in.color.rgb, in.color.a * alpha);
    */

    // Simple: just return the color
    (void)pointCoord;  // Suppress unused warning
    return in.color;
}

// =============================================================================
// ADDITIONAL UTILITY FUNCTIONS (for extension)
// =============================================================================
// You can add more compute kernels or shader functions here.
// Some ideas:
//
// 1. A kernel to sort particles by position for spatial hashing
// 2. A kernel to compute density fields
// 3. A kernel to apply external forces (gravity, wind, attractors)
// 4. Alternative fragment shaders for different particle styles
//
// Example: Simple gravity kernel
/*
kernel void apply_gravity(
    device float2* velocities [[buffer(0)]],
    constant float2& gravity [[buffer(1)]],
    uint gid [[thread_position_in_grid]]
) {
    velocities[gid] += gravity;
}
*/
