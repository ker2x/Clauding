// =============================================================================
// Metal Neural Aura - Compute.metal
// =============================================================================
// Neural particle field simulation shaders.
// Implements a 3-layer neural network in Metal to generate particle forces
// from position and local density context.
//
// Three compute kernels:
//   1. density_map_kernel - Computes particle density grid
//   2. neural_force_kernel - Runs neural network forward pass per particle
//   3. physics_update_kernel - Updates velocity, position, and color
//
// Vertex and fragment shaders for rendering particles with per-particle colors.
// =============================================================================

#include <metal_stdlib>
using namespace metal;

// =============================================================================
// DATA STRUCTURES
// =============================================================================
// Must match main.mm exactly

struct SimParams {
    float width;
    float height;
    float friction;
    float neural_strength;
    float jitter_strength;
    int num_particles;
    int density_res;
    float point_size;
    float2 mouse_pos;
    float mouse_strength;
};

// =============================================================================
// HELPER FUNCTIONS
// =============================================================================

// Fast tanh approximation using rational function
// More accurate than simple polynomial and avoids exp()
float tanh_approx(float x) {
    if (x > 5.0f) return 1.0f;
    if (x < -5.0f) return -1.0f;
    float x2 = x * x;
    return x * (27.0f + x2) / (27.0f + 9.0f * x2);
}

// Matrix-vector multiply: output = W × input + b
// W is stored row-major: W[i * cols + j] = weight from input[j] to output[i]
void matvec(
    device const float* W,
    device const float* b,
    thread const float* input,
    thread float* output,
    int rows,
    int cols
) {
    for (int i = 0; i < rows; i++) {
        float sum = b[i];
        for (int j = 0; j < cols; j++) {
            sum += W[i * cols + j] * input[j];
        }
        output[i] = sum;
    }
}

// =============================================================================
// KERNEL 1: Density Map
// =============================================================================
// Computes a 2D histogram of particle positions on a density grid.
// Each thread represents one particle and atomically adds to the grid cell.
// The first thread also clears the density map.

kernel void density_map_kernel(
    device atomic_float* density_map [[buffer(0)]],
    device const float2* positions   [[buffer(1)]],
    constant SimParams&  params      [[buffer(2)]],
    uint gid [[thread_position_in_grid]]
) {
    // First thread clears the density map
    if (gid == 0) {
        int mapSize = params.density_res * params.density_res;
        for (int i = 0; i < mapSize; i++) {
            atomic_store_explicit(&density_map[i], 0.0f, memory_order_relaxed);
        }
    }

    // Wait for clear to complete
    threadgroup_barrier(mem_flags::mem_device);

    if (gid >= uint(params.num_particles)) return;

    // Convert position from [-1, 1] to grid coordinates [0, density_res-1]
    float2 pos = positions[gid];
    int2 grid_pos = int2((pos + 1.0f) * 0.5f * float(params.density_res - 1));
    grid_pos = clamp(grid_pos, int2(0), int2(params.density_res - 1));

    int idx = grid_pos.y * params.density_res + grid_pos.x;
    atomic_fetch_add_explicit(&density_map[idx], 1.0f, memory_order_relaxed);
}

// =============================================================================
// KERNEL 2: Neural Forces
// =============================================================================
// Runs a 3-layer neural network forward pass for each particle.
// Input: Fourier-encoded position (64 features) + density context (1 feature)
// Output: 2D force vector

kernel void neural_force_kernel(
    device const float2*  positions    [[buffer(0)]],
    device const float*   density_map  [[buffer(1)]],
    device const float2*  fourier_freq [[buffer(2)]],
    device const float*   w1           [[buffer(3)]],
    device const float*   b1           [[buffer(4)]],
    device const float*   w2           [[buffer(5)]],
    device const float*   b2           [[buffer(6)]],
    device const float*   w3           [[buffer(7)]],
    device const float*   b3           [[buffer(8)]],
    device float2*        forces       [[buffer(9)]],
    constant SimParams&   params       [[buffer(10)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= uint(params.num_particles)) return;

    float2 pos = positions[gid];

    // 1. Fourier positional encoding
    // pos_enc = [sin(pos·freq[0]), cos(pos·freq[0]), ..., sin(pos·freq[31]), cos(pos·freq[31])]
    float pos_enc[65];
    for (int i = 0; i < 32; i++) {
        float phase = dot(pos, fourier_freq[i]);
        pos_enc[i * 2]     = sin(phase);
        pos_enc[i * 2 + 1] = cos(phase);
    }

    // 2. Sample density at particle position
    int2 grid_pos = int2((pos + 1.0f) * 0.5f * float(params.density_res - 1));
    grid_pos = clamp(grid_pos, int2(0), int2(params.density_res - 1));
    int idx = grid_pos.y * params.density_res + grid_pos.x;

    float dens = log(1.0f + density_map[idx]) / 5.0f;
    pos_enc[64] = sin(dens * 4.0f);

    // 3. Layer 1: (64+1) → 128 with Tanh activation
    float h1[128];
    matvec(w1, b1, pos_enc, h1, 128, 65);
    for (int i = 0; i < 128; i++) {
        h1[i] = tanh_approx(h1[i]);
    }

    // 4. Layer 2: 128 → 128 with Tanh activation
    float h2[128];
    matvec(w2, b2, h1, h2, 128, 128);
    for (int i = 0; i < 128; i++) {
        h2[i] = tanh_approx(h2[i]);
    }

    // 5. Layer 3: 128 → 2 (linear output)
    float out[2];
    matvec(w3, b3, h2, out, 2, 128);

    forces[gid] = float2(out[0], out[1]);
}

// =============================================================================
// KERNEL 3: Physics Update
// =============================================================================
// Updates particle velocity, position, and color based on neural forces,
// mouse interaction, and random jitter.

kernel void physics_update_kernel(
    device float2*       positions     [[buffer(0)]],
    device float2*       velocities    [[buffer(1)]],
    device float3*       colors        [[buffer(2)]],
    device const float*  density_map   [[buffer(3)]],
    device const float2* forces        [[buffer(4)]],
    constant SimParams&  params        [[buffer(5)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= uint(params.num_particles)) return;

    float2 pos = positions[gid];
    float2 vel = velocities[gid];
    float3 col = colors[gid];

    // 1. Neural force
    float2 neural_force = forces[gid] * params.neural_strength;

    // 2. Mouse interaction force
    float2 delta_mouse = pos - params.mouse_pos;
    float r2 = dot(delta_mouse, delta_mouse) + 0.01f;
    float2 mouse_force = (delta_mouse / r2) * params.mouse_strength;

    // 3. Random jitter (simple LCG for pseudo-random)
    uint seed = gid * 747796405u + 2891336453u;
    uint rng = seed * 747796405u + 2891336453u;
    float rx = float(rng & 0xFFFFu) / 65535.0f * 2.0f - 1.0f;
    rng = rng * 747796405u + 2891336453u;
    float ry = float(rng & 0xFFFFu) / 65535.0f * 2.0f - 1.0f;
    float2 jitter = float2(rx, ry) * params.jitter_strength;

    // 4. Density pressure (anti-clustering)
    int2 grid_pos = int2((pos + 1.0f) * 0.5f * float(params.density_res - 1));
    grid_pos = clamp(grid_pos, int2(0), int2(params.density_res - 1));
    int idx = grid_pos.y * params.density_res + grid_pos.x;
    float dens = log(1.0f + density_map[idx]) / 5.0f;

    rng = rng * 747796405u + 2891336453u;
    float px = float(rng & 0xFFFFu) / 65535.0f * 2.0f - 1.0f;
    rng = rng * 747796405u + 2891336453u;
    float py = float(rng & 0xFFFFu) / 65535.0f * 2.0f - 1.0f;
    float2 pressure_force = float2(px, py) * step(0.7f, dens) * 0.01f;

    // 5. Update velocity with friction
    vel *= params.friction;
    vel += neural_force + mouse_force + jitter + pressure_force;

    // 6. Update position
    pos += vel;

    // 7. Toroidal wrap (map [-1,1] to [0,2], modulo, back to [-1,1])
    pos = fmod(pos + 1.0f + 2.0f, 2.0f) - 1.0f;

    // 8. Update color based on speed and density
    float speed = length(vel);
    col.r = col.r * 0.99f + clamp(speed * 5.0f, 0.0f, 1.0f) * 0.01f;
    col.g = col.g * 0.99f + (dens * 0.5f) * 0.01f;
    col.b = clamp(1.0f - col.r - col.g, 0.2f, 1.0f);

    // Write back
    positions[gid] = pos;
    velocities[gid] = vel;
    colors[gid] = col;
}

// =============================================================================
// VERTEX SHADER OUTPUT
// =============================================================================

struct VertexOut {
    float4 position [[position]];
    float4 color    [[flat]];
    float  pointSize [[point_size]];
};

// =============================================================================
// VERTEX SHADER
// =============================================================================
// Transforms particle positions to clip space and passes per-particle color.

vertex VertexOut vertex_main(
    device const float2* positions [[buffer(0)]],
    device const float3* colors    [[buffer(1)]],
    constant SimParams&  params    [[buffer(2)]],
    uint vid [[vertex_id]]
) {
    VertexOut out;

    float2 pos = positions[vid];

    // Position is already in normalized coordinates [-1, 1]
    // Just need to flip Y (Metal Y is up, our convention is down)
    float2 clipPos = pos;
    clipPos.y = -clipPos.y;

    out.position = float4(clipPos, 0.0f, 1.0f);

    // Get per-particle color and add full alpha
    float3 rgb = colors[vid];
    out.color = float4(rgb, 1.0f);

    out.pointSize = params.point_size;

    return out;
}

// =============================================================================
// FRAGMENT SHADER
// =============================================================================
// Simply returns the color from the vertex shader.
// Could be extended to make circular particles using point_coord.

fragment float4 fragment_main(
    VertexOut in [[stage_in]],
    float2 pointCoord [[point_coord]]
) {
    // Optional: make particles circular
    // Uncomment to enable:
    /*
    float2 centered = pointCoord - 0.5f;
    float dist = length(centered);
    if (dist > 0.5f) {
        discard_fragment();
    }
    float alpha = 1.0f - smoothstep(0.4f, 0.5f, dist);
    return float4(in.color.rgb, in.color.a * alpha);
    */

    (void)pointCoord;
    return in.color;
}
