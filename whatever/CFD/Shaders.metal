#include <metal_stdlib>
using namespace metal;

// ============================================================================
// RENDERING: Fullscreen Quad for Texture Scaling
// ============================================================================

struct VertexOut {
    float4 position [[position]];
    float2 texCoord;
};

vertex VertexOut fullscreen_vertex(uint vertexID [[vertex_id]]) {
    // Generate fullscreen quad: two triangles covering [-1,1] NDC space
    float2 positions[6] = {
        float2(-1.0, -1.0),
        float2( 1.0, -1.0),
        float2(-1.0,  1.0),
        float2(-1.0,  1.0),
        float2( 1.0, -1.0),
        float2( 1.0,  1.0)
    };

    float2 texCoords[6] = {
        float2(0.0, 1.0),
        float2(1.0, 1.0),
        float2(0.0, 0.0),
        float2(0.0, 0.0),
        float2(1.0, 1.0),
        float2(1.0, 0.0)
    };

    VertexOut out;
    out.position = float4(positions[vertexID], 0.0, 1.0);
    out.texCoord = texCoords[vertexID];
    return out;
}

fragment float4 fullscreen_fragment(VertexOut in [[stage_in]],
                                   texture2d<float> inputTexture [[texture(0)]]) {
    constexpr sampler textureSampler(mag_filter::linear, min_filter::linear);
    return inputTexture.sample(textureSampler, in.texCoord);
}

// ============================================================================
// CFD SIMULATION KERNELS
// ============================================================================

// D2Q9 Lattice velocities
constant int2 e[9] = {
    int2(0, 0),   // 0: rest
    int2(1, 0),   // 1: east
    int2(0, 1),   // 2: north
    int2(-1, 0),  // 3: west
    int2(0, -1),  // 4: south
    int2(1, 1),   // 5: northeast
    int2(-1, 1),  // 6: northwest
    int2(-1, -1), // 7: southwest
    int2(1, -1)   // 8: southeast
};

// D2Q9 Lattice weights
constant float w[9] = {
    4.0/9.0,  // 0
    1.0/9.0,  // 1
    1.0/9.0,  // 2
    1.0/9.0,  // 3
    1.0/9.0,  // 4
    1.0/36.0, // 5
    1.0/36.0, // 6
    1.0/36.0, // 7
    1.0/36.0  // 8
};

// Opposite direction indices for bounce-back
constant int opp[9] = {0, 3, 4, 1, 2, 7, 8, 5, 6};

// Compute equilibrium distribution
float feq(int i, float rho, float2 u) {
    float2 ei = float2(e[i]);
    float eu = dot(ei, u);
    float u2 = dot(u, u);
    return rho * w[i] * (1.0 + 3.0*eu + 4.5*eu*eu - 1.5*u2);
}

// ============================================================================
// KERNEL: Stream and Collide
// ============================================================================
kernel void lbm_stream_collide(
    texture2d_array<float, access::read> f_in [[texture(0)]],
    texture2d_array<float, access::write> f_out [[texture(1)]],
    texture2d<float, access::read> obstacles [[texture(2)]],
    constant float &tau [[buffer(0)]],
    constant float2 &inlet_velocity [[buffer(1)]],
    uint2 gid [[thread_position_in_grid]])
{
    uint width = f_in.get_width();
    uint height = f_in.get_height();

    // Check if this is an obstacle cell
    float is_obstacle = obstacles.read(gid).r;

    if (is_obstacle > 0.5) {
        // Bounce-back boundary condition for obstacles
        for (int i = 0; i < 9; i++) {
            // Stream from opposite direction
            int2 source = int2(gid) - e[i];

            // Periodic or clamped boundaries for streaming
            if (source.x < 0) source.x = 0;
            if (source.x >= int(width)) source.x = width - 1;
            if (source.y < 0) source.y = 0;
            if (source.y >= int(height)) source.y = height - 1;

            // Bounce back: read from opposite direction
            float f_val = f_in.read(uint2(source), opp[i]).r;
            f_out.write(float4(f_val, 0, 0, 0), gid, i);
        }
        return;
    }

    // Stream step: gather distributions from neighbors
    float f[9];
    for (int i = 0; i < 9; i++) {
        int2 source = int2(gid) - e[i];

        // Handle boundaries during streaming
        if (source.x < 0 || source.x >= int(width) ||
            source.y < 0 || source.y >= int(height)) {
            // Use current cell value for out-of-bounds
            f[i] = f_in.read(gid, i).r;
        } else {
            f[i] = f_in.read(uint2(source), i).r;
        }
    }

    // Compute macroscopic quantities
    float rho = 0.0;
    float2 u = float2(0.0);

    for (int i = 0; i < 9; i++) {
        rho += f[i];
        u += f[i] * float2(e[i]);
    }

    if (rho > 0.0001) {
        u /= rho;
    }

    // Collision step (BGK)
    float omega = 1.0 / tau;
    for (int i = 0; i < 9; i++) {
        float feq_i = feq(i, rho, u);
        float f_new = f[i] - omega * (f[i] - feq_i);
        f_out.write(float4(f_new, 0, 0, 0), gid, i);
    }
}

// ============================================================================
// KERNEL: Apply Boundary Conditions
// ============================================================================
kernel void lbm_boundary(
    texture2d_array<float, access::read_write> f [[texture(0)]],
    constant float2 &inlet_velocity [[buffer(0)]],
    uint2 gid [[thread_position_in_grid]])
{
    uint width = f.get_width();
    uint height = f.get_height();
    float rho0 = 1.0;

    // Left boundary (inlet) - Zou-He velocity BC
    if (gid.x == 0) {
        float2 u = inlet_velocity;
        float rho = rho0;

        // Set equilibrium at inlet
        for (int i = 0; i < 9; i++) {
            float feq_i = feq(i, rho, u);
            f.write(float4(feq_i, 0, 0, 0), gid, i);
        }
    }

    // Right boundary (outlet) - Zero gradient / open boundary
    if (gid.x == width - 1) {
        uint2 neighbor = uint2(width - 2, gid.y);
        for (int i = 0; i < 9; i++) {
            float f_val = f.read(neighbor, i).r;
            f.write(float4(f_val, 0, 0, 0), gid, i);
        }
    }

    // Top boundary (slip wall) - specular reflection
    if (gid.y == height - 1) {
        // Reflect velocities: swap north/south components
        float f_temp[9];
        for (int i = 0; i < 9; i++) {
            f_temp[i] = f.read(gid, i).r;
        }

        // Swap: 2<->4, 5<->8, 6<->7
        f.write(float4(f_temp[4], 0, 0, 0), gid, 2);
        f.write(float4(f_temp[2], 0, 0, 0), gid, 4);
        f.write(float4(f_temp[8], 0, 0, 0), gid, 5);
        f.write(float4(f_temp[7], 0, 0, 0), gid, 6);
        f.write(float4(f_temp[6], 0, 0, 0), gid, 7);
        f.write(float4(f_temp[5], 0, 0, 0), gid, 8);
    }

    // Bottom boundary (slip wall) - specular reflection
    if (gid.y == 0) {
        float f_temp[9];
        for (int i = 0; i < 9; i++) {
            f_temp[i] = f.read(gid, i).r;
        }

        // Swap: 2<->4, 5<->8, 6<->7
        f.write(float4(f_temp[4], 0, 0, 0), gid, 2);
        f.write(float4(f_temp[2], 0, 0, 0), gid, 4);
        f.write(float4(f_temp[8], 0, 0, 0), gid, 5);
        f.write(float4(f_temp[7], 0, 0, 0), gid, 6);
        f.write(float4(f_temp[6], 0, 0, 0), gid, 7);
        f.write(float4(f_temp[5], 0, 0, 0), gid, 8);
    }
}

// ============================================================================
// KERNEL: Compute Macroscopic Fields (density, velocity, vorticity)
// ============================================================================
kernel void compute_fields(
    texture2d_array<float, access::read> f [[texture(0)]],
    texture2d<float, access::write> velocity_mag [[texture(1)]],
    texture2d<float, access::write> vorticity [[texture(2)]],
    uint2 gid [[thread_position_in_grid]])
{
    uint width = f.get_width();
    uint height = f.get_height();

    // Compute density and velocity
    float rho = 0.0;
    float2 u = float2(0.0);

    for (int i = 0; i < 9; i++) {
        float f_i = f.read(gid, i).r;
        rho += f_i;
        u += f_i * float2(e[i]);
    }

    if (rho > 0.0001) {
        u /= rho;
    }

    float vel_mag = length(u);
    velocity_mag.write(float4(vel_mag, 0, 0, 0), gid);

    // Compute vorticity (curl of velocity field)
    // ω = ∂v/∂x - ∂u/∂y
    float omega = 0.0;

    if (gid.x > 0 && gid.x < width - 1 && gid.y > 0 && gid.y < height - 1) {
        // Get velocities at neighboring cells
        float2 u_xp = float2(0.0), u_xm = float2(0.0);
        float2 u_yp = float2(0.0), u_ym = float2(0.0);

        // x+1
        float rho_xp = 0.0;
        for (int i = 0; i < 9; i++) {
            float f_i = f.read(uint2(gid.x + 1, gid.y), i).r;
            rho_xp += f_i;
            u_xp += f_i * float2(e[i]);
        }
        if (rho_xp > 0.0001) u_xp /= rho_xp;

        // x-1
        float rho_xm = 0.0;
        for (int i = 0; i < 9; i++) {
            float f_i = f.read(uint2(gid.x - 1, gid.y), i).r;
            rho_xm += f_i;
            u_xm += f_i * float2(e[i]);
        }
        if (rho_xm > 0.0001) u_xm /= rho_xm;

        // y+1
        float rho_yp = 0.0;
        for (int i = 0; i < 9; i++) {
            float f_i = f.read(uint2(gid.x, gid.y + 1), i).r;
            rho_yp += f_i;
            u_yp += f_i * float2(e[i]);
        }
        if (rho_yp > 0.0001) u_yp /= rho_yp;

        // y-1
        float rho_ym = 0.0;
        for (int i = 0; i < 9; i++) {
            float f_i = f.read(uint2(gid.x, gid.y - 1), i).r;
            rho_ym += f_i;
            u_ym += f_i * float2(e[i]);
        }
        if (rho_ym > 0.0001) u_ym /= rho_ym;

        // Central difference
        float dvdx = (u_xp.y - u_xm.y) * 0.5;
        float dudy = (u_yp.x - u_ym.x) * 0.5;
        omega = dvdx - dudy;
    }

    vorticity.write(float4(omega, 0, 0, 0), gid);
}

// ============================================================================
// KERNEL: Render Visualization (Velocity + Vorticity Overlay)
// ============================================================================
kernel void render_field(
    texture2d<float, access::read> velocity_mag [[texture(0)]],
    texture2d<float, access::read> vorticity [[texture(1)]],
    texture2d<float, access::read> obstacles [[texture(2)]],
    texture2d<float, access::write> output [[texture(3)]],
    constant float &max_velocity [[buffer(0)]],
    constant float &vorticity_threshold [[buffer(1)]],
    uint2 gid [[thread_position_in_grid]])
{
    float is_obstacle = obstacles.read(gid).r;

    // Draw obstacles as black
    if (is_obstacle > 0.5) {
        output.write(float4(0.2, 0.2, 0.2, 1.0), gid);
        return;
    }

    // Velocity magnitude colormap (Viridis-like)
    float vel = velocity_mag.read(gid).r;
    float t = clamp(vel / max_velocity, 0.0, 1.0);

    // Simplified Viridis colormap
    float3 color;
    if (t < 0.25) {
        float s = t / 0.25;
        color = mix(float3(0.267, 0.005, 0.329), float3(0.282, 0.141, 0.458), s);
    } else if (t < 0.5) {
        float s = (t - 0.25) / 0.25;
        color = mix(float3(0.282, 0.141, 0.458), float3(0.164, 0.471, 0.558), s);
    } else if (t < 0.75) {
        float s = (t - 0.5) / 0.25;
        color = mix(float3(0.164, 0.471, 0.558), float3(0.134, 0.659, 0.518), s);
    } else {
        float s = (t - 0.75) / 0.25;
        color = mix(float3(0.134, 0.659, 0.518), float3(0.993, 0.906, 0.144), s);
    }

    // Overlay vorticity as brightness modulation
    float omega = vorticity.read(gid).r;
    float vort_intensity = abs(omega) / vorticity_threshold;
    vort_intensity = clamp(vort_intensity, 0.0, 1.0);

    // Enhance color where vorticity is high
    if (vort_intensity > 0.3) {
        float enhance = vort_intensity * 0.5;
        color = color * (1.0 + enhance);
    }

    output.write(float4(color, 1.0), gid);
}

// ============================================================================
// KERNEL: Update Obstacles (Drawing)
// ============================================================================
kernel void update_obstacles(
    texture2d<float, access::read_write> obstacles [[texture(0)]],
    constant float2 &brush_center [[buffer(0)]],
    constant float &brush_radius [[buffer(1)]],
    constant float &draw_value [[buffer(2)]],
    uint2 gid [[thread_position_in_grid]])
{
    // Use cell center for distance calculation (add 0.5 to each coordinate)
    float2 cellCenter = float2(gid) + 0.5;
    float dist = distance(cellCenter, brush_center);

    // Use slightly larger radius to ensure full coverage
    if (dist <= brush_radius + 0.5) {
        obstacles.write(float4(draw_value, 0, 0, 0), gid);
    }
}

// ============================================================================
// KERNEL: Initialize Distribution Functions
// ============================================================================
kernel void initialize_distributions(
    texture2d_array<float, access::write> f [[texture(0)]],
    constant float2 &initial_velocity [[buffer(0)]],
    uint2 gid [[thread_position_in_grid]])
{
    float rho = 1.0;
    float2 u = initial_velocity;

    for (int i = 0; i < 9; i++) {
        float feq_i = feq(i, rho, u);
        f.write(float4(feq_i, 0, 0, 0), gid, i);
    }
}
