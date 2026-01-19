// =============================================================================
// Quantum Garden - Compute.metal
// =============================================================================
// GPU kernels for quantum mechanics simulation using Bohmian mechanics.
//
// PHYSICS IMPLEMENTED:
//
// 1. SCHRODINGER EQUATION (wave function evolution):
//    i * hbar * d|psi>/dt = -hbar^2/(2m) * nabla^2|psi> + V|psi>
//
//    In natural units (hbar = m = 1):
//    i * d|psi>/dt = -0.5 * nabla^2|psi> + V|psi>
//
//    Split into real/imaginary parts (psi = psi_r + i*psi_i):
//    d(psi_r)/dt =  0.5 * nabla^2(psi_i) - V * psi_i
//    d(psi_i)/dt = -0.5 * nabla^2(psi_r) + V * psi_r
//
// 2. BOHMIAN MECHANICS (particle guidance):
//    Particles are guided by the wave function's phase gradient.
//    v = (hbar/m) * Im(nabla(psi)/psi)
//
//    For psi = R * exp(i*S) where R = |psi|, S = phase:
//    v = nabla(S) / m = (1/|psi|^2) * (psi_r * nabla(psi_i) - psi_i * nabla(psi_r))
//
// 3. PHASE-BASED COLORING:
//    Color particles based on local wave phase using HSV colorspace
//    Phase phi = atan2(psi_i, psi_r) maps to hue
//    Amplitude |psi| maps to brightness
// =============================================================================

#include <metal_stdlib>
using namespace metal;

// =============================================================================
// DATA STRUCTURES (must match Swift definitions exactly)
// =============================================================================

struct SimParams {
    float width;              // Canvas width in pixels
    float height;             // Canvas height in pixels
    int   grid_width;         // Wave function grid width
    int   grid_height;        // Wave function grid height
    float wave_dt;            // Time step for wave evolution
    float particle_dt;        // Particle guidance strength
    float diffusion;          // Kinetic energy coefficient (hbar^2/2m)
    float potential_strength; // User barrier strength
    int   num_particles;      // Total particle count
    float point_size;         // Render point size
    float time;               // Simulation time
    int   storm_active;       // 1 if quantum storm is happening
    float storm_time;         // Time since storm started
    float storm_center_x;     // Storm epicenter X (grid coords)
    float storm_center_y;     // Storm epicenter Y (grid coords)
};

struct SeedRequest {
    float x;                  // Grid X position
    float y;                  // Grid Y position
    float vx;                 // Momentum X
    float vy;                 // Momentum Y
    int   active;             // 1 if seed should be planted
    float padding;
};

// =============================================================================
// UTILITY FUNCTIONS
// =============================================================================

// Fire/plasma color gradient: black -> red -> orange -> yellow -> white
// Input t in [0, 1], output RGB color
float3 fire_gradient(float t) {
    t = clamp(t, 0.0f, 1.0f);

    // Multi-stop gradient for realistic fire
    // 0.0 = black, 0.2 = dark red, 0.4 = red, 0.6 = orange, 0.8 = yellow, 1.0 = white
    float3 color;

    if (t < 0.2) {
        // Black to dark red
        float f = t / 0.2;
        color = mix(float3(0.0, 0.0, 0.0), float3(0.3, 0.0, 0.0), f);
    } else if (t < 0.4) {
        // Dark red to bright red
        float f = (t - 0.2) / 0.2;
        color = mix(float3(0.3, 0.0, 0.0), float3(1.0, 0.1, 0.0), f);
    } else if (t < 0.6) {
        // Red to orange
        float f = (t - 0.4) / 0.2;
        color = mix(float3(1.0, 0.1, 0.0), float3(1.0, 0.5, 0.0), f);
    } else if (t < 0.8) {
        // Orange to yellow
        float f = (t - 0.6) / 0.2;
        color = mix(float3(1.0, 0.5, 0.0), float3(1.0, 0.9, 0.2), f);
    } else {
        // Yellow to white (hot core)
        float f = (t - 0.8) / 0.2;
        color = mix(float3(1.0, 0.9, 0.2), float3(1.0, 1.0, 0.95), f);
    }

    return color;
}

// Secondary plasma color for phase variation (blue-purple plasma accents)
float3 plasma_accent(float phase, float intensity) {
    // Map phase to blue-purple-magenta spectrum for cool accents
    float3 cool = mix(
        float3(0.2, 0.1, 0.8),  // Blue
        float3(0.8, 0.1, 0.6),  // Magenta
        (sin(phase * 2.0) + 1.0) * 0.5
    );
    return cool * intensity;
}

// Bilinear interpolation for smooth wave function sampling
float sample_wave(device const float* wave, int gw, int gh, float x, float y) {
    // Clamp to valid range
    x = clamp(x, 0.0f, float(gw - 1));
    y = clamp(y, 0.0f, float(gh - 1));

    int x0 = int(x);
    int y0 = int(y);
    int x1 = min(x0 + 1, gw - 1);
    int y1 = min(y0 + 1, gh - 1);

    float fx = x - float(x0);
    float fy = y - float(y0);

    float v00 = wave[y0 * gw + x0];
    float v10 = wave[y0 * gw + x1];
    float v01 = wave[y1 * gw + x0];
    float v11 = wave[y1 * gw + x1];

    return mix(mix(v00, v10, fx), mix(v01, v11, fx), fy);
}

// =============================================================================
// COMPUTE KERNEL: Plant Seed (add wave packet)
// =============================================================================
// Adds a Gaussian wave packet with specified momentum to the wave function.
// The wave packet is: psi(x,y) = A * exp(-r^2/(2*sigma^2)) * exp(i * k.r)
// where k is the momentum vector.

kernel void plant_seed(
    device float*          wave_real  [[buffer(0)]],
    device float*          wave_imag  [[buffer(1)]],
    device const SeedRequest& seed    [[buffer(2)]],
    constant SimParams&    params     [[buffer(3)]],
    uint2 gid [[thread_position_in_grid]]
) {
    if (gid.x >= uint(params.grid_width) || gid.y >= uint(params.grid_height)) return;
    if (seed.active == 0) return;

    int idx = gid.y * params.grid_width + gid.x;

    float dx = float(gid.x) - seed.x;
    float dy = float(gid.y) - seed.y;
    float r2 = dx * dx + dy * dy;

    // Gaussian envelope with sigma = 12 grid units
    float sigma = 12.0;
    float envelope = exp(-r2 / (2.0 * sigma * sigma));

    // Only add if envelope is significant (optimization)
    if (envelope > 0.001) {
        // Phase from momentum: k.r where k = (vx, vy)
        float phase = seed.vx * dx + seed.vy * dy;

        // Add to existing wave (superposition!)
        float amplitude = 0.8;  // New seed amplitude
        wave_real[idx] += amplitude * envelope * cos(phase);
        wave_imag[idx] += amplitude * envelope * sin(phase);
    }
}

// =============================================================================
// COMPUTE KERNEL: Apply Quantum Storm (decoherence)
// =============================================================================
// Simulates decoherence by adding random phase noise that spreads from
// the storm epicenter. This disrupts interference patterns.

kernel void apply_storm(
    device float*        wave_real [[buffer(0)]],
    device float*        wave_imag [[buffer(1)]],
    constant SimParams&  params    [[buffer(2)]],
    uint2 gid [[thread_position_in_grid]]
) {
    if (gid.x >= uint(params.grid_width) || gid.y >= uint(params.grid_height)) return;
    if (params.storm_active == 0) return;

    int idx = gid.y * params.grid_width + gid.x;

    float dx = float(gid.x) - params.storm_center_x;
    float dy = float(gid.y) - params.storm_center_y;
    float dist = sqrt(dx * dx + dy * dy);

    // Storm expands outward over time
    float storm_radius = params.storm_time * 100.0;
    float wave_width = 30.0;

    // Ring of disturbance
    float ring_dist = abs(dist - storm_radius);
    if (ring_dist < wave_width) {
        float intensity = (1.0 - ring_dist / wave_width) * 0.3;

        // Random phase perturbation (deterministic noise based on position)
        float noise = sin(float(gid.x) * 12.9898 + float(gid.y) * 78.233 + params.time * 10.0);
        float phase_shift = noise * intensity * 3.14159;

        // Apply rotation to wave function (phase shift)
        float r = wave_real[idx];
        float i = wave_imag[idx];
        float cs = cos(phase_shift);
        float sn = sin(phase_shift);

        wave_real[idx] = r * cs - i * sn;
        wave_imag[idx] = r * sn + i * cs;

        // Also reduce amplitude slightly (true decoherence)
        wave_real[idx] *= (1.0 - intensity * 0.3);
        wave_imag[idx] *= (1.0 - intensity * 0.3);
    }
}

// =============================================================================
// COMPUTE KERNEL: Evolve Wave Function (Schrodinger equation)
// =============================================================================
// Evolves the wave function using the time-dependent Schrodinger equation.
// Uses explicit Euler integration with a 5-point Laplacian stencil.
//
// The equation (in units where hbar = m = 1):
//   d(psi_r)/dt =  D * nabla^2(psi_i) - V * psi_i
//   d(psi_i)/dt = -D * nabla^2(psi_r) + V * psi_r
// where D = 0.5 (kinetic energy coefficient)

kernel void evolve_wave(
    device const float*  wave_real_in  [[buffer(0)]],  // Read from
    device const float*  wave_imag_in  [[buffer(1)]],
    device float*        wave_real_out [[buffer(2)]],  // Write to
    device float*        wave_imag_out [[buffer(3)]],
    device const float*  potential     [[buffer(4)]],
    constant SimParams&  params        [[buffer(5)]],
    uint2 gid [[thread_position_in_grid]]
) {
    int gw = params.grid_width;
    int gh = params.grid_height;

    if (gid.x >= uint(gw) || gid.y >= uint(gh)) return;

    int idx = gid.y * gw + gid.x;
    float dt = params.wave_dt;
    float D = params.diffusion;  // hbar^2 / (2m)

    // Current values
    float psi_r = wave_real_in[idx];
    float psi_i = wave_imag_in[idx];
    float V = potential[idx] * params.potential_strength;

    // Neighbor indices with absorbing boundary conditions
    // (wave decays at edges to prevent reflection)
    int left  = (gid.x > 0) ? idx - 1 : idx;
    int right = (gid.x < uint(gw - 1)) ? idx + 1 : idx;
    int up    = (gid.y > 0) ? idx - gw : idx;
    int down  = (gid.y < uint(gh - 1)) ? idx + gw : idx;

    // 5-point Laplacian stencil: nabla^2(f) = f(x+1) + f(x-1) + f(y+1) + f(y-1) - 4*f(x,y)
    float laplacian_r = wave_real_in[left] + wave_real_in[right] +
                        wave_real_in[up] + wave_real_in[down] - 4.0 * psi_r;
    float laplacian_i = wave_imag_in[left] + wave_imag_in[right] +
                        wave_imag_in[up] + wave_imag_in[down] - 4.0 * psi_i;

    // Schrodinger evolution:
    // d(psi_r)/dt =  D * nabla^2(psi_i) - V * psi_i
    // d(psi_i)/dt = -D * nabla^2(psi_r) + V * psi_r
    float dpsi_r = D * laplacian_i - V * psi_i;
    float dpsi_i = -D * laplacian_r + V * psi_r;

    // Euler integration
    float new_r = psi_r + dpsi_r * dt;
    float new_i = psi_i + dpsi_i * dt;

    // Absorbing boundary: dampen wave near edges
    float edge_dist_x = min(float(gid.x), float(gw - 1 - gid.x));
    float edge_dist_y = min(float(gid.y), float(gh - 1 - gid.y));
    float edge_dist = min(edge_dist_x, edge_dist_y);
    float absorb_width = 20.0;
    if (edge_dist < absorb_width) {
        float absorb = edge_dist / absorb_width;
        absorb = absorb * absorb;  // Quadratic falloff
        new_r *= absorb;
        new_i *= absorb;
    }

    // Prevent numerical blowup: soft normalization
    float mag2 = new_r * new_r + new_i * new_i;
    if (mag2 > 4.0) {
        float scale = 2.0 / sqrt(mag2);
        new_r *= scale;
        new_i *= scale;
    }

    wave_real_out[idx] = new_r;
    wave_imag_out[idx] = new_i;
}

// =============================================================================
// COMPUTE KERNEL: Update Bohmian Particles
// =============================================================================
// Particles follow the guidance equation from Bohmian mechanics:
//   v = (hbar/m) * Im(grad(psi)/psi)
//
// For psi = psi_r + i*psi_i, this becomes:
//   v = (psi_r * grad(psi_i) - psi_i * grad(psi_r)) / |psi|^2
//
// This makes particles flow along the probability current, creating
// beautiful interference patterns as they trace the wave function's phase.

kernel void update_particles(
    device const float*  wave_real  [[buffer(0)]],
    device const float*  wave_imag  [[buffer(1)]],
    device float2*       positions  [[buffer(2)]],
    device float2*       velocities [[buffer(3)]],
    device float4*       colors     [[buffer(4)]],
    constant SimParams&  params     [[buffer(5)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= uint(params.num_particles)) return;

    float2 pos = positions[gid];
    float2 vel = velocities[gid];

    int gw = params.grid_width;
    int gh = params.grid_height;

    // Convert pixel position to grid coordinates
    float grid_x = pos.x / params.width * float(gw);
    float grid_y = (1.0 - pos.y / params.height) * float(gh);  // Flip Y

    // Clamp to valid range (with margin for gradient calculation)
    grid_x = clamp(grid_x, 1.0f, float(gw - 2));
    grid_y = clamp(grid_y, 1.0f, float(gh - 2));

    // Sample wave function with bilinear interpolation
    float psi_r = sample_wave(wave_real, gw, gh, grid_x, grid_y);
    float psi_i = sample_wave(wave_imag, gw, gh, grid_x, grid_y);

    // Calculate gradients using central differences
    float dx_r = sample_wave(wave_real, gw, gh, grid_x + 1, grid_y) -
                 sample_wave(wave_real, gw, gh, grid_x - 1, grid_y);
    float dx_i = sample_wave(wave_imag, gw, gh, grid_x + 1, grid_y) -
                 sample_wave(wave_imag, gw, gh, grid_x - 1, grid_y);
    float dy_r = sample_wave(wave_real, gw, gh, grid_x, grid_y + 1) -
                 sample_wave(wave_real, gw, gh, grid_x, grid_y - 1);
    float dy_i = sample_wave(wave_imag, gw, gh, grid_x, grid_y + 1) -
                 sample_wave(wave_imag, gw, gh, grid_x, grid_y - 1);

    // Scale gradients (central difference uses 2*h spacing)
    dx_r *= 0.5; dx_i *= 0.5;
    dy_r *= 0.5; dy_i *= 0.5;

    // Probability density |psi|^2
    float rho = psi_r * psi_r + psi_i * psi_i + 0.0001;  // Small epsilon to avoid division by zero

    // Bohmian velocity: v = (psi_r * grad(psi_i) - psi_i * grad(psi_r)) / |psi|^2
    float vx_guidance = (psi_r * dx_i - psi_i * dx_r) / rho;
    float vy_guidance = (psi_r * dy_i - psi_i * dy_r) / rho;

    // Convert grid velocity to screen velocity
    float scale_x = params.width / float(gw);
    float scale_y = params.height / float(gh);

    // Apply guidance with damping
    float guidance_strength = params.particle_dt;
    vel.x = vel.x * 0.95 + vx_guidance * scale_x * guidance_strength;
    vel.y = vel.y * 0.95 - vy_guidance * scale_y * guidance_strength;  // Flip Y back

    // Clamp velocity to prevent instability
    float max_speed = 8.0;
    float speed = length(vel);
    if (speed > max_speed) {
        vel *= max_speed / speed;
    }

    // Update position
    pos += vel;

    // Wrap around screen edges (toroidal topology)
    if (pos.x < 0) pos.x += params.width;
    if (pos.x >= params.width) pos.x -= params.width;
    if (pos.y < 0) pos.y += params.height;
    if (pos.y >= params.height) pos.y -= params.height;

    // Store updated state
    positions[gid] = pos;
    velocities[gid] = vel;

    // =======================================================================
    // FIRE/PLASMA COLORING WITH BLOOM
    // =======================================================================
    // Intensity based on wave amplitude and particle velocity
    // Hot particles (high amplitude, fast moving) = white/yellow core
    // Cooler particles = orange/red

    float amplitude = sqrt(rho);
    float phase = atan2(psi_i, psi_r);

    // Base intensity from amplitude (0 to 1)
    float intensity = clamp(amplitude * 1.5, 0.0f, 1.0f);

    // Velocity adds heat (faster = hotter)
    float vel_heat = clamp(speed * 0.15, 0.0f, 0.4f);
    intensity = clamp(intensity + vel_heat, 0.0f, 1.0f);

    // Get base fire color from gradient
    float3 fire_color = fire_gradient(intensity);

    // Add subtle phase-based plasma accents in cooler regions
    // This creates the "plasma tendril" effect with blue/purple hints
    float accent_strength = (1.0 - intensity) * 0.3;
    float3 accent = plasma_accent(phase, accent_strength);

    // Combine fire and plasma
    float3 rgb = fire_color + accent;

    // Boost brightness for bloom effect - hot particles glow more
    float bloom_boost = intensity * intensity * 0.5;  // Quadratic for dramatic cores
    rgb = rgb * (1.0 + bloom_boost);

    // Blend with previous color (faster transitions for fire flicker)
    float4 prev_color = colors[gid];
    float trail_factor = 0.75;  // Less persistence = more dynamic fire

    float4 new_color = float4(
        rgb.x * (1.0 - trail_factor) + prev_color.x * trail_factor,
        rgb.y * (1.0 - trail_factor) + prev_color.y * trail_factor,
        rgb.z * (1.0 - trail_factor) + prev_color.z * trail_factor,
        0.9 + intensity * 0.1  // Hotter = more opaque
    );

    colors[gid] = new_color;
}

// =============================================================================
// VERTEX SHADER: Transform particle positions to clip space
// =============================================================================

struct VertexOut {
    float4 position [[position]];
    float4 color;
    float  pointSize [[point_size]];
};

vertex VertexOut vertex_main(
    device const float2* positions [[buffer(0)]],
    device const float4* colors    [[buffer(1)]],
    constant SimParams&  params    [[buffer(2)]],
    uint vid [[vertex_id]]
) {
    VertexOut out;

    float2 pos = positions[vid];

    // Transform pixel coordinates to clip space [-1, 1]
    float2 clip;
    clip.x = (pos.x / params.width) * 2.0 - 1.0;
    clip.y = (pos.y / params.height) * 2.0 - 1.0;
    clip.y = -clip.y;  // Flip Y for Metal's coordinate system

    out.position = float4(clip, 0.0, 1.0);
    out.color = colors[vid];

    // Dynamic point size: brighter particles are larger for bloom effect
    float brightness = (colors[vid].r + colors[vid].g + colors[vid].b) / 3.0;
    float size_boost = brightness * 1.5;  // Hot particles slightly larger
    out.pointSize = params.point_size + size_boost;

    return out;
}

// =============================================================================
// FRAGMENT SHADER: Render particles with bloom/glow effect
// =============================================================================
// Creates soft, glowing particles with bright cores and diffuse halos
// Mimics real fire/plasma light emission

fragment float4 fragment_main(
    VertexOut in [[stage_in]],
    float2 pointCoord [[point_coord]]
) {
    float2 centered = pointCoord - 0.5;
    float dist = length(centered);

    // Multi-layer glow for bloom effect:
    // 1. Bright core (small, intense)
    // 2. Inner glow (medium, warm)
    // 3. Outer halo (large, diffuse)

    // Core: tight gaussian, very bright
    float core = exp(-dist * dist * 40.0);

    // Inner glow: medium spread
    float inner = exp(-dist * dist * 12.0) * 0.6;

    // Outer halo: wide spread, subtle
    float outer = exp(-dist * dist * 4.0) * 0.25;

    // Combine layers
    float glow = core + inner + outer;

    // Intensity from color brightness affects glow strength
    float brightness = (in.color.r + in.color.g + in.color.b) / 3.0;
    glow *= (0.5 + brightness * 0.5);

    // Discard nearly invisible fragments
    if (glow < 0.01) {
        discard_fragment();
    }

    // Hot particles get whiter cores (color shifts toward white at center)
    float3 core_color = mix(in.color.rgb, float3(1.0, 0.95, 0.9), core * brightness);

    return float4(core_color * glow, glow * in.color.a);
}
