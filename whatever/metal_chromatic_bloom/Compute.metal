// =============================================================================
// Metal Chromatic Bloom - Compute.metal
// =============================================================================
// Chromatic flower bloom simulation with emergent floral patterns.
// Implements chromatic dispersion where different hues experience
// different forces, creating rainbow-spectrum petal patterns.
//
// Kernels:
//   - update_particles: Chromatic bloom physics with petal formation
//   - vertex_main: Vertex shader with HSV to RGB conversion
//   - fragment_main: Fragment shader with glow and age-based fading
// =============================================================================

#include <metal_stdlib>
using namespace metal;

// =============================================================================
// DATA STRUCTURES
// =============================================================================

struct SimParams {
    float width;
    float height;
    float time;
    int num_particles;
    int num_blooms;
    int num_petals;
    float chromatic_strength;
    float radial_force;
    float petal_attraction;
    float friction;
    float particle_lifetime;
    float point_size;
};

struct VertexOut {
    float4 position [[position]];
    float4 color;
    float age;
    float pointSize [[point_size]];
};

// =============================================================================
// HELPER FUNCTIONS
// =============================================================================

// HSV to RGB color conversion
// h: hue (0.0-1.0), s: saturation (0.0-1.0), v: value/brightness (0.0-1.0)
float3 hsv_to_rgb(float h, float s, float v) {
    float c = v * s;
    float x = c * (1.0 - abs(fmod(h * 6.0, 2.0) - 1.0));
    float m = v - c;

    float3 rgb;
    if (h < 1.0/6.0) {
        rgb = float3(c, x, 0.0);
    } else if (h < 2.0/6.0) {
        rgb = float3(x, c, 0.0);
    } else if (h < 3.0/6.0) {
        rgb = float3(0.0, c, x);
    } else if (h < 4.0/6.0) {
        rgb = float3(0.0, x, c);
    } else if (h < 5.0/6.0) {
        rgb = float3(x, 0.0, c);
    } else {
        rgb = float3(c, 0.0, x);
    }

    return rgb + m;
}

// Simple pseudo-random number generator using thread ID and time
float random(uint seed) {
    seed = (seed * 747796405u + 2891336453u);
    seed = ((seed >> ((seed >> 28u) + 4u)) ^ seed) * 277803737u;
    return float(seed) / 4294967295.0;
}

// =============================================================================
// COMPUTE KERNEL: Chromatic Bloom Physics
// =============================================================================

kernel void update_particles(
    device float2* positions        [[buffer(0)]],
    device float2* velocities       [[buffer(1)]],
    device float* hues              [[buffer(2)]],
    device float* ages              [[buffer(3)]],
    device const float2* bloom_centers [[buffer(4)]],
    device const int* bloom_active  [[buffer(5)]],
    constant SimParams& params      [[buffer(6)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= uint(params.num_particles)) return;

    float2 pos = positions[gid];
    float2 vel = velocities[gid];
    float hue = hues[gid];
    float age = ages[gid];

    // =================================================================
    // STEP 1: Find nearest active bloom center
    // =================================================================
    float2 nearest_center = bloom_centers[0];
    float min_dist = 999999.0;

    for (int i = 0; i < params.num_blooms; i++) {
        if (i >= 8 || !bloom_active[i]) continue;

        float2 to_bloom = pos - bloom_centers[i];
        float d = length(to_bloom);
        if (d < min_dist) {
            min_dist = d;
            nearest_center = bloom_centers[i];
        }
    }

    // =================================================================
    // STEP 2: Calculate radial geometry from bloom center
    // =================================================================
    float2 to_particle = pos - nearest_center;
    float dist = length(to_particle);
    float angle = atan2(to_particle.y, to_particle.x);

    // =================================================================
    // STEP 3: Petal quantization (key to floral shape)
    // =================================================================
    // Snap angle to nearest petal direction
    float petal_angle = 2.0 * M_PI_F / float(params.num_petals);
    float petal_index = round(angle / petal_angle);
    float target_angle = petal_index * petal_angle;
    float angle_diff = target_angle - angle;

    // Normalize angle difference to [-π, π]
    while (angle_diff > M_PI_F) angle_diff -= 2.0 * M_PI_F;
    while (angle_diff < -M_PI_F) angle_diff += 2.0 * M_PI_F;

    // =================================================================
    // STEP 4: Chromatic dispersion (key visual effect)
    // =================================================================
    // Different hues experience different radial forces
    // Red (hue=0) vs Blue (hue=0.67) move at different speeds
    float chromatic_factor = 1.0 + params.chromatic_strength * (hue - 0.5);

    // =================================================================
    // STEP 5: Force accumulation
    // =================================================================

    // Radial outward force (bloom grows)
    float2 radial_force = float2(0.0);
    if (dist > 0.1) {
        radial_force = normalize(to_particle) * params.radial_force * chromatic_factor;
    }

    // Petal alignment force (angular correction toward petal direction)
    float2 tangent = float2(-sin(angle), cos(angle));
    float2 petal_force = tangent * angle_diff * params.petal_attraction;

    // CURVATURE: Add spiral/curl to create organic petal curves
    // Particles farther from center spiral more (like unfurling petals)
    float curl_strength = smoothstep(10.0, 150.0, dist) * 0.03;
    float2 curl_force = tangent * curl_strength * chromatic_factor;

    // WAVE: Add perpendicular oscillation for organic petal shape
    // Creates gentle undulation along the petal
    float wave_freq = 0.02; // Frequency of undulation
    float wave_phase = dist * wave_freq + target_angle * 2.0 + params.time * 0.5;
    float wave_amplitude = 0.015 * (1.0 - dist / 300.0); // Stronger near center
    float2 wave_dir = float2(-sin(target_angle), cos(target_angle)); // Perpendicular to petal
    float2 wave_force = wave_dir * sin(wave_phase) * wave_amplitude;

    // Gentle damping based on distance (prevents excessive spreading)
    float damping = 1.0 - smoothstep(100.0, 300.0, dist);

    // =================================================================
    // STEP 6: Physics update
    // =================================================================
    vel *= params.friction;
    vel += (radial_force + petal_force + curl_force + wave_force) * damping;

    // Clamp velocity to reasonable range
    float speed = length(vel);
    if (speed > 3.0) {
        vel = normalize(vel) * 3.0;
    }

    pos += vel;
    age += 1.0;

    // =================================================================
    // STEP 7: Lifetime management (respawn cycle)
    // =================================================================
    if (age > params.particle_lifetime) {
        // Respawn at nearest bloom center with random offset
        uint seed = gid * 747796405u + uint(params.time * 1000.0);
        float spawn_angle = random(seed) * 2.0 * M_PI_F;
        float spawn_dist = random(seed + 1u) * 5.0;

        pos = nearest_center + float2(cos(spawn_angle), sin(spawn_angle)) * spawn_dist;
        vel = float2(0.0);
        age = 0.0;
        hue = random(seed + 2u);  // New random hue for variety
    }

    // Boundary wrap (keep particles on screen)
    if (pos.x < 0.0) pos.x = params.width;
    if (pos.x > params.width) pos.x = 0.0;
    if (pos.y < 0.0) pos.y = params.height;
    if (pos.y > params.height) pos.y = 0.0;

    // Write back
    positions[gid] = pos;
    velocities[gid] = vel;
    hues[gid] = hue;
    ages[gid] = age;
}

// =============================================================================
// VERTEX SHADER
// =============================================================================

vertex VertexOut vertex_main(
    device const float2* positions [[buffer(0)]],
    device const float* hues [[buffer(1)]],
    device const float* ages [[buffer(2)]],
    constant SimParams& params [[buffer(3)]],
    uint vid [[vertex_id]]
) {
    VertexOut out;

    // Transform position to clip space
    float2 pos = positions[vid];
    float x = (pos.x / params.width) * 2.0 - 1.0;
    float y = 1.0 - (pos.y / params.height) * 2.0;
    out.position = float4(x, y, 0.0, 1.0);

    // HDR-enhanced colors: boost saturation and brightness for young particles
    float age_factor = smoothstep(0.0, 30.0, ages[vid]); // Young particles are brighter
    float saturation = mix(1.0, 0.85, age_factor); // Full saturation when young
    float brightness = mix(1.5, 1.0, age_factor);  // HDR overbright when young

    float3 rgb = hsv_to_rgb(hues[vid], saturation, brightness);
    out.color = float4(rgb, 1.0);

    out.age = ages[vid];
    out.pointSize = params.point_size;

    return out;
}

// =============================================================================
// FRAGMENT SHADER: Glow Effect
// =============================================================================

fragment float4 fragment_main(
    VertexOut in [[stage_in]],
    float2 pointCoord [[point_coord]]
) {
    // Make circular particles
    float2 centered = pointCoord - 0.5;
    float dist = length(centered);
    if (dist > 0.5) {
        discard_fragment();
    }

    // HDR BLOOM: Much more aggressive glow with bright core
    float core = 1.0 - smoothstep(0.0, 0.15, dist);  // Intense hot core
    float glow = 1.0 - smoothstep(0.1, 0.5, dist);   // Extended glow

    // Combine core and glow for HDR effect
    float brightness = core * 2.0 + glow;  // Core is 2x bright (HDR)

    // Age-based fade (fade in first 10 frames, fade out last 30 frames)
    float fade_in = smoothstep(0.0, 10.0, in.age);
    float fade_out = 1.0 - smoothstep(150.0, 180.0, in.age);
    float age_fade = fade_in * fade_out;

    // Young particles glow more intensely
    float youth_boost = 1.0 + (1.0 - smoothstep(0.0, 50.0, in.age)) * 0.5;

    float alpha = age_fade * glow;

    // HDR color: boost intensity for core, allow overbright
    float3 hdr_color = in.color.rgb * brightness * youth_boost;

    return float4(hdr_color, alpha);
}
