#include <metal_stdlib>
using namespace metal;

struct Particle {
    float2 position;
    float2 velocity;
    float3 color;
    float age;
    float hue_phase;
    float speed_amplitude;
};

struct Params {
    uint width;
    uint height;
    float time;
    float time_scale;
    float attraction_strength;
    float color_rotation;
    float bloom_intensity;
    float chaos;
    uint num_particles;
    int padding[3];
};

// Hash-based random
float hash(float2 p) {
    float h = dot(p, float2(127.1, 311.7));
    return fract(sin(h) * 43758.5453);
}

// Perlin-like 2D noise
float noise(float2 p) {
    float2 i = floor(p);
    float2 f = fract(p);
    f = f * f * (3.0 - 2.0 * f);

    float a = hash(i);
    float b = hash(i + float2(1.0, 0.0));
    float c = hash(i + float2(0.0, 1.0));
    float d = hash(i + float2(1.0, 1.0));

    float ab = mix(a, b, f.x);
    float cd = mix(c, d, f.x);
    return mix(ab, cd, f.y);
}

// HSV to RGB
float3 hsv_to_rgb(float h, float s, float v) {
    float c = v * s;
    float x = c * (1.0 - abs(fmod(h * 6.0, 2.0) - 1.0));
    float m = v - c;

    float3 rgb;
    if (h < 1.0/6.0) rgb = float3(c, x, 0.0);
    else if (h < 2.0/6.0) rgb = float3(x, c, 0.0);
    else if (h < 3.0/6.0) rgb = float3(0.0, c, x);
    else if (h < 4.0/6.0) rgb = float3(0.0, x, c);
    else if (h < 5.0/6.0) rgb = float3(x, 0.0, c);
    else rgb = float3(c, 0.0, x);

    return rgb + m;
}

// Rotating attraction wells
float2 get_force(float2 pos, float time) {
    float2 force = float2(0.0);

    for (int i = 0; i < 3; i++) {
        float angle = time * (0.3 + i * 0.1) + i;
        float radius = 150.0 + 80.0 * sin(time * 0.2 + i);
        float2 center = float2(cos(angle) * radius, sin(angle) * radius);

        float2 to_center = center - pos;
        float dist = length(to_center);
        float strength = exp(-dist * dist / 30000.0);

        if (dist > 0.1) {
            force += normalize(to_center) * strength * 0.5;
        }
    }

    float2 noise_force = float2(
        noise(pos * 0.005 + float2(time * 0.1, 0.0)) - 0.5,
        noise(pos * 0.005 + float2(0.0, time * 0.1)) - 0.5
    ) * 0.3;

    return force + noise_force;
}

/**
 * Update particle physics
 */
kernel void update_particles(device Particle* particles [[buffer(0)]],
                            constant Params& params [[buffer(1)]],
                            uint id [[thread_position_in_grid]]) {
    if (id >= params.num_particles) return;

    Particle p = particles[id];
    float t = params.time * params.time_scale;

    // Get forces and smooth-damp velocity
    float2 force = get_force(p.position, t);
    p.velocity = mix(p.velocity * 0.98, force, 0.15);

    // Add chaotic perturbation
    float chaos = noise(p.position * 0.001 + t) - 0.5;
    p.velocity += float2(cos(t + chaos) * 0.1, sin(t + chaos * 2.0) * 0.1) * params.chaos;

    // Limit speed with oscillation
    float speed = length(p.velocity);
    float max_speed = 2.0 + sin(t + p.hue_phase) * 0.5;
    if (speed > max_speed) {
        p.velocity = normalize(p.velocity) * max_speed;
    }

    // Update position
    p.position += p.velocity;
    p.age += 0.01;

    // Wrap at boundaries
    if (p.position.x < 0.0) p.position.x += params.width;
    if (p.position.x > params.width) p.position.x -= params.width;
    if (p.position.y < 0.0) p.position.y += params.height;
    if (p.position.y > params.height) p.position.y -= params.height;

    // Update hue phase based on motion and forces
    p.hue_phase += speed * 0.01 + length(force) * 0.05;

    // Dynamic color: HSV with rotating hue
    float hue = fmod(p.hue_phase * 0.5 + t * 0.3 + params.color_rotation, 1.0);
    float saturation = 0.7 + 0.3 * sin(t + length(p.position) * 0.001);
    float brightness = 0.8 + 0.2 * sin(t * 2.0 + p.hue_phase);

    p.color = hsv_to_rgb(hue, saturation, brightness);

    particles[id] = p;
}

/**
 * Render as textured quads
 */
struct VertexOut {
    float4 position [[position]];
    float4 color;
};

vertex VertexOut vertex_bloom(uint vid [[vertex_id]],
                             constant Particle* particles [[buffer(0)]],
                             constant Params& params [[buffer(1)]]) {
    uint particle_id = vid / 6;  // 6 vertices per particle (2 triangles)
    uint vertex_id = vid % 6;

    if (particle_id >= params.num_particles) {
        return {float4(0), float4(0)};
    }

    Particle p = particles[particle_id];

    // Normalize position to NDC (-1 to 1)
    float x = (p.position.x / float(params.width)) * 2.0 - 1.0;
    float y = 1.0 - (p.position.y / float(params.height)) * 2.0;

    // Size in NDC
    float size = (params.bloom_intensity + 0.5) * 0.02;

    // Create a small quad around the particle
    float2 offsets[6] = {
        {-size, -size},
        {size, -size},
        {size, size},
        {-size, -size},
        {size, size},
        {-size, size}
    };

    VertexOut out;
    out.position = float4(x + offsets[vertex_id].x, y + offsets[vertex_id].y, 0.0, 1.0);
    out.color = float4(p.color, 1.0);

    return out;
}

fragment float4 fragment_bloom(VertexOut in [[stage_in]]) {
    return in.color;
}
