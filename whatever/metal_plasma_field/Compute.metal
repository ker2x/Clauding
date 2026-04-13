// =============================================================================
// Metal Plasma Field Simulation - Compute.metal
// =============================================================================
// GPU shaders for charged particle simulation with electromagnetic fields.
//
// Physics implemented:
// 1. Coulomb force: F = k * q1 * q2 / r^2 (charge interactions)
// 2. Lorentz force: F = q(E + v × B) (EM field effects)
// 3. Thermal noise: Random Brownian motion
// 4. Multiple field modes for visual variety
// =============================================================================

#include <metal_stdlib>
using namespace metal;

// =============================================================================
// DATA STRUCTURES

struct SimParams {
    float width;
    float height;
    float interaction_radius;
    float coulomb_strength;
    float magnetic_strength;
    float friction;
    float thermal_noise;
    float electric_field_x;
    float electric_field_y;
    float magnetic_field_z;  // Reused as field mode selector (0, 1, or 2)
    int num_particles;
    int num_types;
    float point_size;
    float time;
};

// =============================================================================
// RANDOM NUMBER GENERATION
// =============================================================================
// Using a simple hash-based random for GPU compatibility

float hash(uint seed) {
    // Wang hash - good distribution for simple PRNG
    seed = (seed ^ 61) ^ (seed >> 16);
    seed *= 9;
    seed = seed ^ (seed >> 4);
    seed *= 0x27d4eb2d;
    seed = seed ^ (seed >> 15);
    
    // Convert to float [0, 1)
    return float(seed) / float(0xFFFFFFFF);
}

// =============================================================================
// COMPUTE KERNEL: Plasma Physics Simulation
// =============================================================================
// Each thread handles one particle.
// 
// Forces computed:
// 1. Coulomb repulsion between same-sign charges
// 2. Attraction between opposite charges
// 3. Lorentz force from external E and B fields
// 4. Thermal noise for realistic plasma behavior
//
// Field Modes (magnetic_field_z parameter):
// 0: Uniform E and B fields (constant throughout space)
// 1: Central pole field (1/r^2 falloff from center)
// 2: Rotating vortex field (spiral pattern)

kernel void update_plasma(
    device float2* positions [[buffer(0)]],
    device float2* velocities [[buffer(1)]],
    device const int* types [[buffer(2)]],
    device const float* charges [[buffer(3)]],
    constant SimParams& params [[buffer(4)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= uint(params.num_particles)) return;
    
    float2 pos = positions[gid];
    float2 vel = velocities[gid];
    float charge = charges[gid];
    int type = types[gid];
    
    // Accumulate electromagnetic forces
    float2 emForce = float2(0.0f, 0.0f);
    
    // -----------------------------------------------------
    // 1. External Electromagnetic Field
    // -----------------------------------------------------
    int fieldMode = int(params.magnetic_field_z);
    
    if (fieldMode == 0) {
        // Uniform E and B fields (constant)
        // Electric force: F = qE
        emForce += charge * float2(params.electric_field_x, params.electric_field_y);
        
        // Magnetic force (Lorentz): F = q(v × B)
        // B field is perpendicular to screen (Z direction)
        // v × B = (vx, vy, 0) × (0, 0, Bz) = (vy*Bz, -vx*Bz, 0)
        emForce.x += vel.y * params.magnetic_strength;
        emForce.y -= vel.x * params.magnetic_strength;
    }
    else if (fieldMode == 1) {
        // Central pole field (1/r^2 falloff)
        float2 toCenter = float2(params.width * 0.5f, params.height * 0.5f) - pos;
        float dist = length(toCenter);
        float dist2 = dist * dist + 1.0f;  // Avoid division by zero
        
        // Radial E field from center
        float2 radialDir = toCenter / (dist + 0.001f);
        float fieldStrength = 2.0f / (dist2 * 0.001f + 1.0f);
        emForce += charge * radialDir * fieldStrength * params.coulomb_strength * 20.0f;
        
        // Azimuthal B field (creates spiral motion)
        float2 tangent = float2(-radialDir.y, radialDir.x);
        emForce += charge * tangent * fieldStrength * params.magnetic_strength * 10.0f;
    }
    else if (fieldMode == 2) {
        // Rotating vortex field
        float2 center = float2(params.width * 0.5f, params.height * 0.5f);
        float2 offset = pos - center;
        float dist = length(offset);
        float angle = atan2(offset.y, offset.x);
        
        // Rotating field vector
        float rotAngle = angle + params.time * 0.5f;
        float2 fieldDir = float2(cos(rotAngle), sin(rotAngle));
        
        // Field strength decreases with distance
        float fieldStrength = 1.0f / (dist * 0.01f + 1.0f);
        
        emForce += charge * fieldDir * fieldStrength * params.coulomb_strength * 15.0f;
    }
    
    // -----------------------------------------------------
    // 2. Particle-Particle Coulomb Interactions
    // -----------------------------------------------------
    // Only check nearby particles for performance
    // Using simple O(n) loop (could be optimized with spatial hashing)
    
    const float rMax = params.interaction_radius;
    const float rMax2 = rMax * rMax;
    
    for (int j = 0; j < params.num_particles; j++) {
        if (uint(j) == gid) continue;
        
        float2 other_pos = positions[j];
        float other_charge = charges[j];
        
        // Distance calculation with wrapping
        float2 delta = other_pos - pos;
        
        // Handle wrapping (toroidal)
        if (delta.x > params.width * 0.5f)  delta.x -= params.width;
        if (delta.x < -params.width * 0.5f) delta.x += params.width;
        if (delta.y > params.height * 0.5f)  delta.y -= params.height;
        if (delta.y < -params.height * 0.5f) delta.y += params.height;
        
        float dist2 = dot(delta, delta);
        
        if (dist2 > rMax2 || dist2 < 0.0001f) continue;
        
        float dist = sqrt(dist2);
        float2 direction = delta / dist;
        
        // Coulomb force: F = k * q1 * q2 / r^2
        // Like charges repel, opposite charges attract
        float coulombForce = params.coulomb_strength * charge * other_charge / (dist2 + 1.0f);
        
        emForce += direction * coulombForce * 100.0f;
    }
    
    // -----------------------------------------------------
    // 3. Thermal Noise (Brownian Motion)
    // -----------------------------------------------------
    // Adds realistic plasma temperature effects
    uint seed = gid * uint(params.time * 100.0f) + 12345;
    float noiseX = (hash(seed) - 0.5f) * 2.0f;
    float noiseY = (hash(seed + 1) - 0.5f) * 2.0f;
    float2 thermalForce = float2(noiseX, noiseY) * params.thermal_noise;
    
    // -----------------------------------------------------
    // Update Velocity and Position
    // -----------------------------------------------------
    vel += emForce + thermalForce;
    vel *= params.friction;
    
    // Clamp velocity to prevent instability
    float maxSpeed = 10.0f;
    float speed = length(vel);
    if (speed > maxSpeed) {
        vel = vel * (maxSpeed / speed);
    }
    
    pos += vel;
    
    // Wrap position
    if (pos.x < 0.0f) pos.x += params.width;
    if (pos.x >= params.width) pos.x -= params.width;
    if (pos.y < 0.0f) pos.y += params.height;
    if (pos.y >= params.height) pos.y -= params.height;
    
    // Write back
    positions[gid] = pos;
    velocities[gid] = vel;
}

// =============================================================================
// VERTEX SHADER

struct VertexOut {
    float4 position [[position]];
    float4 color [[flat]];
    float pointSize [[point_size]];
};

vertex VertexOut vertex_main(
    device const float2* positions [[buffer(0)]],
    device const int* types [[buffer(1)]],
    device const float4* colors [[buffer(2)]],
    device const float* charges [[buffer(3)]],
    constant SimParams& params [[buffer(4)]],
    uint vid [[vertex_id]]
) {
    VertexOut out;
    
    float2 pos = positions[vid];
    
    // Transform to clip coordinates
    float2 clipPos;
    clipPos.x = (pos.x / params.width) * 2.0f - 1.0f;
    clipPos.y = (pos.y / params.height) * 2.0f - 1.0f;
    clipPos.y = -clipPos.y;
    
    out.position = float4(clipPos, 0.0f, 1.0f);
    
    int type = types[vid];
    out.color = colors[type];
    
    // Size particles based on charge magnitude
    float charge = charges[vid];
    out.pointSize = params.point_size * (0.8f + abs(charge) * 0.4f);
    
    return out;
}

// =============================================================================
// FRAGMENT SHADER

fragment float4 fragment_main(
    VertexOut in [[stage_in]],
    float2 pointCoord [[point_coord]]
) {
    // Circular particles with soft edges
    float2 centered = pointCoord - 0.5f;
    float dist = length(centered);
    
    if (dist > 0.5f) {
        discard_fragment();
    }
    
    // Soft edge glow
    float alpha = 1.0f - smoothstep(0.2f, 0.5f, dist);
    
    // Add slight brightness at center
    float glow = 1.0f - smoothstep(0.0f, 0.3f, dist);
    float3 color = in.color.rgb * (1.0f + glow * 0.5f);
    
    return float4(color, in.color.a * alpha);
}
