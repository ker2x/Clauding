#include <metal_stdlib>
using namespace metal;

// Simulation parameters passed from the host
struct Params {
    float width;
    float height;
    float r_max;
    float force_strength;
    float friction;
    int num_particles;
    int num_types;
};

// Tile size for shared memory. 512 is sweet spot for Apple Silicon occupancy.
#define TILE_SIZE 512

kernel void update_particles(device float2* pos_buffer [[buffer(0)]],      
                             device float2* vel_buffer [[buffer(1)]],      
                             device int* type_buffer [[buffer(2)]],        
                             constant float* matrix [[buffer(3)]],         
                             constant Params& params [[buffer(4)]],        
                             uint id [[thread_position_in_grid]],
                             uint ti [[thread_index_in_threadgroup]],
                             uint lane_id [[thread_index_in_simdgroup]]) {    
    
    // Safety check
    if (id >= (uint)params.num_particles) return;

    // 1. CACHE INTERACTION MATRIX (Local Registers - Half)
    half local_matrix[36];
    for (int m = 0; m < 36; m++) {
        local_matrix[m] = (half)matrix[m];
    }

    // Load current particle data (Float Precision for integration stability)
    float2 p_pos_f = pos_buffer[id];
    float2 p_vel_f = vel_buffer[id];
    int p_type = type_buffer[id];
    
    // Cast constants to half for the interaction loop
    half2 p_pos = (half2)p_pos_f;
    half h_width = (half)params.width;
    half h_height = (half)params.height;
    half h_rmax = (half)params.r_max;
    half h_rmax_sq = h_rmax * h_rmax;
    half h_force_strength = (half)params.force_strength;
    half h_inv_rmax = 1.0h / h_rmax;
    
    half2 total_force_h = half2(0.0h, 0.0h);

    // 2. SHARED MEMORY FOR TILES
    threadgroup half2 shared_pos[TILE_SIZE];
    threadgroup int shared_types[TILE_SIZE];

    for (int step = 0; step < params.num_particles; step += TILE_SIZE) {
        
        // COLLABORATIVE LOAD into Shared Memory
        uint load_idx = step + ti;
        if (load_idx < (uint)params.num_particles) {
            shared_pos[ti] = (half2)pos_buffer[load_idx];
            shared_types[ti] = type_buffer[load_idx];
        } else {
            shared_pos[ti] = half2(-10000.0h); // Far away
            shared_types[ti] = 0;
        }
        
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // 3. SIMD-GROUP SHUFFLE (INNER LOOP)
        for (uint i = 0; i < TILE_SIZE; i += 32) {
            
            half2 other_p = shared_pos[i + lane_id];
            int other_t = shared_types[i + lane_id];

            for (uint j = 0; j < 32; j++) {
                uint other_idx = step + i + j;
                if (other_idx >= (uint)params.num_particles || other_idx == id) continue;

                half2 s_pos = simd_broadcast(other_p, j);
                int s_type = simd_broadcast(other_t, j);
                
                half2 diff = s_pos - p_pos;

                // REVERT TO BRANCHY WRAP (Faster on this hardware/precision)
                if (diff.x > h_width * 0.5h)  diff.x -= h_width;
                if (diff.x < -h_width * 0.5h) diff.x += h_width;
                if (diff.y > h_height * 0.5h)  diff.y -= h_height;
                if (diff.y < -h_height * 0.5h) diff.y += h_height;

                half dist_sq = dot(diff, diff);
                
                if (dist_sq > 0.0001h && dist_sq < h_rmax_sq) {
                    half inv_r = rsqrt(dist_sq);
                    half r = dist_sq * inv_r;
                    half normalized_r = r * h_inv_rmax;
                    
                    half g = local_matrix[p_type * params.num_types + s_type];

                    half force = (normalized_r < 0.3h) 
                        ? (normalized_r / 0.3h) - 1.0h
                        : g * (1.0h - abs(2.0h * normalized_r - 1.0h - 0.3h) / (1.0h - 0.3h));

                    total_force_h += (diff * inv_r) * force * h_force_strength;
                }
            }
        }
        
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // 4. INTEGRATION (Float Precision for stability)
    float2 total_force = (float2)total_force_h;
    float2 mid_velocity = p_vel_f + total_force;
    p_pos_f += mid_velocity;
    p_vel_f = mid_velocity * params.friction;

    // Toroidal wrap position
    p_pos_f.x = fmod(p_pos_f.x, params.width);
    if (p_pos_f.x < 0) p_pos_f.x += params.width;
    p_pos_f.y = fmod(p_pos_f.y, params.height);
    if (p_pos_f.y < 0) p_pos_f.y += params.height;

    pos_buffer[id] = p_pos_f;
    vel_buffer[id] = p_vel_f;
}

// Rendering Shaders
struct VertexOut {
    float4 position [[position]];
    float4 color [[flat]];
    float point_size [[point_size]];
};

vertex VertexOut vertex_main(device float2* pos_buffer [[buffer(0)]],
                             device int* type_buffer [[buffer(1)]],
                             constant float4* colors [[buffer(2)]],
                             constant Params& params [[buffer(3)]],
                             uint id [[vertex_id]]) {
    VertexOut out;
    float2 pos = pos_buffer[id];
    int type = type_buffer[id];
    float x = (pos.x / params.width) * 2.0 - 1.0;
    float y = (pos.y / params.height) * -2.0 + 1.0;
    out.position = float4(x, y, 0, 1);
    out.color = colors[type];
    out.point_size = 4.0;
    return out;
}

fragment float4 fragment_main(VertexOut in [[stage_in]]) {
    return float4(in.color.rgb, 1.0);
}
