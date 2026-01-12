#include <metal_stdlib>
using namespace metal;

/**
 * METAL CONCEPTS FOR BEGINNERS:
 * 
 * 1. SIMD Types: Metal uses types like float2 (x,y), float4 (r,g,b,a) for vector math.
 * 2. Kernels: Functions marked 'kernel' run in parallel on thousands of GPU threads.
 * 3. Buffer indices: [[buffer(0)]] maps to the buffer index set in objective-C/Swift.
 * 4. Grid Positions: [[thread_position_in_grid]] gives each thread its unique ID/Coordinate.
 */

// Represents a single "Slime Mold" inhabitant
struct Agent {
    float2 position; // 8 bytes
    float heading;  // 4 bytes
    int species;     // 4 bytes
    // Total: 16 bytes. Matches 8-byte alignment of float2.
};

// Global simulation settings passed from the CPU every frame
struct Params {
    uint width;
    uint height;
    float move_speed;
    float sensor_angle;
    float sensor_dist;
    float turn_speed;
    float sensor_size;
    float decay_rate;   // How fast the trail disappears (evaporation)
    float diffuse_rate; // How much the trail spreads (blur)
    float dt;           // Delta time (smoothness)
    uint num_agents;
    float combat_strength; // How much they are repelled by the enemy
    float2 mouse_pos;
    int mouse_down;
    int padding2[3];    // Padding to ensure the struct size is consistent across GPU/CPU
};

// A simple random number generator (Hash function) since GPUs don't have rand()
float hash(uint x) {
    x = ((x >> 16) ^ x) * 0x45d9f3b;
    x = ((x >> 16) ^ x) * 0x45d9f3b;
    x = (x >> 16) ^ x;
    return (float)x / 4294967295.0f;
}

/**
 * SENSE: Returns a float2 where:
 * x = density of OWN species pheromones
 * y = density of ENEMY species pheromones
 */
static float2 sense(Agent agent, float angle_offset, texture2d<float, access::sample> trail_map, constant Params& params) {
    float sensor_angle = agent.heading + angle_offset;
    float2 direction = float2(cos(sensor_angle), sin(sensor_angle));
    float2 sensor_pos = agent.position + direction * params.sensor_dist;
    
    // Samplers tell the GPU how to read pixels (e.g., linear interpolation between pixels)
    constexpr sampler s(address::clamp_to_edge, filter::linear);
    float2 uv = sensor_pos / float2(params.width, params.height);
    
    float4 sampled = trail_map.sample(s, uv);
    // Species 0 uses Red (r), Species 1 uses Green (g)
    if (agent.species == 0) {
        return float2(sampled.r, sampled.g);
    } else {
        return float2(sampled.g, sampled.r);
    }
}

/**
 * KERNEL 1: update_agents
 * Runs for every single agent simultaneously.
 * Decide where to turn based on pheromones, and move forward.
 */
kernel void update_agents(device Agent* agents [[buffer(0)]],
                          texture2d<float, access::sample> trail_map [[texture(0)]],
                          constant Params& params [[buffer(1)]],
                          uint id [[thread_position_in_grid]]) {
    if (id >= params.num_agents) return;
    
    Agent agent = agents[id];
    
    // 1. SENSE: Look Forward, Left, and Right
    float2 weight_forward_pair = sense(agent, 0, trail_map, params);
    float2 weight_left_pair = sense(agent, params.sensor_angle, trail_map, params);
    float2 weight_right_pair = sense(agent, -params.sensor_angle, trail_map, params);
    
    // Combine own attraction (+) and enemy repulsion (-)
    float weight_forward = weight_forward_pair.x - weight_forward_pair.y * params.combat_strength;
    float weight_left = weight_left_pair.x - weight_left_pair.y * params.combat_strength;
    float weight_right = weight_right_pair.x - weight_right_pair.y * params.combat_strength;
    
    float random_val = hash(id + (uint)(agent.position.x * 10) + (uint)(agent.position.y * 10));
    
    // 2. STEER: Biological logic - go where it's densest
    if (weight_forward > weight_left && weight_forward > weight_right) {
        // Continue straight
    } else if (weight_forward < weight_left && weight_forward < weight_right) {
        // Random turn if both sides are better than the front
        agent.heading += (random_val - 0.5f) * 2.0f * params.turn_speed * params.dt;
    } else if (weight_right > weight_left) {
        agent.heading -= params.turn_speed * params.dt;
    } else if (weight_left > weight_right) {
        agent.heading += params.turn_speed * params.dt;
    }
    
    // 3. INTERACT: Push agents away from mouse if clicked
    if (params.mouse_down) {
        float2 diff = agent.position - params.mouse_pos;
        float dist_sq = dot(diff, diff);
        if (dist_sq < 2500.0f) { // 50px radius
            float dist = sqrt(dist_sq);
            float2 push = diff / (dist + 0.1f);
            agent.position += push * 1.5f;
            agent.heading += (random_val - 0.5f) * 0.5f;
        }
    }

    // 4. MOVE: Step forward based on heading
    float2 direction = float2(cos(agent.heading), sin(agent.heading));
    agent.position += direction * params.move_speed * params.dt;
    
    // 5. WRAP: Go through walls to the other side (Toroidal field)
    if (agent.position.x < 0) agent.position.x += params.width;
    if (agent.position.x >= params.width) agent.position.x -= params.width;
    if (agent.position.y < 0) agent.position.y += params.height;
    if (agent.position.y >= params.height) agent.position.y -= params.height;
    
    // Save back to GPU memory
    agents[id] = agent;
}

/**
 * KERNEL 2: deposit_pheromones
 * Runs after all agents have moved. 
 * Each agent leaves a "trail" at its current position.
 */
kernel void deposit_pheromones(device Agent* agents [[buffer(0)]],
                               texture2d<float, access::read_write> trail_map [[texture(0)]],
                               constant Params& params [[buffer(1)]],
                               uint id [[thread_position_in_grid]]) {
    if (id >= params.num_agents) return;
    
    uint2 pixel = (uint2)agents[id].position;
    if (pixel.x < params.width && pixel.y < params.height) {
        float4 current = trail_map.read(pixel);
        if (agents[id].species == 0) {
            trail_map.write(float4(min(1.0f, current.r + 0.5f), current.g, 0, 1.0), pixel);
        } else {
            trail_map.write(float4(current.r, min(1.0f, current.g + 0.5f), 0, 1.0), pixel);
        }
    }
}

/**
 * KERNEL 3: process_trail
 * Runs for every pixel on the screen.
 * Diffuse: Blur the map so pheromones spread.
 * Decay: Multiply by a value < 1.0 so they evaporate over time.
 */
kernel void process_trail(texture2d<float, access::read> in_trail [[texture(0)]],
                          texture2d<float, access::write> out_trail [[texture(1)]],
                          constant Params& params [[buffer(0)]],
                          uint2 gid [[thread_position_in_grid]]) {
    if (gid.x >= params.width || gid.y >= params.height) return;
    
    // 3x3 Blur Kernel for both species
    float2 sum = float2(0.0);
    for (int ox = -1; ox <= 1; ox++) {
        for (int oy = -1; oy <= 1; oy++) {
            int sx = (int)gid.x + ox;
            int sy = (int)gid.y + oy;
            // Wrap coordinates manually for the blur
            if (sx < 0) sx += params.width;
            if (sx >= (int)params.width) sx -= params.width;
            if (sy < 0) sy += params.height;
            if (sy >= (int)params.height) sy -= params.height;
            
            float4 val = in_trail.read(uint2(sx, sy));
            sum += val.rg;
        }
    }
    
    float2 blurred = sum / 9.0f;
    float2 decayed = blurred * params.decay_rate;
    
    // COMPETITIVE INHIBITION: 
    // If both species pheromones are in the same spot, they "poison" each other.
    // We store the "poison/clash" intensity in the Blue channel for visualization.
    float interference = blurred.r * blurred.g * 10.0f; 
    decayed = max(0.0f, decayed - interference);
    
    // Write out: R=SpeciesA, G=SpeciesB, B=Clash Highlight
    out_trail.write(float4(decayed.r, decayed.g, min(1.0f, interference * 2.0f), 1.0), gid);
}

/**
 * RENDERING: Vertex & Fragment Shaders
 * Vertex shader: Sets up where on the screen everything is drawn.
 * Fragment shader: Decides what COLOR a pixel should be.
 */
struct VertexOut {
    float4 position [[position]];
    float2 uv;
};

vertex VertexOut vertex_main(uint id [[vertex_id]]) {
    // Generate a full-screen triangle strip (quad)
    float2 positions[4] = { float2(-1, -1), float2(1, -1), float2(-1, 1), float2(1, 1) };
    float2 uvs[4] = { float2(0, 1), float2(1, 1), float2(0, 0), float2(1, 0) };
    
    VertexOut out;
    out.position = float4(positions[id], 0, 1);
    out.uv = uvs[id];
    return out;
}

// Map the "Red Intensity" of the simulation to beautiful Slime colors
fragment float4 fragment_main(VertexOut in [[stage_in]],
                               texture2d<float, access::sample> trail_map [[texture(0)]]) {
    constexpr sampler s(filter::linear);
    
    // Sample texture with a tiny bit of "chromatic spread" for a soft organic look
    float r = trail_map.sample(s, in.uv + float2(0.001, 0)).r;
    float g = trail_map.sample(s, in.uv + float2(0, 0.001)).g;
    float b = trail_map.sample(s, in.uv).b;
    float4 val = float4(r, g, b, 1.0);
    
    // 1. Biological Slime Colors (Cyan & Magenta)
    // Use power functions to make the trails look more "wispy" and less like solid blobs
    float3 cyan = float3(0.0, 0.8, 1.0) * pow(val.r, 1.5);
    float3 magenta = float3(1.0, 0.0, 0.7) * pow(val.g, 1.5);
    
    // 2. Cinematic Lava Frontier
    // Deep reds for the heat shadow, glowing orange for the fire
    float3 lava_glow = float3(1.0, 0.1, 0.0) * smoothstep(0.1, 0.6, val.b);
    float3 lava_core = float3(1.0, 0.6, 0.0) * pow(val.b, 2.5);
    float3 lava = lava_glow + lava_core;
    
    // 3. Composite Colors
    // Use 'max' or soft addition to keep species distinct
    float3 base_color = max(cyan, magenta);
    float3 final_color = base_color + lava;
    
    // 4. Tonemapping & Softening (The "Motion Blur" look)
    // Prevent colors from blowing out to white/yellow by using a filmic curve
    final_color = final_color / (1.0 + final_color); 
    
    return float4(final_color, 1.0);
}
