#include <metal_stdlib>
using namespace metal;

struct Uniforms {
    float2 resolution;
    float time;
    float mouseX;
    float mouseY;
    int mouseDown;
    float mu;       // Lenia growth center
    float sigma;    // Lenia growth width
    float rho;      // Kernel radius
    float dt;       // Time step
};

// --- LENIA MATH ---

// Growth function (Gaussian)
float growth(float U, float mu, float sigma) {
    float d = U - mu;
    return exp(-(d * d) / (2.0 * sigma * sigma)) * 2.0 - 1.0;
}

// Single-Kernel Lenia Update
kernel void lenia_kernel(texture2d<float, access::read> in_state [[texture(0)]],
                        texture2d<float, access::write> out_state [[texture(1)]],
                        constant Uniforms& u [[buffer(0)]],
                        uint2 gid [[thread_position_in_grid]]) {
    if (gid.x >= (uint)u.resolution.x || gid.y >= (uint)u.resolution.y) return;
    
    // 1. CONVOLUTION (Neighborhood Sum)
    float U = 0.0;
    float total_weight = 0.0;
    int r = (int)u.rho;
    
    for(int dy = -r; dy <= r; dy++) {
        for(int dx = -r; dx <= r; dx++) {
            float dist = sqrt(float(dx*dx + dy*dy));
            if (dist > u.rho) continue;
            
            // 3-Ring Kernel (Inner, Middle, Outer)
            // Ring 1: r=0.2, weight=1.0
            float w1 = exp(-pow(dist - u.rho*0.2, 2.0) / (2.0 * pow(u.rho*0.1, 2.0)));
            // Ring 2: r=0.5, weight=-0.5 (Inhibitory)
            float w2 = exp(-pow(dist - u.rho*0.5, 2.0) / (2.0 * pow(u.rho*0.1, 2.0)));
            // Ring 3: r=0.8, weight=0.3
            float w3 = exp(-pow(dist - u.rho*0.8, 2.0) / (2.0 * pow(u.rho*0.1, 2.0)));
            
            float weight = w1 * 1.0 + w2 * -0.2 + w3 * 0.5;
            weight = max(0.0, weight); // rect
            
            uint2 sample_pos = uint2((int(gid.x) + dx + int(u.resolution.x)) % int(u.resolution.x),
                                     (int(gid.y) + dy + int(u.resolution.y)) % int(u.resolution.y));
            
            U += in_state.read(sample_pos).r * weight;
            total_weight += weight;
        }
    }
    
    if (total_weight > 0.0) U /= total_weight;
    
    // 2. GROWTH
    float val = in_state.read(gid).r;
    float g = growth(U, u.mu, u.sigma);
    
    // Smooth update: val + dt * g
    float new_val = saturate(val + u.dt * g);
    
    // 3. INTERACTION (Mouse adds life)
    if (u.mouseDown) {
        float d = distance(float2(gid), float2(u.mouseX, u.resolution.y - u.mouseY));
        if (d < 20.0) {
             // Inject a "seed" value (around 0.2-0.5 is often good for Orbium start)
             new_val = mix(new_val, 0.5, 0.5); 
        }
    }
    
    out_state.write(new_val, gid);
}

// --- DISPLAY ---

struct VertexOut {
    float4 position [[position]];
    float2 uv;
};

vertex VertexOut vertex_main(uint id [[vertex_id]]) {
    float2 positions[4] = { float2(-1, -1), float2(1, -1), float2(-1, 1), float2(1, 1) };
    float2 uvs[4] = { float2(0, 1), float2(1, 1), float2(0, 0), float2(1, 0) };
    VertexOut out;
    out.position = float4(positions[id], 0, 1);
    out.uv = uvs[id];
    return out;
}

fragment float4 fragment_main(VertexOut in [[stage_in]], texture2d<float> tex [[texture(0)]]) {
    constexpr sampler s(filter::linear);
    float v = tex.sample(s, in.uv).r;
    
    // Bioluminescent mapping (Cool cyan-to-white)
    float3 color = mix(float3(0.0, 0.02, 0.05), float3(0.0, 0.8, 1.0), v);
    if (v > 0.8) color = mix(color, float3(1, 1, 1), (v-0.8)*5.0);
    
    return float4(color * (v > 0.01 ? 1.0 : 0.2), 1.0);
}
