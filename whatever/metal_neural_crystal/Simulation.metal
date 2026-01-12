#include <metal_stdlib>
using namespace metal;

struct Uniforms {
    float4x4 viewMatrix;
    float4x4 projectionMatrix;
    float2 resolution;
    float time;
    uint gridSize;
    float growthRate;
    float2 mousePos;
    int mouseDown;
};

// NCA Weights (Small randomized MLP)
constant float4 weights1[8] = {
    float4(0.1, -0.2, 0.4, 0.1), float4(-0.1, 0.5, -0.3, 0.2),
    float4(0.3, -0.1, 0.2, -0.5), float4(0.2, 0.3, -0.1, 0.4),
    float4(-0.4, 0.2, 0.1, -0.3), float4(0.5, -0.4, 0.3, 0.1),
    float4(-0.2, 0.1, -0.5, 0.2), float4(0.1, 0.4, -0.2, 0.3)
};

kernel void nca_update_kernel(device float4* gridA [[buffer(0)]],
                            device float4* gridB [[buffer(1)]],
                            constant Uniforms& uniforms [[buffer(2)]],
                            uint3 gid [[thread_position_in_grid]]) {
    if (gid.x >= uniforms.gridSize || gid.y >= uniforms.gridSize || gid.z >= uniforms.gridSize) return;
    
    uint size = uniforms.gridSize;
    uint idx = gid.z * size * size + gid.y * size + gid.x;
    float4 self = gridA[idx];
    
    // 1.感知 (Perception): 3x3x3 Convolution (Laplacian)
    float4 laplacian = 0.0;
    for (int oz = -1; oz <= 1; oz++) {
        for (int oy = -1; oy <= 1; oy++) {
            for (int ox = -1; ox <= 1; ox++) {
                if (ox == 0 && oy == 0 && oz == 0) continue;
                
                int sx = (int)gid.x + ox;
                int sy = (int)gid.y + oy;
                int sz = (int)gid.z + oz;
                
                // Mirror boundaries
                if (sx < 0 || sx >= (int)size || sy < 0 || sy >= (int)size || sz < 0 || sz >= (int)size) continue;
                
                uint sidx = (uint)sz * size * size + (uint)sy * size + (uint)sx;
                laplacian += gridA[sidx];
            }
        }
    }
    laplacian = laplacian - self * 26.0;
    
    // 2. 变换 (Small Neural Layer)
    float4 h = self + laplacian * 0.5;
    float4 out = 0.0;
    for(int i=0; i<8; i++) {
        out += weights1[i] * dot(h, weights1[7-i]);
    }
    
    // Stochastic update for organic look
    float4 newState = self + out * uniforms.growthRate;
    
    // Activation (e.g., Tanh)
    newState = tanh(newState);
    
    gridB[idx] = newState;
}

struct Ray {
    float3 origin;
    float3 direction;
};

// Helper for spectral dispersion (Rainbow colors)
float3 spectral_color(float t) {
    float3 c = float3(0.0);
    c.r = saturate(sin(t * 6.0 + 0.0) * 0.5 + 0.5);
    c.g = saturate(sin(t * 6.0 + 2.0) * 0.5 + 0.5);
    c.b = saturate(sin(t * 6.0 + 4.0) * 0.5 + 0.5);
    return c;
}

// Trilinear interpolation for smooth voxel rendering
float4 sample_grid(device float4* grid, float3 p, uint size) {
    float3 f = fract(p - 0.5);
    uint3 i = uint3(p - 0.5);
    
    // Clamp to avoid edge artifacts
    i = clamp(i, uint3(0), uint3(size - 2));
    
    uint s2 = size * size;
    uint i000 = i.z * s2 + i.y * size + i.x;
    uint i100 = i000 + 1;
    uint i010 = i000 + size;
    uint i110 = i010 + 1;
    uint i001 = i000 + s2;
    uint i101 = i100 + s2;
    uint i011 = i010 + s2;
    uint i111 = i110 + s2;
    
    float4 v000 = grid[i000]; float4 v100 = grid[i100];
    float4 v010 = grid[i010]; float4 v110 = grid[i110];
    float4 v001 = grid[i001]; float4 v101 = grid[i101];
    float4 v011 = grid[i011]; float4 v111 = grid[i111];
    
    float4 v00 = mix(v000, v100, f.x);
    float4 v10 = mix(v010, v110, f.x);
    float4 v01 = mix(v001, v101, f.x);
    float4 v11 = mix(v011, v111, f.x);
    
    float4 v0 = mix(v00, v10, f.y);
    float4 v1 = mix(v01, v11, f.y);
    
    return mix(v0, v1, f.z);
}

// Raymarching through the volume with Refraction & Dispersion
kernel void raytrace_kernel(device float4* grid [[buffer(0)]],
                           constant Uniforms& uniforms [[buffer(1)]],
                           texture2d<float, access::write> output [[texture(0)]],
                           uint2 gid [[thread_position_in_grid]]) {
    if (gid.x >= (uint)uniforms.resolution.x || gid.y >= (uint)uniforms.resolution.y) return;
    
    float2 uv = (float2(gid) / uniforms.resolution) * 2.0 - 1.0;
    uv.x *= uniforms.resolution.x / uniforms.resolution.y;
    
    // Camera
    float3 ro = float3(0, 0, -2.5);
    float3 rd = normalize(float3(uv, 1.2));
    
    // Rotation
    float3x3 rotX = float3x3(float3(1, 0, 0), float3(0, cos(uniforms.mousePos.y), -sin(uniforms.mousePos.y)), float3(0, sin(uniforms.mousePos.y), cos(uniforms.mousePos.y)));
    float3x3 rotY = float3x3(float3(cos(uniforms.mousePos.x), 0, sin(uniforms.mousePos.x)), float3(0, 1, 0), float3(-sin(uniforms.mousePos.x), 0, cos(uniforms.mousePos.x)));
    float3x3 rot = rotX * rotY;
    ro = rot * ro;
    rd = rot * rd;
    
    float3 final_color = float3(0.002, 0.003, 0.008);
    float3 throughput = float3(1.0);
    float t = 0.5; // Start a bit away to avoid clipping
    bool inside = false;
    
    for(int i=0; i<160; i++) {
        float3 p = ro + rd * t;
        float3 gridP = (p + 1.2) * 0.41 * (float)uniforms.gridSize; // Adjust scaling to fit
        
        if (all(gridP >= 1.0) && all(gridP < (float)uniforms.gridSize - 1.0)) {
            float4 state = sample_grid(grid, gridP, uniforms.gridSize);
            float density = abs(state.r);
            
            if (density > 0.1) {
                // 1. NEURAL COLORING
                // Use RGB channels of state for internal variety
                float3 base_color = spectral_color(state.g * 0.5 + uniforms.time * 0.2);
                float3 emission = base_color * pow(density, 3.0) * 0.4;
                final_color += emission * throughput;
                
                // 2. REFRACTIVE SURFACE
                if (!inside || density > 0.5) {
                    // Estimate normal via smoothed gradient
                    float3 n = float3(
                        sample_grid(grid, gridP + float3(1,0,0), uniforms.gridSize).r - sample_grid(grid, gridP - float3(1,0,0), uniforms.gridSize).r,
                        sample_grid(grid, gridP + float3(0,1,0), uniforms.gridSize).r - sample_grid(grid, gridP - float3(0,1,0), uniforms.gridSize).r,
                        sample_grid(grid, gridP + float3(0,0,1), uniforms.gridSize).r - sample_grid(grid, gridP - float3(0,0,1), uniforms.gridSize).r
                    );
                    
                    if (length(n) > 0.001) {
                        n = normalize(n);
                        float eta = 1.45; // Glass-ish
                        rd = refract(rd, n, inside ? eta : 1.0 / eta);
                        inside = !inside;
                        throughput *= 0.85; // Absorption
                    }
                }
            }
        }
        t += 0.015;
        if (t > 6.0) break;
    }
    
    // Tonemapping
    final_color = 1.0 - exp(-final_color * 1.5);
    output.write(float4(final_color, 1.0), gid);
}

// Display shaders
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
    return tex.sample(s, in.uv);
}
