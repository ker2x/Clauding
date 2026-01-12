#include <metal_stdlib>
using namespace metal;

struct Uniforms {
    float2 resolution;
    float time;
    float dt;
    uint gridSize;
    float2 mousePos;
    int mouseDown;
    float viscosity;
    float buoyancy;
};

// --- FLUID UTILS ---

float3x3 get_rot(float2 mouse) {
    float3x3 rx = float3x3(float3(1,0,0), float3(0,cos(mouse.y),-sin(mouse.y)), float3(0,sin(mouse.y),cos(mouse.y)));
    float3x3 ry = float3x3(float3(cos(mouse.x),0,sin(mouse.x)), float3(0,1,0), float3(-sin(mouse.x),0,cos(mouse.x)));
    return rx * ry;
}

// Neural Turbulence Weights (MLP-ish)
constant float4 turbulence_weights[4] = {
    float4(0.5, -0.2, 0.8, 0.1), float4(-0.1, 0.7, -0.4, 0.3),
    float4(0.2, 0.3, 0.1, -0.6), float4(-0.4, 0.1, 0.5, 0.2)
};

// --- SIMULATION KERNELS ---

kernel void clear_kernel(texture3d<float, access::write> t1 [[texture(0)]],
                        texture3d<float, access::write> t2 [[texture(1)]],
                        texture3d<float, access::write> t3 [[texture(2)]],
                        uint3 gid [[thread_position_in_grid]]) {
    t1.write(0, gid);
    t2.write(0, gid);
    t3.write(0, gid);
}

kernel void advect_kernel(texture3d<float, access::sample> in_vel [[texture(0)]],
                         texture3d<float, access::write> out_vel [[texture(1)]],
                         texture3d<float, access::sample> in_dens [[texture(2)]],
                         texture3d<float, access::write> out_dens [[texture(3)]],
                         constant Uniforms& u [[buffer(0)]],
                         uint3 gid [[thread_position_in_grid]]) {
    if (any(gid >= u.gridSize)) return;
    
    constexpr sampler s(filter::linear, address::clamp_to_edge);
    float3 uvw = (float3(gid) + 0.5) / float(u.gridSize);
    
    // 1. Velocity Advection
    float3 vel = in_vel.sample(s, uvw).rgb;
    float3 back_uvw = uvw - vel * u.dt * (1.0 / float(u.gridSize));
    float3 new_vel = in_vel.sample(s, back_uvw).rgb;
    
    // 2. Neural Turbulence (Injecting small swirls)
    float neural = dot(float4(new_vel, abs(new_vel.x)), turbulence_weights[gid.x % 4]);
    new_vel += float3(sin(u.time + neural), cos(u.time * 1.1 + neural), sin(u.time * 0.9)) * 0.005;
    
    // 3. Buoyancy (Density makes it rise)
    float dens = in_dens.sample(s, uvw).r;
    new_vel.y += dens * u.buoyancy * u.dt;
    
    // CLAMP: Prevent explosion
    new_vel = clamp(new_vel, -1.5, 1.5);
    
    out_vel.write(float4(new_vel, 0), gid);
    
    // 4. Density Advection
    float new_dens = in_dens.sample(s, back_uvw).r;
    
    // SOURCE: Pulsing center source
    float3 center = float3(u.gridSize) * 0.5;
    if (distance(float3(gid), center) < 3.5) {
        new_dens = saturate(new_dens + 0.3 * (sin(u.time * 3.0) * 0.5 + 0.5));
    }
    
    // MOUSE INJECTION: Inject density where user clicks
    if (u.mouseDown) {
        float3 ro = float3(0, 0, -2.5);
        float2 muv = u.mousePos * 2.0 - 1.0;
        muv.y *= -1.0; // Correct flip
        muv.x *= u.resolution.x / u.resolution.y;
        float3 rd = normalize(float3(muv, 1.5));
        
        // Better hit detection: project to a plane at z=0 (center of grid)
        float3x3 rot = get_rot(float2(u.time * 0.2, 0.3));
        float3 rro = rot * ro;
        float3 rrd = rot * rd;
        
        // Ray-Plane intersection: (ro + rd*t).n = d
        float t0 = -dot(rro, float3(0,0,1)) / dot(rrd, float3(0,0,1));
        float3 hit = rro + rrd * t0;
        float3 gridHit = (hit / 1.2 + 0.5) * float(u.gridSize);
        
        if (distance(float3(gid), gridHit) < 4.0) {
            new_dens = saturate(new_dens + 0.8);
        }
    }
    
    out_dens.write(float4(new_dens * 0.99, 0, 0, 0), gid);
}

kernel void divergence_kernel(texture3d<float, access::read> vel [[texture(0)]],
                             texture3d<float, access::write> div [[texture(1)]],
                             constant Uniforms& u [[buffer(0)]],
                             uint3 gid [[thread_position_in_grid]]) {
    if (any(gid >= u.gridSize)) return;
    
    float3 vL = (gid.x > 0) ? vel.read(gid - uint3(1,0,0)).rgb : 0;
    float3 vR = (gid.x < u.gridSize - 1) ? vel.read(gid + uint3(1,0,0)).rgb : 0;
    float3 vB = (gid.y > 0) ? vel.read(gid - uint3(0,1,0)).rgb : 0;
    float3 vT = (gid.y < u.gridSize - 1) ? vel.read(gid + uint3(0,1,0)).rgb : 0;
    float3 vF = (gid.z > 0) ? vel.read(gid - uint3(0,0,1)).rgb : 0;
    float3 vBa = (gid.z < u.gridSize - 1) ? vel.read(gid + uint3(0,0,1)).rgb : 0;
    
    float divergence = 0.5 * ((vR.x - vL.x) + (vT.y - vB.y) + (vBa.z - vF.z));
    div.write(float4(divergence, 0, 0, 0), gid);
}

kernel void pressure_kernel(texture3d<float, access::read> in_p [[texture(0)]],
                           texture3d<float, access::write> out_p [[texture(1)]],
                           texture3d<float, access::read> div [[texture(2)]],
                           constant Uniforms& u [[buffer(0)]],
                           uint3 gid [[thread_position_in_grid]]) {
    if (any(gid >= u.gridSize)) return;
    
    float pL = (gid.x > 0) ? in_p.read(gid - uint3(1,0,0)).r : 0;
    float pR = (gid.x < u.gridSize - 1) ? in_p.read(gid + uint3(1,0,0)).r : 0;
    float pB = (gid.y > 0) ? in_p.read(gid - uint3(0,1,0)).r : 0;
    float pT = (gid.y < u.gridSize - 1) ? in_p.read(gid + uint3(0,1,0)).r : 0;
    float pF = (gid.z > 0) ? in_p.read(gid - uint3(0,0,1)).r : 0;
    float pBa = (gid.z < u.gridSize - 1) ? in_p.read(gid + uint3(0,0,1)).r : 0;
    
    float b = div.read(gid).r;
    float new_p = (pL + pR + pB + pT + pF + pBa - b) / 6.0;
    out_p.write(float4(new_p, 0, 0, 0), gid);
}

kernel void project_kernel(texture3d<float, access::read> in_vel [[texture(0)]],
                          texture3d<float, access::write> out_vel [[texture(1)]],
                          texture3d<float, access::read> p [[texture(2)]],
                          constant Uniforms& u [[buffer(0)]],
                          uint3 gid [[thread_position_in_grid]]) {
    if (any(gid >= u.gridSize)) return;
    
    float pL = (gid.x > 0) ? p.read(gid - uint3(1,0,0)).r : 0;
    float pR = (gid.x < u.gridSize - 1) ? p.read(gid + uint3(1,0,0)).r : 0;
    float pB = (gid.y > 0) ? p.read(gid - uint3(0,1,0)).r : 0;
    float pT = (gid.y < u.gridSize - 1) ? p.read(gid + uint3(0,1,0)).r : 0;
    float pF = (gid.z > 0) ? p.read(gid - uint3(0,0,1)).r : 0;
    float pBa = (gid.z < u.gridSize - 1) ? p.read(gid + uint3(0,0,1)).r : 0;
    
    float3 grad_p = 0.5 * float3(pR - pL, pT - pB, pBa - pF);
    float3 v = in_vel.read(gid).rgb;
    float3 final_v = clamp(v - grad_p, -1.5, 1.5);
    out_vel.write(float4(final_v, 0), gid);
}

// --- RENDERING KERNELS ---

kernel void raytrace_kernel(texture3d<float, access::sample> density [[texture(0)]],
                           texture2d<float, access::write> output [[texture(1)]],
                           constant Uniforms& u [[buffer(0)]],
                           uint2 gid [[thread_position_in_grid]]) {
    if (gid.x >= (uint)u.resolution.x || gid.y >= (uint)u.resolution.y) return;
    
    float2 uv = (float2(gid) / u.resolution) * 2.0 - 1.0;
    uv.x *= u.resolution.x / u.resolution.y;
    
    float3 ro = float3(0, 0, -2.5);
    float3 rd = normalize(float3(uv, 1.2));
    
    // Rotation
    float3x3 mrot = get_rot(u.mousePos * 6.28); // Use mousePos for rotation if not clicking? 
    // Wait, let's use a fixed rotation or mouse-driven rotation properly.
    // Let's use time for some slow auto-rotation and mouse drag for manual.
    float3x3 rot = get_rot(float2(u.time * 0.2, 0.3)); 
    ro = rot * ro;
    rd = rot * rd;
    
    float3 final_color = float3(0.005, 0.005, 0.01);
    float t = 0.5;
    
    for(int i=0; i<128; i++) {
        float3 p = ro + rd * t;
        float3 gridUVW = (p + 1.2) * 0.41;
        
        if (all(gridUVW >= 0.0) && all(gridUVW < 1.0)) {
            constexpr sampler s(filter::linear);
            float d = density.sample(s, gridUVW).r;
            
            if (d > 0.02) {
                // Lava-lamp look
                float3 lava_color = mix(float3(0.2, 0.05, 0.01), float3(1.0, 0.3, 0.05), d);
                if (d > 0.6) lava_color = mix(lava_color, float3(1.0, 0.9, 0.4), (d-0.6)*2.5);
                
                // Emitted light increases with density
                float alpha = smoothstep(0.0, 0.5, d) * 0.15;
                final_color += lava_color * alpha;
                
                if (length(final_color) > 10.0) break; 
            }
        }
        t += 0.02;
    }
    
    // Bloom / Exposure
    final_color = 1.0 - exp(-final_color * 1.5);
    output.write(float4(final_color, 1.0), gid);
}

// Standard Vert/Frag
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
