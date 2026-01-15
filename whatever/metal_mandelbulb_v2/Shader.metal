// =============================================================================
// Mandelbulb Path Tracer v2 - Shader.metal
// =============================================================================
// GPU path tracing with MetalFX upscaling support.
//
// Changes from v1:
// - Renders at lower internal resolution (768x576)
// - Output is upscaled by MetalFX Spatial Scaler to 1024x768
// - Same progressive accumulation for noise reduction
// - Depth output for potential future MetalFX Temporal support
//
// MetalFX Integration:
// --------------------
// MetalFX Spatial Scaler uses ML-based upscaling similar to DLSS/FSR.
// It analyzes the low-res input and reconstructs high-frequency details
// that would be lost in simple bilinear upscaling.
//
// The path tracer outputs to a low-res texture, which is then:
// 1. Tone-mapped and gamma corrected
// 2. Upscaled by MetalFX to the final window resolution
// =============================================================================

#include <metal_stdlib>
using namespace metal;

// =============================================================================
// CONSTANTS AND CONFIGURATION
// =============================================================================

constant float PI = 3.14159265358979323846f;
constant float TWO_PI = 6.28318530717958647692f;
constant float EPSILON = 0.0001f;

// Ray marching parameters
constant int MAX_STEPS = 256;
constant float MAX_DIST = 100.0f;
constant float SURFACE_DIST = 0.0005f;

// Mandelbulb parameters
constant float POWER = 8.0f;
constant int MAX_ITERATIONS_DEFAULT = 20;  // Fallback, actual value from params
constant int MAX_ITERATIONS_LIMIT = 256;   // Hard cap for safety (increased for detail)
constant float BAILOUT = 2.0f;

// Path tracing parameters
constant int MAX_BOUNCES = 6;

// =============================================================================
// DATA STRUCTURES
// =============================================================================

struct RenderParams {
    float4x4 invViewProj;
    float4x4 prevViewProj;      // Previous frame's view-proj for motion vectors
    float3 cameraPos;
    float time;
    uint frameIndex;
    uint width;
    uint height;
    float exposure;
    float jitterX;              // Sub-pixel jitter for temporal AA
    float jitterY;
    int maxIterations;          // Dynamic iteration count based on zoom
    float surfaceDetail;        // Surface distance threshold (smaller = more detail)
};

struct Ray {
    float3 origin;
    float3 direction;
};

struct HitInfo {
    bool hit;
    float distance;
    float3 position;
    float3 normal;
    float iterations;
};

// =============================================================================
// RANDOM NUMBER GENERATION (PCG)
// =============================================================================

uint pcg_hash(uint input) {
    uint state = input * 747796405u + 2891336453u;
    uint word = ((state >> ((state >> 28u) + 4u)) ^ state) * 277803737u;
    return (word >> 22u) ^ word;
}

float random_float(thread uint& seed) {
    seed = pcg_hash(seed);
    return float(seed) / float(0xFFFFFFFFu);
}

// =============================================================================
// HEMISPHERE SAMPLING
// =============================================================================

float3 cosine_weighted_hemisphere(float2 u) {
    float r = sqrt(u.x);
    float theta = TWO_PI * u.y;
    float x = r * cos(theta);
    float y = r * sin(theta);
    float z = sqrt(max(0.0f, 1.0f - u.x));
    return float3(x, y, z);
}

void build_orthonormal_basis(float3 n, thread float3& tangent, thread float3& bitangent) {
    if (n.z < -0.9999999f) {
        tangent = float3(0.0f, -1.0f, 0.0f);
        bitangent = float3(-1.0f, 0.0f, 0.0f);
    } else {
        float a = 1.0f / (1.0f + n.z);
        float b = -n.x * n.y * a;
        tangent = float3(1.0f - n.x * n.x * a, b, -n.x);
        bitangent = float3(b, 1.0f - n.y * n.y * a, -n.y);
    }
}

float3 sample_hemisphere(float3 normal, float2 u) {
    float3 local = cosine_weighted_hemisphere(u);
    float3 tangent, bitangent;
    build_orthonormal_basis(normal, tangent, bitangent);
    return normalize(tangent * local.x + bitangent * local.y + normal * local.z);
}

// =============================================================================
// MANDELBULB DISTANCE ESTIMATOR
// =============================================================================
// maxIter parameter allows dynamic detail based on zoom level.
// More iterations = finer fractal detail visible at close range.

float2 mandelbulb_de(float3 pos, int maxIter) {
    float3 z = pos;
    float dr = 1.0f;
    float r = length(z);
    float iterations = 0.0f;

    for (int i = 0; i < maxIter; i++) {
        if (r > BAILOUT) break;

        float theta = acos(z.z / r);
        float phi = atan2(z.y, z.x);

        dr = pow(r, POWER - 1.0f) * POWER * dr + 1.0f;

        float zr = pow(r, POWER);
        theta *= POWER;
        phi *= POWER;

        z = zr * float3(
            sin(theta) * cos(phi),
            sin(theta) * sin(phi),
            cos(theta)
        );
        z += pos;

        r = length(z);
        iterations += 1.0f;
    }

    float dist = 0.5f * log(r) * r / dr;
    return float2(dist, iterations / float(maxIter));
}

// =============================================================================
// RAY MARCHING
// =============================================================================

HitInfo ray_march(Ray ray, int maxIter, float surfaceThreshold) {
    HitInfo info;
    info.hit = false;
    info.distance = 0.0f;

    float t = 0.0f;

    for (int i = 0; i < MAX_STEPS && t < MAX_DIST; i++) {
        float3 p = ray.origin + ray.direction * t;
        float2 de = mandelbulb_de(p, maxIter);
        float dist = de.x;

        if (dist < surfaceThreshold) {
            info.hit = true;
            info.distance = t;
            info.position = p;
            info.iterations = de.y;
            break;
        }

        t += dist * 0.9f;
    }

    return info;
}

// =============================================================================
// NORMAL ESTIMATION
// =============================================================================

float3 estimate_normal(float3 p, int maxIter, float eps) {
    float2 e = float2(eps, 0.0f);
    float3 n = float3(
        mandelbulb_de(p + e.xyy, maxIter).x - mandelbulb_de(p - e.xyy, maxIter).x,
        mandelbulb_de(p + e.yxy, maxIter).x - mandelbulb_de(p - e.yxy, maxIter).x,
        mandelbulb_de(p + e.yyx, maxIter).x - mandelbulb_de(p - e.yyx, maxIter).x
    );
    return normalize(n);
}

// =============================================================================
// AMBIENT OCCLUSION
// =============================================================================

float calculate_ao(float3 pos, float3 normal, int maxIter) {
    float ao = 0.0f;
    float weight = 0.5f;

    for (int i = 1; i <= 5; i++) {
        float dist = 0.01f + 0.03f * float(i);
        float3 sample_pos = pos + normal * dist;
        float de = mandelbulb_de(sample_pos, maxIter).x;
        ao += weight * (dist - de);
        weight *= 0.5f;
    }

    return clamp(1.0f - 10.0f * ao, 0.0f, 1.0f);
}

// =============================================================================
// SOFT SHADOWS
// =============================================================================

float soft_shadow(float3 origin, float3 light_dir, float min_t, float max_t, float k, int maxIter, float surfaceThreshold) {
    float result = 1.0f;
    float t = min_t;

    for (int i = 0; i < 64 && t < max_t; i++) {
        float3 p = origin + light_dir * t;
        float de = mandelbulb_de(p, maxIter).x;

        if (de < surfaceThreshold) {
            return 0.0f;
        }

        result = min(result, k * de / t);
        t += de;
    }

    return clamp(result, 0.0f, 1.0f);
}

// =============================================================================
// MATERIAL COLORING (Orbit Trap)
// =============================================================================

float3 get_material_color(float iterations, float3 normal) {
    float3 color1 = float3(0.05f, 0.2f, 0.4f);
    float3 color2 = float3(0.8f, 0.4f, 0.1f);
    float3 color3 = float3(0.2f, 0.6f, 0.3f);

    float t = iterations;
    float3 base_color;

    if (t < 0.5f) {
        base_color = mix(color1, color2, t * 2.0f);
    } else {
        base_color = mix(color2, color3, (t - 0.5f) * 2.0f);
    }

    base_color += 0.05f * normal;
    return clamp(base_color, 0.0f, 1.0f);
}

// =============================================================================
// SKY / ENVIRONMENT
// =============================================================================

float3 sky_color(float3 direction) {
    float t = 0.5f * (direction.y + 1.0f);
    float3 horizon = float3(0.8f, 0.85f, 0.9f);
    float3 zenith = float3(0.1f, 0.2f, 0.5f);

    float3 sky = mix(horizon, zenith, clamp(t, 0.0f, 1.0f));

    float3 sun_dir = normalize(float3(0.5f, 0.8f, 0.3f));
    float sun_dot = dot(direction, sun_dir);
    if (sun_dot > 0.995f) {
        sky += float3(10.0f, 9.0f, 7.0f);
    } else if (sun_dot > 0.9f) {
        float glow = pow((sun_dot - 0.9f) / 0.095f, 2.0f);
        sky += float3(1.0f, 0.8f, 0.5f) * glow;
    }

    return sky;
}

// =============================================================================
// PATH TRACING
// =============================================================================

float3 trace_path(Ray ray, thread uint& seed, thread float& out_depth, int maxIter, float surfaceThreshold) {
    float3 throughput = float3(1.0f);
    float3 radiance = float3(0.0f);
    out_depth = MAX_DIST;

    // Epsilon for normal estimation scales with surface threshold
    float normalEps = surfaceThreshold * 0.5f;

    for (int bounce = 0; bounce < MAX_BOUNCES; bounce++) {
        HitInfo hit = ray_march(ray, maxIter, surfaceThreshold);

        if (!hit.hit) {
            radiance += throughput * sky_color(ray.direction);
            break;
        }

        // Record depth on first hit
        if (bounce == 0) {
            out_depth = hit.distance;
        }

        float3 normal = estimate_normal(hit.position, maxIter, normalEps);
        float3 albedo = get_material_color(hit.iterations, normal);
        float ao = (bounce == 0) ? calculate_ao(hit.position, normal, maxIter) : 1.0f;

        // Direct lighting
        float3 sun_dir = normalize(float3(0.5f, 0.8f, 0.3f));
        float ndotl = max(0.0f, dot(normal, sun_dir));

        if (ndotl > 0.0f) {
            float shadow = soft_shadow(hit.position + normal * surfaceThreshold * 2.0f, sun_dir, 0.01f, 10.0f, 16.0f, maxIter, surfaceThreshold);
            float3 sun_color = float3(2.0f, 1.9f, 1.7f);
            radiance += throughput * albedo * sun_color * ndotl * shadow * ao;
        }

        radiance += throughput * albedo * float3(0.05f, 0.08f, 0.12f) * ao;

        // Russian roulette
        if (bounce > 1) {
            float p = max(throughput.x, max(throughput.y, throughput.z));
            if (random_float(seed) > p) break;
            throughput /= p;
        }

        // Next bounce
        float2 u = float2(random_float(seed), random_float(seed));
        float3 new_dir = sample_hemisphere(normal, u);
        throughput *= albedo;

        ray.origin = hit.position + normal * surfaceThreshold * 2.0f;
        ray.direction = new_dir;
    }

    return radiance;
}

// =============================================================================
// COMPUTE KERNEL: Path Trace with Depth Output
// =============================================================================

kernel void path_trace(
    texture2d<float, access::write> output [[texture(0)]],
    texture2d<float, access::read_write> accumulator [[texture(1)]],
    texture2d<float, access::write> depth_texture [[texture(2)]],
    constant RenderParams& params [[buffer(0)]],
    uint2 gid [[thread_position_in_grid]]
) {
    if (gid.x >= params.width || gid.y >= params.height) return;

    uint seed = pcg_hash(gid.x + gid.y * params.width + params.frameIndex * params.width * params.height);

    // Get dynamic iteration count and surface threshold from params
    int maxIter = clamp(params.maxIterations, 8, MAX_ITERATIONS_LIMIT);
    float surfaceThreshold = params.surfaceDetail;

    // Sub-pixel jitter for anti-aliasing
    float2 jitter = float2(random_float(seed), random_float(seed));
    float2 pixel = float2(gid) + jitter;
    float2 ndc = (pixel / float2(params.width, params.height)) * 2.0f - 1.0f;
    ndc.y = -ndc.y;

    float4 near_point = params.invViewProj * float4(ndc, 0.0f, 1.0f);
    float4 far_point = params.invViewProj * float4(ndc, 1.0f, 1.0f);
    near_point /= near_point.w;
    far_point /= far_point.w;

    Ray ray;
    ray.origin = params.cameraPos;
    ray.direction = normalize(far_point.xyz - near_point.xyz);

    float depth;
    float3 color = trace_path(ray, seed, depth, maxIter, surfaceThreshold);

    // Accumulate
    float4 accumulated = accumulator.read(gid);
    float sample_count = accumulated.w + 1.0f;
    float3 new_color = accumulated.xyz + (color - accumulated.xyz) / sample_count;
    accumulator.write(float4(new_color, sample_count), gid);

    // Tone mapping (Reinhard) + gamma
    float3 display_color = new_color * params.exposure;
    display_color = display_color / (display_color + 1.0f);
    display_color = pow(display_color, float3(1.0f / 2.2f));

    output.write(float4(display_color, 1.0f), gid);

    // Write normalized depth for MetalFX (0 = near, 1 = far)
    float normalized_depth = clamp(depth / MAX_DIST, 0.0f, 1.0f);
    depth_texture.write(float4(normalized_depth, 0.0f, 0.0f, 1.0f), gid);
}

// =============================================================================
// COMPUTE KERNEL: Clear Accumulator
// =============================================================================

kernel void clear_accumulator(
    texture2d<float, access::write> accumulator [[texture(0)]],
    uint2 gid [[thread_position_in_grid]]
) {
    accumulator.write(float4(0.0f), gid);
}

// =============================================================================
// FULLSCREEN BLIT (fallback if MetalFX unavailable)
// =============================================================================

struct VertexOut {
    float4 position [[position]];
    float2 uv;
};

vertex VertexOut fullscreen_vertex(uint vid [[vertex_id]]) {
    VertexOut out;
    out.uv = float2((vid << 1) & 2, vid & 2);
    out.position = float4(out.uv * 2.0f - 1.0f, 0.0f, 1.0f);
    out.uv.y = 1.0f - out.uv.y;
    return out;
}

fragment float4 fullscreen_fragment(
    VertexOut in [[stage_in]],
    texture2d<float> tex [[texture(0)]]
) {
    constexpr sampler s(filter::linear);
    return tex.sample(s, in.uv);
}
