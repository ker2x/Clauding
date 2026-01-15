// =============================================================================
// Mandelbulb Path Tracer - Shader.metal
// =============================================================================
// GPU path tracing of the Mandelbulb fractal with progressive rendering.
//
// Key Algorithms:
// 1. Mandelbulb Distance Estimation - SDF for ray marching
// 2. Sphere Tracing - Efficient ray-surface intersection
// 3. Monte Carlo Path Tracing - Physically-based global illumination
// 4. Cosine-Weighted Hemisphere Sampling - Diffuse BRDF importance sampling
//
// Mathematical Background:
// ------------------------
// The Mandelbulb is a 3D fractal defined by iterating:
//   z_{n+1} = z_n^power + c
// where z and c are 3D points using spherical coordinates for the power operation.
//
// The "triplex" power formula in spherical coords:
//   r = |z|
//   theta = atan2(z.y, z.x)  (azimuthal)
//   phi = asin(z.z / r)      (polar)
//
//   z^n = r^n * (cos(n*phi)*cos(n*theta), cos(n*phi)*sin(n*theta), sin(n*phi))
//
// Distance Estimation:
//   DE = 0.5 * |z| * log(|z|) / |dz|
// where |dz| is the derivative magnitude tracked during iteration.
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
constant float MIN_DIST = 0.0001f;
constant float SURFACE_DIST = 0.0005f;

// Mandelbulb parameters
constant float POWER = 8.0f;        // The "n" in z^n + c (8 is classic Mandelbulb)
constant int MAX_ITERATIONS = 20;   // More iterations = more detail but slower
constant float BAILOUT = 2.0f;      // Escape radius

// Path tracing parameters
constant int MAX_BOUNCES = 8;       // Maximum path depth

// =============================================================================
// DATA STRUCTURES
// =============================================================================

struct RenderParams {
    float4x4 invViewProj;    // Inverse view-projection matrix for ray generation
    float3 cameraPos;        // Camera position in world space
    float time;              // Time for animation (unused currently)
    uint frameIndex;         // Frame counter for random seed
    uint width;              // Render target width
    uint height;             // Render target height
    float exposure;          // Tone mapping exposure
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
    float ao;                // Ambient occlusion
    float iterations;        // Normalized iteration count for coloring
};

// =============================================================================
// RANDOM NUMBER GENERATION
// =============================================================================
// PCG hash for high-quality random numbers in path tracing.
// We use the pixel position and frame index to generate unique sequences.

// Hash function from PCG family
uint pcg_hash(uint input) {
    uint state = input * 747796405u + 2891336453u;
    uint word = ((state >> ((state >> 28u) + 4u)) ^ state) * 277803737u;
    return (word >> 22u) ^ word;
}

// Random float in [0, 1)
float random_float(thread uint& seed) {
    seed = pcg_hash(seed);
    return float(seed) / float(0xFFFFFFFFu);
}

// Random float in [min, max)
float random_float(thread uint& seed, float min_val, float max_val) {
    return min_val + (max_val - min_val) * random_float(seed);
}

// =============================================================================
// HEMISPHERE SAMPLING
// =============================================================================
// Cosine-weighted hemisphere sampling for diffuse surfaces.
// This importance samples according to the cosine term in the rendering equation.
//
// PDF = cos(theta) / PI
// Sample generation using Malley's method:
//   1. Sample uniform disk
//   2. Project up to hemisphere

float3 cosine_weighted_hemisphere(float2 u) {
    // Concentric disk mapping (low distortion)
    float r = sqrt(u.x);
    float theta = TWO_PI * u.y;

    float x = r * cos(theta);
    float y = r * sin(theta);
    float z = sqrt(max(0.0f, 1.0f - u.x));  // z = sqrt(1 - r^2)

    return float3(x, y, z);
}

// Build orthonormal basis from normal (Frisvad's method for z-up)
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

// Transform local hemisphere sample to world space
float3 sample_hemisphere(float3 normal, float2 u) {
    float3 local = cosine_weighted_hemisphere(u);

    float3 tangent, bitangent;
    build_orthonormal_basis(normal, tangent, bitangent);

    return normalize(tangent * local.x + bitangent * local.y + normal * local.z);
}

// =============================================================================
// MANDELBULB DISTANCE ESTIMATOR
// =============================================================================
// Returns the estimated distance to the Mandelbulb surface.
// Also outputs iteration count for coloring and derivative for distance estimation.
//
// The algorithm:
// 1. Start with z = position, c = position (Julia variant possible with fixed c)
// 2. Iterate z = z^power + c in spherical coordinates
// 3. Track |dz| for distance estimation
// 4. If |z| > bailout, point is outside
// 5. Distance = 0.5 * |z| * log(|z|) / |dz|

float2 mandelbulb_de(float3 pos) {
    float3 z = pos;
    float dr = 1.0f;           // Running derivative magnitude
    float r = length(z);
    float iterations = 0.0f;

    for (int i = 0; i < MAX_ITERATIONS; i++) {
        if (r > BAILOUT) break;

        // Convert to spherical coordinates
        float theta = acos(z.z / r);           // Polar angle
        float phi = atan2(z.y, z.x);           // Azimuthal angle

        // Derivative update: dr = power * r^(power-1) * dr + 1
        dr = pow(r, POWER - 1.0f) * POWER * dr + 1.0f;

        // Apply the power in spherical coordinates
        float zr = pow(r, POWER);
        theta *= POWER;
        phi *= POWER;

        // Convert back to Cartesian
        z = zr * float3(
            sin(theta) * cos(phi),
            sin(theta) * sin(phi),
            cos(theta)
        );

        // Add c (the original position)
        z += pos;

        r = length(z);
        iterations += 1.0f;
    }

    // Distance estimation formula
    float dist = 0.5f * log(r) * r / dr;

    // Return distance and normalized iteration count
    return float2(dist, iterations / float(MAX_ITERATIONS));
}

// =============================================================================
// RAY MARCHING (SPHERE TRACING)
// =============================================================================
// March along the ray, stepping by the distance estimate at each point.
// This is efficient because the DE gives us a safe step size.

HitInfo ray_march(Ray ray) {
    HitInfo info;
    info.hit = false;
    info.distance = 0.0f;

    float t = 0.0f;

    for (int i = 0; i < MAX_STEPS && t < MAX_DIST; i++) {
        float3 p = ray.origin + ray.direction * t;

        float2 de = mandelbulb_de(p);
        float dist = de.x;

        if (dist < SURFACE_DIST) {
            info.hit = true;
            info.distance = t;
            info.position = p;
            info.iterations = de.y;
            break;
        }

        // Adaptive step: smaller steps near surface for accuracy
        t += dist * 0.9f;  // Slight reduction for safety
    }

    return info;
}

// =============================================================================
// NORMAL ESTIMATION
// =============================================================================
// Compute surface normal using central differences of the distance field.
// This is the gradient of the SDF at the surface point.

float3 estimate_normal(float3 p) {
    float2 e = float2(EPSILON, 0.0f);

    float3 n = float3(
        mandelbulb_de(p + e.xyy).x - mandelbulb_de(p - e.xyy).x,
        mandelbulb_de(p + e.yxy).x - mandelbulb_de(p - e.yxy).x,
        mandelbulb_de(p + e.yyx).x - mandelbulb_de(p - e.yyx).x
    );

    return normalize(n);
}

// =============================================================================
// AMBIENT OCCLUSION
// =============================================================================
// Estimate AO by sampling the distance field along the normal.
// Points inside geometry (low DE values) indicate occlusion.

float calculate_ao(float3 pos, float3 normal) {
    float ao = 0.0f;
    float weight = 0.5f;

    for (int i = 1; i <= 5; i++) {
        float dist = 0.01f + 0.03f * float(i);  // Sample distances
        float3 sample_pos = pos + normal * dist;
        float de = mandelbulb_de(sample_pos).x;

        ao += weight * (dist - de);
        weight *= 0.5f;
    }

    return clamp(1.0f - 10.0f * ao, 0.0f, 1.0f);
}

// =============================================================================
// SOFT SHADOWS
// =============================================================================
// March toward the light, tracking how close we get to surfaces.
// Closer passages = darker shadows (penumbra effect).

float soft_shadow(float3 origin, float3 light_dir, float min_t, float max_t, float k) {
    float result = 1.0f;
    float t = min_t;

    for (int i = 0; i < 64 && t < max_t; i++) {
        float3 p = origin + light_dir * t;
        float de = mandelbulb_de(p).x;

        if (de < SURFACE_DIST) {
            return 0.0f;  // Hard shadow
        }

        // Soft shadow factor: smaller de = darker
        result = min(result, k * de / t);
        t += de;
    }

    return clamp(result, 0.0f, 1.0f);
}

// =============================================================================
// MATERIAL AND SHADING
// =============================================================================
// Simple diffuse + specular material with orbit trap coloring.

float3 get_material_color(float iterations, float3 normal) {
    // Orbit trap coloring based on iteration count
    float3 color1 = float3(0.05f, 0.2f, 0.4f);   // Deep blue
    float3 color2 = float3(0.8f, 0.4f, 0.1f);    // Orange
    float3 color3 = float3(0.2f, 0.6f, 0.3f);    // Green

    // Mix colors based on iteration depth
    float t = iterations;
    float3 base_color;

    if (t < 0.5f) {
        base_color = mix(color1, color2, t * 2.0f);
    } else {
        base_color = mix(color2, color3, (t - 0.5f) * 2.0f);
    }

    // Add subtle variation based on normal
    base_color += 0.05f * normal;

    return clamp(base_color, 0.0f, 1.0f);
}

// =============================================================================
// ENVIRONMENT / SKY
// =============================================================================
// Simple procedural sky for background and ambient lighting.

float3 sky_color(float3 direction) {
    // Gradient from horizon to zenith
    float t = 0.5f * (direction.y + 1.0f);
    float3 horizon = float3(0.8f, 0.85f, 0.9f);
    float3 zenith = float3(0.1f, 0.2f, 0.5f);

    float3 sky = mix(horizon, zenith, clamp(t, 0.0f, 1.0f));

    // Add sun
    float3 sun_dir = normalize(float3(0.5f, 0.8f, 0.3f));
    float sun_dot = dot(direction, sun_dir);
    if (sun_dot > 0.995f) {
        sky += float3(10.0f, 9.0f, 7.0f);  // Bright sun
    } else if (sun_dot > 0.9f) {
        float glow = pow((sun_dot - 0.9f) / 0.095f, 2.0f);
        sky += float3(1.0f, 0.8f, 0.5f) * glow;
    }

    return sky;
}

// =============================================================================
// PATH TRACING
// =============================================================================
// Trace a path through the scene, accumulating radiance.
// Uses Russian roulette for path termination.

float3 trace_path(Ray ray, thread uint& seed) {
    float3 throughput = float3(1.0f);
    float3 radiance = float3(0.0f);

    for (int bounce = 0; bounce < MAX_BOUNCES; bounce++) {
        HitInfo hit = ray_march(ray);

        if (!hit.hit) {
            // Hit the sky
            radiance += throughput * sky_color(ray.direction);
            break;
        }

        // Compute normal
        float3 normal = estimate_normal(hit.position);

        // Get material color
        float3 albedo = get_material_color(hit.iterations, normal);

        // Ambient occlusion (first bounce only for performance)
        float ao = (bounce == 0) ? calculate_ao(hit.position, normal) : 1.0f;

        // Direct lighting from sun
        float3 sun_dir = normalize(float3(0.5f, 0.8f, 0.3f));
        float ndotl = max(0.0f, dot(normal, sun_dir));

        if (ndotl > 0.0f) {
            float shadow = soft_shadow(hit.position + normal * 0.001f, sun_dir, 0.01f, 10.0f, 16.0f);
            float3 sun_color = float3(2.0f, 1.9f, 1.7f);
            radiance += throughput * albedo * sun_color * ndotl * shadow * ao;
        }

        // Add ambient light
        radiance += throughput * albedo * float3(0.05f, 0.08f, 0.12f) * ao;

        // Russian roulette for path termination
        if (bounce > 1) {
            float p = max(throughput.x, max(throughput.y, throughput.z));
            if (random_float(seed) > p) break;
            throughput /= p;
        }

        // Sample next direction (diffuse bounce)
        float2 u = float2(random_float(seed), random_float(seed));
        float3 new_dir = sample_hemisphere(normal, u);

        // Update throughput (diffuse BRDF = albedo / PI, PDF = cos(theta) / PI)
        // These cancel out nicely, leaving just albedo
        throughput *= albedo;

        // Setup ray for next bounce
        ray.origin = hit.position + normal * 0.001f;
        ray.direction = new_dir;
    }

    return radiance;
}

// =============================================================================
// COMPUTE KERNEL: Path Trace One Sample
// =============================================================================
// Each thread traces one ray for one pixel.
// Results are accumulated in the accumulation buffer.

kernel void path_trace(
    texture2d<float, access::write> output [[texture(0)]],     // Final output
    texture2d<float, access::read_write> accumulator [[texture(1)]], // Accumulated samples
    constant RenderParams& params [[buffer(0)]],
    uint2 gid [[thread_position_in_grid]]
) {
    // Bounds check
    if (gid.x >= params.width || gid.y >= params.height) return;

    // Initialize random seed from pixel position and frame
    uint seed = pcg_hash(gid.x + gid.y * params.width + params.frameIndex * params.width * params.height);

    // Generate ray through pixel with jittering for anti-aliasing
    float2 pixel = float2(gid) + float2(random_float(seed), random_float(seed));
    float2 ndc = (pixel / float2(params.width, params.height)) * 2.0f - 1.0f;
    ndc.y = -ndc.y;  // Flip Y for Metal's coordinate system

    // Transform NDC to world space using inverse view-projection
    float4 near_point = params.invViewProj * float4(ndc, 0.0f, 1.0f);
    float4 far_point = params.invViewProj * float4(ndc, 1.0f, 1.0f);

    near_point /= near_point.w;
    far_point /= far_point.w;

    Ray ray;
    ray.origin = params.cameraPos;
    ray.direction = normalize(far_point.xyz - near_point.xyz);

    // Trace the path
    float3 color = trace_path(ray, seed);

    // Accumulate samples
    float4 accumulated = accumulator.read(gid);
    float sample_count = accumulated.w + 1.0f;

    // Running average: new_avg = old_avg + (new_sample - old_avg) / n
    float3 new_color = accumulated.xyz + (color - accumulated.xyz) / sample_count;

    accumulator.write(float4(new_color, sample_count), gid);

    // Tone mapping and gamma correction for display
    float3 display_color = new_color * params.exposure;

    // Reinhard tone mapping
    display_color = display_color / (display_color + 1.0f);

    // Gamma correction (sRGB)
    display_color = pow(display_color, float3(1.0f / 2.2f));

    output.write(float4(display_color, 1.0f), gid);
}

// =============================================================================
// COMPUTE KERNEL: Clear Accumulator
// =============================================================================
// Called when camera moves or parameters change.

kernel void clear_accumulator(
    texture2d<float, access::write> accumulator [[texture(0)]],
    uint2 gid [[thread_position_in_grid]]
) {
    accumulator.write(float4(0.0f), gid);
}

// =============================================================================
// RENDER PIPELINE: Full-screen Quad (for blitting)
// =============================================================================

struct VertexOut {
    float4 position [[position]];
    float2 uv;
};

vertex VertexOut fullscreen_vertex(uint vid [[vertex_id]]) {
    // Generate full-screen triangle (more efficient than quad)
    VertexOut out;

    // Vertices: (-1,-1), (3,-1), (-1,3)
    // This triangle covers the entire screen
    out.uv = float2((vid << 1) & 2, vid & 2);
    out.position = float4(out.uv * 2.0f - 1.0f, 0.0f, 1.0f);
    out.uv.y = 1.0f - out.uv.y;  // Flip Y for texture sampling

    return out;
}

fragment float4 fullscreen_fragment(
    VertexOut in [[stage_in]],
    texture2d<float> tex [[texture(0)]]
) {
    constexpr sampler s(filter::linear);
    return tex.sample(s, in.uv);
}
