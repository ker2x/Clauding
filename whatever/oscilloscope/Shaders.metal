#include <metal_stdlib>
using namespace metal;

struct VertexIn {
  float2 position;
  float2 texCoord;
};

struct VertexOut {
  float4 position [[position]];
  float2 texCoord;
};

struct Uniforms {
  float time;
  float width;
  float height;
  float sampleCount;
};

// Vertex shader (shared by all passes)
vertex VertexOut oscilloscopeVertex(uint vertexID [[vertex_id]],
                                     constant float4 *vertices [[buffer(0)]]) {
  VertexOut out;
  float4 vert = vertices[vertexID];
  out.position = float4(vert.xy, 0.0, 1.0);
  out.texCoord = vert.zw;
  return out;
}

// PASS 1A: Accumulation Fragment (dim previous frame by 5%)
fragment float4 accumulationFragment(VertexOut in [[stage_in]],
                                      texture2d<float> previousFrame [[texture(0)]]) {
  constexpr sampler textureSampler(mag_filter::linear, min_filter::linear);
  float4 previous = previousFrame.sample(textureSampler, in.texCoord);
  
  // Dim by 8% (multiply by 0.92) for faster phosphor decay
  return previous * 0.92;
}

// PASS 1B: Waveform Fragment (SDF-based electron beam glow)
fragment float4 waveformFragment(VertexOut in [[stage_in]],
                                  texture2d<float> audioTexture [[texture(0)]],
                                  constant Uniforms &uniforms [[buffer(0)]]) {
  constexpr sampler textureSampler(mag_filter::linear, min_filter::linear);
  
  float aspect = uniforms.width / uniforms.height;
  float2 uv = in.texCoord;
  
  // Current screen position in normalized coordinates
  float2 screenPos = uv * 2.0 - 1.0;
  screenPos.x *= aspect;
  
  // CRITICAL FIX: Map screen X position to valid sample range
  // This stretches the valid samples across the full screen width
  float textureWidth = 512.0;  // SAMPLE_BUFFER_SIZE
  float validSampleRatio = uniforms.sampleCount / textureWidth;
  
  // Scale UV to only sample from valid portion of texture (0 to validSampleRatio)
  // This stretches the valid samples across the entire screen
  float sampleX = uv.x * validSampleRatio;
  float waveAmplitude = audioTexture.sample(textureSampler, float2(sampleX, 0.5)).r;
  
  // Scale amplitude to screen space (-0.8 to 0.8)
  waveAmplitude *= 0.8;
  
  // Calculate vertical distance from current pixel to the waveform
  float dy = abs(screenPos.y - waveAmplitude);
  
  // Sharp cutoff - only render within 0.06 units of the line (thinner beam)
  if (dy > 0.06) {
    return float4(0.0);
  }
  
  // Exponential falloff for authentic beam physics
  float glow = exp(-dy * 35.0);  // Slightly steeper for thinner look
  
  // Very modest intensity
  glow *= 0.15;
  
  // Classic P1 phosphor green color
  float3 phosphorGreen = float3(0.05, 1.0, 0.1);
  
  return float4(phosphorGreen * glow, 1.0);
}

// PASS 2: Display Fragment (apply CRT effects to accumulated buffer)
fragment float4 displayFragment(VertexOut in [[stage_in]],
                                 texture2d<float> accumulatedFrame [[texture(0)]],
                                 constant Uniforms &uniforms [[buffer(0)]]) {
  constexpr sampler textureSampler(mag_filter::linear, min_filter::linear);
  
  // Apply barrel distortion (CRT curvature)
  float2 uv = in.texCoord;
  float2 centered = uv * 2.0 - 1.0;
  
  // Barrel distortion formula
  float distortion = 0.15;
  float r2 = dot(centered, centered);
  float2 offset = centered * r2 * distortion;
  float2 distortedUV = uv + offset * 0.5;
  
  // Vignette (darken corners)
  float vignette = 1.0 - dot(centered, centered) * 0.4;
  vignette = clamp(vignette, 0.0, 1.0);
  vignette = pow(vignette, 0.8);
  
  // Check if we're outside the curved screen
  if (distortedUV.x < 0.0 || distortedUV.x > 1.0 ||
      distortedUV.y < 0.0 || distortedUV.y > 1.0) {
    return float4(0.0, 0.0, 0.0, 1.0);
  }
  
  // Sample the accumulated phosphor glow
  float4 color = accumulatedFrame.sample(textureSampler, distortedUV);
  
  // Scanlines
  float scanline = sin(distortedUV.y * uniforms.height * 3.14159) * 0.5 + 0.5;
  scanline = mix(0.85, 1.0, scanline);
  color.rgb *= scanline;
  
  // Apply vignette
  color.rgb *= vignette;
  
  // Subtle noise/grain
  float noise = fract(sin(dot(uv + uniforms.time * 0.001, float2(12.9898, 78.233))) * 43758.5453);
  noise = mix(0.97, 1.03, noise);
  color.rgb *= noise;
  
  // Bloom on bright areas
  float brightness = dot(color.rgb, float3(0.2126, 0.7152, 0.0722));
  float bloom = max(brightness - 0.7, 0.0) * 0.3;
  color.rgb +=(color.rgb * bloom);
  
  // Subtle ambient phosphor glow
  color.rgb += float3(0.0, 0.02, 0.0);
  
  return float4(color.rgb, 1.0);
}
