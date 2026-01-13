// =============================================================================
// Metal Neural Aura - main.mm
// =============================================================================
// Neural particle field simulation with Metal compute shaders.
// Implements a 3-layer neural network (65→128→128→2) directly in Metal to
// generate particle forces from position and local density context.
//
// Based on aura_sim.py, ported to Metal/ObjC++ for GPU acceleration.
//
// Architecture:
// - 131k particles with position, velocity, and per-particle colors
// - Density map (128×128) for spatial context
// - Neural network weights in mutable GPU buffers
// - 3-kernel compute pipeline: density → neural forces → physics update
// =============================================================================

// -----------------------------------------------------------------------------
// Framework Imports
// -----------------------------------------------------------------------------
// Metal.h       : Core Metal API (devices, buffers, command queues)
// MetalKit.h    : MTKView and MTKViewDelegate for render loop
// AppKit.h      : macOS application framework (windows, events)
// simd/simd.h   : SIMD vector/matrix types that match Metal's types
#import <AppKit/AppKit.h>
#import <Metal/Metal.h>
#import <MetalKit/MetalKit.h>
#include <simd/simd.h>

// =============================================================================
// CONFIGURATION
// =============================================================================

namespace Config {
// Window dimensions
constexpr float WINDOW_WIDTH = 1280.0f;
constexpr float WINDOW_HEIGHT = 720.0f;

// Particle system settings
constexpr int NUM_PARTICLES = 131072;  // 2^17 particles
constexpr float POINT_SIZE = 2.0f;     // Particle size in pixels

// Neural network architecture
constexpr int FOURIER_DIM = 32;        // Number of Fourier frequency pairs
constexpr int HIDDEN_DIM = 128;        // Hidden layer size
constexpr int INPUT_DIM = 65;          // 64 Fourier features + 1 density
constexpr int OUTPUT_DIM = 2;          // force_x, force_y

// Density map
constexpr int DENSITY_RES = 128;       // Density grid resolution

// Physics settings
constexpr float FRICTION = 0.94f;      // Velocity damping
constexpr float NEURAL_STRENGTH = 0.006f;  // Neural force multiplier
constexpr float MUTATION_RATE = 0.05f; // Weight mutation scale
constexpr float JITTER_STRENGTH = 0.0015f; // Random motion
constexpr float MOUSE_STRENGTH = -0.02f;   // Mouse interaction force

// Rendering settings
constexpr int TARGET_FPS = 60;

// Background color (RGBA, 0.0-1.0)
constexpr float BG_RED = 0.02f;
constexpr float BG_GREEN = 0.02f;
constexpr float BG_BLUE = 0.05f;
constexpr float BG_ALPHA = 1.0f;
} // namespace Config

// =============================================================================
// DATA STRUCTURES
// =============================================================================
// These structures are shared between CPU (this file) and GPU (shader).
// They MUST match exactly with the definitions in Compute.metal.

// Simulation parameters passed to the GPU
struct SimParams {
  float width;               // Canvas width
  float height;              // Canvas height
  float friction;            // Velocity damping
  float neural_strength;     // Neural force multiplier
  float jitter_strength;     // Random motion scale
  int num_particles;         // Total particles
  int density_res;           // Density grid resolution
  float point_size;          // Particle render size
  simd_float2 mouse_pos;     // Mouse position in normalized coords [-1, 1]
  float mouse_strength;      // Mouse interaction force multiplier
};

// =============================================================================
// InputView - Custom MTKView for Keyboard Input
// =============================================================================
// MTKView is MetalKit's view class that provides:
// - A CAMetalLayer for efficient Metal rendering
// - A render loop that calls the delegate's drawInMTKView: method
// - Automatic drawable management
//
// We subclass it to add keyboard input handling.

// Forward declare the renderer protocol for type-safe messaging
@protocol RendererActions <NSObject>
- (void)mutateNeuralField;
- (void)reset;
- (void)togglePause;
@end

@interface InputView : MTKView
// Use 'id' with protocol to allow forward reference while maintaining type
// safety
@property(nonatomic, weak) id<RendererActions> renderer;
@end

@implementation InputView

// Allow this view to receive keyboard events
- (BOOL)acceptsFirstResponder {
  return YES;
}

// Handle key press events
- (void)keyDown:(NSEvent *)event {
  // Get the character that was pressed
  NSString *chars = [event charactersIgnoringModifiers];
  if ([chars length] == 0)
    return;

  unichar key = [chars characterAtIndex:0];

  switch (key) {
  case 'q':
  case 'Q':
    // Quit the application
    [NSApp terminate:nil];
    break;

  case 'r':
  case 'R':
    // Reset the simulation
    if (_renderer) {
      [_renderer reset];
    }
    break;

  case ' ':
    // Mutate neural network weights
    if (_renderer) {
      [_renderer mutateNeuralField];
    }
    break;

  case 'p':
  case 'P':
    // Toggle pause
    if (_renderer) {
      [_renderer togglePause];
    }
    break;

  default:
    // Pass unhandled keys to superclass
    [super keyDown:event];
    break;
  }
}

@end

// =============================================================================
// AppDelegate - Application Lifecycle
// =============================================================================
// The AppDelegate handles application-level events.
// In a simple app, we mainly use it to ensure proper termination.

@interface AppDelegate : NSObject <NSApplicationDelegate>
@end

@implementation AppDelegate

// Quit when the last window closes
- (BOOL)applicationShouldTerminateAfterLastWindowClosed:
    (NSApplication *)sender {
  (void)sender; // Suppress unused parameter warning
  return YES;
}

@end

// =============================================================================
// Renderer - Metal Rendering and Compute Pipeline
// =============================================================================
// This class manages all Metal resources and implements the render loop.
// It conforms to MTKViewDelegate to receive frame callbacks.

@interface Renderer : NSObject <MTKViewDelegate, RendererActions>

// Initialization
- (instancetype)initWithMetalKitView:(MTKView *)view;

// User actions (RendererActions protocol)
- (void)mutateNeuralField;
- (void)reset;
- (void)togglePause;

@end

@implementation Renderer {
  // -----------------------------------------------------
  // Metal Core Objects
  // -----------------------------------------------------
  id<MTLDevice> _device;
  id<MTLCommandQueue> _commandQueue;

  // -----------------------------------------------------
  // Pipeline States
  // -----------------------------------------------------
  // Three compute kernels: density map, neural forces, physics update
  id<MTLComputePipelineState> _densityPipeline;
  id<MTLComputePipelineState> _neuralPipeline;
  id<MTLComputePipelineState> _physicsPipeline;
  id<MTLRenderPipelineState> _renderPipeline;

  // -----------------------------------------------------
  // Particle State Buffers
  // -----------------------------------------------------
  id<MTLBuffer> _positionBuffer;  // float2[NUM_PARTICLES] - normalized [-1, 1]
  id<MTLBuffer> _velocityBuffer;  // float2[NUM_PARTICLES]
  id<MTLBuffer> _colorBuffer;     // float3[NUM_PARTICLES] - per-particle RGB

  // -----------------------------------------------------
  // Density Context Buffer
  // -----------------------------------------------------
  id<MTLBuffer> _densityMapBuffer; // float[DENSITY_RES * DENSITY_RES]

  // -----------------------------------------------------
  // Neural Network Buffers
  // -----------------------------------------------------
  // Layer 1: (64+1) → 128
  id<MTLBuffer> _weightsL1Buffer;  // float[8320]
  id<MTLBuffer> _biasesL1Buffer;   // float[128]

  // Layer 2: 128 → 128
  id<MTLBuffer> _weightsL2Buffer;  // float[16384]
  id<MTLBuffer> _biasesL2Buffer;   // float[128]

  // Layer 3: 128 → 2
  id<MTLBuffer> _weightsL3Buffer;  // float[256]
  id<MTLBuffer> _biasesL3Buffer;   // float[2]

  // Fourier encoding frequencies
  id<MTLBuffer> _fourierFreqsBuffer; // float2[32]

  // Intermediate force storage
  id<MTLBuffer> _forceBuffer;      // float2[NUM_PARTICLES]

  // -----------------------------------------------------
  // Parameters
  // -----------------------------------------------------
  id<MTLBuffer> _paramsBuffer;     // SimParams struct

  // -----------------------------------------------------
  // State
  // -----------------------------------------------------
  MTKView *_view;
  BOOL _paused;
  simd_float2 _mousePos;
  float _mouseStrength;

  // FPS tracking
  double _lastFrameTime;
  int _frameCount;
  double _fpsUpdateTime;
}

// -----------------------------------------------------------------------------
// Initialization
// -----------------------------------------------------------------------------
- (instancetype)initWithMetalKitView:(MTKView *)view {
  self = [super init];
  if (!self)
    return nil;

  _view = view;
  _paused = NO;
  _mousePos = simd_make_float2(0.0f, 0.0f);
  _mouseStrength = 0.0f;
  _lastFrameTime = CACurrentMediaTime();
  _frameCount = 0;
  _fpsUpdateTime = _lastFrameTime;

  // Step 1: Get the Metal device (GPU)
  // ----------------------------------
  // MTLCreateSystemDefaultDevice() returns the default GPU.
  // On Macs with multiple GPUs, this typically returns the discrete GPU.
  _device = MTLCreateSystemDefaultDevice();
  if (!_device) {
    NSLog(@"ERROR: Metal is not supported on this device");
    return nil;
  }
  NSLog(@"Using GPU: %@", _device.name);

  // Step 2: Create the command queue
  // ---------------------------------
  // Commands are submitted to the GPU through a command queue.
  // You typically create one queue and reuse it.
  _commandQueue = [_device newCommandQueue];
  if (!_commandQueue) {
    NSLog(@"ERROR: Failed to create command queue");
    return nil;
  }

  // Step 3: Load and compile shaders
  // ---------------------------------
  if (![self loadShaders]) {
    return nil;
  }

  // Step 4: Create GPU buffers
  // ---------------------------
  [self initBuffers];

  // Configure the view
  view.device = _device;
  view.delegate = self;

  return self;
}

// -----------------------------------------------------------------------------
// Shader Loading
// -----------------------------------------------------------------------------
// Metal shaders can be:
// 1. Pre-compiled into a .metallib file (faster startup)
// 2. Compiled at runtime from source (more flexible, used here)
//
// We load from source so you can modify the shader without rebuilding.

- (BOOL)loadShaders {
  NSError *error = nil;

  // Load shader source from file
  NSString *shaderPath = @"Compute.metal";
  NSString *shaderSource =
      [NSString stringWithContentsOfFile:shaderPath
                                encoding:NSUTF8StringEncoding
                                   error:&error];
  if (!shaderSource) {
    NSLog(@"ERROR: Could not load shader file: %@", error.localizedDescription);
    NSLog(
        @"Make sure Compute.metal is in the same directory as the executable.");
    return NO;
  }

  // Compile shader source into a library
  // A library contains one or more shader functions
  MTLCompileOptions *options = [[MTLCompileOptions alloc] init];
  // Enable fast math optimizations (use mathMode on macOS 15+, fastMathEnabled
  // on older)
  if (@available(macOS 15.0, *)) {
    options.mathMode = MTLMathModeFast;
  } else {
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wdeprecated-declarations"
    options.fastMathEnabled = YES;
#pragma clang diagnostic pop
  }

  id<MTLLibrary> library = [_device newLibraryWithSource:shaderSource
                                                 options:options
                                                   error:&error];
  if (!library) {
    NSLog(@"ERROR: Shader compilation failed: %@", error.localizedDescription);
    return NO;
  }
  NSLog(@"Shaders compiled successfully");

  // Create compute pipelines
  // ------------------------
  // We need three compute kernels for the neural particle simulation

  // 1. Density map kernel
  id<MTLFunction> densityFunc =
      [library newFunctionWithName:@"density_map_kernel"];
  if (!densityFunc) {
    NSLog(@"ERROR: Could not find 'density_map_kernel' function in shader");
    return NO;
  }

  _densityPipeline = [_device newComputePipelineStateWithFunction:densityFunc
                                                            error:&error];
  if (!_densityPipeline) {
    NSLog(@"ERROR: Failed to create density pipeline: %@",
          error.localizedDescription);
    return NO;
  }

  // 2. Neural force kernel
  id<MTLFunction> neuralFunc =
      [library newFunctionWithName:@"neural_force_kernel"];
  if (!neuralFunc) {
    NSLog(@"ERROR: Could not find 'neural_force_kernel' function in shader");
    return NO;
  }

  _neuralPipeline = [_device newComputePipelineStateWithFunction:neuralFunc
                                                           error:&error];
  if (!_neuralPipeline) {
    NSLog(@"ERROR: Failed to create neural pipeline: %@",
          error.localizedDescription);
    return NO;
  }

  // 3. Physics update kernel
  id<MTLFunction> physicsFunc =
      [library newFunctionWithName:@"physics_update_kernel"];
  if (!physicsFunc) {
    NSLog(@"ERROR: Could not find 'physics_update_kernel' function in shader");
    return NO;
  }

  _physicsPipeline = [_device newComputePipelineStateWithFunction:physicsFunc
                                                            error:&error];
  if (!_physicsPipeline) {
    NSLog(@"ERROR: Failed to create physics pipeline: %@",
          error.localizedDescription);
    return NO;
  }

  // Create render pipeline
  // ----------------------
  // A render pipeline processes vertices and produces pixels.
  id<MTLFunction> vertexFunc = [library newFunctionWithName:@"vertex_main"];
  id<MTLFunction> fragmentFunc = [library newFunctionWithName:@"fragment_main"];

  if (!vertexFunc || !fragmentFunc) {
    NSLog(@"ERROR: Could not find vertex or fragment function in shader");
    return NO;
  }

  // Configure the render pipeline
  MTLRenderPipelineDescriptor *pipelineDesc =
      [[MTLRenderPipelineDescriptor alloc] init];
  pipelineDesc.vertexFunction = vertexFunc;
  pipelineDesc.fragmentFunction = fragmentFunc;

  // Set the output pixel format to match the view
  pipelineDesc.colorAttachments[0].pixelFormat = _view.colorPixelFormat;

  // Configure alpha blending for transparent particles
  // This blends new pixels with existing pixels using alpha values
  pipelineDesc.colorAttachments[0].blendingEnabled = YES;
  pipelineDesc.colorAttachments[0].sourceRGBBlendFactor =
      MTLBlendFactorSourceAlpha;
  pipelineDesc.colorAttachments[0].destinationRGBBlendFactor =
      MTLBlendFactorOneMinusSourceAlpha;
  pipelineDesc.colorAttachments[0].rgbBlendOperation = MTLBlendOperationAdd;
  pipelineDesc.colorAttachments[0].sourceAlphaBlendFactor = MTLBlendFactorOne;
  pipelineDesc.colorAttachments[0].destinationAlphaBlendFactor =
      MTLBlendFactorOneMinusSourceAlpha;
  pipelineDesc.colorAttachments[0].alphaBlendOperation = MTLBlendOperationAdd;

  _renderPipeline = [_device newRenderPipelineStateWithDescriptor:pipelineDesc
                                                            error:&error];
  if (!_renderPipeline) {
    NSLog(@"ERROR: Failed to create render pipeline: %@",
          error.localizedDescription);
    return NO;
  }

  return YES;
}

// -----------------------------------------------------------------------------
// Buffer Initialization
// -----------------------------------------------------------------------------
- (void)initBuffers {
  const int numParticles = Config::NUM_PARTICLES;
  const int densitySize = Config::DENSITY_RES * Config::DENSITY_RES;

  // Particle state buffers
  _positionBuffer =
      [_device newBufferWithLength:numParticles * sizeof(simd_float2)
                           options:MTLResourceStorageModeShared];
  _velocityBuffer =
      [_device newBufferWithLength:numParticles * sizeof(simd_float2)
                           options:MTLResourceStorageModeShared];
  _colorBuffer =
      [_device newBufferWithLength:numParticles * sizeof(simd_float3)
                           options:MTLResourceStorageModeShared];

  // Density map buffer
  _densityMapBuffer =
      [_device newBufferWithLength:densitySize * sizeof(float)
                           options:MTLResourceStorageModeShared];

  // Neural network weight buffers
  // Layer 1: (64+1) × 128 = 8320 weights
  _weightsL1Buffer =
      [_device newBufferWithLength:8320 * sizeof(float)
                           options:MTLResourceStorageModeShared];
  _biasesL1Buffer =
      [_device newBufferWithLength:128 * sizeof(float)
                           options:MTLResourceStorageModeShared];

  // Layer 2: 128 × 128 = 16384 weights
  _weightsL2Buffer =
      [_device newBufferWithLength:16384 * sizeof(float)
                           options:MTLResourceStorageModeShared];
  _biasesL2Buffer =
      [_device newBufferWithLength:128 * sizeof(float)
                           options:MTLResourceStorageModeShared];

  // Layer 3: 128 × 2 = 256 weights
  _weightsL3Buffer =
      [_device newBufferWithLength:256 * sizeof(float)
                           options:MTLResourceStorageModeShared];
  _biasesL3Buffer =
      [_device newBufferWithLength:2 * sizeof(float)
                           options:MTLResourceStorageModeShared];

  // Fourier frequencies: 32 frequency pairs
  _fourierFreqsBuffer =
      [_device newBufferWithLength:32 * sizeof(simd_float2)
                           options:MTLResourceStorageModeShared];

  // Intermediate force storage
  _forceBuffer =
      [_device newBufferWithLength:numParticles * sizeof(simd_float2)
                           options:MTLResourceStorageModeShared];

  // Parameters buffer
  _paramsBuffer = [_device newBufferWithLength:sizeof(SimParams)
                                       options:MTLResourceStorageModeShared];

  // Initialize all data
  [self initParticles];
  [self initNeuralNetwork];
  [self updateParams];

  // Initial mutation for interesting startup behavior
  [self mutateNeuralField];
  NSLog(@"Initial neural field mutation applied");
}

// Initialize particle positions, velocities, and colors
- (void)initParticles {
  simd_float2 *positions = (simd_float2 *)_positionBuffer.contents;
  simd_float2 *velocities = (simd_float2 *)_velocityBuffer.contents;
  simd_float3 *colors = (simd_float3 *)_colorBuffer.contents;

  for (int i = 0; i < Config::NUM_PARTICLES; i++) {
    // Random position in normalized coordinates [-1, 1]
    float x = (float)arc4random() / UINT32_MAX * 2.0f - 1.0f;
    float y = (float)arc4random() / UINT32_MAX * 2.0f - 1.0f;
    positions[i] = simd_make_float2(x, y);

    // Start with zero velocity
    velocities[i] = simd_make_float2(0.0f, 0.0f);

    // Random initial color
    float r = (float)arc4random() / UINT32_MAX;
    float g = (float)arc4random() / UINT32_MAX;
    float b = (float)arc4random() / UINT32_MAX;
    colors[i] = simd_make_float3(r, g, b);
  }
}

// Initialize neural network weights and Fourier frequencies
- (void)initNeuralNetwork {
  // Initialize Fourier frequencies: random integers 1-8 times π
  simd_float2 *freqs = (simd_float2 *)_fourierFreqsBuffer.contents;
  for (int i = 0; i < 32; i++) {
    int fx = 1 + (arc4random() % 8);
    int fy = 1 + (arc4random() % 8);
    freqs[i] = simd_make_float2(fx * M_PI, fy * M_PI);
  }

  // Initialize all weights with small random values (orthogonal-like init)
  // Using gain=0.1 for stability (like the Python version)
  float *w1 = (float *)_weightsL1Buffer.contents;
  float *b1 = (float *)_biasesL1Buffer.contents;
  float *w2 = (float *)_weightsL2Buffer.contents;
  float *b2 = (float *)_biasesL2Buffer.contents;
  float *w3 = (float *)_weightsL3Buffer.contents;
  float *b3 = (float *)_biasesL3Buffer.contents;

  const float scale = 0.1f;

  // Layer 1: (64+1) × 128
  for (int i = 0; i < 8320; i++) {
    w1[i] = ((float)arc4random() / UINT32_MAX * 2.0f - 1.0f) * scale;
  }
  for (int i = 0; i < 128; i++) {
    b1[i] = 0.0f;
  }

  // Layer 2: 128 × 128
  for (int i = 0; i < 16384; i++) {
    w2[i] = ((float)arc4random() / UINT32_MAX * 2.0f - 1.0f) * scale;
  }
  for (int i = 0; i < 128; i++) {
    b2[i] = 0.0f;
  }

  // Layer 3: 128 × 2
  for (int i = 0; i < 256; i++) {
    w3[i] = ((float)arc4random() / UINT32_MAX * 2.0f - 1.0f) * scale;
  }
  for (int i = 0; i < 2; i++) {
    b3[i] = 0.0f;
  }

  NSLog(@"Neural network initialized");
}

// Update simulation parameters
- (void)updateParams {
  SimParams *params = (SimParams *)_paramsBuffer.contents;

  params->width = Config::WINDOW_WIDTH;
  params->height = Config::WINDOW_HEIGHT;
  params->friction = Config::FRICTION;
  params->neural_strength = Config::NEURAL_STRENGTH;
  params->jitter_strength = Config::JITTER_STRENGTH;
  params->num_particles = Config::NUM_PARTICLES;
  params->density_res = Config::DENSITY_RES;
  params->point_size = Config::POINT_SIZE;
  params->mouse_pos = _mousePos;
  params->mouse_strength = _mouseStrength;
}

// Helper: Add Gaussian noise to a buffer
- (void)addNoiseToBuffer:(id<MTLBuffer>)buffer count:(int)count scale:(float)scale {
  float *data = (float *)buffer.contents;
  for (int i = 0; i < count; i++) {
    // Simple Gaussian approximation using Box-Muller transform
    float u1 = (float)arc4random() / UINT32_MAX;
    float u2 = (float)arc4random() / UINT32_MAX;
    float noise = sqrtf(-2.0f * logf(u1 + 1e-10f)) * cosf(2.0f * M_PI * u2);
    data[i] += noise * scale;
  }
}

// -----------------------------------------------------------------------------
// User Actions
// -----------------------------------------------------------------------------

- (void)mutateNeuralField {
  // Balanced mutation: 1.5x for noticeable changes without destroying structure
  const float rate = Config::MUTATION_RATE * 1.5f;

  // Mutate all weight layers
  [self addNoiseToBuffer:_weightsL1Buffer count:8320 scale:rate];
  [self addNoiseToBuffer:_weightsL2Buffer count:16384 scale:rate];
  [self addNoiseToBuffer:_weightsL3Buffer count:256 scale:rate];

  // Mutate biases more gently to preserve stability
  [self addNoiseToBuffer:_biasesL1Buffer count:128 scale:rate * 0.3f];
  [self addNoiseToBuffer:_biasesL2Buffer count:128 scale:rate * 0.3f];
  [self addNoiseToBuffer:_biasesL3Buffer count:2 scale:rate * 0.3f];

  // Occasionally mutate 2-3 Fourier frequencies for variety (not every time)
  if (arc4random() % 3 == 0) {  // ~33% chance
    simd_float2 *freqs = (simd_float2 *)_fourierFreqsBuffer.contents;
    int numToChange = 2 + (arc4random() % 2);  // 2-3 frequencies
    for (int i = 0; i < numToChange; i++) {
      int idx = arc4random() % 32;
      int fx = 1 + (arc4random() % 8);
      int fy = 1 + (arc4random() % 8);
      freqs[idx] = simd_make_float2(fx * M_PI, fy * M_PI);
    }
  }

  NSLog(@"Neural field mutated!");
}

- (void)reset {
  [self initParticles];
  NSLog(@"Particles reset!");
}

- (void)togglePause {
  _paused = !_paused;
  NSLog(@"%@", _paused ? @"Paused" : @"Resumed");
}

// -----------------------------------------------------------------------------
// MTKViewDelegate - Render Loop
// -----------------------------------------------------------------------------
// This method is called every frame by MTKView.
// It's where we submit GPU commands for simulation and rendering.

- (void)drawInMTKView:(MTKView *)view {
  // Update FPS counter
  [self updateFPS];

  // Create a command buffer
  // -----------------------
  // A command buffer holds encoded commands that will be sent to the GPU.
  // Commands are encoded, then committed (sent to GPU).
  id<MTLCommandBuffer> commandBuffer = [_commandQueue commandBuffer];
  if (!commandBuffer)
    return;

  // -----------------------------------------------------
  // COMPUTE PASSES: 3-stage neural particle simulation
  // -----------------------------------------------------
  if (!_paused) {
    // Update mouse parameters before compute
    [self updateParams];

    id<MTLComputeCommandEncoder> computeEncoder =
        [commandBuffer computeCommandEncoder];

    const NSUInteger threadsPerGroup = 256;
    const MTLSize gridSize = MTLSizeMake(Config::NUM_PARTICLES, 1, 1);
    const MTLSize groupSize = MTLSizeMake(threadsPerGroup, 1, 1);

    // KERNEL 1: Compute density map
    // ------------------------------
    [computeEncoder setComputePipelineState:_densityPipeline];
    [computeEncoder setBuffer:_densityMapBuffer offset:0 atIndex:0];
    [computeEncoder setBuffer:_positionBuffer offset:0 atIndex:1];
    [computeEncoder setBuffer:_paramsBuffer offset:0 atIndex:2];
    [computeEncoder dispatchThreads:gridSize threadsPerThreadgroup:groupSize];

    // KERNEL 2: Compute neural forces
    // --------------------------------
    [computeEncoder setComputePipelineState:_neuralPipeline];
    [computeEncoder setBuffer:_positionBuffer offset:0 atIndex:0];
    [computeEncoder setBuffer:_densityMapBuffer offset:0 atIndex:1];
    [computeEncoder setBuffer:_fourierFreqsBuffer offset:0 atIndex:2];
    [computeEncoder setBuffer:_weightsL1Buffer offset:0 atIndex:3];
    [computeEncoder setBuffer:_biasesL1Buffer offset:0 atIndex:4];
    [computeEncoder setBuffer:_weightsL2Buffer offset:0 atIndex:5];
    [computeEncoder setBuffer:_biasesL2Buffer offset:0 atIndex:6];
    [computeEncoder setBuffer:_weightsL3Buffer offset:0 atIndex:7];
    [computeEncoder setBuffer:_biasesL3Buffer offset:0 atIndex:8];
    [computeEncoder setBuffer:_forceBuffer offset:0 atIndex:9];
    [computeEncoder setBuffer:_paramsBuffer offset:0 atIndex:10];
    [computeEncoder dispatchThreads:gridSize threadsPerThreadgroup:groupSize];

    // KERNEL 3: Update physics (velocity, position, color)
    // -----------------------------------------------------
    [computeEncoder setComputePipelineState:_physicsPipeline];
    [computeEncoder setBuffer:_positionBuffer offset:0 atIndex:0];
    [computeEncoder setBuffer:_velocityBuffer offset:0 atIndex:1];
    [computeEncoder setBuffer:_colorBuffer offset:0 atIndex:2];
    [computeEncoder setBuffer:_densityMapBuffer offset:0 atIndex:3];
    [computeEncoder setBuffer:_forceBuffer offset:0 atIndex:4];
    [computeEncoder setBuffer:_paramsBuffer offset:0 atIndex:5];
    [computeEncoder dispatchThreads:gridSize threadsPerThreadgroup:groupSize];

    [computeEncoder endEncoding];
  }

  // -----------------------------------------------------
  // RENDER PASS: Draw particles
  // -----------------------------------------------------
  // Get the current drawable (the texture we'll draw to)
  MTLRenderPassDescriptor *renderPassDesc = view.currentRenderPassDescriptor;
  if (!renderPassDesc)
    return;

  // A render command encoder encodes rendering commands
  id<MTLRenderCommandEncoder> renderEncoder =
      [commandBuffer renderCommandEncoderWithDescriptor:renderPassDesc];

  // Set the render pipeline
  [renderEncoder setRenderPipelineState:_renderPipeline];

  // Bind buffers for vertex shader
  // Per-particle colors now (not per-type)
  [renderEncoder setVertexBuffer:_positionBuffer offset:0 atIndex:0];
  [renderEncoder setVertexBuffer:_colorBuffer offset:0 atIndex:1];
  [renderEncoder setVertexBuffer:_paramsBuffer offset:0 atIndex:2];

  // Draw particles as points
  [renderEncoder drawPrimitives:MTLPrimitiveTypePoint
                    vertexStart:0
                    vertexCount:Config::NUM_PARTICLES];

  // End the render pass
  [renderEncoder endEncoding];

  // Present the drawable (display the rendered image)
  [commandBuffer presentDrawable:view.currentDrawable];

  // Commit the command buffer (send to GPU)
  [commandBuffer commit];
}

// Called when the view is about to resize (required by MTKViewDelegate)
- (void)mtkView:(nonnull MTKView *)view drawableSizeWillChange:(CGSize)size {
  // Could update parameters here if needed for responsive resizing
  (void)view;
  (void)size;
}

// -----------------------------------------------------------------------------
// FPS Tracking
// -----------------------------------------------------------------------------
- (void)updateFPS {
  _frameCount++;
  double currentTime = CACurrentMediaTime();

  // Update FPS display every second
  if (currentTime - _fpsUpdateTime >= 1.0) {
    double fps = _frameCount / (currentTime - _fpsUpdateTime);

    // Update window title with FPS
    dispatch_async(dispatch_get_main_queue(), ^{
      NSString *title =
          [NSString stringWithFormat:@"Metal Neural Aura - %.1f FPS", fps];
      [self->_view.window setTitle:title];
    });

    _frameCount = 0;
    _fpsUpdateTime = currentTime;
  }
}

@end

// =============================================================================
// MAIN FUNCTION - Application Entry Point
// =============================================================================
// This sets up the macOS application, creates the window, and starts the
// event loop. This is the minimal setup for a Cocoa application without
// using Interface Builder or storyboards.

int main(int argc, const char *argv[]) {
  (void)argc;
  (void)argv;

  // Create an autorelease pool
  // --------------------------
  // @autoreleasepool manages memory for Objective-C objects.
  // Objects marked for autorelease are freed when the pool is drained.
  @autoreleasepool {
    // Create the application object
    // ------------------------------
    // NSApplication is the central object for a Cocoa app.
    // sharedApplication creates/returns the singleton instance.
    NSApplication *app = [NSApplication sharedApplication];

    // Set activation policy
    // ---------------------
    // This makes the app a regular GUI application (appears in Dock, etc.)
    [app setActivationPolicy:NSApplicationActivationPolicyRegular];

    // Create and set the app delegate
    AppDelegate *delegate = [[AppDelegate alloc] init];
    [app setDelegate:delegate];

    // Create the window
    // -----------------
    NSRect frame =
        NSMakeRect(0, 0, Config::WINDOW_WIDTH, Config::WINDOW_HEIGHT);

    // Window style mask defines the window's appearance and behavior
    NSWindowStyleMask style =
        NSWindowStyleMaskTitled           // Has a title bar
        | NSWindowStyleMaskClosable       // Has a close button
        | NSWindowStyleMaskMiniaturizable // Has a minimize button
        | NSWindowStyleMaskResizable;     // Can be resized

    NSWindow *window =
        [[NSWindow alloc] initWithContentRect:frame
                                    styleMask:style
                                      backing:NSBackingStoreBuffered
                                        defer:NO];

    [window setTitle:@"Metal Neural Aura"];
    [window center]; // Center on screen

    // Create the Metal view
    // ---------------------
    InputView *metalView = [[InputView alloc] initWithFrame:frame];

    // Set the clear color (background)
    metalView.clearColor = MTLClearColorMake(Config::BG_RED, Config::BG_GREEN,
                                             Config::BG_BLUE, Config::BG_ALPHA);

    // Set the target frame rate
    metalView.preferredFramesPerSecond = Config::TARGET_FPS;

    // Create the renderer
    // -------------------
    Renderer *renderer = [[Renderer alloc] initWithMetalKitView:metalView];
    if (!renderer) {
      NSLog(@"ERROR: Failed to create renderer");
      return 1;
    }

    // Connect the renderer to the view
    metalView.renderer = renderer;

    // Set the view as the window's content
    [window setContentView:metalView];

    // Make the window visible and key (receives keyboard input)
    [window makeKeyAndOrderFront:nil];

    // Make the view the first responder (receives keyboard events)
    [window makeFirstResponder:metalView];

    // Activate the application
    [app activateIgnoringOtherApps:YES];

    // Print usage instructions
    NSLog(@"Metal Neural Aura - Neural Particle Field Simulation");
    NSLog(@"Controls:");
    NSLog(@"  Q - Quit");
    NSLog(@"  R - Reset particles");
    NSLog(@"  Space - Mutate neural network");
    NSLog(@"  P - Pause/Resume");

    // Start the event loop
    // --------------------
    // This enters the main run loop and doesn't return until the app quits.
    // The run loop processes events (mouse, keyboard, etc.) and triggers
    // the render loop through MTKView.
    [app run];
  }

  return 0;
}
