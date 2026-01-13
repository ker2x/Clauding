// =============================================================================
// Metal Particle Template - main.mm
// =============================================================================
// This is the main source file for a Metal-based particle simulation.
// It demonstrates the fundamental patterns for building GPU-accelerated
// graphical applications on macOS using Metal and Objective-C.
//
// File Extension: .mm indicates Objective-C++ (mix of Objective-C and C++)
//
// Architecture Overview:
// ----------------------
// 1. InputView    - Custom MTKView subclass for handling keyboard input
// 2. AppDelegate  - Application lifecycle management
// 3. Renderer     - Metal rendering and compute pipeline management
// 4. main()       - Application entry point and window setup
//
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
// Modify these constants to customize the simulation.

namespace Config {
// Window dimensions
constexpr float WINDOW_WIDTH = 1280.0f;
constexpr float WINDOW_HEIGHT = 900.0f;

// Particle system settings
constexpr int NUM_PARTICLES = 50000;      // More particles for lush flowers
constexpr int MAX_BLOOMS = 8;             // Maximum bloom centers
constexpr int DEFAULT_PETALS = 7;         // Default petals per flower
constexpr float PARTICLE_LIFETIME = 180.0f; // Frames before respawn

// Chromatic bloom physics
constexpr float CHROMATIC_STRENGTH = 0.4f;  // Hue dispersion intensity
constexpr float RADIAL_FORCE = 0.02f;     // Outward bloom force
constexpr float PETAL_ATTRACTION = 0.15f;  // Angular correction strength
constexpr float FRICTION = 0.96f;          // Velocity damping

// Rendering settings
constexpr float POINT_SIZE = 20.0f;       // Particle size in pixels (larger for glow)
constexpr int TARGET_FPS = 60;            // Target frame rate

// Background color (RGBA, 0.0-1.0) - dark for glow effect
constexpr float BG_RED = 0.01f;
constexpr float BG_GREEN = 0.01f;
constexpr float BG_BLUE = 0.03f;
constexpr float BG_ALPHA = 1.0f;
} // namespace Config

// =============================================================================
// DATA STRUCTURES
// =============================================================================
// These structures are shared between CPU (this file) and GPU (shader).
// They MUST match exactly with the definitions in Compute.metal.

// Simulation parameters passed to the GPU
struct SimParams {
  float width;                // Canvas width
  float height;               // Canvas height
  float time;                 // Animation time
  int num_particles;
  int num_blooms;             // Number of active blooms
  int num_petals;             // Petals per bloom
  float chromatic_strength;   // Chromatic dispersion intensity
  float radial_force;         // Outward bloom force
  float petal_attraction;     // Angular alignment strength
  float friction;             // Velocity damping
  float particle_lifetime;    // Respawn age threshold
  float point_size;
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
- (void)randomize;
- (void)togglePause;
- (void)cyclePetalCount;
- (void)cycleChromaticStrength;
- (void)setNumBlooms:(int)count;
- (void)addBloomAtPosition:(simd_float2)position;
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
    [NSApp terminate:nil];
    break;

  case 'r':
  case 'R':
    if (_renderer) {
      [_renderer randomize];
    }
    break;

  case ' ':
    if (_renderer) {
      [_renderer togglePause];
    }
    break;

  case 'p':
  case 'P':
    if (_renderer) {
      [_renderer cyclePetalCount];
    }
    break;

  case 'c':
  case 'C':
    if (_renderer) {
      [_renderer cycleChromaticStrength];
    }
    break;

  case '1':
  case '2':
  case '3':
  case '4':
  case '5':
    if (_renderer) {
      [_renderer setNumBlooms:(key - '0')];
    }
    break;

  default:
    [super keyDown:event];
    break;
  }
}

// Handle mouse clicks to add new blooms
- (void)mouseDown:(NSEvent *)event {
  if (_renderer) {
    // Convert click location to view coordinates
    NSPoint location = [self convertPoint:[event locationInWindow] fromView:nil];
    // Flip Y coordinate (AppKit has origin at bottom-left, we want top-left)
    float y = self.bounds.size.height - location.y;
    [_renderer addBloomAtPosition:simd_make_float2(location.x, y)];
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
- (void)randomize;
- (void)togglePause;

@end

@implementation Renderer {
  // -----------------------------------------------------
  // Metal Core Objects
  // -----------------------------------------------------
  // These are the fundamental objects needed for any Metal application.

  id<MTLDevice> _device;             // Represents the GPU
  id<MTLCommandQueue> _commandQueue; // Queue for submitting work to GPU

  // -----------------------------------------------------
  // Pipeline States
  // -----------------------------------------------------
  // Pipeline states are compiled shader programs.

  id<MTLComputePipelineState> _computePipeline; // For physics simulation
  id<MTLRenderPipelineState> _renderPipeline;   // For drawing particles

  // -----------------------------------------------------
  // GPU Buffers
  // -----------------------------------------------------
  // Buffers hold data accessible by both CPU and GPU.

  id<MTLBuffer> _positionBuffer;       // Particle positions (float2 array)
  id<MTLBuffer> _velocityBuffer;       // Particle velocities (float2 array)
  id<MTLBuffer> _hueBuffer;            // Particle hues (float array, 0.0-1.0)
  id<MTLBuffer> _ageBuffer;            // Particle ages (float array)
  id<MTLBuffer> _bloomCentersBuffer;   // Bloom center positions (float2 array)
  id<MTLBuffer> _bloomActiveBuffer;    // Bloom active flags (int array)
  id<MTLBuffer> _paramsBuffer;         // Simulation parameters

  // -----------------------------------------------------
  // State
  // -----------------------------------------------------
  MTKView *_view;
  BOOL _paused;
  int _numActiveBlooms;                // Current number of active blooms
  int _numPetals;                      // Current petal count
  float _chromaticStrength;            // Current chromatic strength
  float _frameCounter;                 // Animation time counter

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

  // Create compute pipeline
  // -----------------------
  // A compute pipeline runs a kernel function on the GPU.
  id<MTLFunction> computeFunc =
      [library newFunctionWithName:@"update_particles"];
  if (!computeFunc) {
    NSLog(@"ERROR: Could not find 'update_particles' function in shader");
    return NO;
  }

  _computePipeline = [_device newComputePipelineStateWithFunction:computeFunc
                                                            error:&error];
  if (!_computePipeline) {
    NSLog(@"ERROR: Failed to create compute pipeline: %@",
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
  const int maxBlooms = Config::MAX_BLOOMS;

  // Particle buffers
  _positionBuffer =
      [_device newBufferWithLength:numParticles * sizeof(simd_float2)
                           options:MTLResourceStorageModeShared];
  _velocityBuffer =
      [_device newBufferWithLength:numParticles * sizeof(simd_float2)
                           options:MTLResourceStorageModeShared];
  _hueBuffer = [_device newBufferWithLength:numParticles * sizeof(float)
                                    options:MTLResourceStorageModeShared];
  _ageBuffer = [_device newBufferWithLength:numParticles * sizeof(float)
                                    options:MTLResourceStorageModeShared];

  // Bloom system buffers
  _bloomCentersBuffer =
      [_device newBufferWithLength:maxBlooms * sizeof(simd_float2)
                           options:MTLResourceStorageModeShared];
  _bloomActiveBuffer = [_device newBufferWithLength:maxBlooms * sizeof(int)
                                             options:MTLResourceStorageModeShared];

  // Parameters buffer
  _paramsBuffer = [_device newBufferWithLength:sizeof(SimParams)
                                       options:MTLResourceStorageModeShared];

  // Initialize state
  _numActiveBlooms = 5;  // Start with pentagon pattern
  _numPetals = Config::DEFAULT_PETALS;
  _chromaticStrength = Config::CHROMATIC_STRENGTH;
  _frameCounter = 0.0f;

  // Initialize bloom centers
  [self initBloomCenters];

  // Initialize particles at bloom centers
  [self initParticles];

  // Initialize parameters
  [self updateParams];
}

// Initialize bloom centers in pentagon pattern
- (void)initBloomCenters {
  simd_float2 *centers = (simd_float2 *)_bloomCentersBuffer.contents;
  int *active = (int *)_bloomActiveBuffer.contents;

  float cx = Config::WINDOW_WIDTH * 0.5f;
  float cy = Config::WINDOW_HEIGHT * 0.5f;
  float radius = 200.0f;

  // Pentagon arrangement (5 blooms)
  for (int i = 0; i < 5; i++) {
    float angle = (i * 2.0f * M_PI / 5.0f) - M_PI_2;
    centers[i] = simd_make_float2(
        cx + cosf(angle) * radius,
        cy + sinf(angle) * radius
    );
    active[i] = 1;
  }

  // Rest inactive
  for (int i = 5; i < Config::MAX_BLOOMS; i++) {
    centers[i] = simd_make_float2(cx, cy);
    active[i] = 0;
  }
}

// Initialize particles at bloom centers with random hues and ages
- (void)initParticles {
  simd_float2 *positions = (simd_float2 *)_positionBuffer.contents;
  simd_float2 *velocities = (simd_float2 *)_velocityBuffer.contents;
  float *hues = (float *)_hueBuffer.contents;
  float *ages = (float *)_ageBuffer.contents;

  simd_float2 *centers = (simd_float2 *)_bloomCentersBuffer.contents;

  for (int i = 0; i < Config::NUM_PARTICLES; i++) {
    // Assign to random active bloom center
    int bloom_idx = arc4random() % _numActiveBlooms;
    simd_float2 center = centers[bloom_idx];

    // Random offset from center
    float angle = (float)arc4random() / UINT32_MAX * 2.0f * M_PI;
    float radius = (float)arc4random() / UINT32_MAX * 10.0f;

    positions[i] = simd_make_float2(
        center.x + cosf(angle) * radius,
        center.y + sinf(angle) * radius
    );
    velocities[i] = simd_make_float2(0.0f, 0.0f);
    hues[i] = (float)arc4random() / UINT32_MAX;  // Random hue 0.0-1.0
    ages[i] = (float)arc4random() / UINT32_MAX * Config::PARTICLE_LIFETIME;  // Stagger ages
  }
}

// Update simulation parameters
- (void)updateParams {
  SimParams *params = (SimParams *)_paramsBuffer.contents;

  params->width = Config::WINDOW_WIDTH;
  params->height = Config::WINDOW_HEIGHT;
  params->time = _frameCounter * 0.016f;  // Assuming 60 FPS
  params->num_particles = Config::NUM_PARTICLES;
  params->num_blooms = _numActiveBlooms;
  params->num_petals = _numPetals;
  params->chromatic_strength = _chromaticStrength;
  params->radial_force = Config::RADIAL_FORCE;
  params->petal_attraction = Config::PETAL_ATTRACTION;
  params->friction = Config::FRICTION;
  params->particle_lifetime = Config::PARTICLE_LIFETIME;
  params->point_size = Config::POINT_SIZE;
}

// -----------------------------------------------------------------------------
// User Actions
// -----------------------------------------------------------------------------

- (void)randomize {
  [self initParticles];
  _frameCounter = 0.0f;
  NSLog(@"Particles reset");
}

- (void)togglePause {
  _paused = !_paused;
  NSLog(@"%@", _paused ? @"Paused" : @"Resumed");
}

- (void)cyclePetalCount {
  // Cycle through 5, 6, 7, 8, 9 petals
  _numPetals++;
  if (_numPetals > 9) {
    _numPetals = 5;
  }
  NSLog(@"Petals: %d", _numPetals);
}

- (void)cycleChromaticStrength {
  // Cycle through 0.2, 0.4, 0.6, 0.8
  _chromaticStrength += 0.2f;
  if (_chromaticStrength > 0.81f) {
    _chromaticStrength = 0.2f;
  }
  NSLog(@"Chromatic strength: %.1f", _chromaticStrength);
}

- (void)setNumBlooms:(int)count {
  if (count < 1 || count > 5) return;
  _numActiveBlooms = count;

  int *active = (int *)_bloomActiveBuffer.contents;
  for (int i = 0; i < Config::MAX_BLOOMS; i++) {
    active[i] = (i < count) ? 1 : 0;
  }

  NSLog(@"Active blooms: %d", count);
}

- (void)addBloomAtPosition:(simd_float2)position {
  if (_numActiveBlooms >= Config::MAX_BLOOMS) {
    NSLog(@"Maximum blooms reached (%d)", Config::MAX_BLOOMS);
    return;
  }

  simd_float2 *centers = (simd_float2 *)_bloomCentersBuffer.contents;
  int *active = (int *)_bloomActiveBuffer.contents;

  // Find first inactive slot
  for (int i = 0; i < Config::MAX_BLOOMS; i++) {
    if (!active[i]) {
      centers[i] = position;
      active[i] = 1;
      _numActiveBlooms++;
      NSLog(@"Bloom added at (%.1f, %.1f)", position.x, position.y);
      return;
    }
  }
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
  // COMPUTE PASS: Update particle physics
  // -----------------------------------------------------
  if (!_paused) {
    // Update animation time
    _frameCounter += 1.0f;
    [self updateParams];

    // A compute command encoder encodes compute (non-rendering) commands
    id<MTLComputeCommandEncoder> computeEncoder =
        [commandBuffer computeCommandEncoder];

    // Set the compute pipeline (which kernel to run)
    [computeEncoder setComputePipelineState:_computePipeline];

    // Bind buffers to the kernel
    [computeEncoder setBuffer:_positionBuffer offset:0 atIndex:0];
    [computeEncoder setBuffer:_velocityBuffer offset:0 atIndex:1];
    [computeEncoder setBuffer:_hueBuffer offset:0 atIndex:2];
    [computeEncoder setBuffer:_ageBuffer offset:0 atIndex:3];
    [computeEncoder setBuffer:_bloomCentersBuffer offset:0 atIndex:4];
    [computeEncoder setBuffer:_bloomActiveBuffer offset:0 atIndex:5];
    [computeEncoder setBuffer:_paramsBuffer offset:0 atIndex:6];

    // Dispatch the compute kernel
    NSUInteger threadsPerGroup = _computePipeline.maxTotalThreadsPerThreadgroup;
    if (threadsPerGroup > 256)
      threadsPerGroup = 256;

    MTLSize gridSize = MTLSizeMake(Config::NUM_PARTICLES, 1, 1);
    MTLSize groupSize = MTLSizeMake(threadsPerGroup, 1, 1);

    [computeEncoder dispatchThreads:gridSize threadsPerThreadgroup:groupSize];

    // End the compute pass
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
  [renderEncoder setVertexBuffer:_positionBuffer offset:0 atIndex:0];
  [renderEncoder setVertexBuffer:_hueBuffer offset:0 atIndex:1];
  [renderEncoder setVertexBuffer:_ageBuffer offset:0 atIndex:2];
  [renderEncoder setVertexBuffer:_paramsBuffer offset:0 atIndex:3];

  // Draw particles as points
  // Each particle is one vertex, drawn as a point primitive
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
          [NSString stringWithFormat:@"Metal Particles - %.1f FPS", fps];
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

    [window setTitle:@"Metal Particles"];
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
    NSLog(@"Metal Chromatic Bloom - Emergent Floral Patterns");
    NSLog(@"Controls:");
    NSLog(@"  Q - Quit");
    NSLog(@"  R - Reset particles");
    NSLog(@"  Space - Pause/Resume");
    NSLog(@"  P - Cycle petal count (5-9)");
    NSLog(@"  C - Cycle chromatic strength");
    NSLog(@"  1-5 - Set number of blooms");
    NSLog(@"  Mouse Click - Add bloom at cursor");

    // Start the event loop
    // --------------------
    // This enters the main run loop and doesn't return until the app quits.
    // The run loop processes events (mouse, keyboard, etc.) and triggers
    // the render loop through MTKView.
    [app run];
  }

  return 0;
}
