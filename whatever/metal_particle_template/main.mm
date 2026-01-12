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
constexpr float WINDOW_WIDTH = 1024.0f;
constexpr float WINDOW_HEIGHT = 768.0f;

// Particle system settings
constexpr int NUM_PARTICLES = 20000; // Total number of particles
constexpr int NUM_TYPES = 4;         // Number of particle types (colors)

// Physics settings
constexpr float INTERACTION_RADIUS =
    80.0f; // Max distance for particle interaction
constexpr float FORCE_STRENGTH =
    0.01f;                        // How strongly particles affect each other
constexpr float FRICTION = 0.80f; // Velocity damping (1.0 = no friction)

// Rendering settings
constexpr float POINT_SIZE = 3.0f; // Particle size in pixels
constexpr int TARGET_FPS = 60;     // Target frame rate

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
  float width;  // Canvas width
  float height; // Canvas height
  float interaction_radius;
  float force_strength;
  float friction;
  int num_particles;
  int num_types;
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
    // Randomize the simulation (delegate to renderer)
    if (_renderer) {
      [_renderer randomize];
    }
    break;

  case ' ':
    // Toggle pause (delegate to renderer)
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

  id<MTLBuffer> _positionBuffer; // Particle positions (float2 array)
  id<MTLBuffer> _velocityBuffer; // Particle velocities (float2 array)
  id<MTLBuffer> _typeBuffer;     // Particle types (int array)
  id<MTLBuffer> _colorBuffer;    // Colors per type (float4 array)
  id<MTLBuffer> _matrixBuffer;   // Interaction matrix (NUM_TYPES x NUM_TYPES)
  id<MTLBuffer> _paramsBuffer;   // Simulation parameters

  // -----------------------------------------------------
  // State
  // -----------------------------------------------------
  MTKView *_view;
  BOOL _paused;

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
  const int numTypes = Config::NUM_TYPES;

  // Position buffer: float2 (x, y) per particle
  // MTLResourceStorageModeShared means the buffer is accessible by both CPU and
  // GPU
  _positionBuffer =
      [_device newBufferWithLength:numParticles * sizeof(simd_float2)
                           options:MTLResourceStorageModeShared];

  // Velocity buffer: float2 (vx, vy) per particle
  _velocityBuffer =
      [_device newBufferWithLength:numParticles * sizeof(simd_float2)
                           options:MTLResourceStorageModeShared];

  // Type buffer: int per particle
  _typeBuffer = [_device newBufferWithLength:numParticles * sizeof(int)
                                     options:MTLResourceStorageModeShared];

  // Color buffer: float4 (RGBA) per type
  _colorBuffer = [_device newBufferWithLength:numTypes * sizeof(simd_float4)
                                      options:MTLResourceStorageModeShared];

  // Interaction matrix: numTypes x numTypes floats
  // Determines how particles of different types interact
  _matrixBuffer =
      [_device newBufferWithLength:numTypes * numTypes * sizeof(float)
                           options:MTLResourceStorageModeShared];

  // Parameters buffer
  _paramsBuffer = [_device newBufferWithLength:sizeof(SimParams)
                                       options:MTLResourceStorageModeShared];

  // Initialize particle data
  [self initParticles];

  // Initialize colors
  [self initColors];

  // Initialize interaction matrix
  [self randomizeMatrix];

  // Initialize parameters
  [self updateParams];
}

// Initialize particle positions, velocities, and types
- (void)initParticles {
  simd_float2 *positions = (simd_float2 *)_positionBuffer.contents;
  simd_float2 *velocities = (simd_float2 *)_velocityBuffer.contents;
  int *types = (int *)_typeBuffer.contents;

  for (int i = 0; i < Config::NUM_PARTICLES; i++) {
    // Random position within the window
    float x = (float)arc4random() / UINT32_MAX * Config::WINDOW_WIDTH;
    float y = (float)arc4random() / UINT32_MAX * Config::WINDOW_HEIGHT;
    positions[i] = simd_make_float2(x, y);

    // Start with zero velocity
    velocities[i] = simd_make_float2(0.0f, 0.0f);

    // Random type
    types[i] = arc4random() % Config::NUM_TYPES;
  }
}

// Define colors for each particle type
- (void)initColors {
  simd_float4 *colors = (simd_float4 *)_colorBuffer.contents;

  // Define distinct colors for each type
  // Format: (R, G, B, A) where values are 0.0-1.0
  simd_float4 palette[] = {
      simd_make_float4(0.0f, 0.9f, 0.9f, 1.0f), // Cyan
      simd_make_float4(0.9f, 0.1f, 0.6f, 1.0f), // Magenta
      simd_make_float4(0.9f, 0.9f, 0.0f, 1.0f), // Yellow
      simd_make_float4(0.2f, 0.9f, 0.2f, 1.0f), // Green
      simd_make_float4(0.9f, 0.5f, 0.0f, 1.0f), // Orange
      simd_make_float4(0.6f, 0.3f, 0.9f, 1.0f), // Purple
  };

  for (int i = 0; i < Config::NUM_TYPES; i++) {
    colors[i] = palette[i % 6];
  }
}

// Generate random interaction matrix
- (void)randomizeMatrix {
  float *matrix = (float *)_matrixBuffer.contents;

  // Each entry determines how type A affects type B
  // Positive = attraction, Negative = repulsion
  for (int i = 0; i < Config::NUM_TYPES * Config::NUM_TYPES; i++) {
    // Random value between -1.0 and 1.0
    matrix[i] = (float)arc4random() / UINT32_MAX * 2.0f - 1.0f;
  }

  NSLog(@"Interaction matrix randomized");
}

// Update simulation parameters
- (void)updateParams {
  SimParams *params = (SimParams *)_paramsBuffer.contents;

  params->width = Config::WINDOW_WIDTH;
  params->height = Config::WINDOW_HEIGHT;
  params->interaction_radius = Config::INTERACTION_RADIUS;
  params->force_strength = Config::FORCE_STRENGTH;
  params->friction = Config::FRICTION;
  params->num_particles = Config::NUM_PARTICLES;
  params->num_types = Config::NUM_TYPES;
  params->point_size = Config::POINT_SIZE;
}

// -----------------------------------------------------------------------------
// User Actions
// -----------------------------------------------------------------------------

- (void)randomize {
  [self randomizeMatrix];
  [self initParticles];
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
  // COMPUTE PASS: Update particle physics
  // -----------------------------------------------------
  if (!_paused) {
    // A compute command encoder encodes compute (non-rendering) commands
    id<MTLComputeCommandEncoder> computeEncoder =
        [commandBuffer computeCommandEncoder];

    // Set the compute pipeline (which kernel to run)
    [computeEncoder setComputePipelineState:_computePipeline];

    // Bind buffers to the kernel
    // The indices (0, 1, 2, ...) must match the buffer indices in the shader
    [computeEncoder setBuffer:_positionBuffer offset:0 atIndex:0];
    [computeEncoder setBuffer:_velocityBuffer offset:0 atIndex:1];
    [computeEncoder setBuffer:_typeBuffer offset:0 atIndex:2];
    [computeEncoder setBuffer:_matrixBuffer offset:0 atIndex:3];
    [computeEncoder setBuffer:_paramsBuffer offset:0 atIndex:4];

    // Dispatch the compute kernel
    // ---------------------------
    // Metal organizes GPU threads into "threadgroups".
    // Each thread processes one particle.
    NSUInteger threadsPerGroup = _computePipeline.maxTotalThreadsPerThreadgroup;
    if (threadsPerGroup > 256)
      threadsPerGroup = 256; // Reasonable default

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
  [renderEncoder setVertexBuffer:_typeBuffer offset:0 atIndex:1];
  [renderEncoder setVertexBuffer:_colorBuffer offset:0 atIndex:2];
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
    NSLog(@"Metal Particle Template");
    NSLog(@"Controls:");
    NSLog(@"  Q - Quit");
    NSLog(@"  R - Randomize particles and interaction matrix");
    NSLog(@"  Space - Pause/Resume");

    // Start the event loop
    // --------------------
    // This enters the main run loop and doesn't return until the app quits.
    // The run loop processes events (mouse, keyboard, etc.) and triggers
    // the render loop through MTKView.
    [app run];
  }

  return 0;
}
