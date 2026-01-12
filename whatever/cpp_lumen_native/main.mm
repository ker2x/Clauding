#import <AppKit/AppKit.h>
#import <Metal/Metal.h>
#import <MetalKit/MetalKit.h>
#include <algorithm>
#include <cmath>
#include <random>
#include <vector>

// Constants defining the simulation scale.
static const int NUM_PARTICLES = 20000;
static const int NUM_TYPES = 6;
static const float WIDTH = 1400.0f;
static const float HEIGHT = 800.0f;
static const float R_MAX = 80.0f;

// Parameters for the simulation, passed to the GPU.
struct Params {
  float width;
  float height;
  float r_max;
  float force_strength;
  float friction;
  int num_particles;
  int num_types;
};

/**
 * InputView: A custom MetalKit view that handles keyboard input.
 *
 * This view is used to handle keyboard input from the user. It is a subclass of
 * MTKView, which is a MetalKit view that is used to display Metal content.
 *
 * @property renderer A handle to the renderer to trigger actions
 */
@interface InputView : MTKView
@property(nonatomic, assign)
    id renderer; // Handle to the renderer to trigger actions
@end

@implementation InputView
- (BOOL)acceptsFirstResponder {
  return YES; // Allow keyboard focus
}
- (void)keyDown:(NSEvent *)event {
  NSString *chars = [event charactersIgnoringModifiers];
  // Quit on 'q'
  if ([chars isEqualToString:@"q"]) {
    [NSApp terminate:nil]; // Quit
  } else if ([chars isEqualToString:@"r"]) {
    // Randomize the interaction matrix on 'r'
    // The pragma is to suppress warnings about undeclared selectors
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wundeclared-selector"
    if ([self.renderer respondsToSelector:@selector(randomizeMatrix)]) {
      // The selector is not known at compile time, so we use performSelector
      // to call it at runtime.
      [self.renderer performSelector:@selector(randomizeMatrix)];
    }
#pragma clang diagnostic pop
  }
}
@end

/**
 * AppDelegate: Handles application-level lifecycle events.
 */
@interface AppDelegate : NSObject <NSApplicationDelegate>
@end

@implementation AppDelegate
- (BOOL)applicationShouldTerminateAfterLastWindowClosed:
    (NSApplication *)sender {
  return YES; // Exit app when window is closed
}
@end

/**
 * MetalRenderer: The core class that manages Metal resources and the frame
 * loop.
 */
@interface MetalRenderer : NSObject <MTKViewDelegate>
- (void)randomizeMatrix; // Randomize the interaction matrix
@property(nonatomic, assign) NSWindow *window;
@end

@implementation MetalRenderer {
  id<MTLDevice> _device;             // The GPU device
  id<MTLCommandQueue> _commandQueue; // Queue to send commands to the GPU
  id<MTLComputePipelineState>
      _computePipeline;                       // The compiled simulation shader
  id<MTLRenderPipelineState> _renderPipeline; // The compiled rendering shader

  // GPU Memory Buffers
  id<MTLBuffer> _posBuffer;    // Position buffer (float2)
  id<MTLBuffer> _velBuffer;    // Velocity buffer (float2)
  id<MTLBuffer> _typeBuffer;   // Type buffer (int)
  id<MTLBuffer> _matrixBuffer; // Stores the interaction matrix
  id<MTLBuffer> _colorBuffer;  // Stores colors for each particle type
  id<MTLBuffer> _paramsBuffer; // Stores simulation parameters

  NSTimeInterval _lastFPSUpdate; // Last time we updated the FPS
  NSInteger _frameCount;         // Number of frames since last update
}

// Generates a new random interaction matrix and uploads it to the GPU.
- (void)randomizeMatrix {
  std::vector<float> matrix(NUM_TYPES * NUM_TYPES);
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_real_distribution<float> dist_force(-1.0f, 1.0f);

  // Generate a random interaction matrix
  for (int i = 0; i < NUM_TYPES * NUM_TYPES; i++)
    matrix[i] = dist_force(gen);

  // Upload the matrix to the GPU
  memcpy([_matrixBuffer contents], matrix.data(),
         sizeof(float) * matrix.size());
  NSLog(@"Matrix Randomized");
}

/**
 * Initializes the renderer with a MetalKit view.
 */
- (instancetype)initWithMetalKitView:(MTKView *)view {
  self = [super init];

  if (self) {
    _device = view.device;
    _commandQueue = [_device newCommandQueue];

    // Load and compile the Metal shader from "Simulation.metal"
    NSError *error = nil;
    NSString *shaderSource =
        [NSString stringWithContentsOfFile:@"Simulation.metal"
                                  encoding:NSUTF8StringEncoding
                                     error:&error];

    if (!shaderSource)
      NSLog(@"Error loading Simulation.metal: %@", error);

    // Compile the shader
    // The library is a collection of compiled functions that can be used
    // by the compute and render pipelines.
    id<MTLLibrary> library = [_device newLibraryWithSource:shaderSource
                                                   options:nil
                                                     error:&error];

    if (!library)
      NSLog(@"Error compiling shaders: %@", error);

    // Setup the Compute Pipeline (Physics simulation)
    // The compute pipeline is used to run the simulation shader.
    id<MTLFunction> computeFunc =
        [library newFunctionWithName:@"update_particles"];
    _computePipeline = [_device newComputePipelineStateWithFunction:computeFunc
                                                              error:&error];

    // Setup the Render Pipeline (Drawing particles)
    // The render pipeline is used to draw the particles.
    id<MTLFunction> vertexFunc = [library newFunctionWithName:@"vertex_main"];
    id<MTLFunction> fragmentFunc =
        [library newFunctionWithName:@"fragment_main"];

    // The render pipeline descriptor is used to create the render pipeline.
    MTLRenderPipelineDescriptor *renderDesc =
        [[MTLRenderPipelineDescriptor alloc] init];
    renderDesc.vertexFunction = vertexFunc;
    renderDesc.fragmentFunction = fragmentFunc;
    renderDesc.colorAttachments[0].pixelFormat = view.colorPixelFormat;
    renderDesc.colorAttachments[0].blendingEnabled = YES;

    // Standard Alpha Blending (Non-additive)
    renderDesc.colorAttachments[0].sourceRGBBlendFactor =
        MTLBlendFactorSourceAlpha;
    renderDesc.colorAttachments[0].destinationRGBBlendFactor =
        MTLBlendFactorOneMinusSourceAlpha;

    // The render pipeline is used to draw the particles.
    _renderPipeline = [_device newRenderPipelineStateWithDescriptor:renderDesc
                                                              error:&error];

    [self initBuffers]; // Initialize the data
  }
  return self;
}

// Allocate and initialize GPU buffers.
- (void)initBuffers {
  std::vector<float> positions(NUM_PARTICLES * 2);
  std::vector<float> velocities(NUM_PARTICLES * 2);
  std::vector<int> types(NUM_PARTICLES);

  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_real_distribution<float> dist_x(0, WIDTH);
  std::uniform_real_distribution<float> dist_y(0, HEIGHT);
  std::uniform_int_distribution<int> dist_type(0, NUM_TYPES - 1);

  for (int i = 0; i < NUM_PARTICLES; i++) {
    positions[i * 2 + 0] = dist_x(gen);
    positions[i * 2 + 1] = dist_y(gen);
    velocities[i * 2 + 0] = 0;
    velocities[i * 2 + 1] = 0;
    types[i] = dist_type(gen);
  }

  _posBuffer = [_device newBufferWithBytes:positions.data()
                                    length:sizeof(float) * 2 * NUM_PARTICLES
                                   options:MTLResourceStorageModeShared];

  _velBuffer = [_device newBufferWithBytes:velocities.data()
                                    length:sizeof(float) * 2 * NUM_PARTICLES
                                   options:MTLResourceStorageModeShared];

  _typeBuffer = [_device newBufferWithBytes:types.data()
                                     length:sizeof(int) * NUM_PARTICLES
                                    options:MTLResourceStorageModeShared];

  _matrixBuffer =
      [_device newBufferWithLength:sizeof(float) * NUM_TYPES * NUM_TYPES
                           options:MTLResourceStorageModeShared];
  [self randomizeMatrix];

  float colors[] = {
      0.0, 1.0, 1.0, 1.0, // Cyan
      1.0, 0.0, 1.0, 1.0, // Magenta
      1.0, 1.0, 0.0, 1.0, // Yellow
      0.0, 1.0, 0.5, 1.0, // Electric Green
      1.0, 0.5, 0.0, 1.0, // Orange
      0.5, 0.0, 1.0, 1.0  // Purple
  };
  _colorBuffer = [_device newBufferWithBytes:colors
                                      length:sizeof(colors)
                                     options:MTLResourceStorageModeShared];

  // Parameters scaled for 40,000 particles: Lowering force to 0.02f to prevent
  // excessive speed.
  Params params = {WIDTH, HEIGHT,        R_MAX,    0.02f,
                   0.85f, NUM_PARTICLES, NUM_TYPES};
  _paramsBuffer = [_device newBufferWithBytes:&params
                                       length:sizeof(Params)
                                      options:MTLResourceStorageModeShared];
}

/**
 * drawInMTKView: The frame loop (called 60 times per second).
 *
 * This function is called every frame to update the simulation and render the
 * particles.
 */
- (void)drawInMTKView:(nonnull MTKView *)view {
  _frameCount++;
  NSTimeInterval now = [[NSDate date] timeIntervalSince1970];

  // Update the window title with the current FPS
  if (now - _lastFPSUpdate >= 1.0) {
    float fps = _frameCount / (now - _lastFPSUpdate);
    self.window.title = [NSString
        stringWithFormat:
            @"LumenParticles Native (Metal) | FPS: %.1f | Particles: %d", fps,
            NUM_PARTICLES];
    _frameCount = 0;
    _lastFPSUpdate = now;
  }

  // Create a command buffer to send commands to the GPU
  id<MTLCommandBuffer> commandBuffer = [_commandQueue commandBuffer];

  // --- 1. COMPUTE PASS (Physics) ---

  id<MTLComputeCommandEncoder> computeEncoder =
      [commandBuffer computeCommandEncoder];
  [computeEncoder setComputePipelineState:_computePipeline];
  [computeEncoder setBuffer:_posBuffer offset:0 atIndex:0];
  [computeEncoder setBuffer:_velBuffer offset:0 atIndex:1];
  [computeEncoder setBuffer:_typeBuffer offset:0 atIndex:2];
  [computeEncoder setBuffer:_matrixBuffer offset:0 atIndex:3];
  [computeEncoder setBuffer:_paramsBuffer offset:0 atIndex:4];

  // Dispatch threads: Using fixed threadgroup size of 256 for tiling
  // optimization.
  [computeEncoder dispatchThreads:MTLSizeMake(NUM_PARTICLES, 1, 1)
            threadsPerThreadgroup:MTLSizeMake(512, 1, 1)];
  [computeEncoder endEncoding];

  // --- 2. RENDER PASS (Drawing) ---

  // The render pass descriptor is used to create the render pass.
  MTLRenderPassDescriptor *renderPassDescriptor =
      view.currentRenderPassDescriptor;
  if (renderPassDescriptor != nil) {
    id<MTLRenderCommandEncoder> renderEncoder =
        [commandBuffer renderCommandEncoderWithDescriptor:renderPassDescriptor];
    [renderEncoder setRenderPipelineState:_renderPipeline];
    [renderEncoder setVertexBuffer:_posBuffer offset:0 atIndex:0];
    [renderEncoder setVertexBuffer:_typeBuffer offset:0 atIndex:1];
    [renderEncoder setVertexBuffer:_colorBuffer offset:0 atIndex:2];
    [renderEncoder setVertexBuffer:_paramsBuffer offset:0 atIndex:3];

    // Draw each particle as a point.
    [renderEncoder drawPrimitives:MTLPrimitiveTypePoint
                      vertexStart:0
                      vertexCount:NUM_PARTICLES];
    [renderEncoder endEncoding];
    [commandBuffer presentDrawable:view.currentDrawable];
  }

  [commandBuffer commit]; // Send all commands to the GPU.
}

/**
 * mtkView:drawableSizeWillChange:
 *
 * This function is called when the drawable size changes (e.g. when the window
 * is resized).
 */
- (void)mtkView:(nonnull MTKView *)view drawableSizeWillChange:(CGSize)size {
}
@end

/**
 * main
 *
 * The entry point of the application.
 */
int main(int argc, char *argv[]) {
  @autoreleasepool { // autoreleasepool is used to manage memory.
    [NSApplication sharedApplication]; // Initialize the application.
    [NSApp
        setActivationPolicy:
            NSApplicationActivationPolicyRegular]; // Set the activation policy.
                                                   // Regular means the app can
                                                   // be activated by the user.

    AppDelegate *delegate =
        [[AppDelegate alloc] init]; // Initialize the delegate.
    [NSApp setDelegate:delegate];   // Set the delegate.

    NSRect frame = NSMakeRect(0, 0, WIDTH, HEIGHT); // Initialize the frame.
    NSWindow *window =
        [[NSWindow alloc] initWithContentRect:frame
                                    styleMask:(NSWindowStyleMaskTitled |
                                               NSWindowStyleMaskClosable |
                                               NSWindowStyleMaskResizable)
                                      backing:NSBackingStoreBuffered
                                        defer:NO];
    [window makeKeyAndOrderFront:nil]; // Make the window visible.

    InputView *mtkView =
        [[InputView alloc] initWithFrame:frame
                                  device:MTLCreateSystemDefaultDevice()];
    mtkView.colorPixelFormat =
        MTLPixelFormatBGRA8Unorm; // Set the color format.
    mtkView.clearColor =
        MTLClearColorMake(0.05, 0.05, 0.1, 1.0); // Set the clear color.
    mtkView.preferredFramesPerSecond =
        60; // Set the preferred frames per second.

    // Initialize the renderer.
    MetalRenderer *renderer =
        [[MetalRenderer alloc] initWithMetalKitView:mtkView];
    renderer.window = window;    // Set the window.
    mtkView.delegate = renderer; // Set the delegate.
    mtkView.renderer = renderer; // Set the renderer.

    // Set the view as the content view of the window.
    [window setContentView:mtkView]; // Set the view as the content view of the
                                     // window.
    [window makeFirstResponder:mtkView]; // Set the view as the first responder.
    [NSApp activateIgnoringOtherApps:YES]; // Activate the app.
    [NSApp run];                           // Start the Cocoa event loop.
  }
  return 0;
}
