// =============================================================================
// Mandelbulb Path Tracer v2 - main.mm
// =============================================================================
// Native macOS application with MetalFX upscaling for enhanced quality.
//
// MetalFX Integration:
// --------------------
// MetalFX is Apple's ML-based upscaling framework (similar to NVIDIA DLSS).
// It uses trained neural networks to reconstruct high-frequency details
// when upscaling from a lower internal resolution.
//
// This version renders at 768x576 and upscales to 1024x768 using
// MTLFXSpatialScaler, which provides:
// - ~1.8x fewer pixels to path trace (major performance gain)
// - ML-based detail reconstruction (sharper than bilinear)
// - Works well with our temporal accumulation for noise reduction
//
// The workflow:
// 1. Path trace at 768x576 → accumulator
// 2. Tone map accumulated result → low-res output
// 3. MetalFX upscale → 1024x768 final
// 4. Blit to screen
//
// Requirements:
// - macOS 13.0+ (Ventura) for MetalFX
// - Apple Silicon or AMD GPU with MetalFX support
// =============================================================================

#import <AppKit/AppKit.h>
#import <Metal/Metal.h>
#import <MetalFX/MetalFX.h>
#import <MetalKit/MetalKit.h>
#include <cmath>
#include <simd/simd.h>

// =============================================================================
// CONFIGURATION
// =============================================================================

namespace Config {
// Window/output dimensions
constexpr float WINDOW_WIDTH = 1024.0f;
constexpr float WINDOW_HEIGHT = 768.0f;

// Internal render resolution (lower for performance, upscaled by MetalFX)
constexpr uint32_t RENDER_WIDTH = 768;
constexpr uint32_t RENDER_HEIGHT = 576;

// Output resolution (after MetalFX upscaling)
constexpr uint32_t OUTPUT_WIDTH = 1024;
constexpr uint32_t OUTPUT_HEIGHT = 768;

// Camera settings
constexpr float CAMERA_DISTANCE = 3.0f;
constexpr float CAMERA_FOV = 60.0f;
constexpr float CAMERA_NEAR = 0.01f;
constexpr float CAMERA_FAR = 100.0f;

// Controls
constexpr float ORBIT_SENSITIVITY = 0.01f;
constexpr float PAN_SENSITIVITY = 0.005f;
constexpr float ZOOM_SENSITIVITY = 0.1f;
constexpr float MOVE_SPEED = 0.05f;

// Rendering
constexpr float INITIAL_EXPOSURE = 1.0f;
constexpr float EXPOSURE_STEP = 0.1f;

// Background
constexpr float BG_RED = 0.1f;
constexpr float BG_GREEN = 0.1f;
constexpr float BG_BLUE = 0.15f;
constexpr float BG_ALPHA = 1.0f;

// Thread groups
constexpr uint32_t THREAD_GROUP_SIZE = 16;
} // namespace Config

// =============================================================================
// DATA STRUCTURES
// =============================================================================

struct RenderParams {
  simd_float4x4 invViewProj;
  simd_float4x4 prevViewProj;
  simd_float3 cameraPos;
  float time;
  uint32_t frameIndex;
  uint32_t width;
  uint32_t height;
  float exposure;
  float jitterX;
  float jitterY;
  int32_t maxIterations; // Dynamic iteration count based on zoom
  float surfaceDetail;   // Surface distance threshold
};

// =============================================================================
// CAMERA
// =============================================================================

class Camera {
public:
  simd_float3 target = simd_make_float3(0.0f, 0.0f, 0.0f);
  float distance = Config::CAMERA_DISTANCE;
  float azimuth = 0.0f;
  float elevation = 0.3f;

  simd_float3 getPosition() const {
    float x = distance * cosf(elevation) * sinf(azimuth);
    float y = distance * sinf(elevation);
    float z = distance * cosf(elevation) * cosf(azimuth);
    return target + simd_make_float3(x, y, z);
  }

  simd_float4x4 getViewMatrix() const {
    simd_float3 pos = getPosition();
    simd_float3 forward = simd_normalize(target - pos);
    simd_float3 right =
        simd_normalize(simd_cross(forward, simd_make_float3(0, 1, 0)));
    simd_float3 up = simd_cross(right, forward);

    simd_float4x4 view;
    view.columns[0] = simd_make_float4(right.x, up.x, -forward.x, 0);
    view.columns[1] = simd_make_float4(right.y, up.y, -forward.y, 0);
    view.columns[2] = simd_make_float4(right.z, up.z, -forward.z, 0);
    view.columns[3] = simd_make_float4(
        -simd_dot(right, pos), -simd_dot(up, pos), simd_dot(forward, pos), 1);
    return view;
  }

  simd_float4x4 getProjectionMatrix(float aspect) const {
    float fov = Config::CAMERA_FOV * (M_PI / 180.0f);
    float f = 1.0f / tanf(fov * 0.5f);
    float near = Config::CAMERA_NEAR;
    float far = Config::CAMERA_FAR;

    simd_float4x4 proj = {};
    proj.columns[0].x = f / aspect;
    proj.columns[1].y = f;
    proj.columns[2].z = far / (near - far);
    proj.columns[2].w = -1.0f;
    proj.columns[3].z = (near * far) / (near - far);
    return proj;
  }

  void orbit(float dx, float dy) {
    azimuth -= dx * Config::ORBIT_SENSITIVITY;
    elevation += dy * Config::ORBIT_SENSITIVITY;
    elevation = fmaxf(-M_PI * 0.49f, fminf(M_PI * 0.49f, elevation));
  }

  void pan(float dx, float dy) {
    simd_float3 pos = getPosition();
    simd_float3 forward = simd_normalize(target - pos);
    simd_float3 right =
        simd_normalize(simd_cross(forward, simd_make_float3(0, 1, 0)));
    simd_float3 up = simd_cross(right, forward);

    target += right * dx * Config::PAN_SENSITIVITY * distance;
    target += up * dy * Config::PAN_SENSITIVITY * distance;
  }

  void zoom(float delta) {
    distance *= (1.0f - delta * Config::ZOOM_SENSITIVITY);
    distance = fmaxf(0.5f, fminf(20.0f, distance));
  }

  void move(float forward, float right, float up) {
    simd_float3 pos = getPosition();
    simd_float3 fwd = simd_normalize(target - pos);
    simd_float3 rgt =
        simd_normalize(simd_cross(fwd, simd_make_float3(0, 1, 0)));

    target += fwd * forward * Config::MOVE_SPEED;
    target += rgt * right * Config::MOVE_SPEED;
    target.y += up * Config::MOVE_SPEED;
  }

  void reset() {
    target = simd_make_float3(0.0f, 0.0f, 0.0f);
    distance = Config::CAMERA_DISTANCE;
    azimuth = 0.0f;
    elevation = 0.3f;
  }
};

inline simd_float4x4 invert_matrix(simd_float4x4 m) { return simd_inverse(m); }

// =============================================================================
// INPUT VIEW
// =============================================================================

@protocol RendererActions <NSObject>
- (void)resetCamera;
- (void)togglePause;
- (void)adjustExposure:(float)delta;
- (void)adjustIterations:(int)delta;
// ZQSD movement (AZERTY layout)
- (void)handleKeyZ:(BOOL)pressed; // Forward
- (void)handleKeyQ:(BOOL)pressed; // Left
- (void)handleKeyS:(BOOL)pressed; // Back
- (void)handleKeyD:(BOOL)pressed; // Right
@end

@interface InputView : MTKView
@property(nonatomic, weak) id<RendererActions> renderer;
@property(nonatomic) BOOL leftMouseDown;
@property(nonatomic) BOOL rightMouseDown;
@property(nonatomic) NSPoint lastMousePos;
@end

@implementation InputView

- (instancetype)initWithFrame:(NSRect)frameRect {
  self = [super initWithFrame:frameRect];
  if (self) {
    self.allowedTouchTypes = NSTouchTypeMaskDirect | NSTouchTypeMaskIndirect;
  }
  return self;
}

- (BOOL)acceptsFirstResponder {
  return YES;
}

- (void)keyDown:(NSEvent *)event {
  NSString *chars = [event charactersIgnoringModifiers];
  if ([chars length] == 0)
    return;

  unichar key = [chars characterAtIndex:0];

  switch (key) {
  // A to quit (AZERTY: A is where Q is on QWERTY)
  case 'a':
  case 'A':
    [NSApp terminate:nil];
    break;
  case 'r':
  case 'R':
    [_renderer resetCamera];
    break;
  case ' ':
    [_renderer togglePause];
    break;
  // Exposure: E to increase
  case 'e':
  case 'E':
    [_renderer adjustExposure:Config::EXPOSURE_STEP];
    break;
  // Iterations: I to increase, K to decrease
  case 'i':
  case 'I':
    [_renderer adjustIterations:4];
    break;
  case 'k':
  case 'K':
    [_renderer adjustIterations:-4];
    break;
  // ZQSD movement (AZERTY layout)
  case 'z':
  case 'Z':
    [_renderer handleKeyZ:YES];
    break;
  case 'q':
  case 'Q':
    [_renderer handleKeyQ:YES];
    break;
  case 's':
  case 'S':
    [_renderer handleKeyS:YES];
    break;
  case 'd':
  case 'D':
    [_renderer handleKeyD:YES];
    break;
  default:
    [super keyDown:event];
    break;
  }
}

- (void)keyUp:(NSEvent *)event {
  NSString *chars = [event charactersIgnoringModifiers];
  if ([chars length] == 0)
    return;

  unichar key = [chars characterAtIndex:0];

  switch (key) {
  case 'z':
  case 'Z':
    [_renderer handleKeyZ:NO];
    break;
  case 'q':
  case 'Q':
    [_renderer handleKeyQ:NO];
    break;
  case 's':
  case 'S':
    [_renderer handleKeyS:NO];
    break;
  case 'd':
  case 'D':
    [_renderer handleKeyD:NO];
    break;
  default:
    [super keyUp:event];
    break;
  }
}

- (void)mouseDown:(NSEvent *)event {
  _leftMouseDown = YES;
  _lastMousePos = [event locationInWindow];
}

- (void)mouseUp:(NSEvent *)event {
  (void)event;
  _leftMouseDown = NO;
}

- (void)rightMouseDown:(NSEvent *)event {
  _rightMouseDown = YES;
  _lastMousePos = [event locationInWindow];
}

- (void)rightMouseUp:(NSEvent *)event {
  (void)event;
  _rightMouseDown = NO;
}

- (void)mouseDragged:(NSEvent *)event {
  _lastMousePos = [event locationInWindow];
}

- (void)rightMouseDragged:(NSEvent *)event {
  _lastMousePos = [event locationInWindow];
}

@end

// =============================================================================
// APP DELEGATE
// =============================================================================

@interface AppDelegate : NSObject <NSApplicationDelegate>
@end

@implementation AppDelegate

- (BOOL)applicationShouldTerminateAfterLastWindowClosed:
    (NSApplication *)sender {
  (void)sender;
  return YES;
}

@end

// =============================================================================
// RENDERER
// =============================================================================

@interface Renderer : NSObject <MTKViewDelegate, RendererActions>
- (instancetype)initWithMetalKitView:(MTKView *)view;
@end

@implementation Renderer {
  // Metal core
  id<MTLDevice> _device;
  id<MTLCommandQueue> _commandQueue;

  // Pipelines
  id<MTLComputePipelineState> _pathTracePipeline;
  id<MTLComputePipelineState> _clearPipeline;
  id<MTLRenderPipelineState> _blitPipeline;

  // Textures (internal resolution)
  id<MTLTexture> _accumulatorTexture;
  id<MTLTexture> _renderTexture; // Low-res path trace output
  id<MTLTexture> _depthTexture;  // For potential future MetalFX Temporal

  // MetalFX upscaled output
  id<MTLTexture> _upscaledTexture;

  // MetalFX Spatial Scaler
  id<MTLFXSpatialScaler> _spatialScaler;
  BOOL _metalFXAvailable;

  // Buffers
  id<MTLBuffer> _paramsBuffer;

  // View
  InputView *_view;

  // Camera
  Camera _camera;

  // State
  BOOL _paused;
  uint32_t _frameIndex;
  BOOL _needsClear;
  float _exposure;
  int32_t _iterations; // Manual iteration count

  // Input (ZQSD for AZERTY)
  BOOL _keyZ, _keyQ, _keyS, _keyD;
  NSPoint _prevMousePos;

  // Stats
  double _lastFrameTime;
  int _frameCount;
  double _fpsUpdateTime;
  uint32_t _sampleCount;
}

- (instancetype)initWithMetalKitView:(MTKView *)view {
  self = [super init];
  if (!self)
    return nil;

  _view = (InputView *)view;
  _paused = NO;
  _frameIndex = 0;
  _needsClear = YES;
  _exposure = Config::INITIAL_EXPOSURE;
  _iterations = 20; // Default iteration count
  _keyZ = _keyQ = _keyS = _keyD = NO;
  _sampleCount = 0;
  _metalFXAvailable = NO;

  _lastFrameTime = CACurrentMediaTime();
  _frameCount = 0;
  _fpsUpdateTime = _lastFrameTime;

  // Initialize Metal
  _device = MTLCreateSystemDefaultDevice();
  if (!_device) {
    NSLog(@"ERROR: Metal is not supported");
    return nil;
  }
  NSLog(@"Using GPU: %@", _device.name);

  _commandQueue = [_device newCommandQueue];
  if (!_commandQueue) {
    NSLog(@"ERROR: Failed to create command queue");
    return nil;
  }

  // Load shaders
  if (![self loadShaders]) {
    return nil;
  }

  // Create textures
  [self createTextures];

  // Initialize MetalFX
  [self initializeMetalFX];

  // Create parameter buffer
  _paramsBuffer = [_device newBufferWithLength:sizeof(RenderParams)
                                       options:MTLResourceStorageModeShared];

  // Configure view
  view.device = _device;
  view.delegate = self;
  view.colorPixelFormat = MTLPixelFormatBGRA8Unorm;
  view.framebufferOnly = NO;

  return self;
}

- (BOOL)loadShaders {
  NSError *error = nil;

  NSString *shaderPath = @"Shader.metal";
  NSString *shaderSource =
      [NSString stringWithContentsOfFile:shaderPath
                                encoding:NSUTF8StringEncoding
                                   error:&error];
  if (!shaderSource) {
    NSLog(@"ERROR: Could not load shader: %@", error.localizedDescription);
    return NO;
  }

  MTLCompileOptions *options = [[MTLCompileOptions alloc] init];
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

  // Path trace pipeline
  id<MTLFunction> pathTraceFunc = [library newFunctionWithName:@"path_trace"];
  _pathTracePipeline =
      [_device newComputePipelineStateWithFunction:pathTraceFunc error:&error];
  if (!_pathTracePipeline) {
    NSLog(@"ERROR: Failed to create path trace pipeline: %@",
          error.localizedDescription);
    return NO;
  }

  // Clear pipeline
  id<MTLFunction> clearFunc =
      [library newFunctionWithName:@"clear_accumulator"];
  _clearPipeline = [_device newComputePipelineStateWithFunction:clearFunc
                                                          error:&error];
  if (!_clearPipeline) {
    NSLog(@"ERROR: Failed to create clear pipeline: %@",
          error.localizedDescription);
    return NO;
  }

  // Blit pipeline (fallback)
  id<MTLFunction> vertexFunc =
      [library newFunctionWithName:@"fullscreen_vertex"];
  id<MTLFunction> fragmentFunc =
      [library newFunctionWithName:@"fullscreen_fragment"];

  MTLRenderPipelineDescriptor *pipelineDesc =
      [[MTLRenderPipelineDescriptor alloc] init];
  pipelineDesc.vertexFunction = vertexFunc;
  pipelineDesc.fragmentFunction = fragmentFunc;
  pipelineDesc.colorAttachments[0].pixelFormat = MTLPixelFormatBGRA8Unorm;

  _blitPipeline = [_device newRenderPipelineStateWithDescriptor:pipelineDesc
                                                          error:&error];
  if (!_blitPipeline) {
    NSLog(@"ERROR: Failed to create blit pipeline: %@",
          error.localizedDescription);
    return NO;
  }

  return YES;
}

- (void)createTextures {
  // Internal resolution textures
  MTLTextureDescriptor *accDesc = [MTLTextureDescriptor
      texture2DDescriptorWithPixelFormat:MTLPixelFormatRGBA32Float
                                   width:Config::RENDER_WIDTH
                                  height:Config::RENDER_HEIGHT
                               mipmapped:NO];
  accDesc.usage = MTLTextureUsageShaderRead | MTLTextureUsageShaderWrite;
  accDesc.storageMode = MTLStorageModePrivate;
  _accumulatorTexture = [_device newTextureWithDescriptor:accDesc];

  // Render output (low-res, will be upscaled)
  MTLTextureDescriptor *renderDesc = [MTLTextureDescriptor
      texture2DDescriptorWithPixelFormat:MTLPixelFormatRGBA16Float
                                   width:Config::RENDER_WIDTH
                                  height:Config::RENDER_HEIGHT
                               mipmapped:NO];
  renderDesc.usage = MTLTextureUsageShaderRead | MTLTextureUsageShaderWrite;
  renderDesc.storageMode = MTLStorageModePrivate;
  _renderTexture = [_device newTextureWithDescriptor:renderDesc];

  // Depth texture
  MTLTextureDescriptor *depthDesc = [MTLTextureDescriptor
      texture2DDescriptorWithPixelFormat:MTLPixelFormatR32Float
                                   width:Config::RENDER_WIDTH
                                  height:Config::RENDER_HEIGHT
                               mipmapped:NO];
  depthDesc.usage = MTLTextureUsageShaderRead | MTLTextureUsageShaderWrite;
  depthDesc.storageMode = MTLStorageModePrivate;
  _depthTexture = [_device newTextureWithDescriptor:depthDesc];

  // Upscaled output texture
  MTLTextureDescriptor *upscaledDesc = [MTLTextureDescriptor
      texture2DDescriptorWithPixelFormat:MTLPixelFormatRGBA16Float
                                   width:Config::OUTPUT_WIDTH
                                  height:Config::OUTPUT_HEIGHT
                               mipmapped:NO];
  upscaledDesc.usage = MTLTextureUsageShaderRead | MTLTextureUsageShaderWrite |
                       MTLTextureUsageRenderTarget;
  upscaledDesc.storageMode = MTLStorageModePrivate;
  _upscaledTexture = [_device newTextureWithDescriptor:upscaledDesc];

  NSLog(@"Created textures:");
  NSLog(@"  Internal: %ux%u", Config::RENDER_WIDTH, Config::RENDER_HEIGHT);
  NSLog(@"  Output:   %ux%u", Config::OUTPUT_WIDTH, Config::OUTPUT_HEIGHT);
}

- (void)initializeMetalFX {
  // Check MetalFX availability (requires macOS 13+)
  if (@available(macOS 13.0, *)) {
    // Create spatial scaler descriptor
    MTLFXSpatialScalerDescriptor *descriptor =
        [[MTLFXSpatialScalerDescriptor alloc] init];

    // Input (low-res render)
    descriptor.inputWidth = Config::RENDER_WIDTH;
    descriptor.inputHeight = Config::RENDER_HEIGHT;

    // Output (upscaled)
    descriptor.outputWidth = Config::OUTPUT_WIDTH;
    descriptor.outputHeight = Config::OUTPUT_HEIGHT;

    // Pixel formats
    descriptor.colorTextureFormat = MTLPixelFormatRGBA16Float;
    descriptor.outputTextureFormat = MTLPixelFormatRGBA16Float;

    // Color processing mode
    descriptor.colorProcessingMode =
        MTLFXSpatialScalerColorProcessingModeLinear;

    // Create the scaler
    _spatialScaler = [descriptor newSpatialScalerWithDevice:_device];

    if (_spatialScaler) {
      _metalFXAvailable = YES;
      NSLog(@"MetalFX Spatial Scaler initialized successfully");
      NSLog(@"  Upscaling: %ux%u -> %ux%u (%.2fx)", Config::RENDER_WIDTH,
            Config::RENDER_HEIGHT, Config::OUTPUT_WIDTH, Config::OUTPUT_HEIGHT,
            (float)Config::OUTPUT_WIDTH / Config::RENDER_WIDTH);
    } else {
      NSLog(@"WARNING: MetalFX Spatial Scaler creation failed, using fallback");
    }
  } else {
    NSLog(@"WARNING: MetalFX requires macOS 13.0+, using bilinear fallback");
  }
}

// RendererActions protocol
- (void)resetCamera {
  _camera.reset();
  _needsClear = YES;
  _sampleCount = 0;
  NSLog(@"Camera reset");
}

- (void)togglePause {
  _paused = !_paused;
  NSLog(@"%@", _paused ? @"Paused" : @"Resumed");
}

- (void)adjustExposure:(float)delta {
  _exposure += delta;
  _exposure = fmaxf(0.1f, fminf(10.0f, _exposure));
  NSLog(@"Exposure: %.2f", _exposure);
}

- (void)adjustIterations:(int)delta {
  _iterations += delta;
  _iterations = fmaxf(8, fminf(256, _iterations)); // Range: 8 to 256
  _needsClear = YES;
  _sampleCount = 0;
  NSLog(@"Iterations: %d", _iterations);
}

// ZQSD movement handlers (AZERTY layout)
- (void)handleKeyZ:(BOOL)pressed {
  _keyZ = pressed;
}
- (void)handleKeyQ:(BOOL)pressed {
  _keyQ = pressed;
}
- (void)handleKeyS:(BOOL)pressed {
  _keyS = pressed;
}
- (void)handleKeyD:(BOOL)pressed {
  _keyD = pressed;
}

- (void)processInput {
  if (_view.leftMouseDown || _view.rightMouseDown) {
    NSPoint currentPos = [_view.window mouseLocationOutsideOfEventStream];
    float dx = currentPos.x - _prevMousePos.x;
    float dy = currentPos.y - _prevMousePos.y;

    if (_view.leftMouseDown) {
      _camera.orbit(dx, dy);
    } else if (_view.rightMouseDown) {
      _camera.pan(-dx, -dy);
    }

    if (fabsf(dx) > 0.1f || fabsf(dy) > 0.1f) {
      _needsClear = YES;
      _sampleCount = 0;
    }

    _prevMousePos = currentPos;
  } else {
    _prevMousePos = [_view.window mouseLocationOutsideOfEventStream];
  }

  // ZQSD movement (AZERTY layout)
  float forward = 0, right = 0, up = 0;
  if (_keyZ)
    forward += 1;
  if (_keyS)
    forward -= 1;
  if (_keyQ)
    right -= 1;
  if (_keyD)
    right += 1;

  if (forward != 0 || right != 0 || up != 0) {
    _camera.move(forward, right, up);
    _needsClear = YES;
    _sampleCount = 0;
  }
}

- (void)drawInMTKView:(MTKView *)view {
  [self processInput];

  // Handle scroll
  NSEvent *scrollEvent = [NSApp currentEvent];
  if (scrollEvent && scrollEvent.type == NSEventTypeScrollWheel) {
    float dy = [scrollEvent scrollingDeltaY];
    if (fabsf(dy) > 0.01f) {
      _camera.zoom(dy * 0.02f);
      _needsClear = YES;
      _sampleCount = 0;
    }
  }

  [self updateFPS];

  if (_paused)
    return;

  id<MTLCommandBuffer> commandBuffer = [_commandQueue commandBuffer];
  if (!commandBuffer)
    return;

  // Clear accumulator if needed
  if (_needsClear) {
    id<MTLComputeCommandEncoder> clearEncoder =
        [commandBuffer computeCommandEncoder];
    [clearEncoder setComputePipelineState:_clearPipeline];
    [clearEncoder setTexture:_accumulatorTexture atIndex:0];

    MTLSize gridSize =
        MTLSizeMake(Config::RENDER_WIDTH, Config::RENDER_HEIGHT, 1);
    MTLSize groupSize =
        MTLSizeMake(Config::THREAD_GROUP_SIZE, Config::THREAD_GROUP_SIZE, 1);
    [clearEncoder dispatchThreads:gridSize threadsPerThreadgroup:groupSize];
    [clearEncoder endEncoding];

    _needsClear = NO;
    _frameIndex = 0;
  }

  // Update render parameters
  float aspect = (float)Config::RENDER_WIDTH / (float)Config::RENDER_HEIGHT;
  simd_float4x4 viewMatrix = _camera.getViewMatrix();
  simd_float4x4 projMatrix = _camera.getProjectionMatrix(aspect);
  simd_float4x4 viewProj = simd_mul(projMatrix, viewMatrix);
  simd_float4x4 invViewProj = invert_matrix(viewProj);

  RenderParams *params = (RenderParams *)_paramsBuffer.contents;
  params->invViewProj = invViewProj;
  params->prevViewProj = viewProj; // For potential motion vectors
  params->cameraPos = _camera.getPosition();
  params->time = CACurrentMediaTime();
  params->frameIndex = _frameIndex;
  params->width = Config::RENDER_WIDTH;
  params->height = Config::RENDER_HEIGHT;
  params->exposure = _exposure;
  params->jitterX = 0.0f;
  params->jitterY = 0.0f;
  params->maxIterations = _iterations;
  // Surface detail scales with iterations (more iters = finer detail)
  params->surfaceDetail = 0.001f / (float)_iterations * 20.0f;

  // Path trace at internal resolution
  id<MTLComputeCommandEncoder> computeEncoder =
      [commandBuffer computeCommandEncoder];
  [computeEncoder setComputePipelineState:_pathTracePipeline];
  [computeEncoder setTexture:_renderTexture atIndex:0];
  [computeEncoder setTexture:_accumulatorTexture atIndex:1];
  [computeEncoder setTexture:_depthTexture atIndex:2];
  [computeEncoder setBuffer:_paramsBuffer offset:0 atIndex:0];

  MTLSize gridSize =
      MTLSizeMake(Config::RENDER_WIDTH, Config::RENDER_HEIGHT, 1);
  MTLSize groupSize =
      MTLSizeMake(Config::THREAD_GROUP_SIZE, Config::THREAD_GROUP_SIZE, 1);
  [computeEncoder dispatchThreads:gridSize threadsPerThreadgroup:groupSize];
  [computeEncoder endEncoding];

  _frameIndex++;
  _sampleCount++;

  // MetalFX upscaling
  id<MTLTexture> displayTexture;

  if (_metalFXAvailable) {
    if (@available(macOS 13.0, *)) {
      // Configure spatial scaler
      _spatialScaler.colorTexture = _renderTexture;
      _spatialScaler.outputTexture = _upscaledTexture;

      // Encode upscaling
      [_spatialScaler encodeToCommandBuffer:commandBuffer];
      displayTexture = _upscaledTexture;
    }
  } else {
    // Fallback: use low-res texture directly
    displayTexture = _renderTexture;
  }

  // Blit to screen
  MTLRenderPassDescriptor *renderPassDesc = view.currentRenderPassDescriptor;
  id<CAMetalDrawable> drawable = view.currentDrawable;

  if (renderPassDesc && drawable) {
    id<MTLRenderCommandEncoder> renderEncoder =
        [commandBuffer renderCommandEncoderWithDescriptor:renderPassDesc];
    [renderEncoder setRenderPipelineState:_blitPipeline];
    [renderEncoder setFragmentTexture:displayTexture atIndex:0];
    [renderEncoder drawPrimitives:MTLPrimitiveTypeTriangle
                      vertexStart:0
                      vertexCount:3];
    [renderEncoder endEncoding];

    [commandBuffer presentDrawable:drawable];
  }

  [commandBuffer commit];
}

- (void)mtkView:(MTKView *)view drawableSizeWillChange:(CGSize)size {
  (void)view;
  (void)size;
}

- (void)updateFPS {
  _frameCount++;
  double currentTime = CACurrentMediaTime();

  if (currentTime - _fpsUpdateTime >= 1.0) {
    double fps = _frameCount / (currentTime - _fpsUpdateTime);

    dispatch_async(dispatch_get_main_queue(), ^{
      NSString *fxStatus = self->_metalFXAvailable ? @"MetalFX" : @"No FX";
      NSString *title = [NSString
          stringWithFormat:@"Mandelbulb v2 - %.1f FPS | %u spp | %d iter | %@",
                           fps, self->_sampleCount, self->_iterations,
                           fxStatus];
      [self->_view.window setTitle:title];
    });

    _frameCount = 0;
    _fpsUpdateTime = currentTime;
  }
}

@end

// =============================================================================
// MAIN
// =============================================================================

int main(int argc, const char *argv[]) {
  (void)argc;
  (void)argv;

  @autoreleasepool {
    NSApplication *app = [NSApplication sharedApplication];
    [app setActivationPolicy:NSApplicationActivationPolicyRegular];

    AppDelegate *delegate = [[AppDelegate alloc] init];
    [app setDelegate:delegate];

    NSRect frame =
        NSMakeRect(0, 0, Config::WINDOW_WIDTH, Config::WINDOW_HEIGHT);
    NSWindowStyleMask style =
        NSWindowStyleMaskTitled | NSWindowStyleMaskClosable |
        NSWindowStyleMaskMiniaturizable | NSWindowStyleMaskResizable;

    NSWindow *window =
        [[NSWindow alloc] initWithContentRect:frame
                                    styleMask:style
                                      backing:NSBackingStoreBuffered
                                        defer:NO];
    [window setTitle:@"Mandelbulb Path Tracer v2"];
    [window center];

    InputView *metalView = [[InputView alloc] initWithFrame:frame];
    metalView.clearColor = MTLClearColorMake(Config::BG_RED, Config::BG_GREEN,
                                             Config::BG_BLUE, Config::BG_ALPHA);
    metalView.preferredFramesPerSecond = 60;

    Renderer *renderer = [[Renderer alloc] initWithMetalKitView:metalView];
    if (!renderer) {
      NSLog(@"ERROR: Failed to create renderer");
      return 1;
    }
    metalView.renderer = renderer;

    [window setContentView:metalView];
    [window makeKeyAndOrderFront:nil];
    [window makeFirstResponder:metalView];

    [app activateIgnoringOtherApps:YES];

    NSLog(@"Mandelbulb Path Tracer v2 (MetalFX)");
    NSLog(@"===================================");
    NSLog(@"Internal resolution: %ux%u", Config::RENDER_WIDTH,
          Config::RENDER_HEIGHT);
    NSLog(@"Output resolution:   %ux%u", Config::OUTPUT_WIDTH,
          Config::OUTPUT_HEIGHT);
    NSLog(@"");
    NSLog(@"Controls (AZERTY):");
    NSLog(@"  Left-drag     : Orbit camera");
    NSLog(@"  Right-drag    : Pan camera");
    NSLog(@"  Scroll        : Zoom in/out");
    NSLog(@"  ZQSD          : Move camera");
    NSLog(@"  I / K         : Iterations up/down");
    NSLog(@"  E             : Exposure up");
    NSLog(@"  R             : Reset camera");
    NSLog(@"  Space         : Pause/Resume");
    NSLog(@"  A             : Quit");

    [app run];
  }

  return 0;
}
