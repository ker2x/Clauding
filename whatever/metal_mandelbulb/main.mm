// =============================================================================
// Mandelbulb Path Tracer - main.mm
// =============================================================================
// Native macOS application for real-time progressive path tracing of the
// Mandelbulb fractal using Metal compute shaders.
//
// Progressive Rendering:
// ----------------------
// Unlike traditional offline rendering that computes all samples before display,
// progressive rendering shows results immediately and refines over time:
//
// 1. Each frame, the GPU traces one sample per pixel
// 2. Samples are accumulated in a buffer (running average)
// 3. The accumulated result is displayed immediately
// 4. Image quality improves continuously without freezing
//
// This approach is essential for interactive previews of computationally
// expensive renders like path tracing.
//
// Controls:
// ---------
// - Left-drag: Orbit camera around the Mandelbulb
// - Right-drag: Pan camera
// - Scroll: Zoom in/out
// - WASD: Move camera position
// - R: Reset camera to initial position
// - E/D: Adjust exposure up/down
// - Q: Quit
// =============================================================================

#import <AppKit/AppKit.h>
#import <Metal/Metal.h>
#import <MetalKit/MetalKit.h>
#include <simd/simd.h>
#include <cmath>

// =============================================================================
// CONFIGURATION
// =============================================================================

namespace Config {
// Window settings
constexpr float WINDOW_WIDTH = 1024.0f;
constexpr float WINDOW_HEIGHT = 768.0f;

// Render target (can be lower for faster preview)
constexpr uint32_t RENDER_WIDTH = 1024;
constexpr uint32_t RENDER_HEIGHT = 768;

// Camera settings
constexpr float CAMERA_DISTANCE = 3.0f;    // Initial distance from origin
constexpr float CAMERA_FOV = 60.0f;        // Field of view in degrees
constexpr float CAMERA_NEAR = 0.01f;
constexpr float CAMERA_FAR = 100.0f;

// Mouse sensitivity
constexpr float ORBIT_SENSITIVITY = 0.01f;
constexpr float PAN_SENSITIVITY = 0.005f;
constexpr float ZOOM_SENSITIVITY = 0.1f;
constexpr float MOVE_SPEED = 0.05f;

// Initial exposure for tone mapping
constexpr float INITIAL_EXPOSURE = 1.0f;
constexpr float EXPOSURE_STEP = 0.1f;

// Background color
constexpr float BG_RED = 0.1f;
constexpr float BG_GREEN = 0.1f;
constexpr float BG_BLUE = 0.15f;
constexpr float BG_ALPHA = 1.0f;

// Thread group size (16x16 = 256, optimal for Apple Silicon)
constexpr uint32_t THREAD_GROUP_SIZE = 16;
} // namespace Config

// =============================================================================
// DATA STRUCTURES
// =============================================================================
// Must match the structures in Shader.metal exactly.

struct RenderParams {
    simd_float4x4 invViewProj;
    simd_float3 cameraPos;
    float time;
    uint32_t frameIndex;
    uint32_t width;
    uint32_t height;
    float exposure;
};

// =============================================================================
// CAMERA
// =============================================================================
// Orbit camera that rotates around a target point.

class Camera {
public:
    simd_float3 target = simd_make_float3(0.0f, 0.0f, 0.0f);
    float distance = Config::CAMERA_DISTANCE;
    float azimuth = 0.0f;     // Horizontal angle (radians)
    float elevation = 0.3f;   // Vertical angle (radians)

    simd_float3 getPosition() const {
        float x = distance * cosf(elevation) * sinf(azimuth);
        float y = distance * sinf(elevation);
        float z = distance * cosf(elevation) * cosf(azimuth);
        return target + simd_make_float3(x, y, z);
    }

    simd_float4x4 getViewMatrix() const {
        simd_float3 pos = getPosition();
        simd_float3 forward = simd_normalize(target - pos);
        simd_float3 right = simd_normalize(simd_cross(forward, simd_make_float3(0, 1, 0)));
        simd_float3 up = simd_cross(right, forward);

        // Build view matrix (column-major)
        simd_float4x4 view;
        view.columns[0] = simd_make_float4(right.x, up.x, -forward.x, 0);
        view.columns[1] = simd_make_float4(right.y, up.y, -forward.y, 0);
        view.columns[2] = simd_make_float4(right.z, up.z, -forward.z, 0);
        view.columns[3] = simd_make_float4(-simd_dot(right, pos), -simd_dot(up, pos), simd_dot(forward, pos), 1);
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
        // Clamp elevation to avoid gimbal lock
        elevation = fmaxf(-M_PI * 0.49f, fminf(M_PI * 0.49f, elevation));
    }

    void pan(float dx, float dy) {
        simd_float3 pos = getPosition();
        simd_float3 forward = simd_normalize(target - pos);
        simd_float3 right = simd_normalize(simd_cross(forward, simd_make_float3(0, 1, 0)));
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
        simd_float3 rgt = simd_normalize(simd_cross(fwd, simd_make_float3(0, 1, 0)));

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

// Matrix inversion helper (for 4x4)
// Note: simd_inverse is available in the simd library
inline simd_float4x4 invert_matrix(simd_float4x4 m) {
    return simd_inverse(m);
}

// =============================================================================
// INPUT VIEW
// =============================================================================

@protocol RendererActions <NSObject>
- (void)resetCamera;
- (void)togglePause;
- (void)adjustExposure:(float)delta;
- (void)handleKeyW:(BOOL)pressed;
- (void)handleKeyA:(BOOL)pressed;
- (void)handleKeyS:(BOOL)pressed;
- (void)handleKeyD:(BOOL)pressed;
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
        // Enable touch/scroll events (modern API)
        self.allowedTouchTypes = NSTouchTypeMaskDirect | NSTouchTypeMaskIndirect;
    }
    return self;
}

- (BOOL)acceptsFirstResponder {
    return YES;
}

- (void)keyDown:(NSEvent *)event {
    NSString *chars = [event charactersIgnoringModifiers];
    if ([chars length] == 0) return;

    unichar key = [chars characterAtIndex:0];

    switch (key) {
        case 'q':
        case 'Q':
            [NSApp terminate:nil];
            break;
        case 'r':
        case 'R':
            [_renderer resetCamera];
            break;
        case ' ':
            [_renderer togglePause];
            break;
        case 'e':
        case 'E':
            [_renderer adjustExposure:Config::EXPOSURE_STEP];
            break;
        case 'd':
        case 'D':
            // D is also used for movement, so only adjust exposure if shift is held
            if ([event modifierFlags] & NSEventModifierFlagShift) {
                [_renderer adjustExposure:-Config::EXPOSURE_STEP];
            } else {
                [_renderer handleKeyD:YES];
            }
            break;
        case 'w':
        case 'W':
            [_renderer handleKeyW:YES];
            break;
        case 'a':
        case 'A':
            [_renderer handleKeyA:YES];
            break;
        case 's':
        case 'S':
            [_renderer handleKeyS:YES];
            break;
        default:
            [super keyDown:event];
            break;
    }
}

- (void)keyUp:(NSEvent *)event {
    NSString *chars = [event charactersIgnoringModifiers];
    if ([chars length] == 0) return;

    unichar key = [chars characterAtIndex:0];

    switch (key) {
        case 'w':
        case 'W':
            [_renderer handleKeyW:NO];
            break;
        case 'a':
        case 'A':
            [_renderer handleKeyA:NO];
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
    // Handled in renderer
    _lastMousePos = [event locationInWindow];
}

- (void)rightMouseDragged:(NSEvent *)event {
    // Handled in renderer
    _lastMousePos = [event locationInWindow];
}

- (void)scrollWheel:(NSEvent *)event {
    // Forward to renderer - handled there
    (void)event;
}

@end

// =============================================================================
// APP DELEGATE
// =============================================================================

@interface AppDelegate : NSObject <NSApplicationDelegate>
@end

@implementation AppDelegate

- (BOOL)applicationShouldTerminateAfterLastWindowClosed:(NSApplication *)sender {
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
    // Metal core objects
    id<MTLDevice> _device;
    id<MTLCommandQueue> _commandQueue;

    // Pipeline states
    id<MTLComputePipelineState> _pathTracePipeline;
    id<MTLComputePipelineState> _clearPipeline;
    id<MTLRenderPipelineState> _blitPipeline;

    // Textures
    id<MTLTexture> _accumulatorTexture;  // Accumulates samples (float4, xyz=color, w=count)
    id<MTLTexture> _outputTexture;       // Display output (RGBA8)

    // Buffers
    id<MTLBuffer> _paramsBuffer;

    // View reference
    InputView *_view;

    // Camera
    Camera _camera;

    // State
    BOOL _paused;
    uint32_t _frameIndex;
    BOOL _needsClear;
    float _exposure;

    // Key states for smooth movement
    BOOL _keyW, _keyA, _keyS, _keyD;

    // Previous mouse position for delta calculation
    NSPoint _prevMousePos;

    // FPS tracking
    double _lastFrameTime;
    int _frameCount;
    double _fpsUpdateTime;
    uint32_t _sampleCount;
}

- (instancetype)initWithMetalKitView:(MTKView *)view {
    self = [super init];
    if (!self) return nil;

    _view = (InputView *)view;
    _paused = NO;
    _frameIndex = 0;
    _needsClear = YES;
    _exposure = Config::INITIAL_EXPOSURE;
    _keyW = _keyA = _keyS = _keyD = NO;
    _sampleCount = 0;

    _lastFrameTime = CACurrentMediaTime();
    _frameCount = 0;
    _fpsUpdateTime = _lastFrameTime;

    // Initialize Metal
    _device = MTLCreateSystemDefaultDevice();
    if (!_device) {
        NSLog(@"ERROR: Metal is not supported on this device");
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

    // Create parameter buffer
    _paramsBuffer = [_device newBufferWithLength:sizeof(RenderParams)
                                         options:MTLResourceStorageModeShared];

    // Configure view
    view.device = _device;
    view.delegate = self;
    view.colorPixelFormat = MTLPixelFormatBGRA8Unorm;
    view.framebufferOnly = NO;  // We need to read from it for blitting

    return self;
}

- (BOOL)loadShaders {
    NSError *error = nil;

    // Load shader source
    NSString *shaderPath = @"Shader.metal";
    NSString *shaderSource = [NSString stringWithContentsOfFile:shaderPath
                                                       encoding:NSUTF8StringEncoding
                                                          error:&error];
    if (!shaderSource) {
        NSLog(@"ERROR: Could not load shader file: %@", error.localizedDescription);
        return NO;
    }

    // Compile shaders
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

    // Create path trace pipeline
    id<MTLFunction> pathTraceFunc = [library newFunctionWithName:@"path_trace"];
    if (!pathTraceFunc) {
        NSLog(@"ERROR: Could not find 'path_trace' function");
        return NO;
    }
    _pathTracePipeline = [_device newComputePipelineStateWithFunction:pathTraceFunc error:&error];
    if (!_pathTracePipeline) {
        NSLog(@"ERROR: Failed to create path trace pipeline: %@", error.localizedDescription);
        return NO;
    }

    // Create clear pipeline
    id<MTLFunction> clearFunc = [library newFunctionWithName:@"clear_accumulator"];
    if (!clearFunc) {
        NSLog(@"ERROR: Could not find 'clear_accumulator' function");
        return NO;
    }
    _clearPipeline = [_device newComputePipelineStateWithFunction:clearFunc error:&error];
    if (!_clearPipeline) {
        NSLog(@"ERROR: Failed to create clear pipeline: %@", error.localizedDescription);
        return NO;
    }

    // Create blit (fullscreen) render pipeline
    id<MTLFunction> vertexFunc = [library newFunctionWithName:@"fullscreen_vertex"];
    id<MTLFunction> fragmentFunc = [library newFunctionWithName:@"fullscreen_fragment"];
    if (!vertexFunc || !fragmentFunc) {
        NSLog(@"ERROR: Could not find fullscreen shader functions");
        return NO;
    }

    MTLRenderPipelineDescriptor *pipelineDesc = [[MTLRenderPipelineDescriptor alloc] init];
    pipelineDesc.vertexFunction = vertexFunc;
    pipelineDesc.fragmentFunction = fragmentFunc;
    pipelineDesc.colorAttachments[0].pixelFormat = MTLPixelFormatBGRA8Unorm;

    _blitPipeline = [_device newRenderPipelineStateWithDescriptor:pipelineDesc error:&error];
    if (!_blitPipeline) {
        NSLog(@"ERROR: Failed to create blit pipeline: %@", error.localizedDescription);
        return NO;
    }

    return YES;
}

- (void)createTextures {
    // Accumulator texture (float4: rgb = accumulated color, a = sample count)
    MTLTextureDescriptor *accDesc = [MTLTextureDescriptor
        texture2DDescriptorWithPixelFormat:MTLPixelFormatRGBA32Float
                                     width:Config::RENDER_WIDTH
                                    height:Config::RENDER_HEIGHT
                                 mipmapped:NO];
    accDesc.usage = MTLTextureUsageShaderRead | MTLTextureUsageShaderWrite;
    accDesc.storageMode = MTLStorageModePrivate;
    _accumulatorTexture = [_device newTextureWithDescriptor:accDesc];

    // Output texture (display quality)
    MTLTextureDescriptor *outDesc = [MTLTextureDescriptor
        texture2DDescriptorWithPixelFormat:MTLPixelFormatRGBA8Unorm
                                     width:Config::RENDER_WIDTH
                                    height:Config::RENDER_HEIGHT
                                 mipmapped:NO];
    outDesc.usage = MTLTextureUsageShaderRead | MTLTextureUsageShaderWrite;
    outDesc.storageMode = MTLStorageModePrivate;
    _outputTexture = [_device newTextureWithDescriptor:outDesc];

    NSLog(@"Created render textures: %ux%u", Config::RENDER_WIDTH, Config::RENDER_HEIGHT);
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
    // Don't clear accumulator for exposure changes - just affects tonemapping
}

- (void)handleKeyW:(BOOL)pressed { _keyW = pressed; }
- (void)handleKeyA:(BOOL)pressed { _keyA = pressed; }
- (void)handleKeyS:(BOOL)pressed { _keyS = pressed; }
- (void)handleKeyD:(BOOL)pressed { _keyD = pressed; }

// Handle continuous input
- (void)processInput {
    // Handle mouse drag
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

    // Handle scroll wheel (zoom)
    // Note: Scroll events come through the event system, handled separately

    // Handle keyboard movement
    float forward = 0, right = 0, up = 0;
    if (_keyW) forward += 1;
    if (_keyS) forward -= 1;
    if (_keyA) right -= 1;
    if (_keyD) right += 1;

    if (forward != 0 || right != 0 || up != 0) {
        _camera.move(forward, right, up);
        _needsClear = YES;
        _sampleCount = 0;
    }
}

// MTKViewDelegate
- (void)drawInMTKView:(MTKView *)view {
    // Process input
    [self processInput];

    // Handle scroll wheel for zoom
    NSEvent *scrollEvent = [NSApp currentEvent];
    if (scrollEvent && scrollEvent.type == NSEventTypeScrollWheel) {
        float dy = [scrollEvent scrollingDeltaY];
        if (fabsf(dy) > 0.01f) {
            _camera.zoom(dy * 0.02f);
            _needsClear = YES;
            _sampleCount = 0;
        }
    }

    // Update FPS
    [self updateFPS];

    if (_paused) return;

    // Create command buffer
    id<MTLCommandBuffer> commandBuffer = [_commandQueue commandBuffer];
    if (!commandBuffer) return;

    // Clear accumulator if needed (camera moved, etc.)
    if (_needsClear) {
        id<MTLComputeCommandEncoder> clearEncoder = [commandBuffer computeCommandEncoder];
        [clearEncoder setComputePipelineState:_clearPipeline];
        [clearEncoder setTexture:_accumulatorTexture atIndex:0];

        MTLSize gridSize = MTLSizeMake(Config::RENDER_WIDTH, Config::RENDER_HEIGHT, 1);
        MTLSize groupSize = MTLSizeMake(Config::THREAD_GROUP_SIZE, Config::THREAD_GROUP_SIZE, 1);
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
    params->cameraPos = _camera.getPosition();
    params->time = CACurrentMediaTime();
    params->frameIndex = _frameIndex;
    params->width = Config::RENDER_WIDTH;
    params->height = Config::RENDER_HEIGHT;
    params->exposure = _exposure;

    // Path trace one sample
    id<MTLComputeCommandEncoder> computeEncoder = [commandBuffer computeCommandEncoder];
    [computeEncoder setComputePipelineState:_pathTracePipeline];
    [computeEncoder setTexture:_outputTexture atIndex:0];
    [computeEncoder setTexture:_accumulatorTexture atIndex:1];
    [computeEncoder setBuffer:_paramsBuffer offset:0 atIndex:0];

    MTLSize gridSize = MTLSizeMake(Config::RENDER_WIDTH, Config::RENDER_HEIGHT, 1);
    MTLSize groupSize = MTLSizeMake(Config::THREAD_GROUP_SIZE, Config::THREAD_GROUP_SIZE, 1);
    [computeEncoder dispatchThreads:gridSize threadsPerThreadgroup:groupSize];
    [computeEncoder endEncoding];

    _frameIndex++;
    _sampleCount++;

    // Blit to screen
    MTLRenderPassDescriptor *renderPassDesc = view.currentRenderPassDescriptor;
    if (renderPassDesc) {
        id<MTLRenderCommandEncoder> renderEncoder = [commandBuffer renderCommandEncoderWithDescriptor:renderPassDesc];
        [renderEncoder setRenderPipelineState:_blitPipeline];
        [renderEncoder setFragmentTexture:_outputTexture atIndex:0];
        [renderEncoder drawPrimitives:MTLPrimitiveTypeTriangle vertexStart:0 vertexCount:3];
        [renderEncoder endEncoding];
    }

    [commandBuffer presentDrawable:view.currentDrawable];
    [commandBuffer commit];
}

- (void)mtkView:(MTKView *)view drawableSizeWillChange:(CGSize)size {
    (void)view;
    (void)size;
    // Could recreate textures here for dynamic resizing
}

- (void)updateFPS {
    _frameCount++;
    double currentTime = CACurrentMediaTime();

    if (currentTime - _fpsUpdateTime >= 1.0) {
        double fps = _frameCount / (currentTime - _fpsUpdateTime);

        dispatch_async(dispatch_get_main_queue(), ^{
            NSString *title = [NSString stringWithFormat:@"Mandelbulb Path Tracer - %.1f FPS | %u samples | Exposure: %.2f",
                               fps, self->_sampleCount, self->_exposure];
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

        // Create window
        NSRect frame = NSMakeRect(0, 0, Config::WINDOW_WIDTH, Config::WINDOW_HEIGHT);
        NSWindowStyleMask style = NSWindowStyleMaskTitled
                                | NSWindowStyleMaskClosable
                                | NSWindowStyleMaskMiniaturizable
                                | NSWindowStyleMaskResizable;

        NSWindow *window = [[NSWindow alloc] initWithContentRect:frame
                                                       styleMask:style
                                                         backing:NSBackingStoreBuffered
                                                           defer:NO];
        [window setTitle:@"Mandelbulb Path Tracer"];
        [window center];

        // Create Metal view
        InputView *metalView = [[InputView alloc] initWithFrame:frame];
        metalView.clearColor = MTLClearColorMake(Config::BG_RED, Config::BG_GREEN,
                                                  Config::BG_BLUE, Config::BG_ALPHA);
        metalView.preferredFramesPerSecond = 60;

        // Create renderer
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

        NSLog(@"Mandelbulb Path Tracer");
        NSLog(@"----------------------");
        NSLog(@"Controls:");
        NSLog(@"  Left-drag     : Orbit camera");
        NSLog(@"  Right-drag    : Pan camera");
        NSLog(@"  Scroll        : Zoom in/out");
        NSLog(@"  WASD          : Move camera");
        NSLog(@"  E / Shift+D   : Exposure up/down");
        NSLog(@"  R             : Reset camera");
        NSLog(@"  Space         : Pause/Resume");
        NSLog(@"  Q             : Quit");
        NSLog(@"");
        NSLog(@"Progressive rendering: image refines continuously");

        [app run];
    }

    return 0;
}
