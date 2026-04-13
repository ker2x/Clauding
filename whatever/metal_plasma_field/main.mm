// =============================================================================
// Metal Plasma Field Simulation - main.mm
// =============================================================================
// Charged particle simulation with electromagnetic fields.
// Particles exhibit spiraling motion through Lorentz force,
// creating plasma-like visual patterns.
//
// Architecture:
// 1. InputView    - Custom MTKView for keyboard input
// 2. AppDelegate  - Application lifecycle
// 3. Renderer     - Metal rendering and compute pipeline
// =============================================================================

#import <AppKit/AppKit.h>
#import <Metal/Metal.h>
#import <MetalKit/MetalKit.h>
#include <simd/simd.h>
#include <cmath>

// =============================================================================
// CONFIGURATION

namespace Config {
constexpr float WINDOW_WIDTH = 1024.0f;
constexpr float WINDOW_HEIGHT = 768.0f;

constexpr int NUM_PARTICLES = 50000;  // High density for plasma effect
constexpr int NUM_TYPES = 3;          // 3 charge types: negative, neutral, positive

// Physics settings
constexpr float INTERACTION_RADIUS = 40.0f;
constexpr float COULOMB_STRENGTH = 0.015f;    // Charge interaction strength
constexpr float MAGNETIC_STRENGTH = 0.008f;  // Magnetic field strength
constexpr float FRICTION = 0.985f;            // Velocity damping
constexpr float THERMAL_NOISE = 0.05f;        // Random thermal motion

// External field settings
constexpr float ELECTRIC_FIELD_X = 0.0f;      // External E-field
constexpr float ELECTRIC_FIELD_Y = 0.3f;      // Vertical E-field
constexpr float MAGNETIC_FIELD_Z = 1.0f;       // Perpendicular B-field

// Rendering
constexpr float POINT_SIZE = 1.5f;
constexpr int TARGET_FPS = 60;

constexpr float BG_RED = 0.0f;
constexpr float BG_GREEN = 0.0f;
constexpr float BG_BLUE = 0.02f;
constexpr float BG_ALPHA = 1.0f;
} // namespace Config

// =============================================================================
// DATA STRUCTURES

struct SimParams {
    float width;
    float height;
    float interaction_radius;
    float coulomb_strength;
    float magnetic_strength;
    float friction;
    float thermal_noise;
    float electric_field_x;
    float electric_field_y;
    float magnetic_field_z;
    int num_particles;
    int num_types;
    float point_size;
    float time;  // For animation
};

// =============================================================================
// InputView - Keyboard Input

@protocol RendererActions <NSObject>
- (void)randomize;
- (void)togglePause;
- (void)cycleFieldMode;
@end

@interface InputView : MTKView
@property(nonatomic, weak) id<RendererActions> renderer;
@end

@implementation InputView

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
            if (_renderer) [_renderer randomize];
            break;
        case ' ':
            if (_renderer) [_renderer togglePause];
            break;
        case 'f':
        case 'F':
            if (_renderer) [_renderer cycleFieldMode];
            break;
        default:
            [super keyDown:event];
            break;
    }
}

@end

// =============================================================================
// AppDelegate

@interface AppDelegate : NSObject <NSApplicationDelegate>
@end

@implementation AppDelegate

- (BOOL)applicationShouldTerminateAfterLastWindowClosed:(NSApplication *)sender {
    (void)sender;
    return YES;
}

@end

// =============================================================================
// Renderer

@interface Renderer : NSObject <MTKViewDelegate, RendererActions>
- (instancetype)initWithMetalKitView:(MTKView *)view;
- (void)randomize;
- (void)togglePause;
- (void)cycleFieldMode;
@end

@implementation Renderer {
    id<MTLDevice> _device;
    id<MTLCommandQueue> _commandQueue;
    id<MTLComputePipelineState> _computePipeline;
    id<MTLRenderPipelineState> _renderPipeline;
    
    id<MTLBuffer> _positionBuffer;
    id<MTLBuffer> _velocityBuffer;
    id<MTLBuffer> _typeBuffer;
    id<MTLBuffer> _chargeBuffer;      // Charge value per particle
    id<MTLBuffer> _colorBuffer;
    id<MTLBuffer> _paramsBuffer;
    
    MTKView *_view;
    BOOL _paused;
    int _fieldMode;  // 0=uniform field, 1=central pole, 2=vortex
    double _startTime;
    double _lastFrameTime;
    int _frameCount;
    double _fpsUpdateTime;
}

- (instancetype)initWithMetalKitView:(MTKView *)view {
    self = [super init];
    if (!self) return nil;
    
    _view = view;
    _paused = NO;
    _fieldMode = 0;
    _startTime = CACurrentMediaTime();
    _lastFrameTime = _startTime;
    _frameCount = 0;
    _fpsUpdateTime = _startTime;
    
    _device = MTLCreateSystemDefaultDevice();
    if (!_device) {
        NSLog(@"ERROR: Metal not supported");
        return nil;
    }
    NSLog(@"Using GPU: %@", _device.name);
    
    _commandQueue = [_device newCommandQueue];
    if (!_commandQueue) {
        NSLog(@"ERROR: Failed to create command queue");
        return nil;
    }
    
    if (![self loadShaders]) return nil;
    [self initBuffers];
    
    view.device = _device;
    view.delegate = self;
    
    return self;
}

- (BOOL)loadShaders {
    NSError *error = nil;
    
    NSString *shaderPath = @"Compute.metal";
    NSString *shaderSource = [NSString stringWithContentsOfFile:shaderPath
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
        options.fastMathEnabled = YES;
    }
    
    id<MTLLibrary> library = [_device newLibraryWithSource:shaderSource
                                                  options:options
                                                    error:&error];
    if (!library) {
        NSLog(@"ERROR: Shader compilation failed: %@", error.localizedDescription);
        return NO;
    }
    NSLog(@"Shaders compiled");
    
    id<MTLFunction> computeFunc = [library newFunctionWithName:@"update_plasma"];
    if (!computeFunc) {
        NSLog(@"ERROR: Could not find 'update_plasma'");
        return NO;
    }
    
    _computePipeline = [_device newComputePipelineStateWithFunction:computeFunc
                                                               error:&error];
    if (!_computePipeline) {
        NSLog(@"ERROR: Compute pipeline failed: %@", error.localizedDescription);
        return NO;
    }
    
    id<MTLFunction> vertexFunc = [library newFunctionWithName:@"vertex_main"];
    id<MTLFunction> fragmentFunc = [library newFunctionWithName:@"fragment_main"];
    
    if (!vertexFunc || !fragmentFunc) {
        NSLog(@"ERROR: Missing vertex/fragment functions");
        return NO;
    }
    
    MTLRenderPipelineDescriptor *pipelineDesc = [[MTLRenderPipelineDescriptor alloc] init];
    pipelineDesc.vertexFunction = vertexFunc;
    pipelineDesc.fragmentFunction = fragmentFunc;
    pipelineDesc.colorAttachments[0].pixelFormat = _view.colorPixelFormat;
    
    pipelineDesc.colorAttachments[0].blendingEnabled = YES;
    pipelineDesc.colorAttachments[0].sourceRGBBlendFactor = MTLBlendFactorSourceAlpha;
    pipelineDesc.colorAttachments[0].destinationRGBBlendFactor = MTLBlendFactorOneMinusSourceAlpha;
    pipelineDesc.colorAttachments[0].rgbBlendOperation = MTLBlendOperationAdd;
    pipelineDesc.colorAttachments[0].sourceAlphaBlendFactor = MTLBlendFactorOne;
    pipelineDesc.colorAttachments[0].destinationAlphaBlendFactor = MTLBlendFactorOneMinusSourceAlpha;
    pipelineDesc.colorAttachments[0].alphaBlendOperation = MTLBlendOperationAdd;
    
    _renderPipeline = [_device newRenderPipelineStateWithDescriptor:pipelineDesc error:&error];
    if (!_renderPipeline) {
        NSLog(@"ERROR: Render pipeline failed: %@", error.localizedDescription);
        return NO;
    }
    
    return YES;
}

- (void)initBuffers {
    const int numParticles = Config::NUM_PARTICLES;
    const int numTypes = Config::NUM_TYPES;
    
    _positionBuffer = [_device newBufferWithLength:numParticles * sizeof(simd_float2)
                                          options:MTLResourceStorageModeShared];
    _velocityBuffer = [_device newBufferWithLength:numParticles * sizeof(simd_float2)
                                          options:MTLResourceStorageModeShared];
    _typeBuffer = [_device newBufferWithLength:numParticles * sizeof(int)
                                       options:MTLResourceStorageModeShared];
    _chargeBuffer = [_device newBufferWithLength:numParticles * sizeof(float)
                                         options:MTLResourceStorageModeShared];
    _colorBuffer = [_device newBufferWithLength:numTypes * sizeof(simd_float4)
                                       options:MTLResourceStorageModeShared];
    _paramsBuffer = [_device newBufferWithLength:sizeof(SimParams)
                                        options:MTLResourceStorageModeShared];
    
    [self initParticles];
    [self initColors];
    [self updateParams];
}

- (void)initParticles {
    simd_float2 *positions = (simd_float2 *)_positionBuffer.contents;
    simd_float2 *velocities = (simd_float2 *)_velocityBuffer.contents;
    int *types = (int *)_typeBuffer.contents;
    float *charges = (float *)_chargeBuffer.contents;
    
    for (int i = 0; i < Config::NUM_PARTICLES; i++) {
        // Initial distribution: clustered near center with some spread
        float angle = (float)arc4random() / UINT32_MAX * 2.0f * (float)M_PI;
        float radius = (float)arc4random() / UINT32_MAX * Config::WINDOW_WIDTH * 0.4f;
        
        positions[i].x = Config::WINDOW_WIDTH * 0.5f + cosf(angle) * radius;
        positions[i].y = Config::WINDOW_HEIGHT * 0.5f + sinf(angle) * radius;
        
        // Initial velocity: tangential for vortex effect
        float speed = 0.5f + (float)arc4random() / UINT32_MAX * 2.0f;
        velocities[i].x = -sinf(angle) * speed;
        velocities[i].y = cosf(angle) * speed;
        
        // Charge distribution: 40% negative, 20% neutral, 40% positive
        float r = (float)arc4random() / UINT32_MAX;
        if (r < 0.4f) {
            types[i] = 0;      // Negative
            charges[i] = -1.0f;
        } else if (r < 0.6f) {
            types[i] = 1;      // Neutral
            charges[i] = 0.0f;
        } else {
            types[i] = 2;      // Positive
            charges[i] = 1.0f;
        }
    }
}

- (void)initColors {
    simd_float4 *colors = (simd_float4 *)_colorBuffer.contents;
    
    // Negative: Cyan/Blue (electrons)
    colors[0] = simd_make_float4(0.1f, 0.6f, 1.0f, 0.8f);
    // Neutral: White/Gray
    colors[1] = simd_make_float4(0.9f, 0.9f, 0.9f, 0.4f);
    // Positive: Orange/Red (ions)
    colors[2] = simd_make_float4(1.0f, 0.4f, 0.1f, 0.8f);
}

- (void)updateParams {
    SimParams *params = (SimParams *)_paramsBuffer.contents;
    
    params->width = Config::WINDOW_WIDTH;
    params->height = Config::WINDOW_HEIGHT;
    params->interaction_radius = Config::INTERACTION_RADIUS;
    params->coulomb_strength = Config::COULOMB_STRENGTH;
    params->magnetic_strength = Config::MAGNETIC_STRENGTH;
    params->friction = Config::FRICTION;
    params->thermal_noise = Config::THERMAL_NOISE;
    params->electric_field_x = Config::ELECTRIC_FIELD_X;
    params->electric_field_y = Config::ELECTRIC_FIELD_Y;
    params->magnetic_field_z = Config::MAGNETIC_FIELD_Z;
    params->num_particles = Config::NUM_PARTICLES;
    params->num_types = Config::NUM_TYPES;
    params->point_size = Config::POINT_SIZE;
    params->time = CACurrentMediaTime() - _startTime;
}

- (void)randomize {
    [self initParticles];
}

- (void)togglePause {
    _paused = !_paused;
    NSLog(@"%@", _paused ? @"Paused" : @"Resumed");
}

- (void)cycleFieldMode {
    _fieldMode = (_fieldMode + 1) % 3;
    const char *modes[] = {"Uniform E+B Field", "Central Pole (1/r^2)", "Rotating Vortex"};
    NSLog(@"Field Mode: %s", modes[_fieldMode]);
}

- (void)drawInMTKView:(MTKView *)view {
    [self updateFPS];
    
    id<MTLCommandBuffer> commandBuffer = [_commandQueue commandBuffer];
    if (!commandBuffer) return;
    
    if (!_paused) {
        [self updateParams];
        
        id<MTLComputeCommandEncoder> computeEncoder = [commandBuffer computeCommandEncoder];
        [computeEncoder setComputePipelineState:_computePipeline];
        
        [computeEncoder setBuffer:_positionBuffer offset:0 atIndex:0];
        [computeEncoder setBuffer:_velocityBuffer offset:0 atIndex:1];
        [computeEncoder setBuffer:_typeBuffer offset:0 atIndex:2];
        [computeEncoder setBuffer:_chargeBuffer offset:0 atIndex:3];
        [computeEncoder setBuffer:_paramsBuffer offset:0 atIndex:4];
        
        // Pass field mode to shader via params buffer
        SimParams *params = (SimParams *)_paramsBuffer.contents;
        params->magnetic_field_z = (float)_fieldMode;
        
        NSUInteger threadsPerGroup = MIN(_computePipeline.maxTotalThreadsPerThreadgroup, 256);
        MTLSize gridSize = MTLSizeMake(Config::NUM_PARTICLES, 1, 1);
        MTLSize groupSize = MTLSizeMake(threadsPerGroup, 1, 1);
        
        [computeEncoder dispatchThreads:gridSize threadsPerThreadgroup:groupSize];
        [computeEncoder endEncoding];
    }
    
    MTLRenderPassDescriptor *renderPassDesc = view.currentRenderPassDescriptor;
    if (!renderPassDesc) return;
    
    id<MTLRenderCommandEncoder> renderEncoder = [commandBuffer renderCommandEncoderWithDescriptor:renderPassDesc];
    [renderEncoder setRenderPipelineState:_renderPipeline];
    
    [renderEncoder setVertexBuffer:_positionBuffer offset:0 atIndex:0];
    [renderEncoder setVertexBuffer:_typeBuffer offset:0 atIndex:1];
    [renderEncoder setVertexBuffer:_colorBuffer offset:0 atIndex:2];
    [renderEncoder setVertexBuffer:_chargeBuffer offset:0 atIndex:3];
    [renderEncoder setVertexBuffer:_paramsBuffer offset:0 atIndex:4];
    
    [renderEncoder drawPrimitives:MTLPrimitiveTypePoint
                      vertexStart:0
                      vertexCount:Config::NUM_PARTICLES];
    
    [renderEncoder endEncoding];
    [commandBuffer presentDrawable:view.currentDrawable];
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
            NSString *title = [NSString stringWithFormat:@"Plasma Field - %.1f FPS", fps];
            [self->_view.window setTitle:title];
        });
        
        _frameCount = 0;
        _fpsUpdateTime = currentTime;
    }
}

@end

// =============================================================================
// MAIN

int main(int argc, const char *argv[]) {
    (void)argc;
    (void)argv;
    
    @autoreleasepool {
        NSApplication *app = [NSApplication sharedApplication];
        [app setActivationPolicy:NSApplicationActivationPolicyRegular];
        
        AppDelegate *delegate = [[AppDelegate alloc] init];
        [app setDelegate:delegate];
        
        NSRect frame = NSMakeRect(0, 0, Config::WINDOW_WIDTH, Config::WINDOW_HEIGHT);
        NSWindowStyleMask style = NSWindowStyleMaskTitled | NSWindowStyleMaskClosable |
                                  NSWindowStyleMaskMiniaturizable | NSWindowStyleMaskResizable;
        
        NSWindow *window = [[NSWindow alloc] initWithContentRect:frame
                                                       styleMask:style
                                                         backing:NSBackingStoreBuffered
                                                           defer:NO];
        [window setTitle:@"Plasma Field"];
        [window center];
        
        InputView *metalView = [[InputView alloc] initWithFrame:frame];
        metalView.clearColor = MTLClearColorMake(Config::BG_RED, Config::BG_GREEN,
                                                 Config::BG_BLUE, Config::BG_ALPHA);
        metalView.preferredFramesPerSecond = Config::TARGET_FPS;
        
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
        
        NSLog(@"Plasma Field Simulation");
        NSLog(@"Controls:");
        NSLog(@"  Q - Quit");
        NSLog(@"  R - Reset particles");
        NSLog(@"  F - Cycle field mode");
        NSLog(@"  Space - Pause/Resume");
        
        [app run];
    }
    
    return 0;
}
