#import <AppKit/AppKit.h>
#import <Metal/Metal.h>
#import <MetalKit/MetalKit.h>
#include <algorithm>
#include <cmath>
#include <random>
#include <simd/simd.h>
#include <vector>

static const uint WIDTH = 1200;
static const uint HEIGHT = 800;
static const uint NUM_PARTICLES = 30000;

struct Particle {
    simd_float2 position;
    simd_float2 velocity;
    simd_float3 color;
    float age;
    float hue_phase;
    float speed_amplitude;
};

struct Params {
    uint width;
    uint height;
    float time;
    float time_scale;
    float attraction_strength;
    float color_rotation;
    float bloom_intensity;
    float chaos;
    uint num_particles;
    int padding[3];
};

@interface BloomRenderer : NSObject <MTKViewDelegate>
@property(nonatomic, weak) MTKView *view;
- (void)initResources;
- (void)resetTime;
- (void)cycleBloom;
- (void)cycleChaos;
@end

@interface BloomView : MTKView
@property(nonatomic, assign) Params params;
@end

@implementation BloomView
- (BOOL)acceptsFirstResponder {
    return YES;
}

- (void)keyDown:(NSEvent *)event {
    BloomRenderer *renderer = (BloomRenderer *)self.delegate;
    if ([[event characters] isEqualToString:@"r"]) {
        [renderer resetTime];
    } else if ([[event characters] isEqualToString:@" "]) {
        [renderer cycleBloom];
    } else if ([[event characters] isEqualToString:@"c"]) {
        [renderer cycleChaos];
    }
}
@end

@implementation BloomRenderer {
    id<MTLDevice> device;
    id<MTLCommandQueue> commandQueue;
    id<MTLComputePipelineState> updatePipeline;
    id<MTLRenderPipelineState> renderPipeline;
    id<MTLBuffer> particleBuffer;
    id<MTLBuffer> paramsBuffer;
    std::vector<Particle> particles;
    Params params;
    float frame_counter;
    float bloom_val;
    float chaos_val;
}

- (void)resetTime {
    frame_counter = 0.0f;
}

- (void)cycleBloom {
    bloom_val = fmod(bloom_val + 0.2f, 2.0f);
    params.bloom_intensity = bloom_val;
}

- (void)cycleChaos {
    chaos_val = fmod(chaos_val + 0.2f, 1.0f);
    params.chaos = chaos_val;
}

- (instancetype)initWithMetalKitView:(BloomView *)view {
    self = [super init];
    if (self) {
        device = view.device;
        commandQueue = [device newCommandQueue];
        _view = view;

        NSError *error = nil;
        NSString *source = [NSString stringWithContentsOfFile:@"Bloom.metal"
                                                     encoding:NSUTF8StringEncoding
                                                        error:&error];
        if (!source) {
            NSLog(@"Failed to load Bloom.metal: %@", error);
            return nil;
        }

        id<MTLLibrary> library = [device newLibraryWithSource:source
                                                      options:nil
                                                        error:&error];
        if (!library) {
            NSLog(@"Metal library compilation failed: %@", error);
            return nil;
        }

        // Compute pipeline for particle updates
        updatePipeline = [device newComputePipelineStateWithFunction:
                                      [library newFunctionWithName:@"update_particles"]
                                                              error:&error];
        if (!updatePipeline)
            NSLog(@"Failed to create updatePipeline: %@", error);

        // Render pipeline for bloom visualization
        MTLRenderPipelineDescriptor *renderDesc = [[MTLRenderPipelineDescriptor alloc] init];
        renderDesc.vertexFunction = [library newFunctionWithName:@"vertex_bloom"];
        renderDesc.fragmentFunction = [library newFunctionWithName:@"fragment_bloom"];
        renderDesc.colorAttachments[0].pixelFormat = view.colorPixelFormat;
        renderDesc.colorAttachments[0].blendingEnabled = YES;
        renderDesc.colorAttachments[0].sourceRGBBlendFactor = MTLBlendFactorOne;
        renderDesc.colorAttachments[0].destinationRGBBlendFactor = MTLBlendFactorOne;
        renderDesc.colorAttachments[0].rgbBlendOperation = MTLBlendOperationAdd;

        renderPipeline = [device newRenderPipelineStateWithDescriptor:renderDesc error:&error];
        if (!renderPipeline)
            NSLog(@"Failed to create renderPipeline: %@", error);

        [self initResources];
    }
    return self;
}

- (void)initResources {
    particles.resize(NUM_PARTICLES);
    std::mt19937 rng(std::random_device{}());
    std::uniform_real_distribution<float> dist(0.0f, 1.0f);

    for (uint i = 0; i < NUM_PARTICLES; i++) {
        auto &p = particles[i];
        p.position = {dist(rng) * WIDTH, dist(rng) * HEIGHT};
        p.velocity = {(dist(rng) - 0.5f) * 2.0f, (dist(rng) - 0.5f) * 2.0f};

        // Initialize with HSV colors spread across hue
        float hue = (float)i / NUM_PARTICLES;
        float h = hue;
        float c = 0.7f;  // saturation
        float x = c * (1.0f - abs(fmod(h * 6.0f, 2.0f) - 1.0f));
        float m = 0.8f - c;  // brightness

        if (h < 1.0f/6.0f) p.color = {c + m, x + m, 0 + m};
        else if (h < 2.0f/6.0f) p.color = {x + m, c + m, 0 + m};
        else if (h < 3.0f/6.0f) p.color = {0 + m, c + m, x + m};
        else if (h < 4.0f/6.0f) p.color = {0 + m, x + m, c + m};
        else if (h < 5.0f/6.0f) p.color = {x + m, 0 + m, c + m};
        else p.color = {c + m, 0 + m, x + m};

        p.age = 0.0f;
        p.hue_phase = hue * 3.14159f;
        p.speed_amplitude = 1.0f;
    }

    particleBuffer = [device newBufferWithBytes:particles.data()
                                         length:sizeof(Particle) * NUM_PARTICLES
                                        options:MTLResourceStorageModeShared];

    params = {WIDTH, HEIGHT, 0.0f, 0.5f, 2.0f, 0.0f, 1.0f, 0.3f, NUM_PARTICLES, {0, 0, 0}};
    paramsBuffer = [device newBufferWithLength:sizeof(Params)
                                       options:MTLResourceStorageModeShared];

    frame_counter = 0.0f;
    bloom_val = 1.0f;
    chaos_val = 0.3f;
}

- (void)drawInMTKView:(nonnull MTKView *)view {
    id<MTLCommandBuffer> commandBuffer = [commandQueue commandBuffer];

    // Update time FIRST
    frame_counter += 0.016f;

    Params p = ((BloomView *)view).params;
    p.width = WIDTH;
    p.height = HEIGHT;
    p.num_particles = NUM_PARTICLES;
    p.time = frame_counter;

    // Update particle parameters
    memcpy([paramsBuffer contents], &p, sizeof(Params));

    // 1. Update particles on GPU
    id<MTLComputeCommandEncoder> computeEncoder = [commandBuffer computeCommandEncoder];
    [computeEncoder setComputePipelineState:updatePipeline];
    [computeEncoder setBuffer:particleBuffer offset:0 atIndex:0];
    [computeEncoder setBuffer:paramsBuffer offset:0 atIndex:1];
    [computeEncoder dispatchThreads:MTLSizeMake(NUM_PARTICLES, 1, 1)
                threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
    [computeEncoder endEncoding];

    // 2. Render particles to screen
    MTLRenderPassDescriptor *rpd = view.currentRenderPassDescriptor;
    if (rpd) {
        rpd.colorAttachments[0].clearColor = MTLClearColorMake(0.05, 0.05, 0.08, 1.0);
        rpd.colorAttachments[0].loadAction = MTLLoadActionClear;

        id<MTLRenderCommandEncoder> renderEncoder =
            [commandBuffer renderCommandEncoderWithDescriptor:rpd];
        [renderEncoder setRenderPipelineState:renderPipeline];
        [renderEncoder setVertexBuffer:particleBuffer offset:0 atIndex:0];
        [renderEncoder setVertexBuffer:paramsBuffer offset:0 atIndex:1];
        [renderEncoder drawPrimitives:MTLPrimitiveTypeTriangle vertexStart:0 vertexCount:NUM_PARTICLES * 6];
        [renderEncoder endEncoding];

        [commandBuffer presentDrawable:view.currentDrawable];
    }

    [commandBuffer commit];
}

- (void)mtkView:(nonnull MTKView *)view drawableSizeWillChange:(CGSize)size {
}
@end

int main(int argc, char *argv[]) {
    @autoreleasepool {
        [NSApplication sharedApplication];
        [NSApp setActivationPolicy:NSApplicationActivationPolicyRegular];

        NSRect frame = NSMakeRect(0, 0, WIDTH, HEIGHT);
        NSWindow *window = [[NSWindow alloc]
            initWithContentRect:frame
                      styleMask:(NSWindowStyleMaskTitled | NSWindowStyleMaskClosable |
                                 NSWindowStyleMaskResizable)
                        backing:NSBackingStoreBuffered
                          defer:NO];
        window.title = @"Chromatic Bloom | Generative Art";
        [window makeKeyAndOrderFront:nil];

        BloomView *view = [[BloomView alloc] initWithFrame:frame
                                                    device:MTLCreateSystemDefaultDevice()];
        view.colorPixelFormat = MTLPixelFormatBGRA8Unorm;
        view.clearColor = MTLClearColorMake(0.05, 0.05, 0.08, 1.0);
        view.drawableSize = CGSizeMake(WIDTH, HEIGHT);
        view.autoResizeDrawable = NO;

        BloomRenderer *renderer = [[BloomRenderer alloc] initWithMetalKitView:view];
        view.delegate = renderer;
        [window setContentView:view];
        [window makeFirstResponder:view];

        NSMenu *mainMenu = [[NSMenu alloc] init];
        NSMenuItem *appMenuItem = [[NSMenuItem alloc] initWithTitle:@"App" action:NULL keyEquivalent:@""];
        NSMenu *appMenu = [[NSMenu alloc] initWithTitle:@"App"];
        [appMenu addItemWithTitle:@"Quit" action:@selector(terminate:) keyEquivalent:@"q"];
        [appMenuItem setSubmenu:appMenu];
        [mainMenu addItem:appMenuItem];
        [NSApp setMainMenu:mainMenu];

        [NSApp activateIgnoringOtherApps:YES];
        [NSApp run];
    }
    return 0;
}
