#import <AppKit/AppKit.h>
#import <Metal/Metal.h>
#import <MetalKit/MetalKit.h>
#import <simd/simd.h>

@interface InputView : MTKView
@property(nonatomic, assign) id renderer;
@end

@interface Renderer : NSObject <MTKViewDelegate>
- (instancetype)initWithMetalKitView:(MTKView *)view;
- (void)reset;
- (void)handleMouseDrag:(NSPoint)delta;
@end

static const uint32_t WIDTH = 1280;
static const uint32_t HEIGHT = 720;
static const uint32_t GRID_SIZE = 128; // 128x128x128 grid for NCA

struct Uniforms {
  simd_float4x4 viewMatrix;
  simd_float4x4 projectionMatrix;
  simd_float2 resolution;
  float time;
  uint32_t gridSize;
  float growthRate;
  simd_float2 mousePos;
  int mouseDown;
};

@implementation InputView
- (BOOL)acceptsFirstResponder {
  return YES;
}
- (void)mouseDragged:(NSEvent *)event {
  [(Renderer *)self.renderer
      handleMouseDrag:NSMakePoint([event deltaX], [event deltaY])];
}
- (void)keyDown:(NSEvent *)event {
  if ([[event characters] isEqualToString:@"r"])
    [(Renderer *)self.renderer reset];
  if ([[event characters] isEqualToString:@"q"])
    [NSApp terminate:nil];
}
@end

@implementation Renderer {
  id<MTLDevice> _device;
  id<MTLCommandQueue> _commandQueue;
  id<MTLRenderPipelineState> _renderPipeline;
  id<MTLComputePipelineState> _ncaPipeline;
  id<MTLComputePipelineState> _rtPipeline;

  id<MTLBuffer> _ncaBufferA;
  id<MTLBuffer> _ncaBufferB;
  id<MTLBuffer> _uniformBuffer;

  id<MTLTexture> _renderTarget;
  float _time;
  simd_float2 _rotation;
}

- (instancetype)initWithMetalKitView:(MTKView *)view {
  self = [super init];
  if (self) {
    _device = view.device;
    _commandQueue = [_device newCommandQueue];

    [self loadShaders];
    [self initResources];
  }
  return self;
}

- (void)loadShaders {
  NSError *error = nil;
  NSString *source = [NSString stringWithContentsOfFile:@"Simulation.metal"
                                               encoding:NSUTF8StringEncoding
                                                  error:&error];
  if (!source) {
    NSLog(@"Failed to load Simulation.metal: %@", error);
    return;
  }

  id<MTLLibrary> library = [_device newLibraryWithSource:source
                                                 options:nil
                                                   error:&error];
  if (!library) {
    NSLog(@"Failed to compile library: %@", error);
    return;
  }

  id<MTLFunction> rtFunc = [library newFunctionWithName:@"raytrace_kernel"];
  _rtPipeline = [_device newComputePipelineStateWithFunction:rtFunc
                                                       error:&error];

  id<MTLFunction> ncaFunc = [library newFunctionWithName:@"nca_update_kernel"];
  _ncaPipeline = [_device newComputePipelineStateWithFunction:ncaFunc
                                                        error:&error];

  MTLRenderPipelineDescriptor *rpDesc = [MTLRenderPipelineDescriptor new];
  rpDesc.vertexFunction = [library newFunctionWithName:@"vertex_main"];
  rpDesc.fragmentFunction = [library newFunctionWithName:@"fragment_main"];
  rpDesc.colorAttachments[0].pixelFormat = MTLPixelFormatBGRA8Unorm;
  _renderPipeline = [_device newRenderPipelineStateWithDescriptor:rpDesc
                                                            error:&error];
}

- (void)initResources {
  // 128^3 grid, each cell is 16 bytes (float4: dens, temp, state, etc)
  NSUInteger bufferSize =
      GRID_SIZE * GRID_SIZE * GRID_SIZE * sizeof(simd_float4);
  _ncaBufferA = [_device newBufferWithLength:bufferSize
                                     options:MTLResourceStorageModeShared];
  _ncaBufferB = [_device newBufferWithLength:bufferSize
                                     options:MTLResourceStorageModeShared];

  _uniformBuffer = [_device newBufferWithLength:sizeof(Uniforms)
                                        options:MTLResourceStorageModeShared];

  MTLTextureDescriptor *tDesc = [MTLTextureDescriptor
      texture2DDescriptorWithPixelFormat:MTLPixelFormatRGBA16Float
                                   width:WIDTH
                                  height:HEIGHT
                               mipmapped:NO];
  tDesc.usage = MTLTextureUsageShaderRead | MTLTextureUsageShaderWrite;
  _renderTarget = [_device newTextureWithDescriptor:tDesc];

  // Initial Seed
  simd_float4 *cells = (simd_float4 *)[_ncaBufferA contents];
  memset(cells, 0, bufferSize);
  cells[(GRID_SIZE / 2) * GRID_SIZE * GRID_SIZE + (GRID_SIZE / 2) * GRID_SIZE +
        (GRID_SIZE / 2)] = (simd_float4){1.0, 0, 0, 0};
}

- (void)reset {
  simd_float4 *cells = (simd_float4 *)[_ncaBufferA contents];
  memset(cells, 0, GRID_SIZE * GRID_SIZE * GRID_SIZE * sizeof(simd_float4));
  cells[(GRID_SIZE / 2) * GRID_SIZE * GRID_SIZE + (GRID_SIZE / 2) * GRID_SIZE +
        (GRID_SIZE / 2)] = (simd_float4){1.0, 0, 0, 0};
}

- (void)handleMouseDrag:(NSPoint)delta {
  _rotation.x += delta.x * 0.01f;
  _rotation.y += delta.y * 0.01f;
}

- (void)drawInMTKView:(nonnull MTKView *)view {
  _time += 1.0f / 60.0f;

  Uniforms uniforms;
  uniforms.resolution = (simd_float2){(float)WIDTH, (float)HEIGHT};
  uniforms.time = _time;
  uniforms.gridSize = GRID_SIZE;
  uniforms.growthRate = 0.15f;
  uniforms.mousePos = _rotation; // Use for rotation
  uniforms.mouseDown = 0; // Not used in this snippet, but added to struct
  memcpy([_uniformBuffer contents], &uniforms, sizeof(Uniforms));

  id<MTLCommandBuffer> cmdBuf = [_commandQueue commandBuffer];

  // 0. NCA Simulation Pass
  id<MTLComputeCommandEncoder> nce = [cmdBuf computeCommandEncoder];
  [nce setComputePipelineState:_ncaPipeline];
  [nce setBuffer:_ncaBufferA offset:0 atIndex:0];
  [nce setBuffer:_ncaBufferB offset:0 atIndex:1];
  [nce setBuffer:_uniformBuffer offset:0 atIndex:2];

  MTLSize ncaGroup = MTLSizeMake(8, 8, 8);
  MTLSize ncaGrid = MTLSizeMake(GRID_SIZE, GRID_SIZE, GRID_SIZE);
  [nce dispatchThreads:ncaGrid threadsPerThreadgroup:ncaGroup];
  [nce endEncoding];

  // Swap buffers
  id<MTLBuffer> tmp = _ncaBufferA;
  _ncaBufferA = _ncaBufferB;
  _ncaBufferB = tmp;

  // 1. Ray Tracing Pass
  id<MTLComputeCommandEncoder> ce = [cmdBuf computeCommandEncoder];
  [ce setComputePipelineState:_rtPipeline];
  [ce setBuffer:_ncaBufferA offset:0 atIndex:0];
  [ce setBuffer:_uniformBuffer offset:0 atIndex:1];
  [ce setTexture:_renderTarget atIndex:0];

  MTLSize threadsPerGroup = MTLSizeMake(16, 16, 1);
  MTLSize groupsPerGrid = MTLSizeMake((WIDTH + 15) / 16, (HEIGHT + 15) / 16, 1);
  [ce dispatchThreadgroups:groupsPerGrid threadsPerThreadgroup:threadsPerGroup];
  [ce endEncoding];

  // 2. Display Pass
  MTLRenderPassDescriptor *rpd = view.currentRenderPassDescriptor;
  if (rpd) {
    id<MTLRenderCommandEncoder> re =
        [cmdBuf renderCommandEncoderWithDescriptor:rpd];
    [re setRenderPipelineState:_renderPipeline];
    [re setFragmentTexture:_renderTarget atIndex:0];
    [re drawPrimitives:MTLPrimitiveTypeTriangleStrip
           vertexStart:0
           vertexCount:4];
    [re endEncoding];
    [cmdBuf presentDrawable:view.currentDrawable];
  }

  [cmdBuf commit];
}

- (void)mtkView:(nonnull MTKView *)view drawableSizeWillChange:(CGSize)size {
}
@end

int main(int argc, char *argv[]) {
  @autoreleasepool {
    [NSApplication sharedApplication];
    [NSApp setActivationPolicy:NSApplicationActivationPolicyRegular];

    NSRect frame = NSMakeRect(0, 0, WIDTH, HEIGHT);
    NSWindow *window =
        [[NSWindow alloc] initWithContentRect:frame
                                    styleMask:(NSWindowStyleMaskTitled |
                                               NSWindowStyleMaskClosable)
                                      backing:NSBackingStoreBuffered
                                        defer:NO];
    window.title = @"M3 Neural Crystal | Ray Tracing & NCA";
    [window makeKeyAndOrderFront:nil];

    InputView *view =
        [[InputView alloc] initWithFrame:frame
                                  device:MTLCreateSystemDefaultDevice()];
    view.colorPixelFormat = MTLPixelFormatBGRA8Unorm;
    view.preferredFramesPerSecond = 60;

    Renderer *renderer = [[Renderer alloc] initWithMetalKitView:view];
    view.delegate = renderer;
    view.renderer = renderer;
    window.contentView = view;
    [window makeFirstResponder:view];

    // Add Menu for Quit
    NSMenu *mainMenu = [[NSMenu alloc] init];
    NSMenuItem *appMenuItem = [[NSMenuItem alloc] init];
    [mainMenu addItem:appMenuItem];
    [NSApp setMainMenu:mainMenu];

    NSMenu *appMenu = [[NSMenu alloc] init];
    [appMenu addItemWithTitle:@"Quit"
                       action:@selector(terminate:)
                keyEquivalent:@"q"];
    [appMenuItem setSubmenu:appMenu];

    [NSApp activateIgnoringOtherApps:YES];
    [NSApp run];
  }
  return 0;
}
