#import <AppKit/AppKit.h>
#import <Metal/Metal.h>
#import <MetalKit/MetalKit.h>
#import <simd/simd.h>

static const uint32_t WIDTH = 1280;
static const uint32_t HEIGHT = 720;

struct Uniforms {
  simd_float2 resolution;
  float time;
  float mouseX;
  float mouseY;
  int mouseDown;
  float mu;    // Lenia growth center
  float sigma; // Lenia growth width
  float rho;   // Kernel radius
  float dt;    // Time step
};

@interface Renderer : NSObject <MTKViewDelegate>
- (instancetype)initWithMetalKitView:(MTKView *)view;
- (void)reset;
- (void)setMouseDown:(BOOL)down;
- (void)updateMousePos:(simd_float2)pos;
@end

@interface InputView : MTKView
@property(nonatomic, assign) id renderer;
@end

@implementation InputView
- (BOOL)acceptsFirstResponder {
  return YES;
}
- (void)mouseDown:(NSEvent *)event {
  [(Renderer *)self.renderer setMouseDown:YES];
}
- (void)mouseUp:(NSEvent *)event {
  [(Renderer *)self.renderer setMouseDown:NO];
}
- (void)mouseDragged:(NSEvent *)event {
  NSPoint p = [self convertPoint:[event locationInWindow] fromView:nil];
  [(Renderer *)self.renderer updateMousePos:simd_make_float2(p.x, p.y)];
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

  id<MTLComputePipelineState> _leniaPipeline;
  id<MTLRenderPipelineState> _displayPipeline;

  id<MTLTexture> _stateTex[2]; // Double buffering for Lenia
  id<MTLTexture> _kernelTex;   // Spatial kernel
  id<MTLBuffer> _uniformBuffer;

  float _time;
  simd_float2 _mousePos;
  BOOL _isMouseDown;

  // Species Parameters
  float _mu;
  float _sigma;
  float _rho;

  uint32_t _frameIdx;
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
  id<MTLLibrary> library = [_device newLibraryWithSource:source
                                                 options:nil
                                                   error:&error];
  if (!library) {
    NSLog(@"Library error: %@", error.localizedDescription);
    [NSApp terminate:nil];
  }

  _leniaPipeline = [_device newComputePipelineStateWithFunction:
                                [library newFunctionWithName:@"lenia_kernel"]
                                                          error:&error];

  MTLRenderPipelineDescriptor *rpDesc = [MTLRenderPipelineDescriptor new];
  rpDesc.vertexFunction = [library newFunctionWithName:@"vertex_main"];
  rpDesc.fragmentFunction = [library newFunctionWithName:@"fragment_main"];
  rpDesc.colorAttachments[0].pixelFormat = MTLPixelFormatBGRA8Unorm;
  _displayPipeline = [_device newRenderPipelineStateWithDescriptor:rpDesc
                                                             error:&error];
}

- (void)initResources {
  MTLTextureDescriptor *tDesc = [MTLTextureDescriptor
      texture2DDescriptorWithPixelFormat:MTLPixelFormatR32Float
                                   width:WIDTH
                                  height:HEIGHT
                               mipmapped:NO];
  tDesc.usage = MTLTextureUsageShaderRead | MTLTextureUsageShaderWrite;
  _stateTex[0] = [_device newTextureWithDescriptor:tDesc];
  _stateTex[1] = [_device newTextureWithDescriptor:tDesc];

  // Kernel texture (smaller, e.g., 64x64)
  MTLTextureDescriptor *kDesc = [MTLTextureDescriptor
      texture2DDescriptorWithPixelFormat:MTLPixelFormatR32Float
                                   width:128
                                  height:128
                               mipmapped:NO];
  kDesc.usage = MTLTextureUsageShaderRead | MTLTextureUsageShaderWrite;
  _kernelTex = [_device newTextureWithDescriptor:kDesc];

  _uniformBuffer = [_device newBufferWithLength:sizeof(Uniforms)
                                        options:MTLResourceStorageModeShared];
  [self reset];
}

- (void)reset {
  _time = 0;

  // Randomize Species Parameters
  _mu = 0.12f + (rand() % 40) / 1000.0f;
  _sigma = 0.012f + (rand() % 80) / 10000.0f;
  _rho = 12.0f + (rand() % 40) / 10.0f;

  NSLog(@"New Species: mu=%.3f, sigma=%.3f, rho=%.1f", _mu, _sigma, _rho);

  // Clear state with some noise
  float *noise = (float *)malloc(WIDTH * HEIGHT * sizeof(float));
  for (int i = 0; i < WIDTH * HEIGHT; i++) {
    noise[i] = (rand() % 100 < 15) ? (float)(rand() % 100) / 100.0f : 0.0f;
    if (noise[i] > 0)
      noise[i] = 0.5 + noise[i] * 0.5;
  }
  for (int i = 0; i < 2; i++) {
    [_stateTex[i] replaceRegion:MTLRegionMake2D(0, 0, WIDTH, HEIGHT)
                    mipmapLevel:0
                      withBytes:noise
                    bytesPerRow:WIDTH * sizeof(float)];
  }
  free(noise);
}

- (void)setMouseDown:(BOOL)down {
  _isMouseDown = down;
}
- (void)updateMousePos:(simd_float2)pos {
  _mousePos = pos;
}

- (void)drawInMTKView:(nonnull MTKView *)view {
  _time += 1.0 / 60.0;
  _frameIdx++;

  Uniforms uniforms;
  uniforms.resolution = (simd_float2){(float)WIDTH, (float)HEIGHT};
  uniforms.time = _time;
  uniforms.mouseX = _mousePos.x;
  uniforms.mouseY = _mousePos.y;
  uniforms.mouseDown = _isMouseDown ? 1 : 0;
  uniforms.mu = _mu;
  uniforms.sigma = _sigma;
  uniforms.rho = _rho;
  uniforms.dt = 0.05f;
  memcpy([_uniformBuffer contents], &uniforms, sizeof(Uniforms));

  id<MTLCommandBuffer> cmdBuf = [_commandQueue commandBuffer];

  // 1. Lenia Simulation (Convolution + Growth)
  id<MTLComputeCommandEncoder> ce = [cmdBuf computeCommandEncoder];
  [ce setComputePipelineState:_leniaPipeline];
  [ce setTexture:_stateTex[0] atIndex:0]; // In
  [ce setTexture:_stateTex[1] atIndex:1]; // Out
  [ce setBuffer:_uniformBuffer offset:0 atIndex:0];
  MTLSize threads = MTLSizeMake(16, 16, 1);
  MTLSize grid = MTLSizeMake((WIDTH + 15) / 16, (HEIGHT + 15) / 16, 1);
  [ce dispatchThreadgroups:grid threadsPerThreadgroup:threads];
  [ce endEncoding];

  // Swap
  id<MTLTexture> tmp = _stateTex[0];
  _stateTex[0] = _stateTex[1];
  _stateTex[1] = tmp;

  // 2. Display
  MTLRenderPassDescriptor *rpd = view.currentRenderPassDescriptor;
  if (rpd) {
    id<MTLRenderCommandEncoder> re =
        [cmdBuf renderCommandEncoderWithDescriptor:rpd];
    [re setRenderPipelineState:_displayPipeline];
    [re setFragmentTexture:_stateTex[0] atIndex:0];
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

int main() {
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
    window.title = @"M3 Neural Lenia | Continuous Emergence";
    [window makeKeyAndOrderFront:nil];
    InputView *view =
        [[InputView alloc] initWithFrame:frame
                                  device:MTLCreateSystemDefaultDevice()];
    view.colorPixelFormat = MTLPixelFormatBGRA8Unorm;
    Renderer *renderer = [[Renderer alloc] initWithMetalKitView:view];
    view.delegate = renderer;
    view.renderer = renderer;
    window.contentView = view;
    [window makeFirstResponder:view];

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
