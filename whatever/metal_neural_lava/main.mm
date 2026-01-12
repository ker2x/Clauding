#import <AppKit/AppKit.h>
#import <Metal/Metal.h>
#import <MetalKit/MetalKit.h>
#import <simd/simd.h>

static const uint32_t WIDTH = 1200;
static const uint32_t HEIGHT = 720;
static const uint32_t GRID_SIZE = 64; // 64^3 for real-time fluid physics + RT

struct Uniforms {
  simd_float2 resolution;
  float time;
  float dt;
  uint32_t gridSize;
  simd_float2 mousePos;
  int mouseDown;
  float viscosity;
  float buoyancy;
};

@interface Renderer : NSObject <MTKViewDelegate>
- (instancetype)initWithMetalKitView:(MTKView *)view;
- (void)reset;
- (void)handleMouseDrag:(NSPoint)delta;
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
  [(id)self.renderer setMouseDown:YES];
}
- (void)mouseUp:(NSEvent *)event {
  [(id)self.renderer setMouseDown:NO];
}
- (void)mouseDragged:(NSEvent *)event {
  [(Renderer *)self.renderer
      handleMouseDrag:NSMakePoint([event deltaX], [event deltaY])];
  // Also update mouse position for injection
  NSPoint p = [self convertPoint:[event locationInWindow] fromView:nil];
  [(Renderer *)self.renderer
      updateMousePos:simd_make_float2(p.x / self.bounds.size.width,
                                      p.y / self.bounds.size.height)];
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

  // Pipelines
  id<MTLComputePipelineState> _advectPipeline;
  id<MTLComputePipelineState> _divergencePipeline;
  id<MTLComputePipelineState> _pressurePipeline;
  id<MTLComputePipelineState> _projectPipeline;
  id<MTLComputePipelineState> _clearPipeline;
  id<MTLComputePipelineState> _rtPipeline;
  id<MTLRenderPipelineState> _displayPipeline;

  // Buffers (Grids as 3D Textures for easier sampling)
  id<MTLTexture> _velocityTex[2];
  id<MTLTexture> _densityTex[2];
  id<MTLTexture> _pressureTex[2];
  id<MTLTexture> _divergenceTex;

  id<MTLBuffer> _uniformBuffer;
  id<MTLTexture> _renderTarget;

  float _time;
  simd_float2 _rotation;
  simd_float2 _mousePos;
  BOOL _isMouseDown;
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

  _advectPipeline = [_device newComputePipelineStateWithFunction:
                                 [library newFunctionWithName:@"advect_kernel"]
                                                           error:&error];
  _divergencePipeline =
      [_device newComputePipelineStateWithFunction:
                   [library newFunctionWithName:@"divergence_kernel"]
                                             error:&error];
  _pressurePipeline =
      [_device newComputePipelineStateWithFunction:
                   [library newFunctionWithName:@"pressure_kernel"]
                                             error:&error];
  _projectPipeline =
      [_device newComputePipelineStateWithFunction:
                   [library newFunctionWithName:@"project_kernel"]
                                             error:&error];
  _clearPipeline = [_device newComputePipelineStateWithFunction:
                                [library newFunctionWithName:@"clear_kernel"]
                                                          error:&error];
  _rtPipeline = [_device newComputePipelineStateWithFunction:
                             [library newFunctionWithName:@"raytrace_kernel"]
                                                       error:&error];

  MTLRenderPipelineDescriptor *rpDesc = [MTLRenderPipelineDescriptor new];
  rpDesc.vertexFunction = [library newFunctionWithName:@"vertex_main"];
  rpDesc.fragmentFunction = [library newFunctionWithName:@"fragment_main"];
  rpDesc.colorAttachments[0].pixelFormat = MTLPixelFormatBGRA8Unorm;
  _displayPipeline = [_device newRenderPipelineStateWithDescriptor:rpDesc
                                                             error:&error];
}

- (void)initResources {
  MTLTextureDescriptor *gridDesc = [MTLTextureDescriptor new];
  gridDesc.textureType = MTLTextureType3D;
  gridDesc.pixelFormat = MTLPixelFormatRGBA16Float;
  gridDesc.width = GRID_SIZE;
  gridDesc.height = GRID_SIZE;
  gridDesc.depth = GRID_SIZE;
  gridDesc.usage = MTLTextureUsageShaderRead | MTLTextureUsageShaderWrite;

  for (int i = 0; i < 2; i++) {
    _velocityTex[i] = [_device newTextureWithDescriptor:gridDesc];
    _densityTex[i] = [_device newTextureWithDescriptor:gridDesc];
    _pressureTex[i] = [_device newTextureWithDescriptor:gridDesc];
  }
  _divergenceTex = [_device newTextureWithDescriptor:gridDesc];

  MTLTextureDescriptor *rtDesc = [MTLTextureDescriptor
      texture2DDescriptorWithPixelFormat:MTLPixelFormatRGBA16Float
                                   width:WIDTH
                                  height:HEIGHT
                               mipmapped:NO];
  rtDesc.usage = MTLTextureUsageShaderRead | MTLTextureUsageShaderWrite;
  _renderTarget = [_device newTextureWithDescriptor:rtDesc];

  _uniformBuffer = [_device newBufferWithLength:sizeof(Uniforms)
                                        options:MTLResourceStorageModeShared];
  [self reset];
}

- (void)reset {
  _rotation = (simd_float2){0, 0};
  _time = 0;

  id<MTLCommandBuffer> cmdBuf = [_commandQueue commandBuffer];
  id<MTLComputeCommandEncoder> ce = [cmdBuf computeCommandEncoder];
  [ce setComputePipelineState:_clearPipeline];

  MTLSize gridThreads = MTLSizeMake(GRID_SIZE, GRID_SIZE, GRID_SIZE);
  MTLSize groupSize = MTLSizeMake(8, 8, 8);

  for (int i = 0; i < 2; i++) {
    [ce setTexture:_velocityTex[i] atIndex:0];
    [ce setTexture:_densityTex[i] atIndex:1];
    [ce setTexture:_pressureTex[i] atIndex:2];
    [ce dispatchThreads:gridThreads threadsPerThreadgroup:groupSize];
  }
  [ce endEncoding];
  [cmdBuf commit];
}

- (void)setMouseDown:(BOOL)down {
  _isMouseDown = down;
}
- (void)updateMousePos:(simd_float2)pos {
  _mousePos = pos;
}

- (void)handleMouseDrag:(NSPoint)delta {
  _rotation.x += delta.x * 0.01f;
  _rotation.y += delta.y * 0.01f;
}

- (void)drawInMTKView:(nonnull MTKView *)view {
  _time += 1.0 / 60.0;
  _frameIdx++;

  Uniforms uniforms;
  uniforms.resolution = (simd_float2){(float)WIDTH, (float)HEIGHT};
  uniforms.time = _time;
  uniforms.dt = 0.05f;
  uniforms.gridSize = GRID_SIZE;
  uniforms.mousePos = _mousePos;
  uniforms.mouseDown = _isMouseDown ? 1 : 0;
  uniforms.viscosity = 0.1f;
  uniforms.buoyancy = 1.0f;
  memcpy([_uniformBuffer contents], &uniforms, sizeof(Uniforms));

  id<MTLCommandBuffer> cmdBuf = [_commandQueue commandBuffer];
  MTLSize gridThreads = MTLSizeMake(GRID_SIZE, GRID_SIZE, GRID_SIZE);
  MTLSize groupSize = MTLSizeMake(8, 8, 8);

  // 1. Advect
  id<MTLComputeCommandEncoder> ce = [cmdBuf computeCommandEncoder];
  [ce setComputePipelineState:_advectPipeline];
  [ce setTexture:_velocityTex[0] atIndex:0]; // In Vel
  [ce setTexture:_velocityTex[1] atIndex:1]; // Out Vel
  [ce setTexture:_densityTex[0] atIndex:2];  // In Dens
  [ce setTexture:_densityTex[1] atIndex:3];  // Out Dens
  [ce setBuffer:_uniformBuffer offset:0 atIndex:0];
  [ce dispatchThreads:gridThreads threadsPerThreadgroup:groupSize];
  [ce endEncoding];

  // 2. Divergence
  ce = [cmdBuf computeCommandEncoder];
  [ce setComputePipelineState:_divergencePipeline];
  [ce setTexture:_velocityTex[1] atIndex:0];
  [ce setTexture:_divergenceTex atIndex:1];
  [ce setBuffer:_uniformBuffer offset:0 atIndex:0];
  [ce dispatchThreads:gridThreads threadsPerThreadgroup:groupSize];
  [ce endEncoding];

  // 3. Pressure (Jacobi)
  for (int i = 0; i < 20; i++) {
    ce = [cmdBuf computeCommandEncoder];
    [ce setComputePipelineState:_pressurePipeline];
    [ce setTexture:_pressureTex[0] atIndex:0];
    [ce setTexture:_pressureTex[1] atIndex:1];
    [ce setTexture:_divergenceTex atIndex:2];
    [ce setBuffer:_uniformBuffer offset:0 atIndex:0];
    [ce dispatchThreads:gridThreads threadsPerThreadgroup:groupSize];
    [ce endEncoding];
    // Swap
    id<MTLTexture> tmp = _pressureTex[0];
    _pressureTex[0] = _pressureTex[1];
    _pressureTex[1] = tmp;
  }

  // 4. Project
  ce = [cmdBuf computeCommandEncoder];
  [ce setComputePipelineState:_projectPipeline];
  [ce setTexture:_velocityTex[1] atIndex:0];
  [ce setTexture:_velocityTex[0] atIndex:1]; // Re-use 0 for output
  [ce setTexture:_pressureTex[0] atIndex:2];
  [ce setBuffer:_uniformBuffer offset:0 atIndex:0];
  [ce dispatchThreads:gridThreads threadsPerThreadgroup:groupSize];
  [ce endEncoding];

  // 5. Ray Trace
  ce = [cmdBuf computeCommandEncoder];
  [ce setComputePipelineState:_rtPipeline];
  [ce setTexture:_densityTex[1] atIndex:0];
  [ce setTexture:_renderTarget atIndex:1];
  [ce setBuffer:_uniformBuffer offset:0 atIndex:0];
  MTLSize rtGroup = MTLSizeMake(16, 16, 1);
  MTLSize rtGrid = MTLSizeMake((WIDTH + 15) / 16, (HEIGHT + 15) / 16, 1);
  [ce dispatchThreadgroups:rtGrid threadsPerThreadgroup:rtGroup];
  [ce endEncoding];

  // 6. Display
  MTLRenderPassDescriptor *rpd = view.currentRenderPassDescriptor;
  if (rpd) {
    id<MTLRenderCommandEncoder> re =
        [cmdBuf renderCommandEncoderWithDescriptor:rpd];
    [re setRenderPipelineState:_displayPipeline];
    [re setFragmentTexture:_renderTarget atIndex:0];
    [re drawPrimitives:MTLPrimitiveTypeTriangleStrip
           vertexStart:0
           vertexCount:4];
    [re endEncoding];
    [cmdBuf presentDrawable:view.currentDrawable];
  }

  [cmdBuf commit];

  // Swap Density for next frame
  id<MTLTexture> tmpD = _densityTex[0];
  _densityTex[0] = _densityTex[1];
  _densityTex[1] = tmpD;
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
    window.title = @"M3 Neural Lava | Fluid Dynamics & NPU Turbulence";
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
    [NSApp run];
  }
  return 0;
}
