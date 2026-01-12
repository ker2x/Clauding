#import <AppKit/AppKit.h>
#import <Metal/Metal.h>
#import <MetalKit/MetalKit.h>
#include <algorithm>
#include <cmath>
#include <random>
#include <simd/simd.h>
#include <vector>

static const uint WIDTH = 1200;
static const uint HEIGHT = 720;
static const uint NUM_AGENTS = 1000000;

struct Agent {
  simd_float2 position;
  float heading;
  int species;
};

struct Params {
  uint width;
  uint height;
  float move_speed;
  float sensor_angle;
  float sensor_dist;
  float turn_speed;
  float sensor_size;
  float decay_rate;
  float diffuse_rate;
  float dt;
  uint num_agents;
  float combat_strength;
  simd_float2 mouse_pos;
  int mouse_down;
  int padding2[3];
};

@interface MetalRenderer : NSObject <MTKViewDelegate>
@property(nonatomic, weak) MTKView *view;
- (void)initResources;
- (void)saveScreenshot;
@end

@interface PhysarumView : MTKView
@property(nonatomic, assign) Params params;
@end

@implementation PhysarumView
- (BOOL)acceptsFirstResponder {
  return YES;
}
- (void)mouseMoved:(NSEvent *)event {
  NSPoint p = [self convertPoint:[event locationInWindow] fromView:nil];
  // Convert points to pixels manually since we set drawableSize fixed
  float scaleX = (float)WIDTH / self.bounds.size.width;
  float scaleY = (float)HEIGHT / self.bounds.size.height;
  _params.mouse_pos = (simd_float2){
      (float)p.x * scaleX, (float)(self.bounds.size.height - p.y) * scaleY};
}
- (void)mouseDown:(NSEvent *)event {
  _params.mouse_down = 1;
}
- (void)mouseUp:(NSEvent *)event {
  _params.mouse_down = 0;
}
- (void)keyDown:(NSEvent *)event {
  if ([[event characters] isEqualToString:@"r"]) {
    [(MetalRenderer *)self.delegate initResources];
  } else if ([[event characters] isEqualToString:@"s"]) {
    [(MetalRenderer *)self.delegate saveScreenshot];
  }
}
@end

@implementation MetalRenderer {
  id<MTLDevice> _device;
  id<MTLCommandQueue> _commandQueue;
  id<MTLComputePipelineState> _updatePipeline;
  id<MTLComputePipelineState> _depositPipeline;
  id<MTLComputePipelineState> _processPipeline;
  id<MTLRenderPipelineState> _renderPipeline;

  id<MTLBuffer> _agentBuffer;
  id<MTLTexture> _trailTexture1;
  id<MTLTexture> _trailTexture2;
  uint _frameIndex;
}

- (instancetype)initWithMetalKitView:(PhysarumView *)view {
  self = [super init];
  if (self) {
    _device = view.device;
    _commandQueue = [_device newCommandQueue];
    _view = view;

    NSError *error = nil;
    NSString *source = [NSString stringWithContentsOfFile:@"Physarum.metal"
                                                 encoding:NSUTF8StringEncoding
                                                    error:&error];
    if (!source) {
      NSLog(@"Failed to load Physarum.metal: %@", error);
      return nil;
    }

    id<MTLLibrary> library = [_device newLibraryWithSource:source
                                                   options:nil
                                                     error:&error];
    if (!library) {
      NSLog(@"Metal library compilation failed: %@", error);
      return nil;
    }

    _updatePipeline =
        [_device newComputePipelineStateWithFunction:
                     [library newFunctionWithName:@"update_agents"]
                                               error:&error];
    if (!_updatePipeline)
      NSLog(@"Failed to create updatePipeline: %@", error);

    _depositPipeline =
        [_device newComputePipelineStateWithFunction:
                     [library newFunctionWithName:@"deposit_pheromones"]
                                               error:&error];
    if (!_depositPipeline)
      NSLog(@"Failed to create depositPipeline: %@", error);

    _processPipeline =
        [_device newComputePipelineStateWithFunction:
                     [library newFunctionWithName:@"process_trail"]
                                               error:&error];
    if (!_processPipeline)
      NSLog(@"Failed to create processPipeline: %@", error);

    MTLRenderPipelineDescriptor *renderDesc =
        [[MTLRenderPipelineDescriptor alloc] init];
    renderDesc.vertexFunction = [library newFunctionWithName:@"vertex_main"];
    renderDesc.fragmentFunction =
        [library newFunctionWithName:@"fragment_main"];
    renderDesc.colorAttachments[0].pixelFormat = view.colorPixelFormat;
    _renderPipeline = [_device newRenderPipelineStateWithDescriptor:renderDesc
                                                              error:&error];
    if (!_renderPipeline)
      NSLog(@"Failed to create renderPipeline: %@", error);

    [self initResources];
  }
  return self;
}

- (void)initResources {
  std::vector<Agent> agents(NUM_AGENTS);
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_real_distribution<float> distX(0, WIDTH);
  std::uniform_real_distribution<float> distY(0, HEIGHT);
  std::uniform_real_distribution<float> distA(0, M_PI * 2);

  for (uint i = 0; i < NUM_AGENTS; i++) {
    agents[i].position = (simd_float2){distX(gen), distY(gen)};
    agents[i].heading = distA(gen);
    // Start them on separate halves to encourage large frontiers
    agents[i].species = (agents[i].position.x < WIDTH / 2) ? 0 : 1;
  }

  _agentBuffer = [_device newBufferWithBytes:agents.data()
                                      length:sizeof(Agent) * NUM_AGENTS
                                     options:MTLResourceStorageModeShared];

  MTLTextureDescriptor *tDesc = [MTLTextureDescriptor
      texture2DDescriptorWithPixelFormat:MTLPixelFormatRGBA16Float
                                   width:WIDTH
                                  height:HEIGHT
                               mipmapped:NO];
  tDesc.usage = MTLTextureUsageShaderRead | MTLTextureUsageShaderWrite;
  _trailTexture1 = [_device newTextureWithDescriptor:tDesc];
  _trailTexture2 = [_device newTextureWithDescriptor:tDesc];

  // Clear textures - not strictly needed as decay will handle it or we can use
  // a kernel
  _frameIndex = 0;
}

- (void)drawInMTKView:(nonnull MTKView *)view {
  id<MTLCommandBuffer> commandBuffer = [_commandQueue commandBuffer];

  Params p = ((PhysarumView *)view).params;
  p.width = WIDTH;
  p.height = HEIGHT;
  p.num_agents = NUM_AGENTS;
  p.dt = 1.0f;
  p.move_speed = 1.2f;
  p.sensor_angle = 0.45f;
  p.sensor_dist = 22.0f;
  p.turn_speed = 0.35f;
  p.decay_rate = 0.94f;
  p.combat_strength = 0.6f;

  id<MTLTexture> inTex =
      (_frameIndex % 2 == 0 ? _trailTexture1 : _trailTexture2);
  id<MTLTexture> outTex =
      (_frameIndex % 2 == 0 ? _trailTexture2 : _trailTexture1);

  // 1. Update Agents
  id<MTLComputeCommandEncoder> ce = [commandBuffer computeCommandEncoder];
  [ce setComputePipelineState:_updatePipeline];
  [ce setBuffer:_agentBuffer offset:0 atIndex:0];
  [ce setTexture:inTex atIndex:0];
  [ce setBytes:&p length:sizeof(Params) atIndex:1];
  [ce dispatchThreads:MTLSizeMake(NUM_AGENTS, 1, 1)
      threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];

  // 2. Deposit
  [ce setComputePipelineState:_depositPipeline];
  [ce setTexture:inTex atIndex:0];
  [ce dispatchThreads:MTLSizeMake(NUM_AGENTS, 1, 1)
      threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];

  // 3. Process Trail (Diffuse/Decay)
  [ce setComputePipelineState:_processPipeline];
  [ce setTexture:inTex atIndex:0];
  [ce setTexture:outTex atIndex:1];
  [ce setBytes:&p length:sizeof(Params) atIndex:0];
  [ce dispatchThreads:MTLSizeMake(WIDTH, HEIGHT, 1)
      threadsPerThreadgroup:MTLSizeMake(16, 16, 1)];
  [ce endEncoding];

  // 4. Render to view
  MTLRenderPassDescriptor *rpd = view.currentRenderPassDescriptor;
  if (rpd) {
    id<MTLRenderCommandEncoder> re =
        [commandBuffer renderCommandEncoderWithDescriptor:rpd];
    [re setRenderPipelineState:_renderPipeline];
    [re setFragmentTexture:outTex atIndex:0];
    [re drawPrimitives:MTLPrimitiveTypeTriangleStrip
           vertexStart:0
           vertexCount:4];
    [re endEncoding];
    [commandBuffer presentDrawable:view.currentDrawable];
  }

  [commandBuffer commit];
  _frameIndex++;
}

- (void)saveScreenshot {
  id<MTLTexture> texture =
      _frameIndex % 2 == 0 ? _trailTexture1 : _trailTexture2;
  uint width = (uint)texture.width;
  uint height = (uint)texture.height;
  uint rowBytes = width * 8; // RGBA16Float is 8 bytes per pixel
  std::vector<uint8_t> data(rowBytes * height);

  [texture getBytes:data.data()
        bytesPerRow:rowBytes
         fromRegion:MTLRegionMake2D(0, 0, width, height)
        mipmapLevel:0];

  std::vector<uint8_t> rgba8(width * height * 4);
  for (size_t i = 0; i < width * height; i++) {
    float *src = (float *)&data[i * 8];
    rgba8[i * 4 + 0] =
        (uint8_t)std::min(255.0f, std::max(0.0f, src[0] * 255.0f));
    rgba8[i * 4 + 1] =
        (uint8_t)std::min(255.0f, std::max(0.0f, src[1] * 255.0f));
    rgba8[i * 4 + 2] =
        (uint8_t)std::min(255.0f, std::max(0.0f, src[2] * 255.0f));
    rgba8[i * 4 + 3] = 255;
  }

  CGDataProviderRef provider =
      CGDataProviderCreateWithData(NULL, rgba8.data(), rgba8.size(), NULL);
  CGColorSpaceRef colorSpace = CGColorSpaceCreateDeviceRGB();
  CGImageRef cgImage =
      CGImageCreate(width, height, 8, 32, width * 4, colorSpace,
                    kCGImageAlphaLast | kCGBitmapByteOrderDefault, provider,
                    NULL, NO, kCGRenderingIntentDefault);

  NSImage *image = [[NSImage alloc] initWithCGImage:cgImage
                                               size:NSMakeSize(width, height)];
  NSData *imageData = [image TIFFRepresentation];
  NSBitmapImageRep *imageRep = [NSBitmapImageRep imageRepWithData:imageData];
  imageData = [imageRep representationUsingType:NSBitmapImageFileTypePNG
                                     properties:@{}];

  NSString *path =
      [NSString stringWithFormat:@"/Users/ker/PycharmProjects/Clauding/"
                                 @"whatever/metal_physarum/capture_%ld.png",
                                 (long)[[NSDate date] timeIntervalSince1970]];
  [imageData writeToFile:path atomically:YES];

  CGImageRelease(cgImage);
  CGColorSpaceRelease(colorSpace);
  CGDataProviderRelease(provider);
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
                                               NSWindowStyleMaskClosable |
                                               NSWindowStyleMaskResizable)
                                      backing:NSBackingStoreBuffered
                                        defer:NO];
    window.title = @"MetalPhysarum | 1M Agents | M4 MacBook Air";
    [window makeKeyAndOrderFront:nil];

    PhysarumView *view =
        [[PhysarumView alloc] initWithFrame:frame
                                     device:MTLCreateSystemDefaultDevice()];
    view.colorPixelFormat = MTLPixelFormatBGRA8Unorm;
    view.clearColor = MTLClearColorMake(0, 0, 0, 1);
    view.drawableSize = CGSizeMake(WIDTH, HEIGHT);
    view.autoResizeDrawable = NO;

    MetalRenderer *renderer = [[MetalRenderer alloc] initWithMetalKitView:view];
    view.delegate = renderer;
    [window setContentView:view];
    [window makeFirstResponder:view];

    NSMenu *mainMenu = [[NSMenu alloc] init];
    NSMenuItem *appMenuItem = [[NSMenuItem alloc] init];
    [mainMenu addItem:appMenuItem];
    [NSApp setMainMenu:mainMenu];

    NSMenu *appMenu = [[NSMenu alloc] init];
    [appMenu addItemWithTitle:@"Save Screenshot (S)"
                       action:@selector(saveScreenshot)
                keyEquivalent:@"s"];
    [appMenu addItemWithTitle:@"Reset (R)"
                       action:@selector(initResources)
                keyEquivalent:@"r"];
    [appMenu addItemWithTitle:@"Quit"
                       action:@selector(terminate:)
                keyEquivalent:@"q"];
    [appMenuItem setSubmenu:appMenu];

    [NSApp activateIgnoringOtherApps:YES];
    [NSApp run];
  }
  return 0;
}
