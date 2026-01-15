#import "OscilloscopeView.h"

#define SAMPLE_BUFFER_SIZE 512

typedef struct {
  simd_float2 position;
  simd_float2 texCoord;
} Vertex;

@implementation OscilloscopeView {
  id<MTLDevice> _device;
  id<MTLCommandQueue> _commandQueue;

  // Three pipeline states for three rendering passes
  id<MTLRenderPipelineState> _waveformPipeline; // Draw waveform to accumulation
  id<MTLRenderPipelineState> _accumulationPipeline; // Dim previous frame
  id<MTLRenderPipelineState> _displayPipeline; // Final display with CRT effects

  id<MTLBuffer> _vertexBuffer;
  id<MTLTexture> _audioTexture;
  id<MTLBuffer> _uniformBuffer;

  // Ping-pong accumulation textures for phosphor persistence
  id<MTLTexture> _accumulationTexture[2];
  int _currentTextureIndex;

  float *_audioSamples;
  NSUInteger _sampleCount;

  CFTimeInterval _startTime;
}

- (instancetype)initWithFrame:(CGRect)frameRect device:(id<MTLDevice>)device {
  self = [super initWithFrame:frameRect device:device];
  if (self) {
    _device = device;
    _audioSamples = (float *)calloc(SAMPLE_BUFFER_SIZE, sizeof(float));
    _sampleCount = SAMPLE_BUFFER_SIZE;
    _startTime = CACurrentMediaTime();
    _currentTextureIndex = 0;

    [self setupMetal];

    self.colorPixelFormat = MTLPixelFormatBGRA8Unorm;
    self.clearColor = MTLClearColorMake(0.0, 0.0, 0.0, 1.0);
    self.delegate = self;
  }
  return self;
}

// ARC automatically calls [super dealloc] - suppress false warning
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wobjc-missing-super-calls"
- (void)dealloc {
  if (_audioSamples) {
    free(_audioSamples);
  }
}
#pragma clang diagnostic pop

- (void)setupMetal {
  _commandQueue = [_device newCommandQueue];

  // Create fullscreen quad vertices
  Vertex vertices[] = {
      {{-1.0, -1.0}, {0.0, 1.0}},
      {{1.0, -1.0}, {1.0, 1.0}},
      {{-1.0, 1.0}, {0.0, 0.0}},
      {{1.0, 1.0}, {1.0, 0.0}},
  };
  _vertexBuffer = [_device newBufferWithBytes:vertices
                                       length:sizeof(vertices)
                                      options:MTLResourceStorageModeShared];

  // Create audio texture (1D, R32Float)
  MTLTextureDescriptor *texDesc = [MTLTextureDescriptor
      texture2DDescriptorWithPixelFormat:MTLPixelFormatR32Float
                                   width:SAMPLE_BUFFER_SIZE
                                  height:1
                               mipmapped:NO];
  texDesc.usage = MTLTextureUsageShaderRead;
  _audioTexture = [_device newTextureWithDescriptor:texDesc];

  // Create ping-pong accumulation textures (RGBA16Float for HDR glow)
  MTLTextureDescriptor *accumDesc = [MTLTextureDescriptor
      texture2DDescriptorWithPixelFormat:MTLPixelFormatRGBA16Float
                                   width:1200
                                  height:800
                               mipmapped:NO];
  accumDesc.usage = MTLTextureUsageRenderTarget | MTLTextureUsageShaderRead;
  _accumulationTexture[0] = [_device newTextureWithDescriptor:accumDesc];
  _accumulationTexture[1] = [_device newTextureWithDescriptor:accumDesc];

  // Create uniform buffer
  _uniformBuffer = [_device newBufferWithLength:sizeof(float) * 4
                                        options:MTLResourceStorageModeShared];

  // Load shaders
  NSError *error = nil;
  id<MTLLibrary> library =
      [_device newDefaultLibraryWithBundle:[NSBundle mainBundle] error:&error];
  if (!library) {
    // Try loading from source
    NSString *shaderPath = [[NSBundle mainBundle] pathForResource:@"Shaders"
                                                           ofType:@"metal"];
    if (!shaderPath) {
      NSString *currentDir =
          [[NSFileManager defaultManager] currentDirectoryPath];
      shaderPath = [currentDir stringByAppendingPathComponent:@"Shaders.metal"];
    }

    if (shaderPath) {
      NSString *shaderSource =
          [NSString stringWithContentsOfFile:shaderPath
                                    encoding:NSUTF8StringEncoding
                                       error:&error];
      library = [_device newLibraryWithSource:shaderSource
                                      options:nil
                                        error:&error];
    }
  }

  if (!library) {
    NSLog(@"Failed to load shader library: %@", error);
    return;
  }

  id<MTLFunction> vertexFunction =
      [library newFunctionWithName:@"oscilloscopeVertex"];
  id<MTLFunction> waveformFragment =
      [library newFunctionWithName:@"waveformFragment"];
  id<MTLFunction> accumulationFragment =
      [library newFunctionWithName:@"accumulationFragment"];
  id<MTLFunction> displayFragment =
      [library newFunctionWithName:@"displayFragment"];

  // Waveform pipeline (renders SDF waveform to accumulation buffer)
  MTLRenderPipelineDescriptor *waveformDesc =
      [[MTLRenderPipelineDescriptor alloc] init];
  waveformDesc.vertexFunction = vertexFunction;
  waveformDesc.fragmentFunction = waveformFragment;
  waveformDesc.colorAttachments[0].pixelFormat = MTLPixelFormatRGBA16Float;
  waveformDesc.colorAttachments[0].blendingEnabled = YES;
  waveformDesc.colorAttachments[0].sourceRGBBlendFactor = MTLBlendFactorOne;
  waveformDesc.colorAttachments[0].destinationRGBBlendFactor =
      MTLBlendFactorOne;
  _waveformPipeline = [_device newRenderPipelineStateWithDescriptor:waveformDesc
                                                              error:&error];

  // Accumulation pipeline (dims previous frame)
  MTLRenderPipelineDescriptor *accumDesc2 =
      [[MTLRenderPipelineDescriptor alloc] init];
  accumDesc2.vertexFunction = vertexFunction;
  accumDesc2.fragmentFunction = accumulationFragment;
  accumDesc2.colorAttachments[0].pixelFormat = MTLPixelFormatRGBA16Float;
  _accumulationPipeline =
      [_device newRenderPipelineStateWithDescriptor:accumDesc2 error:&error];

  // Display pipeline (final output with CRT effects)
  MTLRenderPipelineDescriptor *displayDesc =
      [[MTLRenderPipelineDescriptor alloc] init];
  displayDesc.vertexFunction = vertexFunction;
  displayDesc.fragmentFunction = displayFragment;
  displayDesc.colorAttachments[0].pixelFormat = self.colorPixelFormat;
  _displayPipeline = [_device newRenderPipelineStateWithDescriptor:displayDesc
                                                             error:&error];

  if (!_waveformPipeline || !_accumulationPipeline || !_displayPipeline) {
    NSLog(@"Failed to create pipeline states: %@", error);
  }
}

- (void)updateWithAudioSamples:(const float *)samples count:(NSUInteger)count {
  if (!samples) {
    return;
  }
  NSUInteger copyCount = MIN(count, SAMPLE_BUFFER_SIZE);
  memcpy(_audioSamples, samples, copyCount * sizeof(float));
  _sampleCount = copyCount;
}

- (void)drawInMTKView:(MTKView *)view {
  @autoreleasepool {
    // Update audio texture (Metal requires width > 0)
    if (_sampleCount > 0) {
      MTLRegion region = MTLRegionMake2D(0, 0, _sampleCount, 1);
      [_audioTexture replaceRegion:region
                       mipmapLevel:0
                         withBytes:_audioSamples
                       bytesPerRow:_sampleCount * sizeof(float)];
    }

    // Update uniforms
    float time = (float)(CACurrentMediaTime() - _startTime);
    float *uniforms = (float *)[_uniformBuffer contents];
    uniforms[0] = time;
    uniforms[1] = (float)self.drawableSize.width;
    uniforms[2] = (float)self.drawableSize.height;
    uniforms[3] = (float)_sampleCount;

    id<MTLCommandBuffer> commandBuffer = [_commandQueue commandBuffer];

    int readIndex = _currentTextureIndex;
    int writeIndex = 1 - _currentTextureIndex;

    // PASS 1: Accumulation (dim previous frame + add new waveform)
    MTLRenderPassDescriptor *accumPass =
        [MTLRenderPassDescriptor renderPassDescriptor];
    accumPass.colorAttachments[0].texture = _accumulationTexture[writeIndex];
    accumPass.colorAttachments[0].loadAction = MTLLoadActionClear;
    accumPass.colorAttachments[0].storeAction = MTLStoreActionStore;
    accumPass.colorAttachments[0].clearColor =
        MTLClearColorMake(0.0, 0.0, 0.0, 1.0);

    id<MTLRenderCommandEncoder> accumEncoder =
        [commandBuffer renderCommandEncoderWithDescriptor:accumPass];

    // First: dim the previous frame
    [accumEncoder setRenderPipelineState:_accumulationPipeline];
    [accumEncoder setVertexBuffer:_vertexBuffer offset:0 atIndex:0];
    [accumEncoder setFragmentTexture:_accumulationTexture[readIndex] atIndex:0];
    [accumEncoder drawPrimitives:MTLPrimitiveTypeTriangleStrip
                     vertexStart:0
                     vertexCount:4];

    // Second: add new waveform on top
    [accumEncoder setRenderPipelineState:_waveformPipeline];
    [accumEncoder setFragmentTexture:_audioTexture atIndex:0];
    [accumEncoder setFragmentBuffer:_uniformBuffer offset:0 atIndex:0];
    [accumEncoder drawPrimitives:MTLPrimitiveTypeTriangleStrip
                     vertexStart:0
                     vertexCount:4];
    [accumEncoder endEncoding];

    // PASS 2: Display (render accumulated buffer to screen with CRT effects)
    MTLRenderPassDescriptor *displayPass = self.currentRenderPassDescriptor;
    if (displayPass && _displayPipeline) {
      id<MTLRenderCommandEncoder> displayEncoder =
          [commandBuffer renderCommandEncoderWithDescriptor:displayPass];
      [displayEncoder setRenderPipelineState:_displayPipeline];
      [displayEncoder setVertexBuffer:_vertexBuffer offset:0 atIndex:0];
      [displayEncoder setFragmentTexture:_accumulationTexture[writeIndex]
                                 atIndex:0];
      [displayEncoder setFragmentBuffer:_uniformBuffer offset:0 atIndex:0];
      [displayEncoder drawPrimitives:MTLPrimitiveTypeTriangleStrip
                         vertexStart:0
                         vertexCount:4];
      [displayEncoder endEncoding];

      [commandBuffer presentDrawable:self.currentDrawable];
    }

    [commandBuffer commit];

    // Swap textures
    _currentTextureIndex = writeIndex;
  }
}

- (void)mtkView:(MTKView *)view drawableSizeWillChange:(CGSize)size {
  // Recreate accumulation textures with new size
  MTLTextureDescriptor *accumDesc = [MTLTextureDescriptor
      texture2DDescriptorWithPixelFormat:MTLPixelFormatRGBA16Float
                                   width:(NSUInteger)size.width
                                  height:(NSUInteger)size.height
                               mipmapped:NO];
  accumDesc.usage = MTLTextureUsageRenderTarget | MTLTextureUsageShaderRead;
  _accumulationTexture[0] = [_device newTextureWithDescriptor:accumDesc];
  _accumulationTexture[1] = [_device newTextureWithDescriptor:accumDesc];
}

@end
