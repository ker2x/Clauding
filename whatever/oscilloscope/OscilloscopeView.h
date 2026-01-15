#import <MetalKit/MetalKit.h>

@interface OscilloscopeView : MTKView <MTKViewDelegate>

// Update with new audio samples
- (void)updateWithAudioSamples:(const float *)samples count:(NSUInteger)count;

@end
