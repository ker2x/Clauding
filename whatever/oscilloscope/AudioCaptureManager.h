#import "LockFreeRingBuffer.h"
#import <AudioToolbox/AudioToolbox.h>
#import <Foundation/Foundation.h>

@interface AudioCaptureManager : NSObject

- (instancetype)init;
- (void)start;
- (void)stop;

// Get the latest audio samples for rendering (called from Metal render thread)
- (NSUInteger)getLatestSamples:(float *)outBuffer
                    maxSamples:(NSUInteger)maxSamples;

@property(nonatomic, readonly) BOOL isRunning;
@property(nonatomic, readonly) BOOL permissionGranted;
@property(nonatomic, assign) BOOL agcEnabled;
@property(nonatomic, assign) float gain;

@end
