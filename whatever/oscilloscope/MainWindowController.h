#import <Cocoa/Cocoa.h>
#import <AVFoundation/AVFoundation.h>
#import "OscilloscopeView.h"
#import "AudioCaptureManager.h"

@interface MainWindowController : NSWindowController <NSWindowDelegate, NSToolbarDelegate>

@property (strong, nonatomic) OscilloscopeView *oscilloscopeView;
@property (strong, nonatomic) AudioCaptureManager *audioManager;
@property (strong, nonatomic) NSTimer *updateTimer;
@property (strong, nonatomic) NSView *statusBar;
@property (strong, nonatomic) NSTextField *statusLabel;
@property (strong, nonatomic) NSProgressIndicator *levelMeter;
@property (strong, nonatomic) NSTextField *performanceLabel;
@property (strong, nonatomic) AVCaptureSession *captureSession;

- (instancetype)init;
- (void)startAudioCapture;
- (void)setupCaptureSessionForInterruptionNotifications;
- (void)updateAudio;
- (void)updateStatusDisplay;
- (void)updateWindowTitle;
- (void)toggleAGC:(id)sender;
- (void)increaseGain:(id)sender;
- (void)decreaseGain:(id)sender;
- (void)showPreferences:(id)sender;
- (void)exportScreenshot:(id)sender;

@end