#import "AudioCaptureManager.h"
#import "OscilloscopeView.h"
#import <AVFoundation/AVFoundation.h>
#import <Cocoa/Cocoa.h>
#import <Metal/Metal.h>
#import <MetalKit/MetalKit.h>

@interface AppDelegate : NSObject <NSApplicationDelegate>
@property(strong, nonatomic) NSWindow *window;
@property(strong, nonatomic) OscilloscopeView *oscilloscopeView;
@property(strong, nonatomic) AudioCaptureManager *audioManager;
@property(strong, nonatomic) NSTimer *updateTimer;
@property(strong, nonatomic) AVCaptureSession *captureSession;
@end

@implementation AppDelegate

- (void)applicationDidFinishLaunching:(NSNotification *)notification {
  // Create standard menu bar
  [self createMenuBar];

  // Create window
  NSRect frame = NSMakeRect(0, 0, 1200, 800);
  NSWindowStyleMask styleMask =
      NSWindowStyleMaskTitled | NSWindowStyleMaskClosable |
      NSWindowStyleMaskMiniaturizable | NSWindowStyleMaskResizable;

  self.window = [[NSWindow alloc] initWithContentRect:frame
                                            styleMask:styleMask
                                              backing:NSBackingStoreBuffered
                                                defer:NO];

  [self.window setTitle:@"Vintage Oscilloscope"];
  [self.window center];

  // Create Metal device
  id<MTLDevice> device = MTLCreateSystemDefaultDevice();
  if (!device) {
    NSLog(@"Metal is not supported on this device");
    [NSApp terminate:nil];
    return;
  }

  // Create oscilloscope view
  self.oscilloscopeView = [[OscilloscopeView alloc] initWithFrame:frame
                                                           device:device];
  self.oscilloscopeView.preferredFramesPerSecond = 60;

  [self.window setContentView:self.oscilloscopeView];
  [self.window makeKeyAndOrderFront:nil];

  // Activate app to bring it to front (must be after window creation)
  [NSApp activateIgnoringOtherApps:YES];

  // Request microphone permission and start audio capture
  [self requestMicrophonePermission];

  NSLog(@"Oscilloscope started - requesting microphone access...");

  // Monitor key events
  [NSEvent
      addLocalMonitorForEventsMatchingMask:NSEventMaskKeyDown
                                   handler:^NSEvent *_Nullable(NSEvent *event) {
                                     if ([self handleKeyDown:event]) {
                                       return nil;
                                     }
                                     return event;
                                   }];
}

- (BOOL)handleKeyDown:(NSEvent *)event {
  if (!self.audioManager)
    return NO;

  BOOL handled = NO;

  // 'a' or 'A' to toggle AGC
  if ([event.charactersIgnoringModifiers caseInsensitiveCompare:@"a"] ==
      NSOrderedSame) {
    self.audioManager.agcEnabled = !self.audioManager.agcEnabled;
    [self updateWindowTitle];
    handled = YES;
  }

  // Up/Down arrows to adjust gain
  if (event.keyCode == 126) { // Up Arrow
    self.audioManager.gain += 1.0f;
    if (self.audioManager.gain > 100.0f)
      self.audioManager.gain = 100.0f;
    self.audioManager.agcEnabled = NO;
    [self updateWindowTitle];
    handled = YES;
  }
  if (event.keyCode == 125) { // Down Arrow
    self.audioManager.gain -= 1.0f;
    if (self.audioManager.gain < 1.0f)
      self.audioManager.gain = 1.0f;
    self.audioManager.agcEnabled = NO;
    [self updateWindowTitle];
    handled = YES;
  }

  return handled;
}

- (void)updateWindowTitle {
  if (!self.window || !self.audioManager)
    return;

  NSString *status;
  if (self.audioManager.agcEnabled) {
    status = [NSString
        stringWithFormat:@"Vintage Oscilloscope [AGC: ON] (Gain: %.1fx)",
                         self.audioManager.gain];
  } else {
    status = [NSString
        stringWithFormat:@"Vintage Oscilloscope [AGC: OFF] (Gain: %.1fx)",
                         self.audioManager.gain];
  }
  [self.window setTitle:status];
}

- (void)createMenuBar {
  // Create main menu bar
  NSMenu *menuBar = [[NSMenu alloc] init];

  // App menu
  NSMenuItem *appMenuItem = [[NSMenuItem alloc] init];
  [menuBar addItem:appMenuItem];

  NSMenu *appMenu = [[NSMenu alloc] init];
  [appMenuItem setSubmenu:appMenu];

  // Add Quit menu item with Cmd+Q
  NSMenuItem *quitItem = [[NSMenuItem alloc] initWithTitle:@"Quit Oscilloscope"
                                                    action:@selector(terminate:)
                                             keyEquivalent:@"q"];
  [appMenu addItem:quitItem];

  [NSApp setMainMenu:menuBar];
}

- (void)requestMicrophonePermission {
  // Request microphone access
  switch ([AVCaptureDevice authorizationStatusForMediaType:AVMediaTypeAudio]) {
  case AVAuthorizationStatusAuthorized: {
    NSLog(@"Microphone access already authorized");
    [self startAudioCapture];
    break;
  }

  case AVAuthorizationStatusNotDetermined: {
    NSLog(@"Requesting microphone permission...");
    [AVCaptureDevice requestAccessForMediaType:AVMediaTypeAudio
                             completionHandler:^(BOOL granted) {
                               dispatch_async(dispatch_get_main_queue(), ^{
                                 if (granted) {
                                   NSLog(@"Microphone permission granted!");
                                   [self startAudioCapture];
                                 } else {
                                   NSLog(@"Microphone permission denied!");
                                   [self showPermissionDeniedAlert];
                                 }
                               });
                             }];
    break;
  }

  case AVAuthorizationStatusDenied:
  case AVAuthorizationStatusRestricted: {
    NSLog(@"Microphone access denied or restricted");
    [self showPermissionDeniedAlert];
    break;
  }
  }
}

- (void)startAudioCapture {

  // Set up audio capture
  self.audioManager = [[AudioCaptureManager alloc] init];
  [self.audioManager start];

  // Set up AVCaptureSession to receive interruption notifications
  // Note: macOS kills the app when microphone permission is revoked, but this
  // handles other interruptions (e.g., audio device disconnection)
  [self setupCaptureSessionForInterruptionNotifications];

  // Create timer and add to run loop with common modes
  self.updateTimer = [NSTimer timerWithTimeInterval:1.0 / 60.0
                                             target:self
                                           selector:@selector(updateAudio)
                                           userInfo:nil
                                            repeats:YES];
  [[NSRunLoop mainRunLoop] addTimer:self.updateTimer
                            forMode:NSRunLoopCommonModes];

  NSLog(@"Audio capture started - speak into your microphone!");
  [self updateWindowTitle];
}

- (void)setupCaptureSessionForInterruptionNotifications {
  self.captureSession = [[AVCaptureSession alloc] init];

  // Add audio input to the session
  AVCaptureDevice *audioDevice =
      [AVCaptureDevice defaultDeviceWithMediaType:AVMediaTypeAudio];
  if (audioDevice) {
    NSError *error = nil;
    AVCaptureDeviceInput *audioInput =
        [AVCaptureDeviceInput deviceInputWithDevice:audioDevice error:&error];
    if (audioInput && [self.captureSession canAddInput:audioInput]) {
      [self.captureSession addInput:audioInput];
    } else {
      NSLog(@"Could not add audio input to capture session: %@", error);
    }
  }

  // Register for interruption notifications
  [[NSNotificationCenter defaultCenter]
      addObserver:self
         selector:@selector(handleSessionInterruption:)
             name:AVCaptureSessionWasInterruptedNotification
           object:self.captureSession];

  [[NSNotificationCenter defaultCenter]
      addObserver:self
         selector:@selector(handleSessionInterruptionEnded:)
             name:AVCaptureSessionInterruptionEndedNotification
           object:self.captureSession];

  // Start the session (required to receive notifications)
  [self.captureSession startRunning];
  NSLog(@"AVCaptureSession started for interruption monitoring");
}

- (void)handleSessionInterruption:(NSNotification *)notification {
  NSLog(@"⚠️ Audio session interrupted: %@", notification.userInfo);

  // Stop our audio capture
  [self.audioManager stop];

  // Show alert on main thread
  dispatch_async(dispatch_get_main_queue(), ^{
    NSAlert *alert = [[NSAlert alloc] init];
    [alert setMessageText:@"Audio Interrupted"];
    [alert setInformativeText:
               @"Audio capture was interrupted (device disconnected or in use "
               @"by another application). Please reconnect your audio device "
               @"or close other audio applications."];
    [alert addButtonWithTitle:@"OK"];
    [alert runModal];
  });
}

- (void)handleSessionInterruptionEnded:(NSNotification *)notification {
  NSLog(@"Audio session interruption ended - restarting capture");

  // Restart our audio capture
  [self.audioManager start];
}

- (void)showPermissionDeniedAlert {
  NSAlert *alert = [[NSAlert alloc] init];
  [alert setMessageText:@"Microphone Access Required"];
  [alert
      setInformativeText:
          @"This app needs microphone access to visualize audio. Please grant "
          @"permission in System Settings > Privacy & Security > Microphone."];
  [alert addButtonWithTitle:@"OK"];
  [alert runModal];
}

- (void)updateAudio {

  float samples[512];
  NSUInteger count = [self.audioManager getLatestSamples:samples
                                              maxSamples:512];

  static int debugFrame = 0;
  if (++debugFrame % 60 == 0) {
    NSLog(@"DEBUG: count=%lu, samples[0]=%f", (unsigned long)count, count > 0 ? samples[0] : 0.0f);
  }

  [self.oscilloscopeView updateWithAudioSamples:samples count:count];

  // Update title periodically if AGC is on to show current gain
  if (self.audioManager.agcEnabled) {
    static int frame = 0;
    if (++frame % 10 == 0) { // Update every 10 frames (~6Hz)
      [self updateWindowTitle];
    }
  }
}

- (void)applicationWillTerminate:(NSNotification *)notification {
  [[NSNotificationCenter defaultCenter] removeObserver:self];
  [self.captureSession stopRunning];
  [self.updateTimer invalidate];
  [self.audioManager stop];
}

- (BOOL)applicationShouldTerminateAfterLastWindowClosed:
    (NSApplication *)sender {
  return YES;
}

@end

int main(int argc, const char *argv[]) {
  @autoreleasepool {
    NSApplication *app = [NSApplication sharedApplication];
    AppDelegate *delegate = [[AppDelegate alloc] init];
    [app setDelegate:delegate];
    [app setActivationPolicy:NSApplicationActivationPolicyRegular];
    [app run];
  }
  return 0;
}
