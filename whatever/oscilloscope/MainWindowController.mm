#import "MainWindowController.h"
#import <Metal/Metal.h>
#import <MetalKit/MetalKit.h>

@interface MainWindowController ()

@property (strong, nonatomic) NSToolbar *toolbar;
@property (strong, nonatomic) NSToolbarItem *agcItem;
@property (strong, nonatomic) NSToolbarItem *gainItem;
@property (strong, nonatomic) NSSlider *gainSlider;

@end

@implementation MainWindowController

- (instancetype)init {
    self = [super initWithWindow:nil];
    if (self) {
        [self setupWindow];
        [self setupStatusBar];
        [self setupToolbar];
        [self requestMicrophonePermission];
    }
    return self;
}

- (void)setupWindow {
    // Create beautiful window with modern macOS styling
    NSRect contentRect = NSMakeRect(0, 0, 1200, 800);
    NSWindowStyleMask styleMask = NSWindowStyleMaskTitled |
                                  NSWindowStyleMaskClosable |
                                  NSWindowStyleMaskMiniaturizable |
                                  NSWindowStyleMaskResizable |
                                  NSWindowStyleMaskFullSizeContentView;

    NSWindow *window = [[NSWindow alloc] initWithContentRect:contentRect
                                                    styleMask:styleMask
                                                      backing:NSBackingStoreBuffered
                                                        defer:NO];

    [window setTitle:@"Vintage Oscilloscope"];
    [window setMinSize:NSMakeSize(900, 650)];
    [window center];
    [window setDelegate:self];

    // Modern window appearance
    if (@available(macOS 10.14, *)) {
        [window setAppearance:[NSAppearance appearanceNamed:NSAppearanceNameDarkAqua]];
    }
    [window setTitlebarAppearsTransparent:NO];
    [window setTitleVisibility:NSWindowTitleVisible];

    // Create Metal device and oscilloscope view
    id<MTLDevice> device = MTLCreateSystemDefaultDevice();
    if (!device) {
        [self showErrorAlert:@"Metal Not Supported"
                 description:@"This application requires a Mac with Metal support."];
        [NSApp terminate:nil];
        return;
    }

    self.oscilloscopeView = [[OscilloscopeView alloc] initWithFrame:contentRect device:device];
    self.oscilloscopeView.preferredFramesPerSecond = 60;
    [window setContentView:self.oscilloscopeView];

    self.window = window;
}

- (void)setupStatusBar {
    // Create beautiful status bar with modern design
    self.statusBar = [[NSView alloc] initWithFrame:NSMakeRect(0, 0, 1200, 32)];
    [self.statusBar setWantsLayer:YES];

    // Beautiful gradient background
    NSGradient *gradient = [[NSGradient alloc] initWithColors:@[
        [NSColor colorWithCalibratedWhite:0.15 alpha:1.0],
        [NSColor colorWithCalibratedWhite:0.12 alpha:1.0]
    ]];
    [gradient drawInRect:self.statusBar.bounds angle:90.0];

    // Add subtle border
    NSBox *topBorder = [[NSBox alloc] initWithFrame:NSMakeRect(0, 31, 1200, 1)];
    [topBorder setBoxType:NSBoxSeparator];
    [topBorder setFillColor:[NSColor colorWithCalibratedWhite:0.3 alpha:0.5]];
    [self.statusBar addSubview:topBorder];

    // Status icon and text (left side)
    NSImageView *statusIcon = [[NSImageView alloc] initWithFrame:NSMakeRect(12, 6, 20, 20)];
    [statusIcon setImage:[NSImage imageNamed:NSImageNameStatusAvailable]];
    [statusIcon setImageScaling:NSImageScaleProportionallyUpOrDown];
    [self.statusBar addSubview:statusIcon];

    self.statusLabel = [[NSTextField alloc] initWithFrame:NSMakeRect(40, 6, 350, 20)];
    [self.statusLabel setEditable:NO];
    [self.statusLabel setBordered:NO];
    [self.statusLabel setDrawsBackground:NO];
    [self.statusLabel setFont:[NSFont systemFontOfSize:12 weight:NSFontWeightMedium]];
    [self.statusLabel setTextColor:[NSColor whiteColor]];
    [self.statusLabel setStringValue:@"🎙️ Initializing audio..."];
    [self.statusBar addSubview:self.statusLabel];

    // Beautiful level meter with custom styling
    NSView *meterContainer = [[NSView alloc] initWithFrame:NSMakeRect(420, 8, 180, 16)];
    [meterContainer setWantsLayer:YES];
    [meterContainer.layer setCornerRadius:8.0];
    [meterContainer.layer setBorderWidth:1.0];
    [meterContainer.layer setBorderColor:[[NSColor colorWithCalibratedWhite:0.4 alpha:1.0] CGColor]];
    [meterContainer.layer setBackgroundColor:[[NSColor colorWithCalibratedWhite:0.1 alpha:1.0] CGColor]];

    self.levelMeter = [[NSProgressIndicator alloc] initWithFrame:NSMakeRect(2, 2, 176, 12)];
    [self.levelMeter setStyle:NSProgressIndicatorStyleBar];
    [self.levelMeter setMinValue:0.0];
    [self.levelMeter setMaxValue:1.0];
    [self.levelMeter setDoubleValue:0.0];
    [meterContainer addSubview:self.levelMeter];
    [self.statusBar addSubview:meterContainer];

    // Performance info (right side)
    self.performanceLabel = [[NSTextField alloc] initWithFrame:NSMakeRect(950, 6, 240, 20)];
    [self.performanceLabel setEditable:NO];
    [self.performanceLabel setBordered:NO];
    [self.performanceLabel setDrawsBackground:NO];
    [self.performanceLabel setFont:[NSFont systemFontOfSize:11 weight:NSFontWeightRegular]];
    [self.performanceLabel setTextColor:[NSColor colorWithCalibratedWhite:0.7 alpha:1.0]];
    [self.performanceLabel setAlignment:NSTextAlignmentRight];
    [self.performanceLabel setStringValue:@"48kHz Mono • Ready"];
    [self.statusBar addSubview:self.performanceLabel];

    // Add status bar to window
    NSView *contentView = [self.window contentView];
    [contentView addSubview:self.statusBar positioned:NSWindowBelow relativeTo:nil];

    // Update status bar constraints
    [self updateStatusBarConstraints];
}

- (void)setupToolbar {
    self.toolbar = [[NSToolbar alloc] initWithIdentifier:@"OscilloscopeToolbar"];
    [self.toolbar setDelegate:self];
    [self.toolbar setDisplayMode:NSToolbarDisplayModeIconOnly];
    [self.toolbar setSizeMode:NSToolbarSizeModeRegular];
    [self.toolbar setAllowsUserCustomization:NO];
    [self.toolbar setAutosavesConfiguration:NO];
    [self.window setToolbar:self.toolbar];
    [self.window setToolbarStyle:NSWindowToolbarStyleUnified];
}

- (void)updateStatusBarConstraints {
    NSView *contentView = [self.window contentView];
    NSRect contentBounds = [contentView bounds];
    NSRect statusFrame = NSMakeRect(0, contentBounds.size.height - 24,
                                   contentBounds.size.width, 24);
    [self.statusBar setFrame:statusFrame];
}

- (void)requestMicrophonePermission {
    switch ([AVCaptureDevice authorizationStatusForMediaType:AVMediaTypeAudio]) {
        case AVAuthorizationStatusAuthorized: {
            [self.statusLabel setStringValue:@"🎙️ Connected • AGC: Initializing..."];
            [self startAudioCapture];
            break;
        }

        case AVAuthorizationStatusNotDetermined: {
            [self.statusLabel setStringValue:@"🎙️ Requesting microphone access..."];
            [AVCaptureDevice requestAccessForMediaType:AVMediaTypeAudio
                                     completionHandler:^(BOOL granted) {
                dispatch_async(dispatch_get_main_queue(), ^{
                    if (granted) {
                        [self.statusLabel setStringValue:@"🎙️ Connected • AGC: Initializing..."];
                        [self startAudioCapture];
                    } else {
                        [self showMicrophonePermissionDenied];
                    }
                });
            }];
            break;
        }

        case AVAuthorizationStatusDenied:
        case AVAuthorizationStatusRestricted: {
            [self showMicrophonePermissionDenied];
            break;
        }
    }
}

- (void)showMicrophonePermissionDenied {
    [self.statusLabel setStringValue:@"❌ Microphone access denied"];

    NSAlert *alert = [[NSAlert alloc] init];
    [alert setMessageText:@"Microphone Access Required"];
    [alert setInformativeText:@"Vintage Oscilloscope needs microphone access to visualize your audio in real-time.\n\n"
                             @"To grant permission:\n"
                             @"• Open System Settings\n"
                             @"• Go to Privacy & Security → Microphone\n"
                             @"• Enable Vintage Oscilloscope\n"
                             @"• Restart the application"];

    [alert setAlertStyle:NSAlertStyleWarning];

    [alert addButtonWithTitle:@"Open System Settings"];
    [alert addButtonWithTitle:@"Quit"];

    NSModalResponse response = [alert runModal];
    if (response == NSAlertFirstButtonReturn) {
        [[NSWorkspace sharedWorkspace] openURL:[NSURL URLWithString:@"x-apple.systempreferences:com.apple.preference.security?Privacy_Microphone"]];
    }
    [NSApp terminate:nil];
}

- (void)startAudioCapture {
    self.audioManager = [[AudioCaptureManager alloc] init];
    [self.audioManager start];

    // Set up interruption monitoring
    [self setupCaptureSessionForInterruptionNotifications];

    // Create update timer
    self.updateTimer = [NSTimer timerWithTimeInterval:1.0/60.0
                                              target:self
                                            selector:@selector(updateAudio)
                                            userInfo:nil
                                             repeats:YES];
    [[NSRunLoop mainRunLoop] addTimer:self.updateTimer
                             forMode:NSRunLoopCommonModes];

    // Update AGC button state now that audioManager is initialized
    [self updateAGCButtonState];

    [self updateStatusDisplay];
}

- (void)toggleAGC:(id)sender {
    if (!self.audioManager) return;
    self.audioManager.agcEnabled = !self.audioManager.agcEnabled;

    // Update AGC button state to reflect current AGC status
    [self updateAGCButtonState];

    [self syncGainSlider];
    [self updateStatusDisplay];
}

- (void)updateAGCButtonState {
    // Update the AGC button state to reflect current AGC enabled status
    if (self.agcItem && [self.agcItem view] && [[self.agcItem view] isKindOfClass:[NSButton class]]) {
        NSButton *agcButton = (NSButton *)[self.agcItem view];
        [agcButton setState:self.audioManager.agcEnabled ? NSControlStateValueOn : NSControlStateValueOff];
    }
}

- (void)setupCaptureSessionForInterruptionNotifications {
    self.captureSession = [[AVCaptureSession alloc] init];

    AVCaptureDevice *audioDevice = [AVCaptureDevice defaultDeviceWithMediaType:AVMediaTypeAudio];
    if (audioDevice) {
        NSError *error = nil;
        AVCaptureDeviceInput *audioInput = [AVCaptureDeviceInput deviceInputWithDevice:audioDevice error:&error];
        if (audioInput && [self.captureSession canAddInput:audioInput]) {
            [self.captureSession addInput:audioInput];
        }
    }

    [[NSNotificationCenter defaultCenter] addObserver:self
                                             selector:@selector(handleSessionInterruption:)
                                                 name:AVCaptureSessionWasInterruptedNotification
                                               object:self.captureSession];

    [[NSNotificationCenter defaultCenter] addObserver:self
                                             selector:@selector(handleSessionInterruptionEnded:)
                                                 name:AVCaptureSessionInterruptionEndedNotification
                                               object:self.captureSession];

    [self.captureSession startRunning];
}

- (void)updateAudio {
    float samples[512];
    NSUInteger count = [self.audioManager getLatestSamples:samples maxSamples:512];

    // Update level meter
    if (count > 0) {
        float maxLevel = 0.0f;
        for (NSUInteger i = 0; i < count; i++) {
            maxLevel = MAX(maxLevel, fabsf(samples[i]));
        }
        [self.levelMeter setDoubleValue:maxLevel];
    }

    // Update oscilloscope
    if (count > 0 && count <= 512) {
        [self.oscilloscopeView updateWithAudioSamples:samples count:count];
    } else {
        [self.oscilloscopeView updateWithAudioSamples:NULL count:0];
    }

    // Update gain slider when AGC is active (gain changes automatically)
    if (self.audioManager.agcEnabled) {
        [self syncGainSlider];
    }

    // Update status periodically
    static int frameCounter = 0;
    if (++frameCounter % 60 == 0) { // Update every second
        [self updateStatusDisplay];
    }
}

- (void)updateStatusDisplay {
    if (!self.audioManager) return;

    NSString *agcStatus = self.audioManager.agcEnabled ? @"ON" : @"OFF";
    NSString *status = [NSString stringWithFormat:@"AGC: %@ • Gain: %.1fx",
                       agcStatus, self.audioManager.gain];
    [self.statusLabel setStringValue:status];

    // Update performance info with better formatting
    [self.performanceLabel setStringValue:@"48kHz Mono • Real-time"];
}

- (void)syncGainSlider {
    if (!self.gainSlider || !self.audioManager) return;

    // Update slider position to match current gain value
    [self.gainSlider setDoubleValue:self.audioManager.gain];

    // Update the value label in toolbar
    NSView *container = [self.gainSlider superview];
    for (NSView *subview in [container subviews]) {
        if ([subview isKindOfClass:[NSTextField class]] && [subview tag] == 1001) {
            NSTextField *valueLabel = (NSTextField *)subview;
            [valueLabel setStringValue:[NSString stringWithFormat:@"%.0f", self.audioManager.gain]];
            break;
        }
    }
}

- (void)increaseGain:(id)sender {
    if (!self.audioManager) return;
    self.audioManager.gain += 1.0f;
    // Cap at 50.0 to match AGC_MAX_GAIN and slider range
    if (self.audioManager.gain > 50.0f) self.audioManager.gain = 50.0f;
    self.audioManager.agcEnabled = NO;
    [self syncGainSlider];
    [self updateAGCButtonState];
    [self updateStatusDisplay];
}

- (void)decreaseGain:(id)sender {
    if (!self.audioManager) return;
    self.audioManager.gain -= 1.0f;
    // Cap at 1.0 to match AGC_MIN_GAIN
    if (self.audioManager.gain < 1.0f) self.audioManager.gain = 1.0f;
    self.audioManager.agcEnabled = NO;
    [self syncGainSlider];
    [self updateAGCButtonState];
    [self updateStatusDisplay];
}

- (void)showPreferences:(id)sender {
    // Placeholder for preferences panel
    NSAlert *alert = [[NSAlert alloc] init];
    [alert setMessageText:@"Preferences"];
    [alert setInformativeText:@"Preferences panel coming soon!\n\n"
                             @"Use keyboard shortcuts for now:\n"
                             @"• A: Toggle AGC\n"
                             @"• ↑/↓: Adjust gain"];
    [alert addButtonWithTitle:@"OK"];
    [alert runModal];
}

- (void)exportScreenshot:(id)sender {
    NSSavePanel *savePanel = [NSSavePanel savePanel];
    // Use string-based content types for compatibility
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wdeprecated-declarations"
    [savePanel setAllowedFileTypes:@[@"png", @"jpg", @"jpeg", @"tiff", @"tif"]];
#pragma clang diagnostic pop

    [savePanel beginSheetModalForWindow:self.window completionHandler:^(NSModalResponse result) {
        if (result == NSModalResponseOK) {
            NSBitmapImageRep *imageRep = [self.oscilloscopeView bitmapImageRepForCachingDisplayInRect:self.oscilloscopeView.bounds];
            [self.oscilloscopeView cacheDisplayInRect:self.oscilloscopeView.bounds toBitmapImageRep:imageRep];

            NSDictionary *properties = @{NSImageCompressionFactor: @1.0};
            NSBitmapImageFileType fileType;

            NSString *extension = [[savePanel URL] pathExtension];
            if ([extension isEqualToString:@"jpg"] || [extension isEqualToString:@"jpeg"]) {
                fileType = NSBitmapImageFileTypeJPEG;
            } else if ([extension isEqualToString:@"tiff"] || [extension isEqualToString:@"tif"]) {
                fileType = NSBitmapImageFileTypeTIFF;
            } else {
                fileType = NSBitmapImageFileTypePNG;
            }

            NSData *data = [imageRep representationUsingType:fileType properties:properties];
            [data writeToURL:[savePanel URL] atomically:YES];
        }
    }];
}

- (void)updateWindowTitle {
    // Update window title with current status
    if (!self.window || !self.audioManager) return;

    NSString *status;
    if (self.audioManager.agcEnabled) {
        status = [NSString stringWithFormat:@"AGC: ON (%.1fx)", self.audioManager.gain];
    } else {
        status = [NSString stringWithFormat:@"AGC: OFF (%.1fx)", self.audioManager.gain];
    }
    [self.window setTitle:status];
}

// MARK: - NSToolbarDelegate

- (NSToolbarItem *)toolbar:(NSToolbar *)toolbar itemForItemIdentifier:(NSString *)itemIdentifier willBeInsertedIntoToolbar:(BOOL)flag {
    NSToolbarItem *item = [[NSToolbarItem alloc] initWithItemIdentifier:itemIdentifier];

    if ([itemIdentifier isEqualToString:@"AGC"]) {
        [item setLabel:@"AGC"];
        [item setPaletteLabel:@"AGC"];
        [item setToolTip:@"Toggle Automatic Gain Control"];

        // Beautiful toggle button
        NSButton *button = [[NSButton alloc] initWithFrame:NSMakeRect(0, 0, 50, 32)];
        [button setButtonType:NSButtonTypePushOnPushOff];
        [button setBezelStyle:NSBezelStyleRounded];
        [button setTitle:@"AGC"];
        [button setFont:[NSFont systemFontOfSize:11 weight:NSFontWeightMedium]];
        [button setTarget:self];
        [button setAction:@selector(toggleAGC:)];

        // Style the button
        if (@available(macOS 10.14, *)) {
            [button setContentTintColor:[NSColor controlAccentColor]];
        }

        // Set initial button state based on current AGC state
        [button setState:self.audioManager.agcEnabled ? NSControlStateValueOn : NSControlStateValueOff];

         [item setView:button];
        self.agcItem = item;

    } else if ([itemIdentifier isEqualToString:@"Gain"]) {
        [item setLabel:@"Gain"];
        [item setPaletteLabel:@"Gain"];
        [item setToolTip:@"Manual Gain Control"];

        // Beautiful slider with label
        NSView *container = [[NSView alloc] initWithFrame:NSMakeRect(0, 0, 155, 32)];

        NSTextField *label = [[NSTextField alloc] initWithFrame:NSMakeRect(0, 14, 35, 18)];
        [label setEditable:NO];
        [label setBordered:NO];
        [label setDrawsBackground:NO];
        [label setFont:[NSFont systemFontOfSize:10 weight:NSFontWeightRegular]];
        [label setTextColor:[NSColor secondaryLabelColor]];
        [label setStringValue:@"Gain:"];
        [label setAlignment:NSTextAlignmentRight];
        [container addSubview:label];

        self.gainSlider = [[NSSlider alloc] initWithFrame:NSMakeRect(40, 8, 80, 24)];
        [self.gainSlider setMinValue:1.0];
        [self.gainSlider setMaxValue:50.0];
        [self.gainSlider setDoubleValue:10.0];
        [self.gainSlider setTarget:self];
        [self.gainSlider setAction:@selector(gainSliderChanged:)];
        [container addSubview:self.gainSlider];

        NSTextField *valueLabel = [[NSTextField alloc] initWithFrame:NSMakeRect(125, 14, 25, 18)];
        [valueLabel setEditable:NO];
        [valueLabel setBordered:NO];
        [valueLabel setDrawsBackground:NO];
        [valueLabel setFont:[NSFont systemFontOfSize:10 weight:NSFontWeightRegular]];
        [valueLabel setTextColor:[NSColor secondaryLabelColor]];
        [valueLabel setStringValue:@"10"];
        [valueLabel setTag:1001]; // Tag for updating
        [container addSubview:valueLabel];

         [item setView:container];
        [item setMinSize:NSMakeSize(155, 32)];
        [item setMaxSize:NSMakeSize(155, 32)];

    } else if ([itemIdentifier isEqualToString:@"Preferences"]) {
        [item setLabel:@"Preferences"];
        [item setPaletteLabel:@"Preferences"];
        [item setToolTip:@"Open Preferences"];
        [item setImage:[NSImage imageNamed:NSImageNamePreferencesGeneral]];
        [item setTarget:self];
        [item setAction:@selector(showPreferences:)];

    } else if ([itemIdentifier isEqualToString:@"Screenshot"]) {
        [item setLabel:@"Screenshot"];
        [item setPaletteLabel:@"Screenshot"];
        [item setToolTip:@"Export Screenshot"];

        // Custom camera icon button
        NSButton *button = [[NSButton alloc] initWithFrame:NSMakeRect(0, 0, 32, 32)];
        [button setButtonType:NSButtonTypeMomentaryPushIn];
        [button setBezelStyle:NSBezelStyleRounded];
        [button setImage:[NSImage imageNamed:NSImageNameShareTemplate]];
        [button setImageScaling:NSImageScaleProportionallyUpOrDown];
        [button setTarget:self];
        [button setAction:@selector(exportScreenshot:)];

         [item setView:button];
    }

    return item;
}

- (NSArray<NSString *> *)toolbarAllowedItemIdentifiers:(NSToolbar *)toolbar {
    return @[@"AGC", @"Gain", @"Preferences", @"Screenshot", NSToolbarFlexibleSpaceItemIdentifier];
}

- (NSArray<NSString *> *)toolbarDefaultItemIdentifiers:(NSToolbar *)toolbar {
    return @[@"AGC", @"Gain", NSToolbarFlexibleSpaceItemIdentifier, @"Screenshot", @"Preferences"];
}

- (void)gainSliderChanged:(id)sender {
    if (!self.audioManager) return;
    self.audioManager.gain = [self.gainSlider doubleValue];
    self.audioManager.agcEnabled = NO;

    // Update the value label in toolbar
    NSView *container = [self.gainSlider superview];
    for (NSView *subview in [container subviews]) {
        if ([subview isKindOfClass:[NSTextField class]] && [subview tag] == 1001) {
            NSTextField *valueLabel = (NSTextField *)subview;
            [valueLabel setStringValue:[NSString stringWithFormat:@"%.0f", self.audioManager.gain]];
            break;
        }
    }

    [self updateAGCButtonState];
    [self updateStatusDisplay];
}

// MARK: - NSWindowDelegate

- (void)windowDidResize:(NSNotification *)notification {
    [self updateStatusBarConstraints];
}

- (void)windowWillClose:(NSNotification *)notification {
    [self.audioManager stop];
    [self.updateTimer invalidate];
    [NSApp terminate:nil];
}

// MARK: - Error Handling

- (void)showErrorAlert:(NSString *)title description:(NSString *)description {
    NSAlert *alert = [[NSAlert alloc] init];
    [alert setMessageText:title];
    [alert setInformativeText:description];
    [alert addButtonWithTitle:@"OK"];
    [alert runModal];
}

@end