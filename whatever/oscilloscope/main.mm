#import <Cocoa/Cocoa.h>
#import "MainWindowController.h"

@interface AppDelegate : NSObject <NSApplicationDelegate>
@property(strong, nonatomic) MainWindowController *mainWindowController;
@end

@implementation AppDelegate

- (void)applicationDidFinishLaunching:(NSNotification *)notification {
    // Create modern menu bar
    [self createMenuBar];

    // Create main window controller
    self.mainWindowController = [[MainWindowController alloc] init];
    [self.mainWindowController.window makeKeyAndOrderFront:nil];

    // Activate app to bring it to front
    [NSApp activateIgnoringOtherApps:YES];

    // Set up keyboard event monitoring for global shortcuts
    [NSEvent addLocalMonitorForEventsMatchingMask:NSEventMaskKeyDown
                                          handler:^NSEvent *_Nullable(NSEvent *event) {
        return [self handleGlobalKeyDown:event];
    }];
}

- (NSEvent *)handleGlobalKeyDown:(NSEvent *)event {
    // Handle global keyboard shortcuts
    if (self.mainWindowController) {
        BOOL handled = NO;

        // 'a' or 'A' to toggle AGC
        if ([event.charactersIgnoringModifiers caseInsensitiveCompare:@"a"] == NSOrderedSame) {
            [self.mainWindowController toggleAGC:nil];
            handled = YES;
        }

        // Up/Down arrows to adjust gain
        if (event.keyCode == 126) { // Up Arrow
            [self.mainWindowController increaseGain:nil];
            handled = YES;
        }
        if (event.keyCode == 125) { // Down Arrow
            [self.mainWindowController decreaseGain:nil];
            handled = YES;
        }

        if (handled) {
            return nil; // Event consumed
        }
    }

    return event; // Pass through
}

- (void)createMenuBar {
    // Create comprehensive macOS menu bar
    NSMenu *menuBar = [[NSMenu alloc] init];

    // App menu
    NSMenuItem *appMenuItem = [[NSMenuItem alloc] init];
    [menuBar addItem:appMenuItem];

    NSMenu *appMenu = [[NSMenu alloc] initWithTitle:@"Oscilloscope"];
    [appMenuItem setSubmenu:appMenu];

    // App menu items
    [appMenu addItemWithTitle:@"About Vintage Oscilloscope" action:@selector(orderFrontStandardAboutPanel:) keyEquivalent:@""];
    [appMenu addItem:[NSMenuItem separatorItem]];
    [appMenu addItemWithTitle:@"Preferences..." action:@selector(showPreferences:) keyEquivalent:@","];
    [appMenu addItem:[NSMenuItem separatorItem]];
    [appMenu addItemWithTitle:@"Hide Vintage Oscilloscope" action:@selector(hide:) keyEquivalent:@"h"];
    [appMenu addItemWithTitle:@"Hide Others" action:@selector(hideOtherApplications:) keyEquivalent:@"h"];
    [[appMenu itemWithTitle:@"Hide Others"] setKeyEquivalentModifierMask:NSEventModifierFlagOption | NSEventModifierFlagCommand];
    [appMenu addItemWithTitle:@"Show All" action:@selector(unhideAllApplications:) keyEquivalent:@""];
    [appMenu addItem:[NSMenuItem separatorItem]];
    [appMenu addItemWithTitle:@"Quit Vintage Oscilloscope" action:@selector(terminate:) keyEquivalent:@"q"];

    // File menu
    NSMenuItem *fileMenuItem = [[NSMenuItem alloc] init];
    [fileMenuItem setTitle:@"File"];
    [menuBar addItem:fileMenuItem];

    NSMenu *fileMenu = [[NSMenu alloc] initWithTitle:@"File"];
    [fileMenuItem setSubmenu:fileMenu];

    [fileMenu addItemWithTitle:@"New Window" action:@selector(newDocument:) keyEquivalent:@"n"];
    [fileMenu addItem:[NSMenuItem separatorItem]];
    [fileMenu addItemWithTitle:@"Close Window" action:@selector(performClose:) keyEquivalent:@"w"];

    // Edit menu
    NSMenuItem *editMenuItem = [[NSMenuItem alloc] init];
    [editMenuItem setTitle:@"Edit"];
    [menuBar addItem:editMenuItem];

    NSMenu *editMenu = [[NSMenu alloc] initWithTitle:@"Edit"];
    [editMenuItem setSubmenu:editMenu];

    [editMenu addItemWithTitle:@"Undo" action:@selector(undo:) keyEquivalent:@"z"];
    [editMenu addItemWithTitle:@"Redo" action:@selector(redo:) keyEquivalent:@"Z"];
    [editMenu addItem:[NSMenuItem separatorItem]];
    [editMenu addItemWithTitle:@"Cut" action:@selector(cut:) keyEquivalent:@"x"];
    [editMenu addItemWithTitle:@"Copy" action:@selector(copy:) keyEquivalent:@"c"];
    [editMenu addItemWithTitle:@"Paste" action:@selector(paste:) keyEquivalent:@"v"];

    // View menu
    NSMenuItem *viewMenuItem = [[NSMenuItem alloc] init];
    [viewMenuItem setTitle:@"View"];
    [menuBar addItem:viewMenuItem];

    NSMenu *viewMenu = [[NSMenu alloc] initWithTitle:@"View"];
    [viewMenuItem setSubmenu:viewMenu];

    [viewMenu addItemWithTitle:@"Actual Size" action:@selector(actualSize:) keyEquivalent:@"0"];
    [viewMenu addItemWithTitle:@"Zoom In" action:@selector(zoomIn:) keyEquivalent:@"="];
    [viewMenu addItemWithTitle:@"Zoom Out" action:@selector(zoomOut:) keyEquivalent:@"-"];
    [viewMenu addItem:[NSMenuItem separatorItem]];

    NSMenuItem *agcItem = [[NSMenuItem alloc] initWithTitle:@"Toggle AGC" action:@selector(toggleAGC:) keyEquivalent:@"a"];
    [agcItem setTarget:self.mainWindowController];
    [viewMenu addItem:agcItem];

    [viewMenu addItem:[NSMenuItem separatorItem]];
    [viewMenu addItemWithTitle:@"Enter Full Screen" action:@selector(toggleFullScreen:) keyEquivalent:@"f"];

    // Window menu
    NSMenuItem *windowMenuItem = [[NSMenuItem alloc] init];
    [windowMenuItem setTitle:@"Window"];
    [menuBar addItem:windowMenuItem];

    NSMenu *windowMenu = [[NSMenu alloc] initWithTitle:@"Window"];
    [windowMenuItem setSubmenu:windowMenu];

    [windowMenu addItemWithTitle:@"Minimize" action:@selector(performMiniaturize:) keyEquivalent:@"m"];
    [windowMenu addItemWithTitle:@"Zoom" action:@selector(performZoom:) keyEquivalent:@""];
    [windowMenu addItem:[NSMenuItem separatorItem]];
    [windowMenu addItemWithTitle:@"Bring All to Front" action:@selector(arrangeInFront:) keyEquivalent:@""];

    // Help menu
    NSMenuItem *helpMenuItem = [[NSMenuItem alloc] init];
    [helpMenuItem setTitle:@"Help"];
    [menuBar addItem:helpMenuItem];

    NSMenu *helpMenu = [[NSMenu alloc] initWithTitle:@"Help"];
    [helpMenuItem setSubmenu:helpMenu];

    [helpMenu addItemWithTitle:@"Vintage Oscilloscope Help" action:@selector(showHelp:) keyEquivalent:@"?"];

    [NSApp setMainMenu:menuBar];
}

// MARK: - NSMenuDelegate methods

- (BOOL)validateMenuItem:(NSMenuItem *)menuItem {
    SEL action = [menuItem action];

    if (action == @selector(toggleAGC:)) {
        // Update menu item state based on AGC status
        if (self.mainWindowController.audioManager) {
            [menuItem setState:self.mainWindowController.audioManager.agcEnabled ? NSControlStateValueOn : NSControlStateValueOff];
        }
        return YES;
    }

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
