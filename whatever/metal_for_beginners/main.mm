// =============================================================================
// METAL FOR BEGINNERS - A Guided Tour for C++ Programmers
// =============================================================================
// This file teaches Metal GPU programming to developers familiar with C++
// but new to Objective-C and Apple's Metal framework.
//
// PROJECT: Simple per-pixel shader that runs on the GPU
// GOAL: Understand the minimal boilerplate to get Metal running
//
// File Extension: .mm
// ------------------
// .mm = "Objective-C++" - allows mixing C++, Objective-C, and Metal code
// .m  = Pure Objective-C
// .cpp = Pure C++
// =============================================================================

// -----------------------------------------------------------------------------
// INCLUDES: What each framework does
// -----------------------------------------------------------------------------
#import <AppKit/AppKit.h>   // macOS windows, buttons, events (like Win32 API)
#import <Metal/Metal.h>     // GPU compute & graphics (like Vulkan/DirectX)
#import <MetalKit/MetalKit.h> // Helper classes for Metal rendering
#include <simd/simd.h>      // Vector math types (float2, float4, etc.)

// =============================================================================
// OBJECTIVE-C SYNTAX CRASH COURSE FOR C++ PROGRAMMERS
// =============================================================================
/*
If you know C++, Objective-C will feel alien at first. Here's the translation:

╔═══════════════════════════════════════════════════════════════════════════╗
║                    C++ vs Objective-C Comparison                          ║
╠═══════════════════════════════════════════════════════════════════════════╣
║ C++                          │ Objective-C                                ║
╠══════════════════════════════╪════════════════════════════════════════════╣
║ class MyClass : public Base  │ @interface MyClass : BaseClass             ║
║ {                            │ @end                                       ║
║   public:                    │                                            ║
║     void method();           │ @implementation MyClass                    ║
║ };                           │ - (void)method { }                         ║
║                              │ @end                                       ║
╠══════════════════════════════╪════════════════════════════════════════════╣
║ MyClass* obj = new MyClass() │ MyClass* obj = [[MyClass alloc] init];     ║
╠══════════════════════════════╪════════════════════════════════════════════╣
║ obj->method();               │ [obj method];                              ║
║ obj->setX(5);                │ [obj setX:5];                              ║
║ obj->doThing(a, b, c);       │ [obj doThing:a with:b and:c];              ║
╠══════════════════════════════╪════════════════════════════════════════════╣
║ delete obj;                  │ (Automatic with ARC - see below)           ║
╠══════════════════════════════╪════════════════════════════════════════════╣
║ std::string name;            │ NSString* name;                            ║
║ string name = "hello";       │ NSString* name = @"hello";                 ║
║                              │ (Note the @ prefix for string literals!)   ║
╠══════════════════════════════╪════════════════════════════════════════════╣
║ printf("x = %d\n", x);       │ NSLog(@"x = %d", x);                       ║
╠══════════════════════════════╪════════════════════════════════════════════╣
║ nullptr                      │ nil (for objects), NULL (for pointers)     ║
╠══════════════════════════════╪════════════════════════════════════════════╣
║ static void classMethod()    │ + (void)classMethod;   (note the +)        ║
║ void instanceMethod()        │ - (void)instanceMethod; (note the -)       ║
╚══════════════════════════════╧════════════════════════════════════════════╝

Key Differences:

1. METHOD CALLS USE SQUARE BRACKETS
   C++:       object.method(arg1, arg2)
   Objective-C:  [object method:arg1 with:arg2]

2. METHODS HAVE NAMED PARAMETERS (like Python kwargs)
   Instead of: obj->calculate(10, 20, 30)
   You write:  [obj calculateSum:10 multiplier:20 offset:30]

   The colons are PART OF THE METHOD NAME!
   Method signature: - (int)calculateSum:(int)a multiplier:(int)b offset:(int)c

3. MEMORY MANAGEMENT: ARC (Automatic Reference Counting)
   - Like std::shared_ptr but built into the language
   - Compiler automatically inserts retain/release calls
   - You just create objects and forget about them
   - NO delete, NO free(), just let them go out of scope

4. PROTOCOLS = INTERFACES
   @protocol MyProtocol
   - (void)requiredMethod;
   @end

   Then: @interface MyClass : BaseClass <MyProtocol>

5. PROPERTIES (auto-generated getters/setters)
   @property (nonatomic, strong) NSString* name;

   Access with dot syntax: obj.name = @"hello";
   Or message syntax: [obj setName:@"hello"];

6. id TYPE = void* for objects
   id obj = [[SomeClass alloc] init];  // Type-erased object pointer

7. FORWARD DECLARATIONS
   @class MyClass;  // Like "class MyClass;" in C++

Now let's see this in action...
*/

// =============================================================================
// CONFIGURATION
// =============================================================================
namespace Config {
    constexpr float WINDOW_WIDTH = 800.0f;
    constexpr float WINDOW_HEIGHT = 600.0f;
    constexpr int TARGET_FPS = 60;
}

// =============================================================================
// DATA STRUCTURES
// =============================================================================
// This structure is shared between CPU (this file) and GPU (shader).
// It MUST match exactly with the definition in Compute.metal.

// Uniforms passed to the shader each frame
struct Uniforms {
    float time;         // Time in seconds since app started
    float deltaTime;    // Time since last frame
    int frameCount;     // Total number of frames rendered
    float _padding;     // Align to 16 bytes (Metal requirement)
};
// Note on alignment: Metal requires uniform buffers to be aligned to 16 bytes.
// Each float is 4 bytes, so 4 floats = 16 bytes total.

// =============================================================================
// OBJECTIVE-C CLASS DECLARATIONS
// =============================================================================

// -----------------------------------------------------------------------------
// InputView: Custom MTKView with Keyboard Input
// -----------------------------------------------------------------------------
// MTKView is MetalKit's view class that handles Metal rendering.
// We subclass it to add keyboard event handling.
//
// WHY SUBCLASS?
// By default, MTKView doesn't handle keyboard input. We need to override
// specific methods to receive and process key events.

@interface InputView : MTKView
// Inherits from MTKView, so we get all Metal rendering capabilities
// We just add keyboard handling on top
@end

// @implementation follows later (scroll down)

// -----------------------------------------------------------------------------
// AppDelegate: Handles application lifecycle
// -----------------------------------------------------------------------------
// In Objective-C, you split class definition into two parts:
// 1. @interface = declares what the class looks like (header)
// 2. @implementation = actual code (cpp file)
//
// Syntax breakdown:
// @interface ClassName : ParentClass <Protocol1, Protocol2>

@interface AppDelegate : NSObject <NSApplicationDelegate>
// NSObject = base class for all Objective-C objects (like java.lang.Object)
// <NSApplicationDelegate> = protocol (interface) we implement
//
// No methods or properties to declare here - we'll implement the protocol
@end

// @implementation follows later (scroll down)

// -----------------------------------------------------------------------------
// Renderer: The Metal GPU Manager
// -----------------------------------------------------------------------------
// This class does all the Metal work. It:
// 1. Talks to the GPU (MTLDevice)
// 2. Compiles shaders
// 3. Submits draw commands each frame

// First, a forward declaration (like "class Renderer;" in C++)
@class MTKView; // MTKView is defined in MetalKit framework

@interface Renderer : NSObject <MTKViewDelegate>
// MTKViewDelegate = protocol that gets called each frame to render

// INITIALIZATION METHOD
// Objective-C uses "designated initializers" instead of constructors
// Syntax: - (returnType)methodName:(paramType)paramName
//         - means "instance method" (+ would mean "class method"/"static")
- (instancetype)initWithMetalKitView:(MTKView*)view;
// instancetype = "returns an instance of this class" (like "Renderer*")

@end

// =============================================================================
// IMPLEMENTATION: InputView
// =============================================================================
@implementation InputView

// -----------------------------------------------------------------------------
// Keyboard Input Handling
// -----------------------------------------------------------------------------
// To receive keyboard events, we need to:
// 1. Tell the system this view accepts keyboard input
// 2. Override keyDown: to handle key presses

// METHOD 1: Tell the system we want keyboard events
- (BOOL)acceptsFirstResponder {
    // FIRST RESPONDER = the view that receives keyboard input
    // Like "focus" in web terminology
    //
    // Return YES to indicate: "Yes, send me keyboard events!"
    // Return NO would mean: "Don't send keyboard events to me"
    return YES;
}

// METHOD 2: Handle key presses
- (void)keyDown:(NSEvent*)event {
    // NSEvent contains information about the key press:
    // - Which key was pressed
    // - Modifier keys (Shift, Command, etc.)
    // - Timestamp
    //
    // Syntax note: - (void)keyDown:(NSEvent*)event
    // This is an instance method that takes an NSEvent parameter named "event"

    // Get the character that was pressed
    // [event characters] returns an NSString of the pressed keys
    NSString* chars = [event charactersIgnoringModifiers];
    // charactersIgnoringModifiers = ignore Shift, Cmd, etc.
    // So "Shift+A" gives "a", not "A"

    // Check if any characters were pressed
    if ([chars length] == 0) {
        // No characters (maybe just modifier keys pressed)
        return;
    }

    // Get the first character as a unichar (Unicode character)
    // In C++, this would be: char c = str[0];
    unichar key = [chars characterAtIndex:0];

    // React to specific keys using a switch statement
    // This is just C - nothing Objective-C specific here
    switch (key) {
        case 'q':
        case 'Q':
            // User pressed Q - quit the application
            NSLog(@"Q pressed - quitting application");

            // [NSApp terminate:nil] = quit the app
            // NSApp = global variable for the app (like [NSApplication sharedApplication])
            // terminate: = method that quits the app
            // nil = no sender (we could pass 'self' if we wanted)
            [NSApp terminate:nil];
            break;

        case 27: // ESC key (ASCII code 27)
            NSLog(@"ESC pressed - quitting application");
            [NSApp terminate:nil];
            break;

        default:
            // For any other key, pass it to the superclass
            // This allows the default behavior to happen
            // (e.g., Command+Q to quit, Command+H to hide, etc.)
            [super keyDown:event];
            break;
    }
}

@end
// End of InputView implementation

// =============================================================================
// IMPLEMENTATION: AppDelegate
// =============================================================================
@implementation AppDelegate

// This method is called when the last window closes
// It's part of the NSApplicationDelegate protocol
- (BOOL)applicationShouldTerminateAfterLastWindowClosed:(NSApplication*)sender {
    // Syntax note: (void)sender means "suppress unused parameter warning"
    (void)sender;
    return YES; // Quit app when window closes
}

@end
// End of AppDelegate implementation

// =============================================================================
// IMPLEMENTATION: Renderer
// =============================================================================
@implementation Renderer {
    // -----------------------------------------------------
    // INSTANCE VARIABLES (like C++ private members)
    // -----------------------------------------------------
    // These are inside the @implementation block, so they're private

    // THE "id<Protocol>" TYPE
    // -----------------------
    // id = generic object pointer (like void* but for objects)
    // <MTLDevice> = "any object that implements MTLDevice protocol"
    // Think of it like: std::unique_ptr<IDevice> in C++

    id<MTLDevice> _device;             // The GPU itself
    id<MTLCommandQueue> _commandQueue; // Queue for submitting GPU commands
    id<MTLRenderPipelineState> _pipeline; // Compiled shader program
    id<MTLBuffer> _uniformBuffer;      // Buffer for time/frame data

    MTKView* _view; // The view we're rendering to

    // Timing variables for animation
    double _startTime;    // When the app started
    double _lastFrameTime; // Time of previous frame
    int _frameCount;      // Total frames rendered

    // FPS tracking for window title
    double _fpsUpdateTime; // Last time we updated FPS display
    int _fpsFrameCount;    // Frames since last FPS update

    // Underscore prefix is Objective-C convention for instance variables
    // (like "m_" prefix in C++: m_device, m_commandQueue)
}

// -----------------------------------------------------------------------------
// Initialization (like a constructor)
// -----------------------------------------------------------------------------
- (instancetype)initWithMetalKitView:(MTKView*)view {
    // EVERY Objective-C init method must start with:
    self = [super init];
    // "self" = "this" in C++
    // [super init] = call parent class constructor
    // Pattern: always check if init succeeded

    if (!self) {
        return nil; // nil = nullptr for objects
    }

    // Now initialize our stuff
    _view = view;

    NSLog(@"🔧 Initializing Metal...");
    // NSLog = printf for Objective-C (automatically adds newline)
    // @"..." = Objective-C string literal (NOTE THE @ PREFIX!)

    // -------------------------------------------------------------------------
    // STEP 1: Get a GPU device
    // -------------------------------------------------------------------------
    // This is a C function (not Objective-C method) that returns the GPU
    _device = MTLCreateSystemDefaultDevice();

    if (!_device) {
        NSLog(@"❌ ERROR: Metal is not supported on this Mac");
        return nil;
    }

    // String formatting in NSLog (like printf):
    // %@ = Objective-C object (will call its description method)
    NSLog(@"✓ GPU: %@", _device.name);
    // .name is property access (dot syntax works for properties)

    // -------------------------------------------------------------------------
    // STEP 2: Create a command queue
    // -------------------------------------------------------------------------
    // A command queue is how you submit work to the GPU
    // Think: queue.push(draw_command)

    // METHOD CALL SYNTAX:
    // [object methodName]
    _commandQueue = [_device newCommandQueue];

    if (!_commandQueue) {
        NSLog(@"❌ ERROR: Failed to create command queue");
        return nil;
    }
    NSLog(@"✓ Command queue created");

    // -------------------------------------------------------------------------
    // STEP 3: Load and compile shaders
    // -------------------------------------------------------------------------
    if (![self loadShaders]) {
        // Note: [self methodName] = this->methodName() in C++
        return nil;
    }

    // -------------------------------------------------------------------------
    // STEP 4: Create uniform buffer for passing time to GPU
    // -------------------------------------------------------------------------
    // This buffer will hold our Uniforms struct (time, deltaTime, frameCount)
    _uniformBuffer = [_device newBufferWithLength:sizeof(Uniforms)
                                          options:MTLResourceStorageModeShared];
    // MTLResourceStorageModeShared = accessible by both CPU and GPU
    // This lets us write from CPU and read from GPU efficiently

    if (!_uniformBuffer) {
        NSLog(@"❌ ERROR: Failed to create uniform buffer");
        return nil;
    }
    NSLog(@"✓ Uniform buffer created");

    // -------------------------------------------------------------------------
    // STEP 5: Initialize timing variables
    // -------------------------------------------------------------------------
    // CACurrentMediaTime() returns the current time in seconds (high precision)
    // It's like std::chrono::high_resolution_clock in C++
    _startTime = CACurrentMediaTime();
    _lastFrameTime = _startTime;
    _frameCount = 0;
    _fpsUpdateTime = _startTime;
    _fpsFrameCount = 0;

    // -------------------------------------------------------------------------
    // STEP 6: Configure the view
    // -------------------------------------------------------------------------
    view.device = _view.device = _device; // Set which GPU to use
    view.delegate = self;     // "Call me each frame to render"
    view.clearColor = MTLClearColorMake(0.1, 0.05, 0.15, 1.0); // Dark purple background
    // MTLClearColorMake is a C function: (red, green, blue, alpha)

    NSLog(@"✓ Renderer initialized successfully!");
    return self; // Return the initialized object
}

// -----------------------------------------------------------------------------
// Shader Loading (called from init)
// -----------------------------------------------------------------------------
- (BOOL)loadShaders {
    NSLog(@"📦 Loading shaders...");

    // NSError is Objective-C's error handling type
    // In C++ you'd use exceptions or error codes
    // In Objective-C, methods that can fail take an NSError** parameter
    NSError* error = nil;

    // -------------------------------------------------------------------------
    // LOAD SHADER SOURCE FROM FILE
    // -------------------------------------------------------------------------
    // NSString is Objective-C's string class (immutable)
    NSString* shaderPath = @"Compute.metal";

    // METHOD WITH MULTIPLE NAMED PARAMETERS:
    // [object method:arg1 encoding:arg2 error:arg3]
    // This is ONE method call with THREE parameters!
    NSString* shaderSource = [NSString stringWithContentsOfFile:shaderPath
                                                       encoding:NSUTF8StringEncoding
                                                          error:&error];

    if (!shaderSource) {
        NSLog(@"❌ ERROR: Could not load %@", shaderPath);
        NSLog(@"   Make sure Compute.metal is in the current directory");
        return NO; // NO = false, YES = true (Objective-C booleans)
    }

    NSLog(@"✓ Shader source loaded (%lu bytes)", shaderSource.length);

    // -------------------------------------------------------------------------
    // COMPILE SHADERS
    // -------------------------------------------------------------------------
    // Metal shaders are compiled at runtime from source code
    // (Unlike DirectX where you pre-compile to bytecode)

    // Create compilation options
    // [[Class alloc] init] = new Class() in C++
    MTLCompileOptions* options = [[MTLCompileOptions alloc] init];

    // Enable fast math optimizations
    // Note: fastMathEnabled is deprecated on macOS 15+, use mathMode instead
    if (@available(macOS 15.0, *)) {
        options.mathMode = MTLMathModeFast;
    } else {
        // Suppress deprecation warning for older macOS versions
        #pragma clang diagnostic push
        #pragma clang diagnostic ignored "-Wdeprecated-declarations"
        options.fastMathEnabled = YES;
        #pragma clang diagnostic pop
    }

    // Compile the shader source into a library
    id<MTLLibrary> library = [_device newLibraryWithSource:shaderSource
                                                   options:options
                                                     error:&error];
    if (!library) {
        NSLog(@"❌ Shader compilation failed:");
        NSLog(@"%@", error.localizedDescription);
        return NO;
    }

    NSLog(@"✓ Shaders compiled successfully");

    // -------------------------------------------------------------------------
    // CREATE RENDER PIPELINE
    // -------------------------------------------------------------------------
    // A render pipeline = vertex shader + fragment shader + configuration

    // Get the shader functions by name
    // These names must match the function names in Compute.metal
    id<MTLFunction> vertexFunc = [library newFunctionWithName:@"vertex_main"];
    id<MTLFunction> fragmentFunc = [library newFunctionWithName:@"fragment_main"];

    if (!vertexFunc || !fragmentFunc) {
        NSLog(@"❌ ERROR: Could not find shader functions");
        return NO;
    }

    // Create a pipeline descriptor (configuration object)
    MTLRenderPipelineDescriptor* pipelineDesc = [[MTLRenderPipelineDescriptor alloc] init];
    pipelineDesc.vertexFunction = vertexFunc;
    pipelineDesc.fragmentFunction = fragmentFunc;

    // Configure output format (must match the view's format)
    pipelineDesc.colorAttachments[0].pixelFormat = _view.colorPixelFormat;

    // Compile the pipeline
    _pipeline = [_device newRenderPipelineStateWithDescriptor:pipelineDesc
                                                        error:&error];
    if (!_pipeline) {
        NSLog(@"❌ Failed to create render pipeline:");
        NSLog(@"%@", error.localizedDescription);
        return NO;
    }

    NSLog(@"✓ Render pipeline created");
    return YES;
}

// -----------------------------------------------------------------------------
// MTKViewDelegate Protocol Methods
// -----------------------------------------------------------------------------
// MTKView calls these methods automatically

// Called every frame (60 times per second)
- (void)drawInMTKView:(MTKView*)view {
    // PARAMETER NAME NOTE:
    // Method signature is: - (void)drawInMTKView:(MTKView*)view
    // The "drawInMTKView:" is the method name (including the colon!)
    // "view" is the parameter name
    // So to call this: [renderer drawInMTKView:someView]

    // This is the render loop - runs every frame
    // Here we submit GPU commands to draw stuff

    // -------------------------------------------------------------------------
    // UPDATE TIMING AND UNIFORMS
    // -------------------------------------------------------------------------
    // Calculate elapsed time and delta time
    double currentTime = CACurrentMediaTime();
    double elapsedTime = currentTime - _startTime;
    double deltaTime = currentTime - _lastFrameTime;
    _lastFrameTime = currentTime;
    _frameCount++;

    // Write timing data to the uniform buffer
    // Get a pointer to the buffer's memory
    Uniforms* uniforms = (Uniforms*)_uniformBuffer.contents;
    // Now write the data (like writing to any C struct)
    uniforms->time = (float)elapsedTime;
    uniforms->deltaTime = (float)deltaTime;
    uniforms->frameCount = _frameCount;
    // No need to write _padding - it's just for alignment

    // Note: With MTLResourceStorageModeShared, the GPU can see our writes
    // immediately. No need for manual synchronization!

    // -------------------------------------------------------------------------
    // UPDATE FPS IN WINDOW TITLE
    // -------------------------------------------------------------------------
    // Update FPS display every second (not every frame - too expensive!)
    _fpsFrameCount++;
    if (currentTime - _fpsUpdateTime >= 1.0) {
        // Calculate FPS: frames / time elapsed
        double fps = _fpsFrameCount / (currentTime - _fpsUpdateTime);

        // Update window title on the main thread
        // WHY MAIN THREAD?
        // - UI updates MUST happen on the main thread in macOS/iOS
        // - We're currently on the Metal rendering thread
        // - dispatch_async queues the work for the main thread
        //
        // OBJECTIVE-C BLOCKS (like C++ lambdas):
        // - Syntax: ^{ code }
        // - Can capture variables from surrounding scope
        // - 'self' is captured, so we can access instance variables
        dispatch_async(dispatch_get_main_queue(), ^{
            // INSIDE THE BLOCK:
            // Format string with FPS
            // [NSString stringWithFormat:...] = like sprintf() in C
            // %.1f = float with 1 decimal place
            NSString* title = [NSString stringWithFormat:@"Metal for Beginners - %.1f FPS", fps];

            // Set the window title
            // self->_view = access instance variable from inside block
            // Arrow syntax required inside blocks (not dot syntax)
            [self->_view.window setTitle:title];
        });

        // Reset counters for next second
        _fpsFrameCount = 0;
        _fpsUpdateTime = currentTime;
    }

    // -------------------------------------------------------------------------
    // CREATE COMMAND BUFFER
    // -------------------------------------------------------------------------
    // A command buffer holds encoded GPU commands
    // Think: std::vector<GPUCommand>
    id<MTLCommandBuffer> commandBuffer = [_commandQueue commandBuffer];
    if (!commandBuffer) return;

    // -------------------------------------------------------------------------
    // GET RENDER PASS DESCRIPTOR
    // -------------------------------------------------------------------------
    // This describes what we're rendering to (the screen)
    MTLRenderPassDescriptor* renderPass = view.currentRenderPassDescriptor;
    if (!renderPass) return; // No drawable available

    // -------------------------------------------------------------------------
    // ENCODE RENDER COMMANDS
    // -------------------------------------------------------------------------
    // A render command encoder records drawing commands
    id<MTLRenderCommandEncoder> encoder =
        [commandBuffer renderCommandEncoderWithDescriptor:renderPass];

    // Set the pipeline (which shaders to use)
    [encoder setRenderPipelineState:_pipeline];

    // BIND THE UNIFORM BUFFER TO THE FRAGMENT SHADER
    // This makes the Uniforms data available to the shader
    // atIndex:0 means it will be [[buffer(0)]] in the shader
    [encoder setFragmentBuffer:_uniformBuffer offset:0 atIndex:0];

    // Draw a fullscreen triangle (3 vertices)
    // Metal will call our vertex shader 3 times, then fill in pixels
    // Primitive types: Point, Line, Triangle
    [encoder drawPrimitives:MTLPrimitiveTypeTriangle
                vertexStart:0
                vertexCount:3];

    // Finish encoding commands
    [encoder endEncoding];

    // -------------------------------------------------------------------------
    // PRESENT AND COMMIT
    // -------------------------------------------------------------------------
    // presentDrawable: show the rendered image on screen
    [commandBuffer presentDrawable:view.currentDrawable];

    // commit: send commands to GPU for execution
    [commandBuffer commit];

    // After commit(), the GPU starts working asynchronously
    // We don't wait for it - we return and get called again next frame
}

// Called when window resizes (required by protocol)
- (void)mtkView:(MTKView*)view drawableSizeWillChange:(CGSize)size {
    // CGSize is a struct with .width and .height
    // We could update projection matrices here if needed
    (void)view; // Suppress unused warnings
    (void)size;
}

@end
// End of Renderer implementation

// =============================================================================
// MAIN FUNCTION - Application Entry Point
// =============================================================================
// This is just C - no Objective-C yet
int main(int argc, const char* argv[]) {
    (void)argc;
    (void)argv;

    printf("\n");
    printf("╔════════════════════════════════════════════════════════════╗\n");
    printf("║         METAL FOR BEGINNERS - GPU Programming 101          ║\n");
    printf("║                                                            ║\n");
    printf("║  Animated gradient with time-based color effects           ║\n");
    printf("║  FPS counter displayed in window title                     ║\n");
    printf("║                                                            ║\n");
    printf("║  Controls:                                                 ║\n");
    printf("║    Q or ESC - Quit the application                         ║\n");
    printf("║    Close window - Also quits                               ║\n");
    printf("╚════════════════════════════════════════════════════════════╝\n");
    printf("\n");

    // @autoreleasepool = memory management scope
    // Objects created inside are automatically freed at the end
    // Like RAII in C++: { std::unique_ptr<T> obj; } // auto-freed here
    @autoreleasepool {

        // ---------------------------------------------------------------------
        // CREATE APPLICATION
        // ---------------------------------------------------------------------
        // NSApplication is the main app object (singleton)
        // [NSApplication sharedApplication] = NSApplication::getInstance()
        NSApplication* app = [NSApplication sharedApplication];

        // Make it a regular app (appears in Dock, can be focused)
        [app setActivationPolicy:NSApplicationActivationPolicyRegular];

        // Set up delegate to handle app events
        AppDelegate* delegate = [[AppDelegate alloc] init];
        [app setDelegate:delegate];

        // ---------------------------------------------------------------------
        // CREATE WINDOW
        // ---------------------------------------------------------------------
        // NSRect = rectangle struct {x, y, width, height}
        NSRect frame = NSMakeRect(0, 0, Config::WINDOW_WIDTH, Config::WINDOW_HEIGHT);

        // Window style: what buttons/features it has
        // | = bitwise OR (combining flags)
        NSWindowStyleMask style =
            NSWindowStyleMaskTitled       |  // Has title bar
            NSWindowStyleMaskClosable     |  // Has close button [X]
            NSWindowStyleMaskMiniaturizable; // Has minimize button [-]

        // Create window
        // Multiple parameters with named labels:
        NSWindow* window = [[NSWindow alloc] initWithContentRect:frame
                                                       styleMask:style
                                                         backing:NSBackingStoreBuffered
                                                           defer:NO];

        // PROPERTY SETTING with bracket syntax:
        [window setTitle:@"Metal for Beginners"];

        // PROPERTY SETTING with dot syntax (same as above):
        // window.title = @"Metal for Beginners";

        [window center]; // Center on screen

        // ---------------------------------------------------------------------
        // CREATE METAL VIEW
        // ---------------------------------------------------------------------
        // Use our custom InputView instead of MTKView
        // InputView = MTKView + keyboard handling
        InputView* metalView = [[InputView alloc] initWithFrame:frame];
        metalView.preferredFramesPerSecond = Config::TARGET_FPS;

        // ---------------------------------------------------------------------
        // CREATE RENDERER
        // ---------------------------------------------------------------------
        Renderer* renderer = [[Renderer alloc] initWithMetalKitView:metalView];

        if (!renderer) {
            NSLog(@"❌ Failed to initialize renderer");
            return 1;
        }

        // ---------------------------------------------------------------------
        // ASSEMBLE AND SHOW
        // ---------------------------------------------------------------------
        [window setContentView:metalView]; // Put view in window
        [window makeKeyAndOrderFront:nil]; // Show window and bring to front

        // IMPORTANT: Make the view the first responder
        // This tells the window to send keyboard events to our InputView
        // Without this, keyDown: won't be called!
        [window makeFirstResponder:metalView];

        [app activateIgnoringOtherApps:YES]; // Focus the app

        // ---------------------------------------------------------------------
        // RUN THE APPLICATION
        // ---------------------------------------------------------------------
        // This enters the event loop and doesn't return until app quits
        // The event loop:
        // 1. Processes mouse/keyboard events
        // 2. Calls drawInMTKView: every frame (60 FPS)
        // 3. Updates the window
        [app run];

        // We only get here after quitting
    }

    printf("\n✓ Application exited cleanly\n");
    return 0;
}

// =============================================================================
// SUMMARY FOR C++ PROGRAMMERS
// =============================================================================
/*
╔═══════════════════════════════════════════════════════════════════════════╗
║                    Metal Pipeline Quick Reference                         ║
╠═══════════════════════════════════════════════════════════════════════════╣
║ 1. MTLDevice               │ The GPU (like VkDevice)                      ║
║ 2. MTLCommandQueue         │ Submit commands to GPU                       ║
║ 3. MTLLibrary              │ Compiled shader code                         ║
║ 4. MTLFunction             │ Individual shader function                   ║
║ 5. MTLPipelineState        │ Compiled shader pipeline                     ║
║ 6. MTLCommandBuffer        │ Container for commands                       ║
║ 7. MTLCommandEncoder       │ Records commands into buffer                 ║
║ 8. commit()                │ Send commands to GPU                         ║
╚═══════════════════════════════════════════════════════════════════════════╝

Every Metal app follows this pattern:

    Device → Queue → Library → Pipeline → CommandBuffer → Encoder → Draw → Commit

Key Objective-C patterns you saw:

    1. [object method]                  - Call a method
    2. [object method:arg]              - Method with one parameter
    3. [object doA:a withB:b]           - Method with named parameters
    4. [[Class alloc] init]             - Create an object (new)
    5. @"string"                        - String literal (note @)
    6. @protocol                        - Interface definition
    7. @interface/@implementation       - Class definition
    8. self / super                     - this / base in C++
    9. nil                              - nullptr for objects
    10. id<Protocol>                    - Generic object pointer

Memory Management:
    - ARC (Automatic Reference Counting) handles everything
    - No delete, no free()
    - Objects are freed when nothing references them
    - Like std::shared_ptr built into the language

Next Steps:
    - Read Compute.metal to see GPU shader code
    - Modify the shader to change colors/patterns
    - Add more geometry in drawPrimitives call
    - Add buffers to pass data to shaders (see template project)
*/
