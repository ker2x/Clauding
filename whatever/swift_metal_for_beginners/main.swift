// =============================================================================
// SWIFT + METAL FOR BEGINNERS - A Guided Tour for Programmers
// =============================================================================
// This file teaches Metal GPU programming using Swift to developers familiar
// with other languages but new to Swift and Metal.
//
// PROJECT: Animated gradient with time-based effects and FPS counter
// GOAL: Understand Swift syntax and Metal basics
//
// File Extension: .swift
// ------------------
// .swift = Pure Swift (Apple's modern language)
// Can call Objective-C frameworks seamlessly
//
// WHY SWIFT FOR METAL?
// - More concise than Objective-C (no square brackets!)
// - Type-safe with excellent type inference
// - Modern features: optionals, generics, closures
// - Still full access to all Metal APIs
// - Preferred by Apple for new projects
//
// =============================================================================

// -----------------------------------------------------------------------------
// IMPORTS: What each framework does
// -----------------------------------------------------------------------------
import AppKit      // macOS windows, buttons, events
import Metal       // GPU compute & graphics
import MetalKit    // Helper classes for Metal
import QuartzCore  // High-precision timing

// =============================================================================
// SWIFT SYNTAX CRASH COURSE FOR C++/OBJC PROGRAMMERS
// =============================================================================
/*
Swift is Apple's modern replacement for Objective-C. Key differences:

╔═══════════════════════════════════════════════════════════════════════════╗
║                    C++ / Objective-C vs Swift                             ║
╠═══════════════════════════════════════════════════════════════════════════╣
║ C++/ObjC                     │ Swift                                      ║
╠══════════════════════════════╪════════════════════════════════════════════╣
║ MyClass* obj = new MyClass() │ let obj = MyClass()                        ║
║ [[MyClass alloc] init]       │                                            ║
╠══════════════════════════════╪════════════════════════════════════════════╣
║ obj->method();               │ obj.method()                               ║
║ [obj method]                 │                                            ║
╠══════════════════════════════╪════════════════════════════════════════════╣
║ delete obj;                  │ (Automatic - ARC handles it)               ║
╠══════════════════════════════╪════════════════════════════════════════════╣
║ std::string / NSString*      │ String (native Swift type)                 ║
║ "hello" / @"hello"           │ "hello" (no prefix needed!)                ║
╠══════════════════════════════╪════════════════════════════════════════════╣
║ printf() / NSLog()           │ print()                                    ║
╠══════════════════════════════╪════════════════════════════════════════════╣
║ nullptr / nil                │ nil (only for optionals)                   ║
╠══════════════════════════════╪════════════════════════════════════════════╣
║ int x = 5;                   │ let x = 5  (immutable)                     ║
║                              │ var x = 5  (mutable)                       ║
╠══════════════════════════════╪════════════════════════════════════════════╣
║ void func(int x) { }         │ func myFunc(_ x: Int) { }                  ║
╠══════════════════════════════╪════════════════════════════════════════════╣
║ if (ptr != nullptr)          │ if let unwrapped = optional { }            ║
╠══════════════════════════════╪════════════════════════════════════════════╣
║ auto lambda = []() { };      │ let closure = { }                          ║
║ ^{ } (ObjC block)            │                                            ║
╚══════════════════════════════╧════════════════════════════════════════════╝

KEY SWIFT CONCEPTS:

1. OPTIONALS (handling nil safely)
   var name: String?        // Can be String or nil
   if let unwrapped = name { // Safe unwrapping
       print(unwrapped)
   }

2. let vs var
   let x = 5   // Immutable (like const)
   var y = 10  // Mutable

3. TYPE INFERENCE
   let x = 5          // Compiler knows it's Int
   let name = "Alice" // Compiler knows it's String

4. NO HEADER FILES
   Everything in one .swift file (or split across files in same module)

5. CLOSURES (like lambdas)
   { (param: Type) -> ReturnType in
       // code
   }

Now let's see Swift + Metal in action...
*/

// =============================================================================
// CONFIGURATION
// =============================================================================
struct Config {
    static let windowWidth: CGFloat = 800
    static let windowHeight: CGFloat = 600
    static let targetFPS = 60
}

// =============================================================================
// DATA STRUCTURES
// =============================================================================
// This structure is shared between CPU (Swift) and GPU (Metal shader).
// It MUST match exactly with the definition in Compute.metal.

struct Uniforms {
    var time: Float         // Time in seconds since app started
    var deltaTime: Float    // Time since last frame
    var frameCount: Int32   // Total frames rendered (Int32 to match Metal's int)
    var _padding: Float     // Align to 16 bytes (Metal requirement)
}

// =============================================================================
// InputView - Custom MTKView with Keyboard Input
// =============================================================================
// In Swift, we subclass using `: ParentClass` syntax

class InputView: MTKView {
    // SWIFT PROPERTY
    // This is simpler than Objective-C - no @property needed
    weak var renderer: Renderer?  // weak = don't create retain cycle

    // Override method to accept keyboard input
    // 'override' keyword required when overriding superclass methods
    override var acceptsFirstResponder: Bool {
        return true  // or just: true
    }

    // Handle key presses
    // 'override' + function signature
    override func keyDown(with event: NSEvent) {
        // SWIFT STRING ACCESS
        // No need for [event characters] - just event.characters
        guard let characters = event.charactersIgnoringModifiers else {
            return
        }

        // Get first character
        // String access is different in Swift
        guard let key = characters.first else {
            return
        }

        // SWITCH STATEMENT
        // Swift switch is more powerful than C++ switch
        switch key {
        case "q", "Q":
            print("Q pressed - quitting")
            // NSApp is global constant (like in ObjC)
            NSApp.terminate(nil)

        case "\u{1B}":  // ESC key (Unicode escape)
            print("ESC pressed - quitting")
            NSApp.terminate(nil)

        default:
            // Call superclass for unhandled keys
            super.keyDown(with: event)
        }
    }
}

// =============================================================================
// AppDelegate - Application Lifecycle
// =============================================================================
// In Swift, class conformance to protocols uses `: Protocol1, Protocol2`

class AppDelegate: NSObject, NSApplicationDelegate {
    // Quit when last window closes
    func applicationShouldTerminateAfterLastWindowClosed(_ sender: NSApplication) -> Bool {
        return true
    }
}

// =============================================================================
// Renderer - The Metal GPU Manager
// =============================================================================
// This class handles all Metal operations and conforms to MTKViewDelegate

class Renderer: NSObject, MTKViewDelegate {

    // -------------------------------------------------------------------------
    // PROPERTIES (Swift's version of member variables)
    // -------------------------------------------------------------------------
    // No need for underscore prefix in Swift
    // Type is inferred from initialization or explicitly declared

    let device: MTLDevice           // The GPU (let = immutable reference)
    let commandQueue: MTLCommandQueue
    var pipeline: MTLRenderPipelineState?  // ? = optional (can be nil)
    var uniformBuffer: MTLBuffer?

    weak var view: MTKView?  // weak = prevent retain cycle

    // Timing variables
    var startTime: Double
    var lastFrameTime: Double
    var frameCount: Int = 0

    // FPS tracking
    var fpsUpdateTime: Double
    var fpsFrameCount: Int = 0

    // -------------------------------------------------------------------------
    // INITIALIZATION (Swift's init method)
    // -------------------------------------------------------------------------
    // Swift has designated initializers (like init methods in ObjC)
    // Can fail by returning nil (use init?)

    init?(view: MTKView) {
        // GUARD STATEMENT - early exit if condition fails
        // "Guard lets you exit early if things aren't right"
        guard let device = MTLCreateSystemDefaultDevice() else {
            print("❌ ERROR: Metal is not supported on this Mac")
            return nil  // init failed
        }

        guard let commandQueue = device.makeCommandQueue() else {
            print("❌ ERROR: Failed to create command queue")
            return nil
        }

        // Set properties before calling super.init
        self.device = device
        self.commandQueue = commandQueue
        self.view = view

        // Initialize timing
        let currentTime = CACurrentMediaTime()
        self.startTime = currentTime
        self.lastFrameTime = currentTime
        self.fpsUpdateTime = currentTime

        // MUST call super.init before using 'self' fully
        super.init()

        print("🔧 Initializing Metal...")
        print("✓ GPU: \(device.name)")
        print("✓ Command queue created")

        // Load shaders (can fail)
        guard loadShaders() else {
            return nil
        }

        // Create uniform buffer
        guard let uniformBuffer = device.makeBuffer(
            length: MemoryLayout<Uniforms>.stride,  // Size of struct
            options: .storageModeShared
        ) else {
            print("❌ ERROR: Failed to create uniform buffer")
            return nil
        }
        self.uniformBuffer = uniformBuffer
        print("✓ Uniform buffer created")

        // Configure the view
        view.device = device
        view.delegate = self
        view.clearColor = MTLClearColor(red: 0.1, green: 0.05, blue: 0.15, alpha: 1.0)

        print("✓ Renderer initialized successfully!")
    }

    // -------------------------------------------------------------------------
    // Shader Loading
    // -------------------------------------------------------------------------
    // Swift functions: func name(paramLabel paramName: Type) -> ReturnType

    func loadShaders() -> Bool {
        print("📦 Loading shaders...")

        // SWIFT ERROR HANDLING with do-try-catch
        do {
            // Load shader source from file
            // String(contentsOfFile:) can throw an error
            let shaderPath = "Compute.metal"
            let shaderSource = try String(contentsOfFile: shaderPath, encoding: .utf8)
            print("✓ Shader source loaded (\(shaderSource.count) bytes)")

            // Compile shaders
            let options = MTLCompileOptions()
            if #available(macOS 15.0, *) {
                options.mathMode = .fast
            } else {
                options.fastMathEnabled = true
            }

            // try = can throw error
            let library = try device.makeLibrary(source: shaderSource, options: options)
            print("✓ Shaders compiled successfully")

            // Get shader functions
            // guard let = unwrap optional or fail
            guard let vertexFunc = library.makeFunction(name: "vertex_main"),
                  let fragmentFunc = library.makeFunction(name: "fragment_main") else {
                print("❌ ERROR: Could not find shader functions")
                return false
            }

            // Create render pipeline
            let pipelineDesc = MTLRenderPipelineDescriptor()
            pipelineDesc.vertexFunction = vertexFunc
            pipelineDesc.fragmentFunction = fragmentFunc
            pipelineDesc.colorAttachments[0].pixelFormat = view?.colorPixelFormat ?? .bgra8Unorm

            // Alpha blending
            pipelineDesc.colorAttachments[0].isBlendingEnabled = true
            pipelineDesc.colorAttachments[0].sourceRGBBlendFactor = .sourceAlpha
            pipelineDesc.colorAttachments[0].destinationRGBBlendFactor = .oneMinusSourceAlpha

            pipeline = try device.makeRenderPipelineState(descriptor: pipelineDesc)
            print("✓ Render pipeline created")

            return true

        } catch {
            // 'error' is automatically available in catch block
            print("❌ Shader loading failed: \(error)")
            return false
        }
    }

    // -------------------------------------------------------------------------
    // MTKViewDelegate - Render Loop
    // -------------------------------------------------------------------------
    // Called every frame by MTKView

    func draw(in view: MTKView) {
        // GUARD STATEMENTS for early exit if resources aren't ready
        guard let pipeline = pipeline,
              let uniformBuffer = uniformBuffer else {
            return
        }

        // -------------------------------------------------------------------------
        // UPDATE TIMING AND UNIFORMS
        // -------------------------------------------------------------------------
        let currentTime = CACurrentMediaTime()
        let elapsedTime = currentTime - startTime
        let deltaTime = currentTime - lastFrameTime
        lastFrameTime = currentTime
        frameCount += 1

        // Update uniform buffer
        // POINTER ACCESS in Swift
        let uniformsPointer = uniformBuffer.contents().bindMemory(
            to: Uniforms.self,
            capacity: 1
        )
        uniformsPointer.pointee.time = Float(elapsedTime)
        uniformsPointer.pointee.deltaTime = Float(deltaTime)
        uniformsPointer.pointee.frameCount = Int32(frameCount)

        // -------------------------------------------------------------------------
        // UPDATE FPS IN WINDOW TITLE
        // -------------------------------------------------------------------------
        fpsFrameCount += 1
        if currentTime - fpsUpdateTime >= 1.0 {
            let fps = Double(fpsFrameCount) / (currentTime - fpsUpdateTime)

            // SWIFT CLOSURES (like lambdas)
            // Syntax: { parameters in code }
            DispatchQueue.main.async { [weak self] in
                // [weak self] = capture self weakly to prevent retain cycle
                // Swift's version of Objective-C blocks
                self?.view?.window?.title = String(format: "Swift Metal - %.1f FPS", fps)
            }

            fpsFrameCount = 0
            fpsUpdateTime = currentTime
        }

        // -------------------------------------------------------------------------
        // CREATE COMMAND BUFFER
        // -------------------------------------------------------------------------
        guard let commandBuffer = commandQueue.makeCommandBuffer() else {
            return
        }

        // -------------------------------------------------------------------------
        // GET RENDER PASS
        // -------------------------------------------------------------------------
        guard let renderPass = view.currentRenderPassDescriptor else {
            return
        }

        // -------------------------------------------------------------------------
        // ENCODE RENDER COMMANDS
        // -------------------------------------------------------------------------
        guard let encoder = commandBuffer.makeRenderCommandEncoder(descriptor: renderPass) else {
            return
        }

        encoder.setRenderPipelineState(pipeline)

        // Bind uniform buffer to fragment shader
        encoder.setFragmentBuffer(uniformBuffer, offset: 0, index: 0)

        // Draw triangle
        encoder.drawPrimitives(type: .triangle, vertexStart: 0, vertexCount: 3)

        encoder.endEncoding()

        // -------------------------------------------------------------------------
        // PRESENT AND COMMIT
        // -------------------------------------------------------------------------
        if let drawable = view.currentDrawable {
            commandBuffer.present(drawable)
        }

        commandBuffer.commit()
    }

    // Called when window resizes (required by MTKViewDelegate)
    func mtkView(_ view: MTKView, drawableSizeWillChange size: CGSize) {
        // Could update projection matrices here
    }
}

// =============================================================================
// MAIN FUNCTION - Application Entry Point
// =============================================================================
// Swift doesn't need @autoreleasepool - memory is automatically managed

print("\n")
print("╔════════════════════════════════════════════════════════════╗")
print("║         SWIFT + METAL FOR BEGINNERS - GPU 101              ║")
print("║                                                            ║")
print("║  Animated gradient with time-based color effects           ║")
print("║  FPS counter displayed in window title                     ║")
print("║                                                            ║")
print("║  Controls:                                                 ║")
print("║    Q or ESC - Quit the application                         ║")
print("║    Close window - Also quits                               ║")
print("╚════════════════════════════════════════════════════════════╝")
print("\n")

// Create the application
// NSApplication.shared = singleton instance (like [NSApplication sharedApplication])
let app = NSApplication.shared
app.setActivationPolicy(.regular)

// Create and set delegate
let delegate = AppDelegate()
app.delegate = delegate

// Create window
let frame = NSRect(x: 0, y: 0, width: Config.windowWidth, height: Config.windowHeight)
let window = NSWindow(
    contentRect: frame,
    styleMask: [.titled, .closable, .miniaturizable],  // Array of options
    backing: .buffered,
    defer: false
)
window.title = "Swift Metal for Beginners"
window.center()

// Create Metal view
let metalView = InputView(frame: frame)
metalView.preferredFramesPerSecond = Config.targetFPS

// Create renderer
guard let renderer = Renderer(view: metalView) else {
    print("❌ Failed to initialize renderer")
    exit(1)
}

// Connect renderer to view
metalView.renderer = renderer

// Show window
window.contentView = metalView
window.makeKeyAndOrderFront(nil)
window.makeFirstResponder(metalView)

app.activate(ignoringOtherApps: true)

// Run the app (doesn't return until app quits)
app.run()

print("\n✓ Application exited cleanly\n")

// =============================================================================
// SUMMARY FOR C++/OBJC PROGRAMMERS
// =============================================================================
/*
╔═══════════════════════════════════════════════════════════════════════════╗
║                    Swift Syntax Quick Reference                           ║
╠═══════════════════════════════════════════════════════════════════════════╣
║ VARIABLES                                                                 ║
║   let x = 5         - Immutable (like const)                              ║
║   var y = 10        - Mutable                                             ║
║                                                                           ║
║ OPTIONALS                                                                 ║
║   var name: String?      - Can be nil                                     ║
║   if let n = name { }    - Safe unwrapping                                ║
║   guard let n = name else { return }  - Early exit                        ║
║                                                                           ║
║ FUNCTIONS                                                                 ║
║   func name(label param: Type) -> ReturnType { }                          ║
║   func name(_ param: Type)  - No external label                           ║
║                                                                           ║
║ CLASSES                                                                   ║
║   class MyClass: Parent, Protocol1, Protocol2 { }                         ║
║   init() { }        - Constructor                                         ║
║   init?() { }       - Failable initializer (can return nil)               ║
║                                                                           ║
║ CLOSURES                                                                  ║
║   { (param: Type) -> ReturnType in code }                                 ║
║   { code }          - If types can be inferred                            ║
║   [weak self]       - Capture self weakly (prevent cycles)                ║
║                                                                           ║
║ ERROR HANDLING                                                            ║
║   do { try risky() } catch { handle(error) }                              ║
║                                                                           ║
║ PROPERTY WRAPPERS                                                         ║
║   weak var delegate: Protocol?  - Weak reference                          ║
║   @IBOutlet         - Interface Builder connection                        ║
╚═══════════════════════════════════════════════════════════════════════════╝

Swift vs Objective-C for Metal:
- Cleaner syntax (no square brackets!)
- Better type safety
- Optionals prevent nil crashes
- Modern features (generics, protocols, etc.)
- Exact same Metal API underneath
- Better error handling with do-try-catch

Next Steps:
- Read Compute.metal (same as Objective-C version!)
- Modify the shader to experiment
- Compare this to the Objective-C version
- Add more Swift features (protocols, extensions, etc.)
*/
