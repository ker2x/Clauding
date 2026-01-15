// =============================================================================
// Swift Particle Template - main.swift
// =============================================================================
// A comprehensive template for building GPU-accelerated particle simulations
// using Swift and Metal. This is the Swift equivalent of metal_particle_template.
//
// This file demonstrates:
// - Swift + Metal integration
// - Particle physics simulation on GPU
// - Keyboard input handling
// - FPS monitoring
// - Buffer management between CPU and GPU
//
// =============================================================================

// -----------------------------------------------------------------------------
// IMPORTS
// -----------------------------------------------------------------------------
import AppKit       // macOS application and windowing
import Metal        // GPU compute and graphics
import MetalKit     // Metal helper utilities
import QuartzCore   // High-precision timing

// =============================================================================
// CONFIGURATION
// =============================================================================
// All simulation parameters are centralized in this Config struct.
// Modify these values to customize the particle behavior.
//
// SWIFT NOTE: This uses a struct with static properties, similar to a
// namespace in C++ or a static class in other languages.

struct Config {
    // Window dimensions
    // CGFloat is the standard type for UI measurements in AppKit/UIKit
    static let windowWidth: CGFloat = 1024
    static let windowHeight: CGFloat = 768

    // Particle system settings
    static let numParticles = 20000      // Total number of particles (try 50000 for more!)
    static let numTypes = 4              // Number of particle types/colors (creates groups)

    // Physics settings
    // These control the emergent behavior of the particle system
    static let interactionRadius: Float = 80.0   // Max distance for forces (larger = slower)
    static let forceStrength: Float = 0.01       // Force multiplier (higher = more chaotic)
    static let friction: Float = 0.80            // Velocity damping 0-1 (lower = more friction)

    // Rendering settings
    static let pointSize: Float = 3.0    // Particle size in pixels (rendered as squares)
    static let targetFPS = 60            // Target frame rate (unused, MTKView handles this)

    // Background color (RGBA, 0.0-1.0)
    // Dark blue background provides good contrast for colored particles
    static let bgColor = (r: Float(0.02), g: Float(0.02), b: Float(0.05), a: Float(1.0))
}

// =============================================================================
// DATA STRUCTURES
// =============================================================================
// These structures are shared between CPU (Swift) and GPU (Metal shader).
// They MUST match exactly with the definitions in Compute.metal.
//
// MEMORY LAYOUT: Swift structs use the same memory layout as C structs when
// passed to Metal, making them perfect for CPU-GPU data transfer.
//
// IMPORTANT: The order, type, and padding of fields must match exactly between
// Swift and Metal, or you'll get garbage data on the GPU!

struct SimParams {
    var width: Float                // Canvas width in pixels
    var height: Float               // Canvas height in pixels
    var interaction_radius: Float   // Max distance for particle interactions
    var force_strength: Float       // Force multiplier
    var friction: Float             // Velocity damping (0-1)
    var num_particles: Int32        // Total particle count (Int32 matches Metal's int)
    var num_types: Int32            // Number of particle types
    var point_size: Float           // Render size in pixels
}

// =============================================================================
// InputView - Custom MTKView with Keyboard Input
// =============================================================================
// MTKView is MetalKit's view class that provides:
// - A CAMetalLayer for efficient Metal rendering
// - A render loop that calls draw(in:) on its delegate
// - Automatic drawable management
//
// We subclass it to add keyboard input handling.
//
// SWIFT INHERITANCE: Uses `: ParentClass` syntax (like C++/Java)

class InputView: MTKView {
    // SWIFT PROPERTY: Simpler than Objective-C's @property
    // 'weak' prevents retain cycles (similar to weak_ptr in C++)
    weak var renderer: Renderer?

    // SWIFT COMPUTED PROPERTY: Returns a value when accessed
    // 'override' required when overriding superclass members
    override var acceptsFirstResponder: Bool {
        return true  // Must return true to receive keyboard events
    }

    // SWIFT METHOD OVERRIDE: Handle key presses
    // 'with event:' is the external parameter label
    override func keyDown(with event: NSEvent) {
        // SWIFT GUARD: Early exit pattern for safer code
        // Unwraps the optional and exits if nil
        guard let characters = event.charactersIgnoringModifiers else {
            return
        }

        // Get first character from the string
        guard let key = characters.first else {
            return
        }

        // SWIFT SWITCH: More powerful than C++ switch
        // No fallthrough by default, can match multiple values per case
        switch key {
        case "q", "Q":
            // Quit the application
            // NSApp is a global constant for NSApplication.shared
            NSApp.terminate(nil)

        case "r", "R":
            // Randomize particles and interaction matrix
            // '?' is optional chaining - only calls if renderer is not nil
            renderer?.randomize()

        case " ":  // Space bar
            // Toggle pause
            renderer?.togglePause()

        default:
            // Pass unhandled keys to superclass
            super.keyDown(with: event)
        }
    }
}

// =============================================================================
// AppDelegate - Application Lifecycle
// =============================================================================
// The AppDelegate handles application-level events like launch, termination,
// and window management.
//
// SWIFT PROTOCOL CONFORMANCE: Uses `: Protocol1, Protocol2` syntax
// NSObject is required for Objective-C interop (NSApplicationDelegate)

class AppDelegate: NSObject, NSApplicationDelegate {
    // DELEGATE METHOD: Called when user tries to close the last window
    // The '_' means no external label (unlike 'with event:' above)
    func applicationShouldTerminateAfterLastWindowClosed(_ sender: NSApplication) -> Bool {
        return true  // Quit the app when the window closes
    }
}

// =============================================================================
// Renderer - Metal Rendering and Compute Pipeline
// =============================================================================
// This class manages all Metal resources and implements the render loop.
// It conforms to MTKViewDelegate to receive frame callbacks.
//
// ARCHITECTURE:
// 1. Initialization: Load shaders, create buffers
// 2. Render loop: Update physics (compute), draw particles (render)
// 3. User actions: Randomize, pause/resume
//
// METAL PIPELINE:
// CPU: Create command buffer → Encode commands → Commit
// GPU: Execute compute kernel → Execute render pipeline → Present

class Renderer: NSObject, MTKViewDelegate {

    // -------------------------------------------------------------------------
    // Metal Core Objects
    // -------------------------------------------------------------------------
    // These are the fundamental objects needed for any Metal application.

    let device: MTLDevice              // Represents the GPU
    let commandQueue: MTLCommandQueue  // Queue for submitting GPU work

    // -------------------------------------------------------------------------
    // Pipeline States
    // -------------------------------------------------------------------------
    // Pipeline states are compiled shader programs.
    // They're expensive to create, so we create once and reuse.

    var computePipeline: MTLComputePipelineState?  // For particle physics kernel
    var renderPipeline: MTLRenderPipelineState?    // For drawing particles

    // -------------------------------------------------------------------------
    // GPU Buffers
    // -------------------------------------------------------------------------
    // Buffers hold data accessible by both CPU and GPU.
    // Using storageModeShared for easy CPU↔GPU data transfer.

    var positionBuffer: MTLBuffer?   // Particle positions (SIMD2<Float> array)
    var velocityBuffer: MTLBuffer?   // Particle velocities (SIMD2<Float> array)
    var typeBuffer: MTLBuffer?       // Particle types (Int32 array)
    var colorBuffer: MTLBuffer?      // Colors per type (SIMD4<Float> array)
    var matrixBuffer: MTLBuffer?     // Interaction matrix (Float array)
    var paramsBuffer: MTLBuffer?     // Simulation parameters (SimParams struct)

    // -------------------------------------------------------------------------
    // State
    // -------------------------------------------------------------------------

    weak var view: MTKView?  // Reference to the view (weak to avoid retain cycle)
    var paused = false       // Simulation pause state

    // -------------------------------------------------------------------------
    // FPS Tracking
    // -------------------------------------------------------------------------

    var lastFrameTime: Double = 0    // Time of last frame (for delta time)
    var frameCount = 0               // Frames since last FPS update
    var fpsUpdateTime: Double = 0    // Time of last FPS display update

    // MARK: - Initialization

    // SWIFT FAILABLE INITIALIZER: Returns nil if initialization fails
    // The '?' after init means this can return nil
    // This is safer than throwing errors for resource initialization
    init?(view: MTKView) {

        // =====================================================================
        // Step 1: Get the Metal device (GPU)
        // =====================================================================
        // MTLCreateSystemDefaultDevice() returns the default GPU.
        // On Macs with multiple GPUs, this typically returns the discrete GPU.
        //
        // SWIFT GUARD: If device is nil, print error and return nil
        guard let device = MTLCreateSystemDefaultDevice() else {
            print("❌ Metal is not supported on this device")
            return nil
        }

        // =====================================================================
        // Step 2: Create the command queue
        // =====================================================================
        // Commands are submitted to the GPU through a command queue.
        // You typically create one queue and reuse it for the app's lifetime.
        guard let commandQueue = device.makeCommandQueue() else {
            print("❌ Failed to create command queue")
            return nil
        }

        // Store Metal objects
        self.device = device
        self.commandQueue = commandQueue
        self.view = view

        // Initialize timing
        let currentTime = CACurrentMediaTime()  // High-precision system time
        self.lastFrameTime = currentTime
        self.fpsUpdateTime = currentTime

        // =====================================================================
        // IMPORTANT: Must call super.init() before using 'self' fully
        // =====================================================================
        // In Swift, you must initialize all properties before calling super.init()
        super.init()

        print("Using GPU: \(device.name)")

        // =====================================================================
        // Step 3: Load and compile shaders
        // =====================================================================
        guard loadShaders() else {
            return nil
        }

        // =====================================================================
        // Step 4: Create GPU buffers
        // =====================================================================
        initBuffers()

        // =====================================================================
        // Step 5: Configure the view
        // =====================================================================
        view.device = device           // Associate view with our Metal device
        view.delegate = self           // Set renderer as delegate to receive draw calls
        view.clearColor = MTLClearColor(  // Background color
            red: Double(Config.bgColor.r),
            green: Double(Config.bgColor.g),
            blue: Double(Config.bgColor.b),
            alpha: Double(Config.bgColor.a)
        )

        // Explicit return (optional in Swift, but clarifies intent)
        return
    }

    // MARK: - Shader Loading

    // -------------------------------------------------------------------------
    // loadShaders: Compile Metal shaders from source
    // -------------------------------------------------------------------------
    // Metal shaders can be:
    // 1. Pre-compiled into a .metallib file (faster startup)
    // 2. Compiled at runtime from source (more flexible, used here)
    //
    // We load from source so you can modify shaders without rebuilding the app.
    //
    // SWIFT ERROR HANDLING: Uses do-try-catch for operations that can throw
    func loadShaders() -> Bool {
        do {
            // =================================================================
            // Load shader source from file
            // =================================================================
            // SWIFT TRY: Marks operations that can throw errors
            let shaderPath = "Compute.metal"
            let shaderSource = try String(contentsOfFile: shaderPath, encoding: .utf8)

            // =================================================================
            // Compile shader source into a library
            // =================================================================
            // A library contains one or more shader functions (kernels, vertex, fragment)
            let options = MTLCompileOptions()

            // Enable fast math optimizations (trade precision for speed)
            // SWIFT AVAILABILITY CHECK: Different API for macOS 15+
            if #available(macOS 15.0, *) {
                options.mathMode = .fast
            } else {
                options.fastMathEnabled = true
            }

            let library = try device.makeLibrary(source: shaderSource, options: options)
            print("✓ Shaders compiled successfully")

            // =================================================================
            // Create compute pipeline (for particle physics kernel)
            // =================================================================
            guard let computeFunc = library.makeFunction(name: "update_particles") else {
                print("❌ Could not find 'update_particles' function in shader")
                return false
            }

            // Compile the compute function into a pipeline state
            computePipeline = try device.makeComputePipelineState(function: computeFunc)

            // =================================================================
            // Create render pipeline (for drawing particles)
            // =================================================================
            // A render pipeline needs both a vertex and fragment function
            guard let vertexFunc = library.makeFunction(name: "vertex_main"),
                  let fragmentFunc = library.makeFunction(name: "fragment_main") else {
                print("❌ Could not find vertex/fragment functions in shader")
                return false
            }

            let pipelineDescriptor = MTLRenderPipelineDescriptor()
            pipelineDescriptor.vertexFunction = vertexFunc
            pipelineDescriptor.fragmentFunction = fragmentFunc
            // Match the view's pixel format (usually .bgra8Unorm on macOS)
            pipelineDescriptor.colorAttachments[0].pixelFormat = view?.colorPixelFormat ?? .bgra8Unorm

            // Compile the render pipeline
            renderPipeline = try device.makeRenderPipelineState(descriptor: pipelineDescriptor)

            return true

        } catch {
            // SWIFT CATCH: The 'error' variable is automatically available
            print("❌ Shader loading/compilation failed: \(error)")
            print("Make sure Compute.metal is in the same directory as the executable.")
            return false
        }
    }

    // MARK: - Buffer Initialization

    // -------------------------------------------------------------------------
    // initBuffers: Create GPU buffers and initialize particle data
    // -------------------------------------------------------------------------
    // This function creates all the buffers needed for the simulation:
    // - Position, velocity, type buffers for particles
    // - Color palette buffer
    // - Interaction matrix buffer
    // - Parameters buffer
    //
    // BUFFER STORAGE MODE: Using .storageModeShared allows both CPU and GPU
    // to access the same memory, making it easy to read/write from Swift.
    func initBuffers() {
        let numParticles = Config.numParticles
        let numTypes = Config.numTypes

        // =====================================================================
        // Allocate CPU arrays
        // =====================================================================
        // SWIFT ARRAY INITIALIZATION: Create arrays filled with default values
        // SIMD2<Float> is Swift's type for 2D float vectors (matches Metal's float2)
        var positions = [SIMD2<Float>](repeating: SIMD2(0, 0), count: numParticles)
        var velocities = [SIMD2<Float>](repeating: SIMD2(0, 0), count: numParticles)
        var types = [Int32](repeating: 0, count: numParticles)

        // =====================================================================
        // Initialize particles with random positions and types
        // =====================================================================
        // SWIFT RANGE: '0..<n' is a half-open range [0, n)
        for i in 0..<numParticles {
            positions[i] = SIMD2(
                Float.random(in: 0..<Float(Config.windowWidth)),
                Float.random(in: 0..<Float(Config.windowHeight))
            )
            velocities[i] = SIMD2(0, 0)  // Start at rest
            types[i] = Int32.random(in: 0..<Int32(numTypes))
        }

        // =====================================================================
        // Create GPU buffers from CPU arrays
        // =====================================================================
        // makeBuffer copies the data from CPU to GPU memory
        // MemoryLayout<T>.stride gives the size of T in bytes
        positionBuffer = device.makeBuffer(
            bytes: positions,
            length: MemoryLayout<SIMD2<Float>>.stride * numParticles,
            options: .storageModeShared  // CPU and GPU can both access
        )

        velocityBuffer = device.makeBuffer(
            bytes: velocities,
            length: MemoryLayout<SIMD2<Float>>.stride * numParticles,
            options: .storageModeShared
        )

        typeBuffer = device.makeBuffer(
            bytes: types,
            length: MemoryLayout<Int32>.stride * numParticles,
            options: .storageModeShared
        )

        // =====================================================================
        // Create color palette buffer
        // =====================================================================
        // SIMD4<Float> is Swift's type for 4D float vectors (matches Metal's float4)
        // Used for RGBA colors
        let colors: [SIMD4<Float>] = [
            SIMD4(1.0, 0.0, 0.0, 1.0),  // Red
            SIMD4(0.0, 1.0, 0.0, 1.0),  // Green
            SIMD4(0.0, 0.5, 1.0, 1.0),  // Blue
            SIMD4(1.0, 1.0, 0.0, 1.0),  // Yellow
        ]

        colorBuffer = device.makeBuffer(
            bytes: colors,
            length: MemoryLayout<SIMD4<Float>>.stride * numTypes,
            options: .storageModeShared
        )

        // =====================================================================
        // Create interaction matrix (random attraction/repulsion values)
        // =====================================================================
        randomizeMatrix()

        // =====================================================================
        // Create simulation parameters buffer
        // =====================================================================
        // This struct is passed to both compute and render shaders
        // 'var' is required for the '&' inout parameter below
        var params = SimParams(
            width: Float(Config.windowWidth),
            height: Float(Config.windowHeight),
            interaction_radius: Config.interactionRadius,
            force_strength: Config.forceStrength,
            friction: Config.friction,
            num_particles: Int32(numParticles),
            num_types: Int32(numTypes),
            point_size: Config.pointSize
        )

        // '&params' passes the address of the struct (like C++)
        paramsBuffer = device.makeBuffer(
            bytes: &params,
            length: MemoryLayout<SimParams>.stride,
            options: .storageModeShared
        )
    }

    // MARK: - User Actions

    // -------------------------------------------------------------------------
    // randomizeMatrix: Create a new random interaction matrix
    // -------------------------------------------------------------------------
    // The interaction matrix controls how particle types attract/repel each other.
    // matrix[typeA * numTypes + typeB] = strength (-1 to 1)
    //   Positive = attraction
    //   Negative = repulsion
    func randomizeMatrix() {
        let numTypes = Config.numTypes
        var matrix = [Float](repeating: 0, count: numTypes * numTypes)

        // Fill with random values between -1 and 1
        // SWIFT RANGE: '...' is a closed range [a, b]
        for i in 0..<(numTypes * numTypes) {
            matrix[i] = Float.random(in: -1.0...1.0)
        }

        matrixBuffer = device.makeBuffer(
            bytes: matrix,
            length: MemoryLayout<Float>.stride * numTypes * numTypes,
            options: .storageModeShared
        )
    }

    // -------------------------------------------------------------------------
    // randomize: Reset simulation with random positions and interactions
    // -------------------------------------------------------------------------
    // This function directly modifies GPU buffer contents from the CPU.
    // Works because we used .storageModeShared.
    func randomize() {
        // Randomize particle positions
        guard let positionBuffer = positionBuffer else { return }

        // Get a pointer to the buffer's memory
        // bindMemory converts the raw pointer to a typed pointer
        let positions = positionBuffer.contents().bindMemory(
            to: SIMD2<Float>.self,
            capacity: Config.numParticles
        )

        for i in 0..<Config.numParticles {
            positions[i] = SIMD2(
                Float.random(in: 0..<Float(Config.windowWidth)),
                Float.random(in: 0..<Float(Config.windowHeight))
            )
        }

        // Reset velocities to zero
        guard let velocityBuffer = velocityBuffer else { return }

        let velocities = velocityBuffer.contents().bindMemory(
            to: SIMD2<Float>.self,
            capacity: Config.numParticles
        )

        for i in 0..<Config.numParticles {
            velocities[i] = SIMD2(0, 0)
        }

        // Randomize interaction matrix
        randomizeMatrix()

        print("🔄 Randomized")
    }

    // -------------------------------------------------------------------------
    // togglePause: Pause/resume the physics simulation
    // -------------------------------------------------------------------------
    func togglePause() {
        // SWIFT TOGGLE: Flips a boolean value
        paused.toggle()
        // SWIFT TERNARY: condition ? true_value : false_value
        print(paused ? "⏸️  Paused" : "▶️  Resumed")
    }

    // MARK: - MTKViewDelegate

    // -------------------------------------------------------------------------
    // mtkView(_:drawableSizeWillChange:): Handle window resize
    // -------------------------------------------------------------------------
    // Called when the view's size changes (window resize, fullscreen, etc.)
    func mtkView(_ view: MTKView, drawableSizeWillChange size: CGSize) {
        // Handle window resize if needed
        // For this simple simulation, we don't need to do anything special
    }

    // -------------------------------------------------------------------------
    // draw(in:): Main render loop - called 60 times per second
    // -------------------------------------------------------------------------
    // This is the heart of the application. Called automatically by MTKView.
    //
    // METAL COMMAND FLOW:
    // 1. Create command buffer (holds commands)
    // 2. Encode compute commands (physics simulation)
    // 3. Encode render commands (draw particles)
    // 4. Commit command buffer (send to GPU)
    // 5. GPU executes commands asynchronously
    //
    // SWIFT MULTIPLE UNWRAPPING: guard let can unwrap many optionals at once
    func draw(in view: MTKView) {
        // =====================================================================
        // Ensure we have all required resources
        // =====================================================================
        // If any resource is nil, exit early (likely initialization failed)
        guard let computePipeline = computePipeline,
              let renderPipeline = renderPipeline,
              let positionBuffer = positionBuffer,
              let velocityBuffer = velocityBuffer,
              let typeBuffer = typeBuffer,
              let colorBuffer = colorBuffer,
              let matrixBuffer = matrixBuffer,
              let paramsBuffer = paramsBuffer else {
            return
        }

        // =====================================================================
        // Update FPS counter
        // =====================================================================
        let currentTime = CACurrentMediaTime()
        frameCount += 1

        // Update window title with FPS once per second
        if currentTime - fpsUpdateTime >= 1.0 {
            let fps = Double(frameCount) / (currentTime - fpsUpdateTime)
            // SWIFT STRING INTERPOLATION: String(format:) works like printf
            view.window?.title = String(format: "Particle Simulation - %.1f FPS", fps)
            frameCount = 0
            fpsUpdateTime = currentTime
        }

        lastFrameTime = currentTime

        // =====================================================================
        // Create command buffer
        // =====================================================================
        // A command buffer holds a list of commands to execute on the GPU
        guard let commandBuffer = commandQueue.makeCommandBuffer() else {
            return
        }

        // =====================================================================
        // COMPUTE PASS: Update particle physics on GPU
        // =====================================================================
        // Skip compute if paused (still render the existing state)
        if !paused {
            guard let computeEncoder = commandBuffer.makeComputeCommandEncoder() else {
                return
            }

            // Set the compute pipeline (which kernel to run)
            computeEncoder.setComputePipelineState(computePipeline)

            // Bind buffers to kernel arguments (matches [[buffer(N)]] in shader)
            computeEncoder.setBuffer(positionBuffer, offset: 0, index: 0)
            computeEncoder.setBuffer(velocityBuffer, offset: 0, index: 1)
            computeEncoder.setBuffer(typeBuffer, offset: 0, index: 2)
            computeEncoder.setBuffer(matrixBuffer, offset: 0, index: 3)
            computeEncoder.setBuffer(paramsBuffer, offset: 0, index: 4)

            // Dispatch compute threads
            // THREAD ORGANIZATION:
            // - 256 threads per threadgroup (optimal for Apple Silicon)
            // - (numParticles + 255) / 256 threadgroups (rounds up)
            // - Each thread processes one particle
            let threadGroupSize = MTLSize(width: 256, height: 1, depth: 1)
            let threadGroups = MTLSize(
                width: (Config.numParticles + 255) / 256,
                height: 1,
                depth: 1
            )

            computeEncoder.dispatchThreadgroups(threadGroups, threadsPerThreadgroup: threadGroupSize)
            computeEncoder.endEncoding()
        }

        // =====================================================================
        // RENDER PASS: Draw particles to screen
        // =====================================================================
        // Get the render pass descriptor (includes render target info)
        // Get the drawable (the texture we'll render into)
        guard let renderPassDescriptor = view.currentRenderPassDescriptor,
              let drawable = view.currentDrawable else {
            return
        }

        guard let renderEncoder = commandBuffer.makeRenderCommandEncoder(descriptor: renderPassDescriptor) else {
            return
        }

        // Set the render pipeline (vertex + fragment shaders)
        renderEncoder.setRenderPipelineState(renderPipeline)

        // Bind vertex buffers (matches [[buffer(N)]] in vertex shader)
        renderEncoder.setVertexBuffer(positionBuffer, offset: 0, index: 0)
        renderEncoder.setVertexBuffer(typeBuffer, offset: 0, index: 1)
        renderEncoder.setVertexBuffer(colorBuffer, offset: 0, index: 2)
        renderEncoder.setVertexBuffer(paramsBuffer, offset: 0, index: 3)

        // Draw all particles as points
        // Each point becomes a square of size Config.pointSize
        renderEncoder.drawPrimitives(type: .point, vertexStart: 0, vertexCount: Config.numParticles)
        renderEncoder.endEncoding()

        // =====================================================================
        // Present and commit
        // =====================================================================
        // Schedule the drawable to be presented to the screen
        commandBuffer.present(drawable)

        // Commit the command buffer to the GPU
        // The GPU will execute all encoded commands asynchronously
        commandBuffer.commit()
    }
}

// =============================================================================
// MAIN - Application Entry Point
// =============================================================================
// This code runs when the program starts. It sets up the application, window,
// and Metal rendering, then enters the run loop.
//
// SWIFT TOP-LEVEL CODE: Unlike C++ which requires main(), Swift can execute
// code at the top level of a file. This code runs immediately.
//
// macOS APP REQUIREMENTS:
// For keyboard events to work when running from command line, we need:
// 1. Set activation policy to .regular (makes app appear in Dock)
// 2. Create an application menu
// 3. Call app.activate() to bring app to foreground

// =============================================================================
// Create and configure application
// =============================================================================
let app = NSApplication.shared  // Get the shared application instance

// ACTIVATION POLICY: Required for command-line apps to receive keyboard events
// .regular = Normal application (appears in Dock, can receive focus)
// Without this, the app runs in background and keyboard events are ignored
app.setActivationPolicy(.regular)

let delegate = AppDelegate()
app.delegate = delegate

// =============================================================================
// Create application menu
// =============================================================================
// MENU REQUIREMENT: macOS apps need a menu bar to properly receive keyboard events
// Even if we don't use the menu, it must exist for the app to be "fully activated"

let mainMenu = NSMenu()

// App menu (first menu, usually shows app name)
let appMenuItem = NSMenuItem()
mainMenu.addItem(appMenuItem)

let appMenu = NSMenu()
appMenuItem.submenu = appMenu

// Add Quit menu item (Cmd+Q)
// SWIFT SELECTOR: #selector() creates an Objective-C selector for a method
let quitMenuItem = NSMenuItem(
    title: "Quit",
    action: #selector(NSApplication.terminate(_:)),
    keyEquivalent: "q"  // Cmd+Q shortcut
)
appMenu.addItem(quitMenuItem)

app.mainMenu = mainMenu

// =============================================================================
// Create window
// =============================================================================
// SWIFT ARRAY LITERAL: [.titled, .closable] creates an OptionSet
let window = NSWindow(
    contentRect: NSRect(x: 0, y: 0, width: Config.windowWidth, height: Config.windowHeight),
    styleMask: [.titled, .closable, .miniaturizable],  // Window style
    backing: .buffered,   // Double-buffered for smooth rendering
    defer: false          // Create window immediately
)
window.title = "Particle Simulation"
window.center()  // Center window on screen

// =============================================================================
// Create Metal view
// =============================================================================
// SWIFT FORCE UNWRAP: '!' assumes the optional has a value (crashes if nil)
// Safe here because we just created the window
let metalView = InputView(frame: window.contentView!.bounds)

// Allow view to resize with window
// SWIFT ARRAY: [.width, .height] is shorthand for [.flexibleWidth, .flexibleHeight]
metalView.autoresizingMask = [.width, .height]

// =============================================================================
// Create renderer
// =============================================================================
// SWIFT GUARD with EXIT: Exit program if renderer creation fails
guard let renderer = Renderer(view: metalView) else {
    print("❌ Failed to create renderer")
    exit(1)  // Non-zero exit code indicates error
}

metalView.renderer = renderer

// =============================================================================
// Set up window and activate app
// =============================================================================
window.contentView = metalView         // Set Metal view as window content
window.makeFirstResponder(metalView)   // Make view the first responder for keyboard events
window.makeKeyAndOrderFront(nil)       // Show window and bring to front

// ACTIVATION: Bring app to foreground and enable keyboard input
// ignoringOtherApps:true = Take focus even if another app is active
// This is CRITICAL for command-line apps to receive keyboard events
app.activate(ignoringOtherApps: true)

// =============================================================================
// Run application
// =============================================================================
// Enter the run loop - this blocks until the app quits
// The run loop handles events (keyboard, mouse, timers) and calls
// our draw(in:) method ~60 times per second
app.run()
