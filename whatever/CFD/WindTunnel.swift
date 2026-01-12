import SwiftUI
import MetalKit
import Foundation

// MARK: - Configuration Structure
struct SimConfig: Codable {
    struct Simulation: Codable {
        let grid_width: Int
        let grid_height: Int
        let steps_per_frame: Int
        let relaxation_time: Float
        let initial_wind_speed: Float
        let max_wind_speed: Float
    }

    struct Visualization: Codable {
        let velocity_colormap: String
        let vorticity_contour_levels: Int
        let vorticity_threshold: Float
    }

    struct Interaction: Codable {
        let initial_brush_size: Int
        let min_brush_size: Int
        let max_brush_size: Int
    }

    let simulation: Simulation
    let visualization: Visualization
    let interaction: Interaction
}

// MARK: - Configuration Loader
func loadConfig() -> SimConfig {
    let defaultConfig = SimConfig(
        simulation: .init(grid_width: 512, grid_height: 256, steps_per_frame: 5,
                        relaxation_time: 0.7, initial_wind_speed: 0.05, max_wind_speed: 0.15),
        visualization: .init(velocity_colormap: "viridis", vorticity_contour_levels: 8,
                           vorticity_threshold: 0.01),
        interaction: .init(initial_brush_size: 5, min_brush_size: 1, max_brush_size: 20)
    )

    // Try to load config from multiple locations
    var data: Data?

    // Try 1: Bundle resources
    if let bundlePath = Bundle.main.resourcePath,
       let configData = try? Data(contentsOf: URL(fileURLWithPath: "\(bundlePath)/config.json")) {
        data = configData
        print("✓ Loaded config from bundle")
    }

    // Try 2: Current directory
    if data == nil,
       let configData = try? Data(contentsOf: URL(fileURLWithPath: "./CFD/config.json")) {
        data = configData
        print("✓ Loaded config from ./CFD/config.json")
    }

    // Try 3: Relative path
    if data == nil,
       let configData = try? Data(contentsOf: URL(fileURLWithPath: "config.json")) {
        data = configData
        print("✓ Loaded config from config.json")
    }

    guard let configData = data else {
        print("⚠️  Could not load config.json from any location, using defaults")
        return defaultConfig
    }

    guard let config = try? JSONDecoder().decode(SimConfig.self, from: configData) else {
        print("⚠️  Could not parse config.json, using defaults")
        return defaultConfig
    }

    return config
}

// MARK: - Simulation Mode
enum SimulationMode {
    case draw
    case erase
    case run
}

// MARK: - LBM Simulator
class LBMSimulator {
    let device: MTLDevice
    let commandQueue: MTLCommandQueue
    let config: SimConfig

    // Pipelines
    var streamCollidePipeline: MTLComputePipelineState!
    var boundaryPipeline: MTLComputePipelineState!
    var computeFieldsPipeline: MTLComputePipelineState!
    var renderPipeline: MTLComputePipelineState!
    var updateObstaclesPipeline: MTLComputePipelineState!
    var initializePipeline: MTLComputePipelineState!
    var displayPipeline: MTLRenderPipelineState!

    // Textures
    var f_in: MTLTexture!
    var f_out: MTLTexture!
    var obstacles: MTLTexture!
    var velocityMag: MTLTexture!
    var vorticity: MTLTexture!
    var outputTexture: MTLTexture!

    // Simulation parameters
    var windSpeed: Float
    var relaxationTime: Float

    // Ping-pong flag
    var useFirstBuffer = true

    init(device: MTLDevice, config: SimConfig) {
        self.device = device
        self.config = config
        self.windSpeed = config.simulation.initial_wind_speed
        self.relaxationTime = config.simulation.relaxation_time

        guard let queue = device.makeCommandQueue() else {
            fatalError("Failed to create command queue")
        }
        self.commandQueue = queue

        setupPipelines()
        setupTextures()
        clearObstacles()
        initializeSimulation()
    }

    func setupPipelines() {
        // Try to load Metal library from different sources
        var library: MTLLibrary?

        // Try 1: Load from bundle (compiled metallib)
        if let bundlePath = Bundle.main.resourcePath,
           let url = URL(string: "file://\(bundlePath)/default.metallib") {
            library = try? device.makeLibrary(URL: url)
            if library != nil {
                print("✓ Loaded precompiled Metal library from bundle")
            }
        }

        // Try 2: Load from source in bundle
        if library == nil,
           let bundlePath = Bundle.main.resourcePath,
           let source = try? String(contentsOfFile: "\(bundlePath)/default.metal") {
            library = try? device.makeLibrary(source: source, options: nil)
            if library != nil {
                print("✓ Compiled Metal library from source in bundle")
            }
        }

        // Try 3: Load from current directory (for development)
        if library == nil,
           let source = try? String(contentsOfFile: "./CFD/Shaders.metal") {
            library = try? device.makeLibrary(source: source, options: nil)
            if library != nil {
                print("✓ Compiled Metal library from ./CFD/Shaders.metal")
            }
        }

        // Try 4: Load from relative path
        if library == nil,
           let source = try? String(contentsOfFile: "Shaders.metal") {
            library = try? device.makeLibrary(source: source, options: nil)
            if library != nil {
                print("✓ Compiled Metal library from Shaders.metal")
            }
        }

        // Try 5: Default library (Xcode builds)
        if library == nil {
            library = device.makeDefaultLibrary()
            if library != nil {
                print("✓ Loaded default Metal library")
            }
        }

        guard let library = library else {
            fatalError("Failed to create Metal library from any source")
        }

        do {
            streamCollidePipeline = try device.makeComputePipelineState(
                function: library.makeFunction(name: "lbm_stream_collide")!)
            boundaryPipeline = try device.makeComputePipelineState(
                function: library.makeFunction(name: "lbm_boundary")!)
            computeFieldsPipeline = try device.makeComputePipelineState(
                function: library.makeFunction(name: "compute_fields")!)
            renderPipeline = try device.makeComputePipelineState(
                function: library.makeFunction(name: "render_field")!)
            updateObstaclesPipeline = try device.makeComputePipelineState(
                function: library.makeFunction(name: "update_obstacles")!)
            initializePipeline = try device.makeComputePipelineState(
                function: library.makeFunction(name: "initialize_distributions")!)

            // Create render pipeline for displaying
            let renderPipelineDescriptor = MTLRenderPipelineDescriptor()
            renderPipelineDescriptor.vertexFunction = library.makeFunction(name: "fullscreen_vertex")
            renderPipelineDescriptor.fragmentFunction = library.makeFunction(name: "fullscreen_fragment")
            renderPipelineDescriptor.colorAttachments[0].pixelFormat = .bgra8Unorm

            if renderPipelineDescriptor.vertexFunction == nil {
                print("⚠️  Warning: fullscreen_vertex not found!")
            }
            if renderPipelineDescriptor.fragmentFunction == nil {
                print("⚠️  Warning: fullscreen_fragment not found!")
            }

            displayPipeline = try device.makeRenderPipelineState(descriptor: renderPipelineDescriptor)

            print("✓ Metal pipelines created successfully")
        } catch {
            fatalError("Failed to create pipelines: \(error)")
        }
    }

    func setupTextures() {
        let width = config.simulation.grid_width
        let height = config.simulation.grid_height

        // Distribution function textures (9 layers for D2Q9)
        let arrayDesc = MTLTextureDescriptor()
        arrayDesc.textureType = .type2DArray
        arrayDesc.pixelFormat = .r32Float
        arrayDesc.width = width
        arrayDesc.height = height
        arrayDesc.arrayLength = 9
        arrayDesc.usage = [.shaderRead, .shaderWrite]

        f_in = device.makeTexture(descriptor: arrayDesc)!
        f_out = device.makeTexture(descriptor: arrayDesc)!

        // Obstacle texture
        let obstacleDesc = MTLTextureDescriptor.texture2DDescriptor(
            pixelFormat: .r32Float, width: width, height: height, mipmapped: false)
        obstacleDesc.usage = [.shaderRead, .shaderWrite]
        obstacles = device.makeTexture(descriptor: obstacleDesc)!

        // Obstacles will be cleared after pipeline initialization

        // Field textures
        velocityMag = device.makeTexture(descriptor: obstacleDesc)!
        vorticity = device.makeTexture(descriptor: obstacleDesc)!

        // Output texture
        let outputDesc = MTLTextureDescriptor.texture2DDescriptor(
            pixelFormat: .rgba8Unorm, width: width, height: height, mipmapped: false)
        outputDesc.usage = [.shaderRead, .shaderWrite]
        outputTexture = device.makeTexture(descriptor: outputDesc)!

        print("✓ Textures allocated: \(width)x\(height)")
    }

    func clearTexture(_ texture: MTLTexture) {
        // Use a compute kernel to clear the texture
        guard let commandBuffer = commandQueue.makeCommandBuffer(),
              let encoder = commandBuffer.makeComputeCommandEncoder() else { return }

        encoder.setComputePipelineState(updateObstaclesPipeline)
        encoder.setTexture(texture, index: 0)

        // Set brush center outside the texture to avoid drawing anything
        var center = SIMD2<Float>(-1000, -1000)
        var radius: Float = 0.0
        var value: Float = 0.0
        encoder.setBytes(&center, length: MemoryLayout<SIMD2<Float>>.size, index: 0)
        encoder.setBytes(&radius, length: MemoryLayout<Float>.size, index: 1)
        encoder.setBytes(&value, length: MemoryLayout<Float>.size, index: 2)

        // Actually, let's just dispatch with value 0 everywhere
        center = SIMD2<Float>(Float(texture.width) / 2, Float(texture.height) / 2)
        radius = Float(max(texture.width, texture.height))

        encoder.setBytes(&center, length: MemoryLayout<SIMD2<Float>>.size, index: 0)
        encoder.setBytes(&radius, length: MemoryLayout<Float>.size, index: 1)
        encoder.setBytes(&value, length: MemoryLayout<Float>.size, index: 2)

        dispatchThreads(encoder: encoder, width: texture.width, height: texture.height,
                       pipeline: updateObstaclesPipeline)

        encoder.endEncoding()
        commandBuffer.commit()
        commandBuffer.waitUntilCompleted()
    }

    func initializeSimulation() {
        guard let commandBuffer = commandQueue.makeCommandBuffer(),
              let encoder = commandBuffer.makeComputeCommandEncoder() else { return }

        encoder.setComputePipelineState(initializePipeline)
        encoder.setTexture(f_in, index: 0)

        var initialVel = SIMD2<Float>(windSpeed, 0.0)
        encoder.setBytes(&initialVel, length: MemoryLayout<SIMD2<Float>>.size, index: 0)

        dispatchThreads(encoder: encoder, width: config.simulation.grid_width,
                       height: config.simulation.grid_height, pipeline: initializePipeline)

        encoder.endEncoding()
        commandBuffer.commit()
        commandBuffer.waitUntilCompleted()

        print("✓ Simulation initialized")
    }

    func updateVisualization() {
        guard let commandBuffer = commandQueue.makeCommandBuffer(),
              let encoder = commandBuffer.makeComputeCommandEncoder() else { return }

        let currentF = useFirstBuffer ? f_in! : f_out!

        // Compute macroscopic fields
        encoder.setComputePipelineState(computeFieldsPipeline)
        encoder.setTexture(currentF, index: 0)
        encoder.setTexture(velocityMag, index: 1)
        encoder.setTexture(vorticity, index: 2)

        dispatchThreads(encoder: encoder, width: config.simulation.grid_width,
                       height: config.simulation.grid_height, pipeline: computeFieldsPipeline)

        // Render visualization
        encoder.setComputePipelineState(renderPipeline)
        encoder.setTexture(velocityMag, index: 0)
        encoder.setTexture(vorticity, index: 1)
        encoder.setTexture(obstacles, index: 2)
        encoder.setTexture(outputTexture, index: 3)

        var maxVel = config.simulation.max_wind_speed
        var vortThresh = config.visualization.vorticity_threshold
        encoder.setBytes(&maxVel, length: MemoryLayout<Float>.size, index: 0)
        encoder.setBytes(&vortThresh, length: MemoryLayout<Float>.size, index: 1)

        dispatchThreads(encoder: encoder, width: config.simulation.grid_width,
                       height: config.simulation.grid_height, pipeline: renderPipeline)

        encoder.endEncoding()
        commandBuffer.commit()
    }

    func step() {
        guard let commandBuffer = commandQueue.makeCommandBuffer() else { return }

        let steps = config.simulation.steps_per_frame

        for _ in 0..<steps {
            guard let encoder = commandBuffer.makeComputeCommandEncoder() else { return }

            let input = useFirstBuffer ? f_in! : f_out!
            let output = useFirstBuffer ? f_out! : f_in!

            // 1. Stream and Collide
            encoder.setComputePipelineState(streamCollidePipeline)
            encoder.setTexture(input, index: 0)
            encoder.setTexture(output, index: 1)
            encoder.setTexture(obstacles, index: 2)
            var tau = relaxationTime
            var inletVel = SIMD2<Float>(windSpeed, 0.0)
            encoder.setBytes(&tau, length: MemoryLayout<Float>.size, index: 0)
            encoder.setBytes(&inletVel, length: MemoryLayout<SIMD2<Float>>.size, index: 1)

            dispatchThreads(encoder: encoder, width: config.simulation.grid_width,
                           height: config.simulation.grid_height, pipeline: streamCollidePipeline)

            // 2. Apply Boundaries
            encoder.setComputePipelineState(boundaryPipeline)
            encoder.setTexture(output, index: 0)
            encoder.setBytes(&inletVel, length: MemoryLayout<SIMD2<Float>>.size, index: 0)

            dispatchThreads(encoder: encoder, width: config.simulation.grid_width,
                           height: config.simulation.grid_height, pipeline: boundaryPipeline)

            encoder.endEncoding()

            // Swap buffers
            useFirstBuffer.toggle()
        }

        commandBuffer.commit()

        // Update visualization after simulation steps
        updateVisualization()
    }

    func drawObstacle(at point: SIMD2<Float>, radius: Float, value: Float) {
        guard let commandBuffer = commandQueue.makeCommandBuffer(),
              let encoder = commandBuffer.makeComputeCommandEncoder() else { return }

        encoder.setComputePipelineState(updateObstaclesPipeline)
        encoder.setTexture(obstacles, index: 0)

        var center = point
        var rad = radius
        var val = value
        encoder.setBytes(&center, length: MemoryLayout<SIMD2<Float>>.size, index: 0)
        encoder.setBytes(&rad, length: MemoryLayout<Float>.size, index: 1)
        encoder.setBytes(&val, length: MemoryLayout<Float>.size, index: 2)

        dispatchThreads(encoder: encoder, width: config.simulation.grid_width,
                       height: config.simulation.grid_height, pipeline: updateObstaclesPipeline)

        encoder.endEncoding()
        commandBuffer.commit()
    }

    func clearObstacles() {
        clearTexture(obstacles)
    }

    func resetSimulation() {
        clearObstacles()
        initializeSimulation()
        useFirstBuffer = true
    }

    func dispatchThreads(encoder: MTLComputeCommandEncoder, width: Int, height: Int,
                        pipeline: MTLComputePipelineState) {
        let w = pipeline.threadExecutionWidth
        let h = pipeline.maxTotalThreadsPerThreadgroup / w
        let threadsPerGroup = MTLSize(width: w, height: h, depth: 1)
        let numGroups = MTLSize(width: (width + w - 1) / w,
                               height: (height + h - 1) / h,
                               depth: 1)
        encoder.dispatchThreadgroups(numGroups, threadsPerThreadgroup: threadsPerGroup)
    }
}

// MARK: - Metal View Coordinator
class MetalViewCoordinator: NSObject, MTKViewDelegate {
    var simulator: LBMSimulator
    var mode: SimulationMode = .run
    var onFPSUpdate: ((Double) -> Void)?

    private var lastFrameTime = CFAbsoluteTimeGetCurrent()
    private var frameCount = 0
    private var fpsAccumulator: Double = 0.0

    init(simulator: LBMSimulator) {
        self.simulator = simulator
    }

    func mtkView(_ view: MTKView, drawableSizeWillChange size: CGSize) {}

    var hasLoggedSizes = false

    func draw(in view: MTKView) {
        guard let drawable = view.currentDrawable else { return }

        // Calculate FPS
        let currentTime = CFAbsoluteTimeGetCurrent()
        let deltaTime = currentTime - lastFrameTime
        lastFrameTime = currentTime

        frameCount += 1
        fpsAccumulator += deltaTime

        // Update FPS every 30 frames
        if frameCount >= 30 {
            let avgFPS = Double(frameCount) / fpsAccumulator
            onFPSUpdate?(avgFPS)
            frameCount = 0
            fpsAccumulator = 0.0
        }

        // Debug: Log sizes once
        if !hasLoggedSizes {
            print("🔍 Debug Info:")
            print("   View bounds: \(view.bounds.size)")
            print("   Drawable size: \(view.drawableSize)")
            print("   Drawable texture size: \(drawable.texture.width)x\(drawable.texture.height)")
            print("   Simulation texture size: \(simulator.outputTexture.width)x\(simulator.outputTexture.height)")
            hasLoggedSizes = true
        }

        // Step simulation in run mode, otherwise just update visualization
        if mode == .run {
            simulator.step()
        } else {
            // In draw/erase mode, update visualization to show obstacles
            simulator.updateVisualization()
        }

        // Render scaled texture to drawable
        guard let commandBuffer = simulator.commandQueue.makeCommandBuffer() else { return }

        let renderPassDescriptor = MTLRenderPassDescriptor()
        renderPassDescriptor.colorAttachments[0].texture = drawable.texture
        renderPassDescriptor.colorAttachments[0].loadAction = .clear
        renderPassDescriptor.colorAttachments[0].clearColor = MTLClearColor(red: 0, green: 0, blue: 0, alpha: 1)
        renderPassDescriptor.colorAttachments[0].storeAction = .store

        guard let renderEncoder = commandBuffer.makeRenderCommandEncoder(descriptor: renderPassDescriptor) else {
            print("⚠️  Failed to create render encoder!")
            return
        }

        renderEncoder.setRenderPipelineState(simulator.displayPipeline)
        renderEncoder.setFragmentTexture(simulator.outputTexture, index: 0)
        renderEncoder.drawPrimitives(type: .triangle, vertexStart: 0, vertexCount: 6)

        renderEncoder.endEncoding()
        commandBuffer.present(drawable)
        commandBuffer.commit()
    }
}

// MARK: - Metal View Wrapper
struct MetalView: NSViewRepresentable {
    @Binding var simulator: LBMSimulator
    @Binding var mode: SimulationMode
    @Binding var brushSize: Float
    @Binding var viewSize: CGSize
    var onFPSUpdate: ((Double) -> Void)?

    func makeCoordinator() -> MetalViewCoordinator {
        let coordinator = MetalViewCoordinator(simulator: simulator)
        coordinator.onFPSUpdate = onFPSUpdate
        return coordinator
    }

    func makeNSView(context: Context) -> MTKView {
        let mtkView = MTKView()
        mtkView.device = simulator.device
        mtkView.delegate = context.coordinator
        mtkView.preferredFramesPerSecond = 60
        mtkView.enableSetNeedsDisplay = false
        mtkView.isPaused = false
        mtkView.framebufferOnly = false
        mtkView.clearColor = MTLClearColor(red: 0, green: 0, blue: 0, alpha: 1)

        // Set drawable to match simulation grid for 1:1 pixel mapping
        let width = simulator.config.simulation.grid_width
        let height = simulator.config.simulation.grid_height
        mtkView.drawableSize = CGSize(width: width, height: height)

        return mtkView
    }

    func updateNSView(_ nsView: MTKView, context: Context) {
        context.coordinator.mode = mode

        // Update view size for coordinate conversion
        DispatchQueue.main.async {
            viewSize = nsView.bounds.size
        }
    }
}

// MARK: - Main Content View
struct ContentView: View {
    @State private var config = loadConfig()
    @State private var simulator: LBMSimulator?
    @State private var mode: SimulationMode = .run
    @State private var windSpeed: Float = 0.05
    @State private var brushSize: Float = 5.0
    @State private var mousePosition: CGPoint = .zero
    @State private var isDragging = false
    @State private var viewSize: CGSize = CGSize(width: 800, height: 600)
    @State private var fps: Double = 60.0
    @State private var lastDrawPoint: SIMD2<Float>?

    var body: some View {
        VStack(spacing: 0) {
            // Metal View (1:1 pixel mapping with simulation grid)
            ZStack {
                if let sim = simulator {
                    MetalView(
                        simulator: .constant(sim),
                        mode: $mode,
                        brushSize: $brushSize,
                        viewSize: $viewSize,
                        onFPSUpdate: { newFPS in
                            fps = newFPS
                        }
                    )
                    .frame(
                        width: CGFloat(config.simulation.grid_width),
                        height: CGFloat(config.simulation.grid_height)
                    )
                    .gesture(
                        DragGesture(minimumDistance: 0)
                            .onChanged { value in
                                handleDrag(at: value.location, sim: sim)
                            }
                            .onEnded { _ in
                                isDragging = false
                                lastDrawPoint = nil
                            }
                    )
                } else {
                    ProgressView("Initializing Metal...")
                }
            }
            .frame(
                width: CGFloat(config.simulation.grid_width),
                height: CGFloat(config.simulation.grid_height)
            )

            // Control Panel
            VStack(spacing: 12) {
                // Mode buttons
                HStack(spacing: 20) {
                    Button(action: { mode = .draw }) {
                        Text("✏️ Draw")
                            .frame(width: 100)
                            .padding(8)
                            .background(mode == .draw ? Color.blue : Color.gray.opacity(0.3))
                            .foregroundColor(.white)
                            .cornerRadius(8)
                    }

                    Button(action: { mode = .erase }) {
                        Text("🧹 Erase")
                            .frame(width: 100)
                            .padding(8)
                            .background(mode == .erase ? Color.orange : Color.gray.opacity(0.3))
                            .foregroundColor(.white)
                            .cornerRadius(8)
                    }

                    Button(action: { mode = .run }) {
                        Text("▶️ Run")
                            .frame(width: 100)
                            .padding(8)
                            .background(mode == .run ? Color.green : Color.gray.opacity(0.3))
                            .foregroundColor(.white)
                            .cornerRadius(8)
                    }
                }

                Divider()

                // Wind speed slider
                HStack {
                    Text("Wind Speed:")
                        .frame(width: 100, alignment: .leading)
                    Slider(value: $windSpeed,
                           in: 0...config.simulation.max_wind_speed,
                           onEditingChanged: { _ in
                        simulator?.windSpeed = windSpeed
                    })
                    Text(String(format: "%.3f", windSpeed))
                        .frame(width: 50, alignment: .trailing)
                        .monospacedDigit()
                }

                // Brush size slider
                HStack {
                    Text("Brush Size:")
                        .frame(width: 100, alignment: .leading)
                    Slider(value: $brushSize,
                           in: Float(config.interaction.min_brush_size)...Float(config.interaction.max_brush_size),
                           step: 1)
                    Text("\(Int(brushSize))")
                        .frame(width: 50, alignment: .trailing)
                        .monospacedDigit()
                }

                Divider()

                // Action buttons
                HStack(spacing: 20) {
                    Button("Clear Obstacles") {
                        simulator?.clearObstacles()
                    }
                    .padding(8)
                    .background(Color.red.opacity(0.7))
                    .foregroundColor(.white)
                    .cornerRadius(8)

                    Button("Reset Simulation") {
                        simulator?.resetSimulation()
                    }
                    .padding(8)
                    .background(Color.purple.opacity(0.7))
                    .foregroundColor(.white)
                    .cornerRadius(8)
                }

                // Info
                Text("Steps/Frame: \(config.simulation.steps_per_frame) | Grid: \(config.simulation.grid_width)x\(config.simulation.grid_height) (1:1)")
                    .font(.caption)
                    .foregroundColor(.gray)
            }
            .padding()
            .background(Color(NSColor.windowBackgroundColor))
        }
        .onAppear {
            initializeSimulator()
            windSpeed = config.simulation.initial_wind_speed
            brushSize = Float(config.interaction.initial_brush_size)
            updateWindowTitle(fps: fps)

            // Set window size to match simulation grid (1:1 mapping)
            DispatchQueue.main.async {
                if let window = NSApplication.shared.windows.first {
                    let width = CGFloat(config.simulation.grid_width)
                    let height = CGFloat(config.simulation.grid_height) + 150 // Extra space for controls
                    window.setContentSize(NSSize(width: width, height: height))
                }
            }
        }
        .onChange(of: fps) { newFPS in
            updateWindowTitle(fps: newFPS)
        }
    }

    func updateWindowTitle(fps: Double) {
        DispatchQueue.main.async {
            if let window = NSApplication.shared.windows.first {
                window.title = String(format: "Virtual Wind Tunnel | %.1f FPS", fps)
            }
        }
    }

    func initializeSimulator() {
        guard let device = MTLCreateSystemDefaultDevice() else {
            print("Metal is not supported on this device")
            return
        }

        print("✓ Metal device: \(device.name)")
        simulator = LBMSimulator(device: device, config: config)
    }

    func handleDrag(at location: CGPoint, sim: LBMSimulator) {
        guard mode == .draw || mode == .erase else { return }

        // Direct 1:1 mapping - no scaling!
        let gridX = Float(location.x)
        let gridY = Float(location.y)

        // Clamp to grid bounds
        let clampedX = max(0, min(Float(config.simulation.grid_width - 1), gridX))
        let clampedY = max(0, min(Float(config.simulation.grid_height - 1), gridY))

        let currentPoint = SIMD2<Float>(clampedX, clampedY)
        let value: Float = (mode == .draw) ? 1.0 : 0.0
        let effectiveRadius = brushSize + 0.5

        // If we have a previous point, interpolate to fill gaps
        if let lastPoint = lastDrawPoint, isDragging {
            let dist = distance(currentPoint, lastPoint)
            let steps = max(1, Int(ceil(dist / (brushSize * 0.5))))

            for i in 0...steps {
                let t = Float(i) / Float(steps)
                let interpPoint = lastPoint + (currentPoint - lastPoint) * t
                sim.drawObstacle(at: interpPoint, radius: effectiveRadius, value: value)
            }
        } else {
            // First point or not dragging yet
            sim.drawObstacle(at: currentPoint, radius: effectiveRadius, value: value)
        }

        lastDrawPoint = currentPoint
        isDragging = true
    }
}

// MARK: - App Delegate
class AppDelegate: NSObject, NSApplicationDelegate {
    func applicationShouldTerminateAfterLastWindowClosed(_ sender: NSApplication) -> Bool {
        return true
    }
}

// MARK: - App Entry Point
@main
struct WindTunnelApp: App {
    @NSApplicationDelegateAdaptor(AppDelegate.self) var appDelegate

    var body: some Scene {
        WindowGroup("Virtual Wind Tunnel") {
            ContentView()
        }
    }
}
