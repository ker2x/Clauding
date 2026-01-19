// =============================================================================
// Quantum Garden - main.swift
// =============================================================================
// An interactive quantum mechanics simulation where you cultivate a garden of
// Bohmian particles guided by wave functions. Plant seeds (wave packets), draw
// potential barriers, and watch beautiful quantum interference patterns emerge.
//
// PHYSICS:
// - Wave function evolves via time-dependent Schrodinger equation
// - Particles follow Bohmian mechanics: v = (hbar/m) * Im(grad psi / psi)
// - User-drawn potentials bend wave propagation and particle paths
//
// CONTROLS:
// - Left click: Plant a quantum seed (Gaussian wave packet)
// - Right click / Option+click: Draw potential barriers
// - Shift+click: Erase potentials
// - Space: Toggle pause
// - R: Reset simulation
// - C: Clear all potentials
// - S: Trigger quantum storm (decoherence event)
// - Q: Quit
// =============================================================================

import AppKit
import Metal
import MetalKit
import QuartzCore

// =============================================================================
// CONFIGURATION
// =============================================================================

struct Config {
    // Window
    static let windowWidth: CGFloat = 1200
    static let windowHeight: CGFloat = 900

    // Quantum simulation grid (wave function resolution)
    static let gridWidth = 512
    static let gridHeight = 384

    // Bohmian particles (guided by wave function)
    static let numParticles = 1000000

    // Physics parameters (in natural units where hbar = m = 1)
    static let waveDt: Float = 0.15          // Wave evolution time step
    static let particleDt: Float = 1.0        // Particle guidance strength
    static let diffusionCoeff: Float = 0.25   // Kinetic term coefficient (-hbar^2/2m)

    // Gameplay
    static let seedRadius: Float = 12.0       // Initial wave packet width (sigma)
    static let seedMomentum: Float = 1.5      // Initial wave packet momentum
    static let potentialStrength: Float = 2.0 // Barrier strength
    static let stormDuration: Float = 2.0     // Decoherence storm duration
    static let stormStrength: Float = 0.3     // How much storms disrupt the wave

    // Rendering
    static let pointSize: Float = 2.0         // Small for 1M particle fluid look
    static let trailDecay: Float = 0.92       // Faster color transitions for fire
    static let bgColor = (r: Float(0.01), g: Float(0.005), b: Float(0.015), a: Float(1.0))
}

// =============================================================================
// DATA STRUCTURES (must match Metal shader)
// =============================================================================

struct SimParams {
    var width: Float
    var height: Float
    var grid_width: Int32
    var grid_height: Int32
    var wave_dt: Float
    var particle_dt: Float
    var diffusion: Float
    var potential_strength: Float
    var num_particles: Int32
    var point_size: Float
    var time: Float
    var storm_active: Int32       // 1 if storm is happening
    var storm_time: Float         // Time since storm started
    var storm_center_x: Float     // Storm epicenter X
    var storm_center_y: Float     // Storm epicenter Y
}

// Seed request structure for planting new wave packets
struct SeedRequest {
    var x: Float
    var y: Float
    var vx: Float               // Momentum direction X
    var vy: Float               // Momentum direction Y
    var active: Int32           // 1 if this seed should be planted
    var padding: Float = 0
}

// =============================================================================
// InputView - Handles mouse and keyboard input
// =============================================================================

class InputView: MTKView {
    weak var renderer: Renderer?
    var lastMousePos: NSPoint = .zero

    override var acceptsFirstResponder: Bool { true }

    override func keyDown(with event: NSEvent) {
        guard let key = event.charactersIgnoringModifiers?.first else { return }

        switch key {
        case "q", "Q":
            NSApp.terminate(nil)
        case "r", "R":
            renderer?.reset()
        case " ":
            renderer?.togglePause()
        case "c", "C":
            renderer?.clearPotentials()
        case "s", "S":
            renderer?.triggerStorm()
        default:
            super.keyDown(with: event)
        }
    }

    override func mouseDown(with event: NSEvent) {
        let location = convert(event.locationInWindow, from: nil)
        lastMousePos = location

        if event.modifierFlags.contains(.option) {
            // Option+click: Draw potential
            renderer?.drawPotential(at: location, erase: false)
        } else if event.modifierFlags.contains(.shift) {
            // Shift+click: Erase potential
            renderer?.drawPotential(at: location, erase: true)
        } else {
            // Regular click: Plant seed
            renderer?.plantSeed(at: location, momentum: nil)
        }
    }

    override func mouseDragged(with event: NSEvent) {
        let location = convert(event.locationInWindow, from: nil)

        if event.modifierFlags.contains(.option) || event.modifierFlags.contains(.control) {
            renderer?.drawPotential(at: location, erase: false)
        } else if event.modifierFlags.contains(.shift) {
            renderer?.drawPotential(at: location, erase: true)
        } else {
            // While dragging from a seed, calculate momentum direction
            let dx = Float(location.x - lastMousePos.x)
            let dy = Float(location.y - lastMousePos.y)
            let dist = sqrt(dx*dx + dy*dy)
            if dist > 5 {
                let momentum = SIMD2<Float>(dx / dist, -dy / dist) * Config.seedMomentum
                renderer?.plantSeed(at: lastMousePos, momentum: momentum)
            }
        }
    }

    override func rightMouseDown(with event: NSEvent) {
        let location = convert(event.locationInWindow, from: nil)
        renderer?.drawPotential(at: location, erase: false)
    }

    override func rightMouseDragged(with event: NSEvent) {
        let location = convert(event.locationInWindow, from: nil)
        renderer?.drawPotential(at: location, erase: false)
    }
}

// =============================================================================
// AppDelegate
// =============================================================================

class AppDelegate: NSObject, NSApplicationDelegate {
    func applicationShouldTerminateAfterLastWindowClosed(_ sender: NSApplication) -> Bool {
        return true
    }
}

// =============================================================================
// Renderer - Metal compute and rendering
// =============================================================================

class Renderer: NSObject, MTKViewDelegate {
    let device: MTLDevice
    let commandQueue: MTLCommandQueue

    // Pipelines
    var wavePipeline: MTLComputePipelineState?
    var particlePipeline: MTLComputePipelineState?
    var seedPipeline: MTLComputePipelineState?
    var stormPipeline: MTLComputePipelineState?
    var renderPipeline: MTLRenderPipelineState?

    // Buffers - double buffered for wave function
    var waveRealBuffers: [MTLBuffer] = []
    var waveImagBuffers: [MTLBuffer] = []
    var potentialBuffer: MTLBuffer?
    var positionBuffer: MTLBuffer?
    var velocityBuffer: MTLBuffer?
    var colorBuffer: MTLBuffer?           // Accumulated color trails
    var paramsBuffer: MTLBuffer?
    var seedBuffer: MTLBuffer?            // For planting new seeds

    var currentWaveBuffer = 0             // Which buffer to read from (0 or 1)

    // State
    weak var view: MTKView?
    var paused = false
    var simulationTime: Float = 0
    var score = 0
    var seedsPlanted = 0

    // Storm state
    var stormActive = false
    var stormStartTime: Float = 0
    var stormCenterX: Float = 0
    var stormCenterY: Float = 0

    // FPS tracking
    var lastFrameTime: Double = 0
    var frameCount = 0
    var fpsUpdateTime: Double = 0

    // Pending seed to plant
    var pendingSeed: SeedRequest?

    init?(view: MTKView) {
        guard let device = MTLCreateSystemDefaultDevice(),
              let commandQueue = device.makeCommandQueue() else {
            print("Metal is not supported")
            return nil
        }

        self.device = device
        self.commandQueue = commandQueue
        self.view = view

        let currentTime = CACurrentMediaTime()
        self.lastFrameTime = currentTime
        self.fpsUpdateTime = currentTime

        super.init()

        print("Quantum Garden initializing on: \(device.name)")

        guard loadShaders() else { return nil }
        initBuffers()

        // Configure view
        view.device = device
        view.colorPixelFormat = .bgra8Unorm
        view.enableSetNeedsDisplay = false
        view.isPaused = false
        view.delegate = self
        view.clearColor = MTLClearColor(
            red: Double(Config.bgColor.r),
            green: Double(Config.bgColor.g),
            blue: Double(Config.bgColor.b),
            alpha: Double(Config.bgColor.a)
        )

        print("Quantum Garden ready!")
        print("  Grid: \(Config.gridWidth) x \(Config.gridHeight)")
        print("  Particles: \(Config.numParticles)")
        print("Controls:")
        print("  Click: Plant quantum seed")
        print("  Drag: Plant seed with momentum")
        print("  Option+drag: Draw potential barrier")
        print("  Shift+drag: Erase potential")
        print("  S: Trigger quantum storm")
        print("  Space: Pause | R: Reset | C: Clear | Q: Quit")
    }

    func loadShaders() -> Bool {
        do {
            let shaderPath = "Compute.metal"
            let shaderSource = try String(contentsOfFile: shaderPath, encoding: .utf8)

            let options = MTLCompileOptions()
            if #available(macOS 15.0, *) {
                options.mathMode = .fast
            } else {
                options.fastMathEnabled = true
            }

            let library = try device.makeLibrary(source: shaderSource, options: options)

            // Create compute pipelines
            guard let waveFunc = library.makeFunction(name: "evolve_wave"),
                  let particleFunc = library.makeFunction(name: "update_particles"),
                  let seedFunc = library.makeFunction(name: "plant_seed"),
                  let stormFunc = library.makeFunction(name: "apply_storm") else {
                print("Could not find compute functions")
                return false
            }

            wavePipeline = try device.makeComputePipelineState(function: waveFunc)
            particlePipeline = try device.makeComputePipelineState(function: particleFunc)
            seedPipeline = try device.makeComputePipelineState(function: seedFunc)
            stormPipeline = try device.makeComputePipelineState(function: stormFunc)

            // Create render pipeline
            guard let vertexFunc = library.makeFunction(name: "vertex_main"),
                  let fragmentFunc = library.makeFunction(name: "fragment_main") else {
                print("Could not find render functions")
                return false
            }

            let pipelineDesc = MTLRenderPipelineDescriptor()
            pipelineDesc.vertexFunction = vertexFunc
            pipelineDesc.fragmentFunction = fragmentFunc
            pipelineDesc.colorAttachments[0].pixelFormat = .bgra8Unorm

            // Enable additive blending for glowing effect
            pipelineDesc.colorAttachments[0].isBlendingEnabled = true
            pipelineDesc.colorAttachments[0].rgbBlendOperation = .add
            pipelineDesc.colorAttachments[0].alphaBlendOperation = .add
            pipelineDesc.colorAttachments[0].sourceRGBBlendFactor = .sourceAlpha
            pipelineDesc.colorAttachments[0].destinationRGBBlendFactor = .one
            pipelineDesc.colorAttachments[0].sourceAlphaBlendFactor = .one
            pipelineDesc.colorAttachments[0].destinationAlphaBlendFactor = .one

            renderPipeline = try device.makeRenderPipelineState(descriptor: pipelineDesc)

            print("Shaders compiled successfully")
            return true

        } catch {
            print("Shader compilation failed: \(error)")
            return false
        }
    }

    func initBuffers() {
        let gridSize = Config.gridWidth * Config.gridHeight
        let numParticles = Config.numParticles

        // Double-buffered wave function (for ping-pong updates)
        for _ in 0..<2 {
            let realBuf = device.makeBuffer(length: MemoryLayout<Float>.stride * gridSize, options: .storageModeShared)
            let imagBuf = device.makeBuffer(length: MemoryLayout<Float>.stride * gridSize, options: .storageModeShared)
            waveRealBuffers.append(realBuf!)
            waveImagBuffers.append(imagBuf!)
        }

        // Potential field
        potentialBuffer = device.makeBuffer(length: MemoryLayout<Float>.stride * gridSize, options: .storageModeShared)

        // Particles
        var positions = [SIMD2<Float>](repeating: .zero, count: numParticles)
        let velocities = [SIMD2<Float>](repeating: .zero, count: numParticles)

        // Distribute particles according to probability distribution (initially uniform)
        for i in 0..<numParticles {
            positions[i] = SIMD2(
                Float.random(in: 0..<Float(Config.windowWidth)),
                Float.random(in: 0..<Float(Config.windowHeight))
            )
        }

        positionBuffer = device.makeBuffer(bytes: positions, length: MemoryLayout<SIMD2<Float>>.stride * numParticles, options: .storageModeShared)
        velocityBuffer = device.makeBuffer(bytes: velocities, length: MemoryLayout<SIMD2<Float>>.stride * numParticles, options: .storageModeShared)
        colorBuffer = device.makeBuffer(length: MemoryLayout<SIMD4<Float>>.stride * numParticles, options: .storageModeShared)

        // Parameters
        var params = SimParams(
            width: Float(Config.windowWidth),
            height: Float(Config.windowHeight),
            grid_width: Int32(Config.gridWidth),
            grid_height: Int32(Config.gridHeight),
            wave_dt: Config.waveDt,
            particle_dt: Config.particleDt,
            diffusion: Config.diffusionCoeff,
            potential_strength: Config.potentialStrength,
            num_particles: Int32(numParticles),
            point_size: Config.pointSize,
            time: 0,
            storm_active: 0,
            storm_time: 0,
            storm_center_x: 0,
            storm_center_y: 0
        )
        paramsBuffer = device.makeBuffer(bytes: &params, length: MemoryLayout<SimParams>.stride, options: .storageModeShared)

        // Seed request buffer
        var seed = SeedRequest(x: 0, y: 0, vx: 0, vy: 0, active: 0)
        seedBuffer = device.makeBuffer(bytes: &seed, length: MemoryLayout<SeedRequest>.stride, options: .storageModeShared)
    }

    func plantInitialSeed() {
        // Plant an initial wave packet in the center
        guard let realBuf = waveRealBuffers.first,
              let imagBuf = waveImagBuffers.first else { return }

        let real = realBuf.contents().bindMemory(to: Float.self, capacity: Config.gridWidth * Config.gridHeight)
        let imag = imagBuf.contents().bindMemory(to: Float.self, capacity: Config.gridWidth * Config.gridHeight)

        let centerX = Float(Config.gridWidth) / 2
        let centerY = Float(Config.gridHeight) / 2
        let sigma = Config.seedRadius
        let k = Config.seedMomentum * 0.5  // Initial momentum

        for y in 0..<Config.gridHeight {
            for x in 0..<Config.gridWidth {
                let dx = Float(x) - centerX
                let dy = Float(y) - centerY
                let r2 = dx*dx + dy*dy

                // Gaussian envelope with plane wave
                let envelope = exp(-r2 / (2 * sigma * sigma))
                let phase = k * dx  // Momentum in +x direction

                let idx = y * Config.gridWidth + x
                real[idx] = envelope * cos(phase)
                imag[idx] = envelope * sin(phase)
            }
        }

        seedsPlanted = 1
    }

    func plantSeed(at location: NSPoint, momentum: SIMD2<Float>?) {
        // Convert screen coordinates to grid coordinates
        // macOS has Y=0 at bottom, but our grid/render convention has Y=0 at top
        let gridX = Float(location.x) / Float(Config.windowWidth) * Float(Config.gridWidth)
        let gridY = Float(location.y) / Float(Config.windowHeight) * Float(Config.gridHeight)

        var vx: Float = 0
        var vy: Float = 0
        if let mom = momentum {
            vx = mom.x
            vy = mom.y
        } else {
            // Random momentum if not specified
            let angle = Float.random(in: 0..<Float.pi * 2)
            vx = cos(angle) * Config.seedMomentum
            vy = sin(angle) * Config.seedMomentum
        }

        // Store seed request to be processed on GPU
        pendingSeed = SeedRequest(x: gridX, y: gridY, vx: vx, vy: vy, active: 1)
        seedsPlanted += 1
        score += 10
    }

    func drawPotential(at location: NSPoint, erase: Bool) {
        guard let potentialBuffer = potentialBuffer else { return }

        let gridX = Int(Float(location.x) / Float(Config.windowWidth) * Float(Config.gridWidth))
        let gridY = Int(Float(location.y) / Float(Config.windowHeight) * Float(Config.gridHeight))

        let potential = potentialBuffer.contents().bindMemory(to: Float.self, capacity: Config.gridWidth * Config.gridHeight)

        let radius = 15
        for dy in -radius...radius {
            for dx in -radius...radius {
                let dist = sqrt(Float(dx*dx + dy*dy))
                if dist <= Float(radius) {
                    let x = gridX + dx
                    let y = gridY + dy
                    if x >= 0 && x < Config.gridWidth && y >= 0 && y < Config.gridHeight {
                        let idx = y * Config.gridWidth + x
                        if erase {
                            potential[idx] = max(0, potential[idx] - 0.5)
                        } else {
                            let strength = (1.0 - dist / Float(radius)) * Config.potentialStrength
                            potential[idx] = min(potential[idx] + strength, Config.potentialStrength * 2)
                        }
                    }
                }
            }
        }
    }

    func triggerStorm() {
        stormActive = true
        stormStartTime = simulationTime
        stormCenterX = Float.random(in: 0..<Float(Config.gridWidth))
        stormCenterY = Float.random(in: 0..<Float(Config.gridHeight))
        print("Quantum storm triggered at (\(Int(stormCenterX)), \(Int(stormCenterY)))")
    }

    func clearPotentials() {
        guard let potentialBuffer = potentialBuffer else { return }
        let potential = potentialBuffer.contents().bindMemory(to: Float.self, capacity: Config.gridWidth * Config.gridHeight)
        for i in 0..<(Config.gridWidth * Config.gridHeight) {
            potential[i] = 0
        }
        print("Potentials cleared")
    }

    func reset() {
        // Clear wave function
        for i in 0..<2 {
            let real = waveRealBuffers[i].contents().bindMemory(to: Float.self, capacity: Config.gridWidth * Config.gridHeight)
            let imag = waveImagBuffers[i].contents().bindMemory(to: Float.self, capacity: Config.gridWidth * Config.gridHeight)
            for j in 0..<(Config.gridWidth * Config.gridHeight) {
                real[j] = 0
                imag[j] = 0
            }
        }

        clearPotentials()

        // Redistribute particles
        let positions = positionBuffer!.contents().bindMemory(to: SIMD2<Float>.self, capacity: Config.numParticles)
        let velocities = velocityBuffer!.contents().bindMemory(to: SIMD2<Float>.self, capacity: Config.numParticles)
        for i in 0..<Config.numParticles {
            positions[i] = SIMD2(
                Float.random(in: 0..<Float(Config.windowWidth)),
                Float.random(in: 0..<Float(Config.windowHeight))
            )
            velocities[i] = .zero
        }

        score = 0
        seedsPlanted = 0
        simulationTime = 0
        stormActive = false
        currentWaveBuffer = 0

        print("Simulation reset")
    }

    func togglePause() {
        paused.toggle()
        print(paused ? "Paused" : "Resumed")
    }

    func calculateComplexity() -> Float {
        // Calculate wave function complexity for scoring
        let real = waveRealBuffers[currentWaveBuffer].contents().bindMemory(to: Float.self, capacity: Config.gridWidth * Config.gridHeight)
        let imag = waveImagBuffers[currentWaveBuffer].contents().bindMemory(to: Float.self, capacity: Config.gridWidth * Config.gridHeight)

        var totalEnergy: Float = 0
        var phaseVariation: Float = 0
        var lastPhase: Float = 0

        let sampleStep = 4  // Sample every 4th point for performance
        for y in stride(from: 0, to: Config.gridHeight, by: sampleStep) {
            for x in stride(from: 0, to: Config.gridWidth, by: sampleStep) {
                let idx = y * Config.gridWidth + x
                let r = real[idx]
                let i = imag[idx]
                let mag2 = r*r + i*i
                totalEnergy += mag2

                let phase = atan2(i, r)
                if x > 0 {
                    phaseVariation += abs(phase - lastPhase)
                }
                lastPhase = phase
            }
        }

        return totalEnergy * 0.01 + phaseVariation * 0.001
    }

    // MARK: - MTKViewDelegate

    func mtkView(_ view: MTKView, drawableSizeWillChange size: CGSize) {}

    func draw(in view: MTKView) {
        let currentTime = CACurrentMediaTime()
        frameCount += 1

        if currentTime - fpsUpdateTime >= 1.0 {
            let fps = Double(frameCount) / (currentTime - fpsUpdateTime)
            let complexity = calculateComplexity()
            score += Int(complexity)
            view.window?.title = String(format: "Quantum Garden - Seeds: %d  Score: %d  -  %.1f FPS", seedsPlanted, score, fps)
            frameCount = 0
            fpsUpdateTime = currentTime
        }

        lastFrameTime = currentTime

        guard let wavePipeline = wavePipeline,
              let particlePipeline = particlePipeline,
              let seedPipeline = seedPipeline,
              let stormPipeline = stormPipeline,
              let renderPipeline = renderPipeline,
              let paramsBuffer = paramsBuffer,
              let potentialBuffer = potentialBuffer,
              let positionBuffer = positionBuffer,
              let velocityBuffer = velocityBuffer,
              let colorBuffer = colorBuffer,
              let seedBuffer = seedBuffer else { return }

        // Update parameters
        let params = paramsBuffer.contents().bindMemory(to: SimParams.self, capacity: 1)
        if !paused {
            simulationTime += Config.waveDt
        }
        params.pointee.time = simulationTime

        // Update storm state
        if stormActive {
            let stormTime = simulationTime - stormStartTime
            if stormTime > Config.stormDuration {
                stormActive = false
                params.pointee.storm_active = 0
            } else {
                params.pointee.storm_active = 1
                params.pointee.storm_time = stormTime
                params.pointee.storm_center_x = stormCenterX
                params.pointee.storm_center_y = stormCenterY
            }
        } else {
            params.pointee.storm_active = 0
        }

        // Handle pending seed
        if let seed = pendingSeed {
            let seedPtr = seedBuffer.contents().bindMemory(to: SeedRequest.self, capacity: 1)
            seedPtr.pointee = seed
            pendingSeed = nil
        } else {
            let seedPtr = seedBuffer.contents().bindMemory(to: SeedRequest.self, capacity: 1)
            seedPtr.pointee.active = 0
        }

        guard let commandBuffer = commandQueue.makeCommandBuffer() else { return }

        // COMPUTE PASS
        if !paused {
            guard let computeEncoder = commandBuffer.makeComputeCommandEncoder() else { return }

            let readBuffer = currentWaveBuffer
            let writeBuffer = 1 - currentWaveBuffer

            // 1. Plant seed if requested
            computeEncoder.setComputePipelineState(seedPipeline)
            computeEncoder.setBuffer(waveRealBuffers[readBuffer], offset: 0, index: 0)
            computeEncoder.setBuffer(waveImagBuffers[readBuffer], offset: 0, index: 1)
            computeEncoder.setBuffer(seedBuffer, offset: 0, index: 2)
            computeEncoder.setBuffer(paramsBuffer, offset: 0, index: 3)

            let seedThreads = MTLSize(width: (Config.gridWidth + 15) / 16, height: (Config.gridHeight + 15) / 16, depth: 1)
            computeEncoder.dispatchThreadgroups(seedThreads, threadsPerThreadgroup: MTLSize(width: 16, height: 16, depth: 1))

            // 2. Apply storm if active
            if stormActive {
                computeEncoder.setComputePipelineState(stormPipeline)
                computeEncoder.setBuffer(waveRealBuffers[readBuffer], offset: 0, index: 0)
                computeEncoder.setBuffer(waveImagBuffers[readBuffer], offset: 0, index: 1)
                computeEncoder.setBuffer(paramsBuffer, offset: 0, index: 2)
                computeEncoder.dispatchThreadgroups(seedThreads, threadsPerThreadgroup: MTLSize(width: 16, height: 16, depth: 1))
            }

            // 3. Evolve wave function (Schrodinger equation)
            computeEncoder.setComputePipelineState(wavePipeline)
            computeEncoder.setBuffer(waveRealBuffers[readBuffer], offset: 0, index: 0)
            computeEncoder.setBuffer(waveImagBuffers[readBuffer], offset: 0, index: 1)
            computeEncoder.setBuffer(waveRealBuffers[writeBuffer], offset: 0, index: 2)
            computeEncoder.setBuffer(waveImagBuffers[writeBuffer], offset: 0, index: 3)
            computeEncoder.setBuffer(potentialBuffer, offset: 0, index: 4)
            computeEncoder.setBuffer(paramsBuffer, offset: 0, index: 5)

            let waveThreads = MTLSize(width: (Config.gridWidth + 15) / 16, height: (Config.gridHeight + 15) / 16, depth: 1)
            computeEncoder.dispatchThreadgroups(waveThreads, threadsPerThreadgroup: MTLSize(width: 16, height: 16, depth: 1))

            // Swap buffers
            currentWaveBuffer = writeBuffer

            // 4. Update Bohmian particles
            computeEncoder.setComputePipelineState(particlePipeline)
            computeEncoder.setBuffer(waveRealBuffers[currentWaveBuffer], offset: 0, index: 0)
            computeEncoder.setBuffer(waveImagBuffers[currentWaveBuffer], offset: 0, index: 1)
            computeEncoder.setBuffer(positionBuffer, offset: 0, index: 2)
            computeEncoder.setBuffer(velocityBuffer, offset: 0, index: 3)
            computeEncoder.setBuffer(colorBuffer, offset: 0, index: 4)
            computeEncoder.setBuffer(paramsBuffer, offset: 0, index: 5)

            let particleThreads = MTLSize(width: (Config.numParticles + 255) / 256, height: 1, depth: 1)
            computeEncoder.dispatchThreadgroups(particleThreads, threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))

            computeEncoder.endEncoding()
        }

        // RENDER PASS
        guard let renderPassDesc = view.currentRenderPassDescriptor,
              let drawable = view.currentDrawable else { return }

        guard let renderEncoder = commandBuffer.makeRenderCommandEncoder(descriptor: renderPassDesc) else { return }

        renderEncoder.setRenderPipelineState(renderPipeline)
        renderEncoder.setVertexBuffer(positionBuffer, offset: 0, index: 0)
        renderEncoder.setVertexBuffer(colorBuffer, offset: 0, index: 1)
        renderEncoder.setVertexBuffer(paramsBuffer, offset: 0, index: 2)

        renderEncoder.drawPrimitives(type: .point, vertexStart: 0, vertexCount: Config.numParticles)
        renderEncoder.endEncoding()

        commandBuffer.present(drawable)
        commandBuffer.commit()
    }
}

// =============================================================================
// MAIN
// =============================================================================

let app = NSApplication.shared
app.setActivationPolicy(.regular)

let delegate = AppDelegate()
app.delegate = delegate

// Create menu
let mainMenu = NSMenu()
let appMenuItem = NSMenuItem()
mainMenu.addItem(appMenuItem)
let appMenu = NSMenu()
appMenuItem.submenu = appMenu
appMenu.addItem(NSMenuItem(title: "Quit", action: #selector(NSApplication.terminate(_:)), keyEquivalent: "q"))
app.mainMenu = mainMenu

// Create window
let window = NSWindow(
    contentRect: NSRect(x: 0, y: 0, width: Config.windowWidth, height: Config.windowHeight),
    styleMask: [.titled, .closable, .miniaturizable],
    backing: .buffered,
    defer: false
)
window.title = "Quantum Garden"
window.center()

// Create Metal view
let metalView = InputView(frame: window.contentView!.bounds)
metalView.autoresizingMask = [.width, .height]

guard let renderer = Renderer(view: metalView) else {
    print("Failed to create renderer")
    exit(1)
}

metalView.renderer = renderer
window.contentView = metalView
window.makeFirstResponder(metalView)
window.makeKeyAndOrderFront(nil)
window.orderFrontRegardless()
app.activate(ignoringOtherApps: true)

app.run()
