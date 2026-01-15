# Swift Particle Template

A comprehensive template for building GPU-accelerated particle simulations using Swift and Metal. This is the **Swift equivalent** of `metal_particle_template`, demonstrating the same particle life simulation using Swift's modern syntax instead of Objective-C++.

## Quick Start

```bash
# Build
make

# Run
./SwiftParticles

# Or build and run in one command
make run
```

### Controls

| Key   | Action |
|-------|--------|
| Q     | Quit application |
| R     | Randomize particles and interaction matrix |
| Space | Pause/Resume simulation |

## Project Structure

```
swift_particle_template/
├── Makefile          # Build configuration
├── main.swift        # Application code (Pure Swift)
├── Compute.metal     # GPU shaders (Metal Shading Language)
└── README.md         # This file
```

## What's Different from the Objective-C++ Version?

This template demonstrates **identical functionality** to `metal_particle_template/` but uses Swift instead of Objective-C++. The key differences are:

| Aspect | Objective-C++ | Swift |
|--------|---------------|-------|
| **File extension** | `.mm` | `.swift` |
| **Syntax** | `[obj method]` | `obj.method()` |
| **Type safety** | Weak | Strong |
| **Optionals** | Can crash with nil | Safe unwrapping required |
| **Memory** | ARC (manual) | ARC (fully automatic) |
| **Line count** | ~650 lines | ~500 lines |
| **Metal shaders** | Identical `Compute.metal` | Identical `Compute.metal` |

**Key Insight**: The Metal shader code (`Compute.metal`) is **100% identical** between the Swift and Objective-C++ versions. Only the host language changes!

## Requirements

- macOS 10.14 or later (Metal support)
- Xcode Command Line Tools (for `swiftc`)

Install command line tools if needed:
```bash
xcode-select --install
```

## Libraries and Dependencies

**Zero external dependencies**. This template uses only built-in macOS frameworks:

- **Metal** - Core GPU framework for compute and graphics
- **MetalKit** - High-level utilities (MTKView, render loop)
- **AppKit** - macOS windowing and event handling
- **QuartzCore** - High-precision timing (CACurrentMediaTime)

**Pure Metal Implementation** - This template does NOT use:
- ❌ Metal Performance Shaders (MPS) for neural networks
- ❌ MPSGraph for computation graphs
- ❌ CoreML for machine learning
- ❌ Accelerate framework
- ❌ Any third-party libraries

All compute operations (particle physics, force calculations) are **hand-coded in Metal Shading Language**. This approach provides:
- Full control over GPU execution
- Direct buffer management
- Educational clarity - see exactly how GPU programming works
- Minimal binary size (~110 KB for Swift vs ~77 KB for Objective-C++)
- No external dependencies to manage

## How It Works

This template implements a **particle life simulation** where thousands of particles interact based on configurable attraction/repulsion rules. The simulation runs entirely on the GPU for high performance.

### Architecture Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                         Application                              │
├─────────────────────────────────────────────────────────────────┤
│  main()           Create window, view, and start event loop     │
│  AppDelegate      Handle application lifecycle                   │
│  InputView        Capture keyboard input                         │
│  Renderer         Manage Metal resources and render loop         │
├─────────────────────────────────────────────────────────────────┤
│                          Metal GPU                               │
├─────────────────────────────────────────────────────────────────┤
│  Compute Kernel   update_particles() - physics simulation       │
│  Vertex Shader    vertex_main() - transform positions           │
│  Fragment Shader  fragment_main() - color pixels                │
└─────────────────────────────────────────────────────────────────┘
```

### Frame Loop

Each frame follows this sequence:

1. **Compute Pass**: GPU runs the `update_particles` kernel
   - Each particle calculates forces from all other particles
   - Velocities and positions are updated

2. **Render Pass**: GPU draws all particles
   - Vertex shader transforms world coordinates to screen coordinates
   - Fragment shader colors each pixel

3. **Present**: The rendered frame is displayed

### macOS App Requirements

For proper keyboard input and window management, the Swift version includes:

```swift
// 1. Set activation policy (makes app appear in Dock)
app.setActivationPolicy(.regular)

// 2. Create application menu (required for keyboard events)
let mainMenu = NSMenu()
// ... menu setup

// 3. Activate the app (brings to foreground)
app.activate(ignoringOtherApps: true)
```

These are **required** for macOS apps built from command line. Without them:
- ❌ Keyboard events won't be captured
- ❌ App won't appear in Dock
- ❌ App won't receive focus properly

The Objective-C++ version needs similar setup (already included in `metal_particle_template`).

## Understanding Swift for Metal

If you're coming from C++ or Objective-C, here are key Swift concepts used in this template:

### Optionals (Handling nil Safely)

```swift
// Optional: can be nil
var device: MTLDevice?

// Safe unwrapping with guard
guard let device = MTLCreateSystemDefaultDevice() else {
    print("No Metal support")
    return nil
}
// device is safely unwrapped here
```

### Type Inference

```swift
let numParticles = 20000        // Compiler knows: Int
let friction: Float = 0.80      // Explicit type
var positions = [SIMD2<Float>]  // Array of 2D vectors
```

### Properties and Methods

```swift
// Class with properties
class Renderer: NSObject {
    let device: MTLDevice           // Immutable
    var paused = false              // Mutable

    func draw(in view: MTKView) {   // Method
        // ...
    }
}
```

### Guard Statements (Early Exit)

```swift
guard let computePipeline = computePipeline,
      let renderPipeline = renderPipeline else {
    return  // Exit early if any is nil
}
// Both are safely unwrapped here
```

### SIMD Types

```swift
// Swift has native SIMD types that match Metal
let position = SIMD2<Float>(100.0, 200.0)  // float2 in Metal
let color = SIMD4<Float>(1.0, 0.0, 0.0, 1.0)  // float4 in Metal
```

## Customization Guide

### Changing Particle Count

Edit `Config.numParticles` in `main.swift`:

```swift
struct Config {
    static let numParticles = 20000  // Increase for more particles
}
```

### Changing Colors

Edit the color palette in `Renderer.initBuffers()`:

```swift
let colors: [SIMD4<Float>] = [
    SIMD4(1.0, 0.0, 0.0, 1.0),  // Red
    SIMD4(0.0, 1.0, 0.0, 1.0),  // Green
    SIMD4(0.0, 0.5, 1.0, 1.0),  // Blue
    SIMD4(1.0, 1.0, 0.0, 1.0),  // Yellow
]
```

### Modifying Physics

The physics behavior is defined in `Compute.metal` (identical to Objective-C++ version):

```metal
// Adjust the force calculation in update_particles
if (normalizedDist < 0.3f) {
    // Short range behavior
    forceMagnitude = normalizedDist / 0.3f - 1.0f;
} else {
    // Medium range behavior
    float t = (normalizedDist - 0.3f) / 0.7f;
    forceMagnitude = attraction * (1.0f - abs(2.0f * t - 1.0f));
}
```

## Comparing Swift and Objective-C++ Versions

This project has a **nearly identical** Objective-C++ version at `../metal_particle_template/`.

### Code Comparison Examples

**Creating a buffer:**

```objc
// Objective-C++
id<MTLBuffer> buffer = [device newBufferWithLength:size
                                           options:MTLResourceStorageModeShared];
```

```swift
// Swift
let buffer = device.makeBuffer(
    length: size,
    options: .storageModeShared
)
```

**Handling errors:**

```objc
// Objective-C++
NSError* error = nil;
NSString* source = [NSString stringWithContentsOfFile:path
                                            encoding:NSUTF8StringEncoding
                                               error:&error];
if (!source) {
    NSLog(@"Error: %@", error);
    return NO;
}
```

```swift
// Swift
do {
    let source = try String(contentsOfFile: path, encoding: .utf8)
} catch {
    print("Error: \(error)")
    return false
}
```

**Creating objects:**

```objc
// Objective-C++
MTKView* view = [[MTKView alloc] initWithFrame:frame];
```

```swift
// Swift
let view = MTKView(frame: frame)
```

### Which Should You Use?

**Use Swift if:**
- ✅ Starting a new project
- ✅ Want cleaner, more readable code
- ✅ Prefer modern language features
- ✅ Building for Apple platforms only

**Use Objective-C++ if:**
- ✅ Integrating with existing C++ code
- ✅ Need maximum portability
- ✅ Working with legacy codebases
- ✅ Prefer explicit memory control

## Performance Comparison

| Metric | Objective-C++ | Swift |
|--------|---------------|-------|
| **Compile time** | ~0.5s | ~2-3s |
| **Binary size** | ~77 KB | ~110 KB |
| **Runtime FPS** | 60 FPS | 60 FPS |
| **GPU performance** | Identical | Identical |

**Conclusion**: Both versions have **identical GPU performance** because they use the same Metal shaders. The Swift version has longer compile times but the same runtime speed.

## Common Issues

### Shader Compilation Errors

If you see "Could not load shader file", make sure `Compute.metal` is in the same directory as the executable:

```bash
ls -la
# Should show both SwiftParticles and Compute.metal
```

### Metal Not Supported

If you see "Metal is not supported on this device", you're on older hardware without Metal support (pre-2012 Macs).

### Swift Version Mismatch

If you see "Module compiled with Swift X.Y cannot be imported", your Swift version might be incompatible. Check your version:

```bash
swiftc --version
```

## Extending This Template

Ideas for extensions:

1. **Add mouse interaction**: Implement mouse events in InputView
2. **Add trails**: Store particle history and draw lines
3. **Add spatial hashing**: Improve performance for large particle counts
4. **Add external forces**: Gravity, wind, attractors
5. **Add different particle types**: Different sizes, behaviors
6. **Save/load configurations**: JSON for interaction matrices

## Resources

### Swift Resources
- [Swift.org](https://swift.org/) - Official Swift website
- [Swift Book](https://docs.swift.org/swift-book/) - Free official documentation
- [Swift Forums](https://forums.swift.org/) - Community discussion

### Metal Resources
- [Metal Programming Guide](https://developer.apple.com/documentation/metal)
- [Metal Shading Language Specification](https://developer.apple.com/metal/Metal-Shading-Language-Specification.pdf)
- [Metal by Example](https://metalbyexample.com/)

### Related Projects
- `../metal_particle_template/` - Objective-C++ version (compare!)
- `../swift_metal_for_beginners/` - Beginner Swift + Metal tutorial
- `../metal_lumen/` - More advanced particle system
- `../metal_physarum/` - Slime mold simulation

## Documentation Philosophy

This template follows the **"What, Why, How"** documentation framework (see `../CLAUDE.md`):

- **What**: Pure Swift + Metal, no external dependencies
- **Why**: Educational clarity, full control, modern language
- **How**: GPU compute kernels, vertex/fragment shaders, buffer management

## License

This template is released into the public domain. Use it however you like.

---

**Recommendation**: Study both the Swift and Objective-C++ versions side-by-side to see how the same Metal concepts are expressed in different languages. The shader code is identical, demonstrating that Metal works the same way regardless of host language!
