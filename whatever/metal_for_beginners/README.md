# Metal for Beginners

**A comprehensive tutorial for C++ programmers learning GPU programming with Metal and Objective-C**

This project is designed for programmers who:
- ✅ Know how to code (C++, Java, C#, etc.)
- ✅ Understand basic graphics concepts (vertices, pixels, shaders)
- ❌ Have never used Objective-C
- ❌ Have never programmed GPUs with Metal

By the end of this tutorial, you'll understand:
- Objective-C syntax and how it relates to C++
- How to set up a Metal rendering pipeline
- How to write GPU shaders in Metal Shading Language
- How to create a macOS window without Xcode

---

## Table of Contents

1. [Quick Start](#quick-start)
2. [What This Program Does](#what-this-program-does)
3. [Prerequisites](#prerequisites)
4. [File Overview](#file-overview)
5. [Objective-C Crash Course](#objective-c-crash-course)
6. [Metal Architecture](#metal-architecture)
7. [Code Walkthrough](#code-walkthrough)
8. [Shader Explained](#shader-explained)
9. [Exercises](#exercises)
10. [Next Steps](#next-steps)
11. [Troubleshooting](#troubleshooting)
12. [Resources](#resources)

---

## Quick Start

```bash
# Navigate to this directory
cd whatever/metal_for_beginners

# Build the project
make

# Run it
./MetalBeginner

# Or build and run in one command
make run
```

You should see a window with an **animated** colorful gradient triangle that pulses and cycles through colors! The window title displays the current FPS (frames per second).

**Controls:**
- **Q** or **ESC** - Quit the application
- **Close window** - Also quits

**What to expect:**
- Should run at ~60 FPS on most Macs
- Window title updates every second with current FPS
- Smooth color animations with multiple effects

---

## What This Program Does

This is a comprehensive Metal tutorial that demonstrates:

1. **Creates a macOS window** (no Xcode, no Interface Builder, pure code)
2. **Initializes Metal** (gets GPU, creates command queue)
3. **Compiles shaders** at runtime from `Compute.metal`
4. **Renders an animated gradient** 60 times per second
5. **Passes time data from CPU to GPU** (uniform buffers)
6. **Animates colors with math** (sin/cos for pulsing and cycling)
7. **Displays FPS in window title** (updated every second)
8. **Handles keyboard input** (Q/ESC to quit)

Total code: ~800 lines with extensive comments (~330 without comments)

**No dependencies** - just built-in macOS frameworks:
- Metal (GPU API)
- MetalKit (rendering helpers)
- AppKit (window/app management)
- QuartzCore (high-precision timing)

---

## Prerequisites

### Required
- **macOS 10.14+** (Metal is macOS/iOS only, like DirectX is Windows-only)
- **Xcode Command Line Tools** (for the compiler)

Install command line tools if needed:
```bash
xcode-select --install
```

You do **NOT** need the full Xcode IDE. Just the command line tools.

### Recommended Knowledge
- **C++ basics**: classes, pointers, functions
- **Basic graphics**: what a vertex is, what a pixel is
- **Command line**: how to `cd` and run programs

---

## File Overview

```
metal_for_beginners/
├── README.md          ← You are here
├── main.mm            ← Application code (Objective-C++)
├── Compute.metal      ← GPU shader code (Metal Shading Language)
└── Makefile           ← Build configuration
```

### File Extensions Explained

| Extension | Language | Used For |
|-----------|----------|----------|
| `.mm` | Objective-C++ | Can mix C++, C, and Objective-C in one file |
| `.m` | Objective-C | Pure Objective-C (no C++) |
| `.metal` | Metal Shading Language | GPU shaders (compute, vertex, fragment) |
| `.cpp` | C++ | Pure C++ (no Objective-C) |
| `.h` / `.hpp` | Header | Declarations (works with any of the above) |

We use `.mm` because we want both C++ features (`namespace`, `constexpr`) and Objective-C features (frameworks, classes).

---

## Objective-C Crash Course

Objective-C feels alien if you come from C++. Here's a quick translation guide:

### Object Creation

```cpp
// C++
MyClass* obj = new MyClass();
delete obj; // Manual memory management
```

```objc
// Objective-C
MyClass* obj = [[MyClass alloc] init];
// No delete! ARC (Automatic Reference Counting) handles it
```

**ARC** = like `std::shared_ptr` built into the language. Objects are freed when nothing references them.

### Method Calls

```cpp
// C++
obj->doSomething();
obj->calculate(10, 20, 30);
```

```objc
// Objective-C
[obj doSomething];
[obj calculateWithA:10 b:20 c:30];
```

**Note**: Square brackets `[ ]` are used for method calls (messages).

### Method Definitions

```cpp
// C++ header
class MyClass {
public:
    void doSomething();
    int calculate(int a, int b);
};

// C++ implementation
void MyClass::doSomething() { ... }
int MyClass::calculate(int a, int b) { ... }
```

```objc
// Objective-C interface (header)
@interface MyClass : NSObject
- (void)doSomething;
- (int)calculateWithA:(int)a b:(int)b;
@end

// Objective-C implementation
@implementation MyClass
- (void)doSomething { ... }
- (int)calculateWithA:(int)a b:(int)b { ... }
@end
```

**Syntax breakdown**:
- `-` = instance method (like non-static in C++)
- `+` = class method (like static in C++)
- `(returnType)` = what the method returns
- `methodName:` = method name (colons are part of the name!)
- `(paramType)paramName` = parameter

### Named Parameters

This is the weirdest part for C++ programmers:

```objc
[window initWithContentRect:frame styleMask:style backing:NSBackingStoreBuffered defer:NO];
```

This is **ONE** method call with **FOUR** parameters!

The method signature is:
```objc
- (instancetype)initWithContentRect:(NSRect)rect
                          styleMask:(NSWindowStyleMask)style
                            backing:(NSBackingStoreType)backing
                              defer:(BOOL)flag;
```

The parts between the colons are **parameter labels** (like Python's keyword arguments).

### String Literals

```cpp
// C++
std::string str = "hello";
printf("Hello %s\n", name);
```

```objc
// Objective-C
NSString* str = @"hello";  // Note the @ prefix!
NSLog(@"Hello %@", name);
```

**Important**: Objective-C string literals need an `@` prefix. `@"text"` creates an `NSString*`.

### Protocols (Interfaces)

```cpp
// C++ (pure virtual functions)
class IDelegate {
public:
    virtual void onEvent() = 0;
};

class MyClass : public IDelegate {
    void onEvent() override { ... }
};
```

```objc
// Objective-C
@protocol IDelegate
- (void)onEvent;
@end

@interface MyClass : NSObject <IDelegate>
@end

@implementation MyClass
- (void)onEvent { ... }
@end
```

Angle brackets `<Protocol>` indicate protocol conformance (like C++ `: public Interface`).

### Blocks (Closures/Lambdas)

```cpp
// C++ lambda
auto lambda = [](int x) { return x * 2; };
std::thread t([this]() {
    this->updateUI();
});
```

```objc
// Objective-C block
int (^myBlock)(int) = ^(int x) { return x * 2; };

dispatch_async(dispatch_get_main_queue(), ^{
    [self updateUI];
});
```

**Block syntax**: `^(parameters) { code }`

Common uses:
- Callbacks and completion handlers
- Asynchronous operations (like `dispatch_async`)
- Collection operations (like `enumerateObjectsUsingBlock:`)

**Capturing variables**:
- Blocks capture variables from surrounding scope (like lambdas)
- Inside block, use `self->ivar` to access instance variables
- Variables are captured by value unless marked with `__block`

---

## Metal Architecture

Metal is Apple's low-level GPU API. Think of it as Apple's version of Vulkan or DirectX 12.

### Core Concepts

```
┌─────────────────────────────────────────────────────────┐
│                    Your Application                      │
│                        (CPU)                             │
└────────────────────┬────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────┐
│                   Metal Framework                        │
│  • MTLDevice          (The GPU)                          │
│  • MTLCommandQueue    (Submit work to GPU)               │
│  • MTLLibrary         (Compiled shaders)                 │
│  • MTLBuffer          (GPU memory)                       │
│  • MTLTexture         (Images on GPU)                    │
│  • MTLPipeline        (Shader configuration)             │
└────────────────────┬────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────┐
│                   GPU Hardware                           │
│  • Runs shaders in parallel                              │
│  • Processes vertices and pixels                         │
│  • Executes compute kernels                              │
└─────────────────────────────────────────────────────────┘
```

### The Metal Pipeline (Step by Step)

Every Metal app follows this pattern:

```cpp
// 1. Get the GPU
id<MTLDevice> device = MTLCreateSystemDefaultDevice();

// 2. Create a command queue (reuse this)
id<MTLCommandQueue> queue = [device newCommandQueue];

// 3. Compile shaders
id<MTLLibrary> library = [device newLibraryWithSource:shaderCode ...];
id<MTLFunction> vertexFunc = [library newFunctionWithName:@"vertex_main"];
id<MTLFunction> fragmentFunc = [library newFunctionWithName:@"fragment_main"];

// 4. Create a pipeline (compiled shader program)
id<MTLRenderPipelineState> pipeline = [device newRenderPipelineStateWith...];

// 5. Each frame:
{
    // Create command buffer (holds GPU commands for this frame)
    id<MTLCommandBuffer> commandBuffer = [queue commandBuffer];

    // Create encoder (records commands into the buffer)
    id<MTLRenderCommandEncoder> encoder = [commandBuffer renderCommandEncoderWith...];

    // Set pipeline and draw
    [encoder setRenderPipelineState:pipeline];
    [encoder drawPrimitives:MTLPrimitiveTypeTriangle vertexStart:0 vertexCount:3];

    // Finish encoding
    [encoder endEncoding];

    // Send to GPU
    [commandBuffer commit];
}
```

### Key Objects Explained

| Object | C++ Analogy | Purpose |
|--------|-------------|---------|
| `id<MTLDevice>` | Graphics card handle | Represents the GPU |
| `id<MTLCommandQueue>` | `std::queue<Command>` | Submits work to GPU |
| `id<MTLLibrary>` | Compiled DLL | Contains shader functions |
| `id<MTLFunction>` | Function pointer | One shader function |
| `id<MTLPipelineState>` | Compiled shader program | Ready-to-use pipeline |
| `id<MTLCommandBuffer>` | Frame's command list | Commands for this frame |
| `id<MTLCommandEncoder>` | Command recorder | Records draw/compute commands |

---

## Code Walkthrough

Let's trace what happens when you run `./MetalBeginner`:

### 1. `main()` Function (main.mm:652-746)

```objc
int main() {
    @autoreleasepool {
        // Create app
        NSApplication* app = [NSApplication sharedApplication];

        // Create window
        NSWindow* window = [[NSWindow alloc] init...];

        // Create Metal view
        MTKView* metalView = [[MTKView alloc] initWithFrame:...];

        // Create renderer
        Renderer* renderer = [[Renderer alloc] initWithMetalKitView:metalView];

        // Show window
        [window makeKeyAndOrderFront:nil];

        // Run event loop
        [app run]; // Doesn't return until app quits
    }
}
```

**What happens**:
1. Creates `NSApplication` (the app itself)
2. Creates `NSWindow` (the window frame)
3. Creates `MTKView` (a Metal-capable view that provides a render loop)
4. Creates `Renderer` (our class that does Metal stuff)
5. Shows the window
6. Enters the event loop (processes mouse/keyboard, calls render every frame)

### 2. `Renderer` Initialization (main.mm:231-277)

```objc
- (instancetype)initWithMetalKitView:(MTKView*)view {
    self = [super init];

    // Get GPU
    _device = MTLCreateSystemDefaultDevice();

    // Create command queue
    _commandQueue = [_device newCommandQueue];

    // Load shaders
    [self loadShaders];

    // Configure view
    view.device = _device;
    view.delegate = self;

    return self;
}
```

**What happens**:
1. Gets the default GPU (`_device`)
2. Creates a command queue for submitting work
3. Loads and compiles shaders from `Compute.metal`
4. Configures the view to use this GPU and call us every frame

### 3. Shader Loading (main.mm:289-388)

```objc
- (BOOL)loadShaders {
    // Load shader source from file
    NSString* shaderSource = [NSString stringWithContentsOfFile:@"Compute.metal" ...];

    // Compile into library
    id<MTLLibrary> library = [_device newLibraryWithSource:shaderSource ...];

    // Get specific functions
    id<MTLFunction> vertexFunc = [library newFunctionWithName:@"vertex_main"];
    id<MTLFunction> fragmentFunc = [library newFunctionWithName:@"fragment_main"];

    // Create pipeline
    MTLRenderPipelineDescriptor* desc = [[MTLRenderPipelineDescriptor alloc] init];
    desc.vertexFunction = vertexFunc;
    desc.fragmentFunction = fragmentFunc;
    desc.colorAttachments[0].pixelFormat = _view.colorPixelFormat;

    _pipeline = [_device newRenderPipelineStateWithDescriptor:desc ...];

    return YES;
}
```

**What happens**:
1. Reads `Compute.metal` as a string
2. Compiles it into a library of shader functions
3. Extracts the `vertex_main` and `fragment_main` functions
4. Creates a pipeline that connects them together

### 4. Render Loop (main.mm:447-484)

```objc
- (void)drawInMTKView:(MTKView*)view {
    // Create command buffer for this frame
    id<MTLCommandBuffer> commandBuffer = [_commandQueue commandBuffer];

    // Get render pass (describes where to draw)
    MTLRenderPassDescriptor* renderPass = view.currentRenderPassDescriptor;

    // Create encoder (records draw commands)
    id<MTLRenderCommandEncoder> encoder =
        [commandBuffer renderCommandEncoderWithDescriptor:renderPass];

    // Use our pipeline
    [encoder setRenderPipelineState:_pipeline];

    // Draw 3 vertices as a triangle
    [encoder drawPrimitives:MTLPrimitiveTypeTriangle
                vertexStart:0
                vertexCount:3];

    // Finish encoding
    [encoder endEncoding];

    // Present result on screen
    [commandBuffer presentDrawable:view.currentDrawable];

    // Send to GPU
    [commandBuffer commit];
}
```

**What happens** (every frame, 60 times per second):
1. Creates a command buffer (holds this frame's commands)
2. Gets the render pass (tells GPU where to draw)
3. Creates an encoder (records draw commands)
4. Sets the pipeline (which shaders to use)
5. Issues a draw call (draw 3 vertices)
6. Presents the result (show on screen)
7. Commits to GPU (GPU starts executing asynchronously)

---

## Shader Explained

The shader code in `Compute.metal` runs **on the GPU**, not the CPU.

### Vertex Shader (Compute.metal:178-226)

```metal
vertex VertexOut vertex_main(uint vertexID [[vertex_id]]) {
    // Generate positions for a fullscreen triangle
    float2 positions[3] = {
        float2(-1.0, -1.0),  // Bottom-left
        float2( 3.0, -1.0),  // Off-screen right
        float2(-1.0,  3.0)   // Off-screen top
    };

    // Generate colors
    float3 colors[3] = {
        float3(1.0, 0.2, 0.4),  // Reddish
        float3(0.2, 0.6, 1.0),  // Bluish
        float3(0.3, 1.0, 0.5)   // Greenish
    };

    VertexOut out;
    out.position = float4(positions[vertexID], 0.0, 1.0);
    out.color = float4(colors[vertexID], 1.0);
    return out;
}
```

**What it does**:
- Runs **3 times** (once per vertex)
- Generates vertex positions procedurally (no vertex buffer needed!)
- Assigns a color to each vertex
- Returns position and color for each vertex

**Coordinate system**:
- `(-1, -1)` = bottom-left of screen
- `(1, 1)` = top-right of screen
- `(0, 0)` = center

We draw a triangle larger than the screen so it covers the whole viewport.

### Fragment Shader with Animation (Compute.metal:227-281)

```metal
fragment float4 fragment_main(
    VertexOut in [[stage_in]],
    constant Uniforms& uniforms [[buffer(0)]]  // ← Time data from CPU!
) {
    float3 color = in.color.rgb;

    // Pulsating brightness
    float pulse = 0.5 + 0.5 * sin(uniforms.time * 2.0);
    color *= (0.5 + 0.5 * pulse);

    // Color cycling (hue rotation)
    float hueShift = uniforms.time * 0.5;
    float r = color.r * cos(hueShift) - color.g * sin(hueShift);
    float g = color.r * sin(hueShift) + color.g * cos(hueShift);
    color = float3(r, g, color.b);

    // Saturation pulse
    float gray = dot(color, float3(0.299, 0.587, 0.114));
    float saturation = 0.5 + 0.5 * sin(uniforms.time * 3.0);
    color = mix(float3(gray), color, saturation);

    return float4(color, 1.0);
}
```

**What it does**:
- Runs **once per pixel** (800×600 = 480,000 times per frame!)
- Receives interpolated color from vertex shader
- **Receives time data from CPU via uniform buffer**
- Applies three animation effects:
  1. **Brightness pulse** - Uses `sin(time)` to fade in/out
  2. **Hue rotation** - Rotates colors through the spectrum
  3. **Saturation pulse** - Oscillates between grayscale and full color
- Returns the animated pixel color

**Key concept**: The `uniforms` parameter contains data sent from the CPU each frame. This is how we pass time to the GPU!

**Interpolation**: Metal automatically interpolates vertex data across the triangle. Pixels near vertex 0 are more red, pixels near vertex 1 are more blue, etc.

### The Graphics Pipeline Flow

```
CPU: [encoder drawPrimitives:MTLPrimitiveTypeTriangle vertexStart:0 vertexCount:3]
        │
        ▼
GPU: vertex_main() called 3 times
        │
        ├─► Vertex 0: position(-1,-1), color(red)
        ├─► Vertex 1: position( 3,-1), color(blue)
        └─► Vertex 2: position(-1, 3), color(green)
        │
        ▼
GPU: Rasterizer determines which pixels are inside the triangle
        │
        ▼
GPU: fragment_main() called for each pixel (480,000 times!)
        │
        ├─► Pixel at (100, 200): interpolated color
        ├─► Pixel at (101, 200): interpolated color
        └─► ... (all pixels)
        │
        ▼
GPU: Write pixels to framebuffer
        │
        ▼
Screen: Display!
```

All of this happens in **~16ms** for 60 FPS!

### CPU ↔ GPU Data Flow (Uniform Buffers)

One of the most important concepts in GPU programming is **passing data from CPU to GPU**. Here's how we animate with time:

```
EVERY FRAME (60 times per second):

CPU (main.mm):
┌─────────────────────────────────────────────┐
│ 1. Calculate time:                          │
│    double time = CACurrentMediaTime();      │
│                                             │
│ 2. Write to buffer:                         │
│    Uniforms* u = _uniformBuffer.contents;  │
│    u->time = time;                          │
│    u->frameCount = _frameCount;             │
│                                             │
│ 3. Bind to shader:                          │
│    [encoder setFragmentBuffer:              │
│        _uniformBuffer ... atIndex:0];       │
└─────────────────────────────────────────────┘
            │
            ▼ (Buffer stays in GPU memory)
            │
GPU (Compute.metal):
┌─────────────────────────────────────────────┐
│ fragment float4 fragment_main(              │
│     ...,                                    │
│     constant Uniforms& u [[buffer(0)]]      │
│ ) {                                         │
│     // Access time instantly!               │
│     float pulse = sin(u.time);              │
│     return float4(color * pulse, 1.0);      │
│ }                                           │
└─────────────────────────────────────────────┘
```

**Key points**:
- The `Uniforms` struct is **defined identically** on both CPU and GPU
- We use `MTLResourceStorageModeShared` so both can access the same memory
- The GPU reads the data, but **never writes** (read-only via `constant` keyword)
- This pattern works for any data: mouse position, physics parameters, etc.

**Why this is powerful**:
- The shader code is the same for all pixels
- Only the **input data changes** each frame
- 480,000 pixels all use the same time value (efficient!)

---

## Exercises

Once you understand the basics, try these modifications:

### Easy

1. **Change animation speed**
   - In `Compute.metal`, line 251, change `uniforms.time * 2.0` to `* 5.0`
   - Makes the brightness pulse faster

2. **Disable specific animations**
   - Comment out individual animation sections (lines 248-270)
   - Try disabling just the hue rotation or saturation pulse
   - See which effect you like best

3. **Change the colors**
   - In `Compute.metal`, line 209-213, modify the `colors` array
   - Try `float3(1.0, 0.0, 0.0)` for pure red

4. **Change the background**
   - In `main.mm`, line 409, change `MTLClearColorMake(...)`
   - Try `(0.0, 0.0, 0.0, 1.0)` for black

### Medium

5. **Use frame count instead of time**
   - In `Compute.metal`, replace `uniforms.time` with `(float)uniforms.frameCount * 0.016`
   - This gives frame-based animation (useful for debugging)

6. **Add position-based effects**
   - In fragment shader, use `in.position.xy` to create patterns
   - Try: `color += 0.1 * sin(in.position.x * 0.1 + uniforms.time);`
   - Creates moving wave pattern

7. **Make it monochrome**
   - Remove the animation code
   - Calculate grayscale: `float gray = dot(in.color.rgb, float3(0.299, 0.587, 0.114));`
   - Return: `float4(float3(gray), 1.0);`

8. **Create a different animation**
   - Try: `color *= abs(sin(uniforms.time));` for sharper pulses
   - Try: `color.r += 0.2 * sin(uniforms.time);` for red channel flicker

### Advanced

9. **Combine time with position**
   - Create ripple effect: `float ripple = sin(length(in.position.xy) * 0.01 - uniforms.time);`
   - Apply to color: `color *= 0.5 + 0.5 * ripple;`

10. **Animate vertex positions**
    - In vertex shader, modify `out.position` based on time
    - You'll need to pass uniforms to vertex shader too
    - Try: `pos.y += sin(uniforms.time) * 0.1;` for bouncing

11. **Add mouse interaction**
    - Add mouse position to `Uniforms` struct
    - Pass it from CPU each frame
    - Use in shader to create interactive effects

---

## Next Steps

### Learn More Metal

1. **Study the template project**
   - Look at `metal_particle_template/` in this repo
   - It shows buffers, compute kernels, and particle physics
   - More advanced but still well-commented

2. **Add buffers**
   - Instead of hardcoding positions, store them in a `MTLBuffer`
   - Pass the buffer to the shader
   - Modify the buffer data each frame

3. **Try compute kernels**
   - Write a `kernel` function instead of vertex/fragment
   - Process data in parallel (like CUDA or OpenCL)
   - See example in `Compute.metal` lines 408-444

4. **Add textures**
   - Load an image with `MTKTextureLoader`
   - Sample it in the fragment shader with `texture2d`

### Read Documentation

- [Metal Programming Guide](https://developer.apple.com/documentation/metal) - Official docs
- [Metal Shading Language Spec](https://developer.apple.com/metal/Metal-Shading-Language-Specification.pdf) - Language reference
- [Metal Best Practices](https://developer.apple.com/documentation/metal/gpu_selection_in_macos) - Performance tips

### Compare with Other APIs

If you know other GPU APIs:

| Metal | Vulkan | DirectX | OpenGL |
|-------|--------|---------|--------|
| `MTLDevice` | `VkDevice` | `ID3D12Device` | Context |
| `MTLCommandQueue` | `VkQueue` | `ID3D12CommandQueue` | Immediate mode |
| `MTLCommandBuffer` | `VkCommandBuffer` | `ID3D12CommandList` | (implicit) |
| `MTLBuffer` | `VkBuffer` | `ID3D12Resource` | `glGenBuffers` |
| `MTLRenderPipelineState` | `VkPipeline` | `ID3D12PipelineState` | Program object |
| `kernel` function | Compute shader | Compute shader | Compute shader |

Metal is closest to Vulkan/DX12 in philosophy (explicit, low-level) but simpler.

---

## Troubleshooting

### Build Errors

**"Metal/Metal.h: No such file or directory"**
- Metal only works on macOS (and iOS)
- Ensure you're on macOS 10.14 or later
- Check: `ls /System/Library/Frameworks/Metal.framework`

**"clang: command not found"**
- Install Xcode Command Line Tools: `xcode-select --install`

**"ld: framework not found Metal"**
- Your macOS might be too old
- Metal requires macOS 10.14+ (Mojave, 2018)

### Runtime Errors

**"Metal is not supported on this device"**
- Your Mac is too old (pre-2012 Macs don't support Metal)
- Check: System Preferences → About This Mac → Graphics
- If it says "Intel GMA" or very old GPU, Metal won't work

**"Could not load shader file"**
- Make sure `Compute.metal` is in the current directory
- Run from the project directory: `cd metal_for_beginners && ./MetalBeginner`

**"Shader compilation failed"**
- Check syntax errors in `Compute.metal`
- Error message will tell you which line is wrong
- Common mistake: forgetting `metal::` namespace or semicolons

**Window doesn't respond to keyboard**
- Make sure `[window makeFirstResponder:metalView]` is called
- Check if app is actually focused (click the window)

### Performance Issues

**Stuttering / Low FPS**
- Check Activity Monitor → GPU usage
- Try reducing resolution or shader complexity
- Old MacBooks might struggle with 60 FPS

**High CPU usage**
- Make sure shaders are compiled once, not every frame
- Check you're not calling `waitUntilCompleted` on command buffers

---

## Resources

### Official Documentation
- [Metal Homepage](https://developer.apple.com/metal/)
- [Metal Programming Guide](https://developer.apple.com/documentation/metal)
- [Metal Shading Language Specification](https://developer.apple.com/metal/Metal-Shading-Language-Specification.pdf)
- [Metal Best Practices Guide](https://developer.apple.com/documentation/metal/metal_sample_code_library)

### Tutorials
- [Metal by Example](https://metalbyexample.com/) - Excellent book/blog
- [Ray Wenderlich Metal Tutorials](https://www.raywenderlich.com/metal-tutorial-swift-3-part-1-getting-started)
- [Metal Sample Code](https://developer.apple.com/documentation/metal/metal_sample_code_library) - Official examples

### Books
- *Metal by Tutorials* by Ray Wenderlich (up to date, very clear)
- *Metal Programming Guide* by Janie Clayton (comprehensive reference)

### Community
- [r/MetalProgramming](https://reddit.com/r/metalprogramming) - Small but helpful
- [Apple Developer Forums - Metal](https://developer.apple.com/forums/tags/metal)
- [Stack Overflow [metal] tag](https://stackoverflow.com/questions/tagged/metal)

### Related Projects in This Repo
- `metal_particle_template/` - More advanced particle simulation
- `metal_lumen/` - Complex particle system
- `metal_physarum/` - Slime mold simulation

---

## About This Tutorial

This tutorial was created to lower the barrier to entry for Metal programming. Most Metal tutorials assume:
- You know Objective-C already
- You use Xcode IDE
- You understand iOS/macOS app architecture

This tutorial assumes none of that. It's for **programmers who want to learn GPU programming** without getting lost in platform specifics.

### Philosophy

1. **Comment everything weird** - Especially Objective-C syntax
2. **Show the minimal viable program** - Not production-ready, just educational
3. **Explain the "why"** - Not just "what" but "why this design"
4. **No magic** - Every line is explained

### Contributing

Found a mistake? Confusing explanation? Typo?

This is part of a larger repo. Feel free to:
- Suggest clarifications
- Point out errors
- Share what confused you

The best way to improve tutorials is feedback from beginners!

---

## License

This tutorial code is released into the **public domain**. Use it however you want:
- Learn from it
- Copy it
- Modify it
- Include it in commercial projects
- Don't even need to give credit (but appreciated!)

The goal is education. Make something cool with Metal!

---

## Summary

You learned:

✅ **Objective-C syntax** - How to read and write Objective-C code
✅ **Metal architecture** - Device, queue, pipeline, command buffer
✅ **Shader basics** - Vertex and fragment shaders
✅ **Metal Shading Language** - GPU programming with MSL
✅ **macOS app structure** - Window, app delegate, event loop
✅ **Build system** - Makefile, frameworks, compiler flags

Next steps:
- Modify the shader (change colors, add effects)
- Study the particle template (more advanced techniques)
- Read Apple's Metal Programming Guide
- Build something cool!

**You're now ready to write your own Metal projects from scratch.**

Happy GPU programming! 🚀
