# Swift + Metal for Beginners

**A comprehensive tutorial for programmers learning GPU programming with Swift and Metal**

This project is designed for programmers who:
- ✅ Know how to code (C++, Python, Java, etc.)
- ✅ Want to learn Swift (Apple's modern language)
- ✅ Want to learn Metal GPU programming
- ❌ Don't need to know Objective-C

By the end of this tutorial, you'll understand:
- Swift syntax and how it compares to C++/Objective-C
- How to set up a Metal rendering pipeline in Swift
- How to write GPU shaders in Metal Shading Language
- How to create a macOS window without Xcode
- Modern Swift features: optionals, closures, guards, property wrappers

---

## Table of Contents

1. [Quick Start](#quick-start)
2. [What This Program Does](#what-this-program-does)
3. [Prerequisites](#prerequisites)
4. [Why Swift for Metal?](#why-swift-for-metal)
5. [Swift Crash Course](#swift-crash-course)
6. [Code Walkthrough](#code-walkthrough)
7. [Comparing to Objective-C](#comparing-to-objective-c)
8. [Exercises](#exercises)
9. [Next Steps](#next-steps)
10. [Resources](#resources)

---

## Quick Start

```bash
# Navigate to this directory
cd whatever/swift_metal_for_beginners

# Build the project
make

# Run it
./SwiftMetal

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
- Identical visuals to the Objective-C version (same shaders!)

---

## What This Program Does

This is a comprehensive Swift + Metal tutorial that demonstrates:

1. **Creates a macOS window** (no Xcode IDE, pure code)
2. **Initializes Metal** (gets GPU, creates command queue)
3. **Compiles shaders** at runtime from `Compute.metal`
4. **Renders an animated gradient** 60 times per second
5. **Passes time data from CPU to GPU** (uniform buffers)
6. **Animates colors with math** (sin/cos for pulsing and cycling)
7. **Displays FPS in window title** (updated every second)
8. **Handles keyboard input** (Q/ESC to quit)

Total code: ~420 lines with extensive comments (~200 without comments)

**No dependencies** - just built-in macOS frameworks:
- Metal (GPU API)
- MetalKit (rendering helpers)
- AppKit (window/app management)
- QuartzCore (high-precision timing)

**Note**: The shader file (`Compute.metal`) is identical to the Objective-C version. Metal Shading Language works the same regardless of whether you use Swift or Objective-C!

---

## Prerequisites

### Required
- **macOS 10.14+** (Metal is macOS/iOS only)
- **Xcode Command Line Tools** (for swiftc compiler)

Install command line tools if needed:
```bash
xcode-select --install
```

You do **NOT** need the full Xcode IDE. Just the command line tools.

### Recommended Knowledge
- Basic programming (any language)
- Basic graphics concepts (what is a vertex, pixel, shader)
- Command line comfort (cd, running programs)

**No Swift experience needed** - this tutorial teaches Swift from scratch!

---

## Why Swift for Metal?

### Swift vs Objective-C

| Feature | Objective-C | Swift |
|---------|-------------|-------|
| **Syntax** | Square brackets `[obj method]` | Dot syntax `obj.method()` |
| **Type Safety** | Weak typing, runtime checks | Strong typing, compile-time checks |
| **Optionals** | nil crashes possible | Optionals prevent nil crashes |
| **Memory** | ARC (manual retain/release) | ARC (fully automatic) |
| **Verbosity** | More verbose | Concise, less boilerplate |
| **Error Handling** | NSError*, return values | do-try-catch, throws |
| **Closures** | Blocks `^{ }` | Closures `{ }` |
| **Learning Curve** | C/C++ + ObjC syntax | Just Swift |

### Why Choose Swift?

**Pros:**
- ✅ Cleaner, more readable code
- ✅ Safer (optionals, type checking)
- ✅ Modern language features
- ✅ Apple's preferred language (better future support)
- ✅ One language for iOS, macOS, visionOS
- ✅ Faster to write once you know it

**Cons:**
- ❌ Slightly slower compile times
- ❌ Some legacy code only in Objective-C
- ❌ Younger ecosystem (but growing fast)

**For new projects**: Use Swift. It's Apple's future.

---

## Swift Crash Course

### Variables: let vs var

```swift
// Immutable (like const in C++)
let x = 5        // Can't change
let name = "Bob"

// Mutable
var y = 10       // Can change
y = 20           // OK
```

**Rule of thumb**: Use `let` unless you need to mutate.

### Type Inference

```swift
let x = 5           // Compiler knows it's Int
let name = "Alice"  // Compiler knows it's String
let view = MTKView() // Compiler knows it's MTKView

// Or explicit types:
let x: Int = 5
let name: String = "Alice"
```

### Optionals (Handling nil Safely)

The biggest difference from C++ and most languages:

```swift
// Optional String (can be nil)
var name: String? = "Alice"
name = nil  // OK

// Regular String (CANNOT be nil)
var name: String = "Alice"
name = nil  // ❌ COMPILE ERROR

// Safely unwrap with if let
if let unwrapped = name {
    print(unwrapped)  // Only runs if name isn't nil
}

// Early exit with guard let
guard let unwrapped = name else {
    return  // Exit if name is nil
}
// unwrapped is available here

// Force unwrap (dangerous!)
let unwrapped = name!  // Crashes if name is nil
```

### Functions

```swift
// Basic function
func greet(name: String) -> String {
    return "Hello, \(name)!"
}

// Call it
greet(name: "Alice")  // "Hello, Alice!"

// Function with no external label
func greet(_ name: String) -> String {
    return "Hello, \(name)!"
}

greet("Bob")  // No label needed

// Multiple parameters
func add(a: Int, to b: Int) -> Int {
    return a + b
}

add(a: 5, to: 10)  // Named parameters
```

### Classes and Structs

```swift
// Class (reference type)
class MyClass {
    var property: Int = 0

    init() {
        // Constructor
    }

    func method() {
        print("Called")
    }
}

// Struct (value type)
struct MyStruct {
    var x: Int
    var y: Int
}

// Inheritance
class ChildClass: ParentClass {
    override func method() {
        super.method()  // Call parent
    }
}

// Protocol conformance
class MyClass: ParentClass, Protocol1, Protocol2 {
    // ...
}
```

### Closures (Like Lambdas)

```swift
// Basic closure
let sayHello = {
    print("Hello!")
}
sayHello()

// Closure with parameters
let add = { (a: Int, b: Int) -> Int in
    return a + b
}
add(5, 10)

// Trailing closure syntax
DispatchQueue.main.async {
    print("On main thread")
}

// Capturing variables
var count = 0
let increment = {
    count += 1
}
```

### Error Handling

```swift
// Function that can throw
func loadFile() throws -> String {
    // If something goes wrong:
    throw NSError(...)
}

// Call with do-try-catch
do {
    let contents = try loadFile()
    print(contents)
} catch {
    print("Error: \(error)")
}
```

### Guard Statements (Early Exit)

```swift
func process(name: String?) {
    // Guard: "This must be true, or exit early"
    guard let name = name else {
        return  // Exit if name is nil
    }

    // name is safely unwrapped here
    print(name)
}
```

---

## Code Walkthrough

### Main Structure

The Swift version has the same structure as the Objective-C version:

1. **InputView** - Handles keyboard input
2. **AppDelegate** - Manages app lifecycle
3. **Renderer** - Does all Metal work
4. **main code** - Sets up window and runs app

But the Swift code is **much more concise**!

### Key Differences from Objective-C

#### 1. Creating Objects

```objc
// Objective-C
MTKView* view = [[MTKView alloc] initWithFrame:frame];
```

```swift
// Swift
let view = MTKView(frame: frame)
```

#### 2. Method Calls

```objc
// Objective-C
[device newCommandQueue];
[device makeBufferWithLength:size options:options];
```

```swift
// Swift
device.makeCommandQueue()
device.makeBuffer(length: size, options: options)
```

#### 3. Optional Handling

```objc
// Objective-C
MTLDevice* device = MTLCreateSystemDefaultDevice();
if (!device) {
    NSLog(@"No Metal");
    return;
}
```

```swift
// Swift
guard let device = MTLCreateSystemDefaultDevice() else {
    print("No Metal")
    return
}
```

#### 4. Closures vs Blocks

```objc
// Objective-C block
dispatch_async(dispatch_get_main_queue(), ^{
    [self updateUI];
});
```

```swift
// Swift closure
DispatchQueue.main.async {
    self.updateUI()
}
```

### Renderer Initialization

```swift
init?(view: MTKView) {
    // Get GPU (returns nil if no Metal support)
    guard let device = MTLCreateSystemDefaultDevice() else {
        print("❌ No Metal")
        return nil  // Failable initializer
    }

    // Create command queue
    guard let commandQueue = device.makeCommandQueue() else {
        print("❌ No command queue")
        return nil
    }

    // Set properties
    self.device = device
    self.commandQueue = commandQueue
    self.view = view

    // Initialize timing
    let currentTime = CACurrentMediaTime()
    self.startTime = currentTime
    self.lastFrameTime = currentTime
    self.fpsUpdateTime = currentTime

    // MUST call super.init before using self
    super.init()

    // Load shaders (can fail)
    guard loadShaders() else {
        return nil
    }

    // ... rest of initialization
}
```

### Render Loop

```swift
func draw(in view: MTKView) {
    // Guard: make sure we have resources
    guard let pipeline = pipeline,
          let uniformBuffer = uniformBuffer else {
        return
    }

    // Update timing
    let currentTime = CACurrentMediaTime()
    frameCount += 1

    // Update uniforms
    let pointer = uniformBuffer.contents().bindMemory(
        to: Uniforms.self,
        capacity: 1
    )
    pointer.pointee.time = Float(elapsedTime)

    // Create command buffer
    guard let commandBuffer = commandQueue.makeCommandBuffer() else {
        return
    }

    // Encode and submit
    // ...
}
```

---

## Comparing to Objective-C

This project has a **nearly identical** Objective-C version at `../metal_for_beginners/`.

### Line Count Comparison

| Version | With Comments | Without Comments |
|---------|---------------|------------------|
| Objective-C++ | ~800 lines | ~330 lines |
| Swift | ~420 lines | ~200 lines |

**Swift is 50% more concise!**

### Feature Comparison

| Feature | Objective-C | Swift | Winner |
|---------|-------------|-------|--------|
| **Syntax clarity** | `[obj method:arg]` | `obj.method(arg)` | Swift |
| **Type safety** | Weak | Strong | Swift |
| **Compile time** | Faster | Slower | ObjC |
| **Runtime speed** | Same | Same | Tie |
| **Learning curve** | Steeper (two syntaxes) | Gentler | Swift |
| **Legacy support** | Better | Limited | ObjC |
| **Future support** | Maintenance | Active dev | Swift |

### Code Comparison Examples

**Creating a buffer:**

```objc
// Objective-C (19 characters)
[device newBufferWithLength:size options:MTLResourceStorageModeShared];
```

```swift
// Swift (17 characters)
device.makeBuffer(length: size, options: .storageModeShared)
```

**Handling errors:**

```objc
// Objective-C
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

**Recommendation**: Study both versions side-by-side to see how the same Metal concepts are expressed differently!

---

## Exercises

### Easy

1. **Change animation speed**
   - In `Compute.metal`, line 251, change `uniforms.time * 2.0` to `* 5.0`

2. **Modify colors**
   - In `Compute.metal`, lines 209-213, change the RGB values

3. **Experiment with optionals**
   - In `main.swift`, try force-unwrapping (!) vs safe unwrapping (if let)
   - See what happens if you force-unwrap a nil

4. **Print debug info**
   - Add `print("Frame: \(frameCount)")` in the render loop
   - See how fast it prints (hint: very fast!)

### Medium

5. **Convert to struct**
   - Try converting `Renderer` from class to struct
   - See what errors you get (hint: protocols require class)

6. **Add a new uniform**
   - Add `mouseX: Float` to `Uniforms` struct
   - Update both Swift and Metal sides
   - Pass mouse position to shader

7. **Use lazy properties**
   - Make some properties `lazy var` to defer initialization
   - See when they're actually created

8. **Add error types**
   - Define a custom error enum
   - Throw specific errors instead of returning nil

### Advanced

9. **Compare performance**
   - Build with different optimization levels (-Onone vs -O)
   - Compare FPS between Swift and Objective-C versions
   - Profile with Instruments

10. **Add protocol extensions**
    - Extend MTLDevice with helper methods
    - Use Swift's protocol extensions feature

11. **Convert to Swift Package**
    - Create a Package.swift file
    - Convert this to a Swift Package Manager project
    - Add dependencies if needed

---

## Next Steps

### Learn More Swift

1. **Official Swift Book**
   - Free from Apple: https://docs.swift.org/swift-book/
   - Comprehensive and well-written

2. **Swift Playgrounds**
   - Interactive learning environment
   - Great for experimenting with syntax

3. **Stanford CS193p**
   - Free iOS development course using Swift
   - Excellent for learning Swift + SwiftUI

### Learn More Metal

1. **Study the Objective-C version**
   - Same Metal API, different syntax
   - Compare and contrast

2. **Metal by Example**
   - Excellent book: https://metalbyexample.com/
   - Has Swift examples

3. **Apple Sample Code**
   - Search "Metal Swift" on Apple Developer
   - Real-world examples from Apple

### Build Something!

Ideas for projects:
- **Particle system** - Thousands of particles with physics
- **Image filters** - Apply effects to photos
- **Fractals** - Mandelbrot set, Julia sets
- **Raytracer** - Simple ray tracing on GPU
- **Fluid simulation** - Navier-Stokes on GPU
- **Game** - 2D game with Metal rendering

---

## Resources

### Swift Resources
- [Swift.org](https://swift.org/) - Official Swift website
- [Swift Book](https://docs.swift.org/swift-book/) - Free official book
- [Swift Forums](https://forums.swift.org/) - Community discussion
- [Hacking with Swift](https://www.hackingwithswift.com/) - Excellent tutorials

### Metal Resources
- [Metal Documentation](https://developer.apple.com/documentation/metal)
- [Metal Shading Language Spec](https://developer.apple.com/metal/Metal-Shading-Language-Specification.pdf)
- [Metal by Example](https://metalbyexample.com/)
- [Metal Sample Code](https://developer.apple.com/documentation/metal/metal_sample_code_library)

### Related Projects in This Repo
- `../metal_for_beginners/` - Objective-C version (compare!)
- `../metal_particle_template/` - More advanced particle simulation
- `../metal_lumen/` - Complex particle system
- `../metal_physarum/` - Slime mold simulation

---

## Troubleshooting

### Build Errors

**"swiftc: command not found"**
- Install Xcode Command Line Tools: `xcode-select --install`

**"Could not load shader file"**
- Make sure `Compute.metal` is in the current directory
- Run from project directory: `cd swift_metal_for_beginners && ./SwiftMetal`

**"Module compiled with Swift X.Y cannot be imported by Swift X.Z"**
- Your Swift version might be incompatible
- Update Xcode or specify Swift version: `swiftc -swift-version 5`

### Runtime Errors

**"Unexpectedly found nil while unwrapping an Optional value"**
- You force-unwrapped (!) a nil optional
- Use `if let` or `guard let` instead
- Check the stack trace to find which line

**"Metal is not supported on this Mac"**
- Your Mac is too old (pre-2012 Macs lack Metal)
- No workaround - Metal requires compatible GPU

**Low FPS**
- Check Activity Monitor → GPU usage
- Make sure you're not running in debug mode (use -O flag)
- Older Macs might struggle with 60 FPS

### Compilation is Slow

- Swift can be slow with complex type inference
- Break complex expressions into multiple statements
- Add explicit type annotations
- Use `swiftc -Onone` for faster debug builds

---

## Summary

You learned:

✅ **Swift syntax** - Variables, optionals, functions, closures, classes
✅ **Swift vs Objective-C** - Same Metal API, cleaner syntax
✅ **Metal with Swift** - How to call Metal APIs from Swift
✅ **Modern Swift features** - Guard, optionals, do-try-catch
✅ **Command-line Swift** - Build without Xcode IDE

**Key Insight**: The Metal API is **identical** in Swift and Objective-C. The shaders are exactly the same. Only the host language syntax changes.

**Recommendation**: If you're starting fresh, use Swift. It's cleaner, safer, and Apple's future. But knowing Objective-C helps for reading legacy code.

Happy GPU programming in Swift! 🚀
