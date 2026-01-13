# =============================================================================
# Makefile for Metal Beginners Project
# =============================================================================
# This Makefile compiles Objective-C++ code with Metal on macOS.
# If you're coming from C++, Makefiles work the same way!
#
# Quick Start:
#   make        - Build the program
#   make run    - Build and run
#   make clean  - Delete the built executable
#   make debug  - Build with debug symbols (for lldb/gdb debugging)
#
# =============================================================================

# -----------------------------------------------------------------------------
# MAKEFILE BASICS (if you're new to Make)
# -----------------------------------------------------------------------------
# A Makefile contains "rules" that look like:
#
#   target: dependencies
#       command
#
# Example:
#   program: main.cpp utils.cpp
#       g++ main.cpp utils.cpp -o program
#
# When you run "make program", Make will:
# 1. Check if main.cpp or utils.cpp changed
# 2. If yes, run the command to rebuild
# 3. If no, skip (already up to date)
#
# Variables:
#   VAR = value      - Define a variable
#   $(VAR)           - Use the variable
#
# Special variables:
#   $@    - The target name (left side of :)
#   $<    - First dependency
#   $^    - All dependencies
#
# .PHONY: Marks targets that aren't real files (like "clean" or "run")
# -----------------------------------------------------------------------------

# -----------------------------------------------------------------------------
# COMPILER SELECTION
# -----------------------------------------------------------------------------
# For macOS, we use Apple's clang++ compiler (part of Xcode Command Line Tools)
# Install with: xcode-select --install

CXX = clang++
# CXX is the conventional variable name for C++ compiler
# Make uses this automatically for .cpp/.mm files

# -----------------------------------------------------------------------------
# COMPILER FLAGS EXPLAINED
# -----------------------------------------------------------------------------
# These flags control how the compiler builds your code.
# Each flag is explained below:

CXXFLAGS = -O2 -std=c++17 -fobjc-arc -Wall -Wextra -Wpedantic

# Let's break down what each flag does:
#
# -O2
#   Optimization level 2 (good balance of speed and compile time)
#   O0 = no optimization (fastest compile, slowest runtime)
#   O1 = basic optimization
#   O2 = moderate optimization (recommended for development)
#   O3 = aggressive optimization (best for release builds)
#
# -std=c++17
#   Use C++17 language standard
#   Enables: auto, range-for, std::optional, if constexpr, etc.
#   You can change to c++20 or c++23 if you want newer features
#
# -fobjc-arc
#   Enable Automatic Reference Counting for Objective-C objects
#   This is CRITICAL for Objective-C/Objective-C++ code!
#   Without this, you'd have to manually call [obj retain]/[obj release]
#   With this, the compiler adds memory management code automatically
#   Think: automatic std::shared_ptr for all Objective-C objects
#
# -Wall
#   Enable all common warnings
#   Examples: unused variables, missing return statements, etc.
#
# -Wextra
#   Enable additional warnings beyond -Wall
#   Examples: comparing signed/unsigned, unused parameters, etc.
#
# -Wpedantic
#   Warn about code that doesn't follow the C++ standard strictly
#   Good for learning proper C++ style

# Debug flags (used by "make debug")
DEBUGFLAGS = -g -O0 -std=c++17 -fobjc-arc -Wall -Wextra -Wpedantic -DDEBUG

# -g        Add debug symbols (for debugger like lldb)
# -O0       No optimization (easier to debug)
# -DDEBUG   Define DEBUG preprocessor macro (#ifdef DEBUG will be true)

# -----------------------------------------------------------------------------
# LINKER FLAGS - Connecting to macOS Frameworks
# -----------------------------------------------------------------------------
# macOS uses "frameworks" - bundles of code, headers, and resources.
# Think of them like .dll files on Windows or .so files on Linux.
#
# To use a framework, we need to tell the linker with -framework

LDFLAGS = -framework Metal -framework MetalKit -framework AppKit -framework QuartzCore

# Let's explain each framework:
#
# -framework Metal
#   The core GPU programming API
#   Provides: MTLDevice, MTLBuffer, MTLCommandQueue, MTLTexture, etc.
#   This is like Vulkan or DirectX - low-level GPU control
#
# -framework MetalKit
#   Higher-level utilities built on Metal
#   Provides: MTKView (a view that renders Metal content with a built-in loop)
#   Handles drawable management, timing, and view setup
#
# -framework AppKit
#   The macOS UI framework (also called "Cocoa")
#   Provides: NSWindow, NSApplication, NSView, NSEvent, etc.
#   This is like Win32 API on Windows or Xlib on Linux
#   Note: iOS uses UIKit instead of AppKit
#
# -framework QuartzCore
#   Core Animation and graphics timing
#   Provides: CACurrentMediaTime (high-precision time for animations)
#   Used for smooth, time-based animations

# -----------------------------------------------------------------------------
# PROJECT CONFIGURATION
# -----------------------------------------------------------------------------
# These variables define what we're building

TARGET = MetalBeginner
# Name of the executable we'll create

SRCS = main.mm
# Source files to compile
# .mm = Objective-C++ (mix of C++ and Objective-C)
# If you had multiple files: SRCS = main.mm utils.mm renderer.mm

SHADERS = Compute.metal
# Shader files (not compiled by Make - Metal compiles them at runtime)
# We list them here so Make knows to rebuild if shaders change

# -----------------------------------------------------------------------------
# BUILD RULES
# -----------------------------------------------------------------------------
# These tell Make how to build the project

# Default target (runs when you type just "make")
all: $(TARGET)
	@echo ""
	@echo "✓ Build complete!"
	@echo "  Run with: ./$(TARGET)"
	@echo "  Or use:   make run"
	@echo ""

# Main build rule
# Pattern: target: dependencies
#          command
$(TARGET): $(SRCS) $(SHADERS)
	@echo "🔨 Compiling $(SRCS)..."
	$(CXX) $(CXXFLAGS) $(SRCS) -o $(TARGET) $(LDFLAGS)
	@echo "🔗 Linking with frameworks: Metal, MetalKit, AppKit"

# Explanation:
# $(TARGET) = MetalBeginner (the file we're creating)
# $(SRCS) = main.mm (if this changes, rebuild)
# $(SHADERS) = Compute.metal (if this changes, rebuild)
#
# The command does:
# 1. clang++ compiles main.mm
# 2. -O2 -std=c++17 ... are the compile flags
# 3. -o MetalBeginner sets the output name
# 4. -framework Metal -framework ... links the frameworks
#
# @ prefix = don't print the command itself, just the echo
# Without @: you'd see "echo Building..."
# With @:    you just see "Building..."

# Build with debug symbols (for debugging with lldb)
debug: $(SRCS) $(SHADERS)
	@echo "🐛 Building with debug symbols..."
	$(CXX) $(DEBUGFLAGS) $(SRCS) -o $(TARGET) $(LDFLAGS)
	@echo "✓ Debug build complete"
	@echo "  Debug with: lldb ./$(TARGET)"

# Clean up compiled files
clean:
	@echo "🧹 Cleaning build artifacts..."
	rm -f $(TARGET)
	rm -rf $(TARGET).dSYM
	@echo "✓ Clean complete"

# Explanation:
# rm -f $(TARGET)          Remove the executable (if it exists)
# rm -rf $(TARGET).dSYM    Remove debug symbols directory
# -f = "force" (don't error if file doesn't exist)
# -r = "recursive" (for directories)

# Build and run in one command
run: all
	@echo "▶️  Running $(TARGET)..."
	@echo ""
	./$(TARGET)

# Explanation:
# run depends on "all", so it will build first if needed
# Then it runs the executable with ./$(TARGET)

# Help command (shows available targets)
help:
	@echo "Available commands:"
	@echo "  make        - Build the project"
	@echo "  make run    - Build and run"
	@echo "  make clean  - Remove built files"
	@echo "  make debug  - Build with debug symbols"
	@echo "  make help   - Show this help message"

# -----------------------------------------------------------------------------
# PHONY TARGETS
# -----------------------------------------------------------------------------
# These are targets that don't create files.
# Without .PHONY, if you had a file named "clean", "make clean" wouldn't work!

.PHONY: all clean run debug help

# =============================================================================
# BUILDING AND RUNNING - STEP BY STEP
# =============================================================================
# Let's walk through what happens when you type "make run":
#
# 1. Make reads this Makefile
# 2. You asked for target "run"
# 3. "run" depends on "all"
# 4. "all" depends on $(TARGET) which is "MetalBeginner"
# 5. "MetalBeginner" depends on main.mm and Compute.metal
# 6. Make checks: do main.mm or Compute.metal have changes?
#    - If YES: run the compile command
#    - If NO:  skip to next step
# 7. After building, execute the "run" command: ./MetalBeginner
# 8. Your program runs!
#
# =============================================================================

# =============================================================================
# TROUBLESHOOTING
# =============================================================================
#
# Error: "make: command not found"
# → Install Xcode Command Line Tools: xcode-select --install
#
# Error: "clang: command not found"
# → Install Xcode Command Line Tools: xcode-select --install
#
# Error: "Metal/Metal.h: No such file or directory"
# → Metal only works on macOS. Make sure you're on a Mac.
# → Also ensure your macOS is 10.14+ (Metal requirement)
#
# Error: "ld: framework not found Metal"
# → Check you have /System/Library/Frameworks/Metal.framework
# → Update macOS if needed
#
# Warning: "unused variable"
# → Suppress with: (void)variableName;
# → Or remove the -Wall flag (not recommended)
#
# Slow compilation?
# → Change -O2 to -O0 for faster builds (slower runtime)
# → Remove -Wpedantic if you don't need strict warnings
#
# Want to see what the compiler is actually doing?
# → Remove @ symbols from commands
# → Run: make clean && make
# → You'll see the full clang++ command
#
# =============================================================================

# =============================================================================
# EXERCISES
# =============================================================================
#
# 1. CHANGE OPTIMIZATION LEVEL
#    - Change -O2 to -O3 and rebuild
#    - Notice any difference in compile time?
#
# 2. ADD ANOTHER SOURCE FILE
#    - Create utils.mm with a helper function
#    - Add it to SRCS: SRCS = main.mm utils.mm
#    - Run make - both files will be compiled
#
# 3. TRY C++20 FEATURES
#    - Change -std=c++17 to -std=c++20
#    - Use C++20 features like std::format, concepts, etc.
#
# 4. ENABLE SANITIZERS (for debugging)
#    - Add to DEBUGFLAGS: -fsanitize=address -fsanitize=undefined
#    - This catches memory bugs, buffer overflows, etc.
#
# 5. PROFILE COMPILATION TIME
#    - Add -ftime-report to see where compilation time is spent
#    - Add -ftime-trace to generate a Chrome tracing JSON
#
# 6. STUDY THE PARTICLE TEMPLATE
#    - Look at metal_particle_template/Makefile
#    - It's similar but adds more features
#    - Compare and see what's different!
#
# =============================================================================
