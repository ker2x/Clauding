# =============================================================================
# Makefile for Swift + Metal Beginners Project
# =============================================================================
# This Makefile compiles Swift code with Metal on macOS.
#
# Quick Start:
#   make        - Build the program
#   make run    - Build and run
#   make clean  - Delete the built executable
#
# =============================================================================

# -----------------------------------------------------------------------------
# COMPILER CONFIGURATION
# -----------------------------------------------------------------------------
# Swift uses 'swiftc' (Swift compiler) instead of clang++
SWIFTC = swiftc

# -----------------------------------------------------------------------------
# COMPILER FLAGS EXPLAINED
# -----------------------------------------------------------------------------
# Swift compilation is simpler than Objective-C++ - fewer flags needed!

SWIFTFLAGS = -O

# Let's explain the flag:
#
# -O
#   Optimize for speed (release mode)
#   -Onone = no optimization (fastest compile, slowest runtime)
#   -O     = optimize for speed (recommended for release)
#   -Osize = optimize for binary size

# -----------------------------------------------------------------------------
# LINKER FLAGS - Connecting to macOS Frameworks
# -----------------------------------------------------------------------------
# Swift uses -framework just like Objective-C

LDFLAGS = -framework Metal -framework MetalKit -framework AppKit -framework QuartzCore

# Framework explanations (same as Objective-C version):
#
# -framework Metal
#   The core GPU programming API
#   Provides: MTLDevice, MTLBuffer, MTLCommandQueue, etc.
#
# -framework MetalKit
#   Higher-level Metal utilities
#   Provides: MTKView with built-in render loop
#
# -framework AppKit
#   The macOS UI framework (Cocoa)
#   Provides: NSWindow, NSApplication, NSView, etc.
#
# -framework QuartzCore
#   Core Animation and timing
#   Provides: CACurrentMediaTime for high-precision timing

# -----------------------------------------------------------------------------
# PROJECT CONFIGURATION
# -----------------------------------------------------------------------------
TARGET = SwiftMetal
SRCS = main.swift
SHADERS = Compute.metal

# -----------------------------------------------------------------------------
# BUILD RULES
# -----------------------------------------------------------------------------

# Default target: build the application
all: $(TARGET)
	@echo ""
	@echo "✓ Build complete!"
	@echo "  Run with: ./$(TARGET)"
	@echo "  Or use:   make run"
	@echo ""

# Main build rule
$(TARGET): $(SRCS) $(SHADERS)
	@echo "🔨 Compiling $(SRCS)..."
	$(SWIFTC) $(SWIFTFLAGS) $(SRCS) -o $(TARGET) $(LDFLAGS)
	@echo "🔗 Linking with frameworks: Metal, MetalKit, AppKit, QuartzCore"

# Clean build artifacts
clean:
	@echo "🧹 Cleaning..."
	rm -f $(TARGET)
	rm -rf $(TARGET).dSYM
	@echo "✓ Clean complete."

# Build and run
run: all
	@echo "▶️  Running $(TARGET)..."
	@echo ""
	./$(TARGET)

# Help command
help:
	@echo "Available commands:"
	@echo "  make        - Build the project"
	@echo "  make run    - Build and run"
	@echo "  make clean  - Remove built files"
	@echo "  make help   - Show this help message"

# Phony targets (not files)
.PHONY: all clean run help

# =============================================================================
# SWIFT vs OBJECTIVE-C++ COMPILATION
# =============================================================================
# Key differences when compiling Swift:
#
# 1. COMPILER
#    ObjC++: clang++ (C/C++/Objective-C compiler)
#    Swift:  swiftc (Swift-specific compiler)
#
# 2. FLAGS
#    ObjC++: -std=c++17, -fobjc-arc, -Wall, -Wextra
#    Swift:  Much simpler! Just -O for optimization
#            No need for ARC flag (always on in Swift)
#            No C++ version flag (Swift is its own language)
#
# 3. FILE EXTENSIONS
#    ObjC++: .mm (Objective-C++)
#    Swift:  .swift
#
# 4. FRAMEWORKS
#    Same! Both use -framework Metal, MetalKit, AppKit
#    Swift can call Objective-C APIs directly
#
# 5. COMPILATION SPEED
#    Swift: Can be slower to compile (more type checking)
#    But: Incremental compilation helps with large projects
#
# =============================================================================

# =============================================================================
# TROUBLESHOOTING
# =============================================================================
#
# Error: "swiftc: command not found"
# → Install Xcode Command Line Tools: xcode-select --install
#
# Error: "Could not load shader file"
# → Make sure Compute.metal is in the same directory
# → Run from the project directory: cd swift_metal_for_beginners && ./SwiftMetal
#
# Compilation is slow?
# → Swift type checking can be slow in complex expressions
# → Break down complex expressions into multiple statements
# → Use explicit types instead of relying on inference in complex cases
#
# Want faster debug builds?
# → Change -O to -Onone for debug builds
# → Add -g for debug symbols: make debug
#
# =============================================================================

# =============================================================================
# EXERCISES
# =============================================================================
#
# 1. COMPARE BUILD TIMES
#    - make clean && time make
#    - Compare to the Objective-C version
#    - Swift might be slightly slower to compile but the binary is similar
#
# 2. TRY DIFFERENT OPTIMIZATION LEVELS
#    - Change -O to -Onone (no optimization)
#    - Change -O to -Osize (optimize for size)
#    - Compare binary sizes and performance
#
# 3. ADD MORE SWIFT FILES
#    - Split the code into multiple .swift files
#    - Add them to SRCS: SRCS = main.swift utils.swift
#    - Swift automatically handles cross-file dependencies
#
# 4. USE SWIFT PACKAGE MANAGER (advanced)
#    - Create a Package.swift file
#    - Convert this Makefile project to SPM
#    - SPM handles dependencies and builds automatically
#
# =============================================================================
