# =============================================================================
# Metal Particle Template - Makefile
# =============================================================================
# This Makefile compiles an Objective-C/Metal application for macOS without
# requiring Xcode IDE. It uses Apple's clang compiler directly.
#
# Usage:
#   make        - Build the application
#   make run    - Build and run the application
#   make clean  - Remove compiled binary
#   make debug  - Build with debug symbols (no optimization)
# =============================================================================

# -----------------------------------------------------------------------------
# Compiler Configuration
# -----------------------------------------------------------------------------
# CXX: We use clang++ (Apple's C++ compiler) which also handles Objective-C++
# The .mm extension tells the compiler to treat the file as Objective-C++
CXX = clang++

# -----------------------------------------------------------------------------
# Compiler Flags
# -----------------------------------------------------------------------------
# -O3           : Maximum optimization level (fastest runtime, longer compile)
# -std=c++17    : Enable C++17 features (structured bindings, if constexpr, etc.)
# -fobjc-arc    : Enable Automatic Reference Counting for Objective-C objects
#                 This automatically manages memory for Objective-C objects,
#                 inserting retain/release calls as needed
# -Wall         : Enable all common warnings
# -Wextra       : Enable extra warnings
CXXFLAGS = -O3 -std=c++17 -fobjc-arc -Wall -Wextra

# Debug flags (used by 'make debug')
DEBUGFLAGS = -g -O0 -std=c++17 -fobjc-arc -Wall -Wextra -DDEBUG

# -----------------------------------------------------------------------------
# Linker Flags (Frameworks)
# -----------------------------------------------------------------------------
# macOS uses "frameworks" to bundle headers, libraries, and resources.
# We need these three frameworks for a Metal application:
#
# -framework Metal     : Core Metal API for GPU programming
#                        Provides: MTLDevice, MTLBuffer, MTLCommandQueue, etc.
#
# -framework MetalKit  : Higher-level Metal utilities
#                        Provides: MTKView (drawable Metal view with render loop)
#
# -framework AppKit    : macOS application framework (Cocoa)
#                        Provides: NSApplication, NSWindow, NSView, event handling
#
# -framework QuartzCore: Core Animation framework (optional, for advanced timing)
#                        Provides: CADisplayLink, CAMetalLayer
LDFLAGS = -framework Metal -framework MetalKit -framework AppKit -framework QuartzCore

# -----------------------------------------------------------------------------
# Project Files
# -----------------------------------------------------------------------------
TARGET = MetalNeuralAura
SRCS = main.mm

# Metal shaders are compiled at runtime from source, not at build time.
# This variable is for documentation purposes.
SHADERS = Compute.metal

# -----------------------------------------------------------------------------
# Build Rules
# -----------------------------------------------------------------------------

# Default target: build the application
all: $(TARGET)

# Main build rule
# $@ = target name (MetalParticles)
# $< = first prerequisite (main.mm)
$(TARGET): $(SRCS) $(SHADERS)
	@echo "Building $(TARGET)..."
	$(CXX) $(CXXFLAGS) $(SRCS) -o $(TARGET) $(LDFLAGS)
	@echo "Build complete: ./$(TARGET)"

# Debug build (with symbols, no optimization)
debug: $(SRCS) $(SHADERS)
	@echo "Building $(TARGET) with debug symbols..."
	$(CXX) $(DEBUGFLAGS) $(SRCS) -o $(TARGET) $(LDFLAGS)
	@echo "Debug build complete: ./$(TARGET)"

# Clean build artifacts
clean:
	@echo "Cleaning..."
	rm -f $(TARGET)
	rm -rf $(TARGET).dSYM
	@echo "Clean complete."

# Build and run
run: all
	@echo "Running $(TARGET)..."
	./$(TARGET)

# Phony targets (not files)
.PHONY: all clean run debug
