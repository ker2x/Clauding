# =============================================================================
# Swift Particle Template - Makefile
# =============================================================================
# This Makefile compiles a Swift/Metal application for macOS without requiring
# Xcode IDE. It uses Apple's swiftc compiler directly.
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
SWIFT = swiftc

# -----------------------------------------------------------------------------
# Compiler Flags
# -----------------------------------------------------------------------------
# -O              : Optimize for speed (release build)
# -whole-module-optimization : Optimize across entire module (faster runtime)
# -suppress-warnings : Suppress all warnings for cleaner output (optional)
SWIFTFLAGS = -O -whole-module-optimization

# Debug flags (used by 'make debug')
DEBUGFLAGS = -g -Onone

# -----------------------------------------------------------------------------
# Linker Flags (Frameworks)
# -----------------------------------------------------------------------------
# We need these frameworks for a Metal application:
# - Metal      : Core Metal API for GPU programming
# - MetalKit   : MTKView and Metal utilities
# - AppKit     : macOS application framework
# - QuartzCore : High-precision timing
FRAMEWORKS = -framework Metal -framework MetalKit -framework AppKit -framework QuartzCore

# -----------------------------------------------------------------------------
# Project Files
# -----------------------------------------------------------------------------
TARGET = SwiftParticles
SRCS = main.swift

# Metal shaders are compiled at runtime from source, not at build time
SHADERS = Compute.metal

# -----------------------------------------------------------------------------
# Build Rules
# -----------------------------------------------------------------------------

# Default target: build the application
all: $(TARGET)

# Main build rule
$(TARGET): $(SRCS) $(SHADERS)
	@echo "Building $(TARGET)..."
	$(SWIFT) $(SWIFTFLAGS) $(SRCS) -o $(TARGET) $(FRAMEWORKS)
	@echo "Build complete: ./$(TARGET)"

# Debug build (with symbols, no optimization)
debug: $(SRCS) $(SHADERS)
	@echo "Building $(TARGET) with debug symbols..."
	$(SWIFT) $(DEBUGFLAGS) $(SRCS) -o $(TARGET) $(FRAMEWORKS)
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
