# =============================================================================
# Mandelbulb Path Tracer - Makefile
# =============================================================================
# Builds the Mandelbulb path tracing application for macOS using Metal.
#
# Usage:
#   make        - Build the application
#   make run    - Build and run
#   make clean  - Remove compiled binary
#   make debug  - Build with debug symbols
# =============================================================================

# Compiler
CXX = clang++

# Compiler flags
# -O3           : Maximum optimization (important for path tracing performance)
# -std=c++17    : C++17 features
# -fobjc-arc    : Automatic Reference Counting
# -Wall -Wextra : All warnings
CXXFLAGS = -O3 -std=c++17 -fobjc-arc -Wall -Wextra

# Debug flags
DEBUGFLAGS = -g -O0 -std=c++17 -fobjc-arc -Wall -Wextra -DDEBUG

# Frameworks
LDFLAGS = -framework Metal -framework MetalKit -framework AppKit -framework QuartzCore

# Project files
TARGET = MandelbulbPathTracer
SRCS = main.mm
SHADERS = Shader.metal

# Build rules
all: $(TARGET)

$(TARGET): $(SRCS) $(SHADERS)
	@echo "Building $(TARGET)..."
	$(CXX) $(CXXFLAGS) $(SRCS) -o $(TARGET) $(LDFLAGS)
	@echo "Build complete: ./$(TARGET)"

debug: $(SRCS) $(SHADERS)
	@echo "Building $(TARGET) with debug symbols..."
	$(CXX) $(DEBUGFLAGS) $(SRCS) -o $(TARGET) $(LDFLAGS)
	@echo "Debug build complete: ./$(TARGET)"

clean:
	@echo "Cleaning..."
	rm -f $(TARGET)
	rm -rf $(TARGET).dSYM
	@echo "Clean complete."

run: all
	@echo "Running $(TARGET)..."
	./$(TARGET)

.PHONY: all clean run debug
