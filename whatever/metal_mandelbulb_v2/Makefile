# =============================================================================
# Mandelbulb Path Tracer v2 - Makefile
# =============================================================================
# Builds with MetalFX framework for ML-based upscaling.
# Requires macOS 13.0+ (Ventura) for MetalFX.
# =============================================================================

CXX = clang++

# Note: -mmacosx-version-min=13.0 required for MetalFX
CXXFLAGS = -O3 -std=c++17 -fobjc-arc -Wall -Wextra -mmacosx-version-min=13.0

DEBUGFLAGS = -g -O0 -std=c++17 -fobjc-arc -Wall -Wextra -mmacosx-version-min=13.0 -DDEBUG

# MetalFX framework added
LDFLAGS = -framework Metal -framework MetalKit -framework MetalFX -framework AppKit -framework QuartzCore

TARGET = MandelbulbV2
SRCS = main.mm
SHADERS = Shader.metal

all: $(TARGET)

$(TARGET): $(SRCS) $(SHADERS)
	@echo "Building $(TARGET) with MetalFX..."
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
