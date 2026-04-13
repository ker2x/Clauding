# Makefile for Metal Plasma Field Simulation

TARGET = PlasmaField
SOURCES = main.mm
FRAMEWORKS = -framework Metal -framework MetalKit -framework AppKit -framework QuartzCore
LDFLAGS = $(FRAMEWORKS) -fobjc-link-runtime -lz

# Compiler
CXX = clang++

# Build flags
CXXFLAGS = -O3 -std=c++17 -Wall -fobjc-arc

.PHONY: all clean run

all: $(TARGET)

$(TARGET): $(SOURCES)
	$(CXX) $(CXXFLAGS) -o $@ $(SOURCES) $(LDFLAGS)

run: $(TARGET)
	./$(TARGET)

clean:
	rm -f $(TARGET)
