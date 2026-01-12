CXX = clang++
CXXFLAGS = -O3 -std=c++17 -fobjc-arc
LDFLAGS = -framework Metal -framework MetalKit -framework AppKit -framework CoreGraphics

TARGET = MetalPhysarum
SRCS = main.mm
METALS = Physarum.metal

all: $(TARGET)

$(TARGET): $(SRCS)
	$(CXX) $(CXXFLAGS) $(SRCS) -o $(TARGET) $(LDFLAGS)

clean:
	rm -f $(TARGET)

run: all
	./$(TARGET)
