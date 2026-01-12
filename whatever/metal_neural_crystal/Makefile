CC=clang++
CFLAGS=-O3 -std=c++17 -fobjc-arc
FRAMEWORKS=-framework AppKit -framework Metal -framework MetalKit -framework Foundation -framework QuartzCore
SOURCES=main.mm
EXECUTABLE=NeuralCrystal

all: $(EXECUTABLE)

$(EXECUTABLE): $(SOURCES)
	$(CC) $(CFLAGS) $(SOURCES) $(FRAMEWORKS) -o $(EXECUTABLE)

clean:
	rm -f $(EXECUTABLE)
