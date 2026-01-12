CXX = clang++
CXXFLAGS = -O3 -std=c++17 -I/opt/homebrew/include
LDFLAGS = -L/opt/homebrew/lib -lSDL2

TARGET = lumen_particles
SRCS = main.cpp

$(TARGET): $(SRCS)
	$(CXX) $(CXXFLAGS) $(SRCS) -o $(TARGET) $(LDFLAGS)

clean:
	rm -f $(TARGET)

run: $(TARGET)
	./$(TARGET)
