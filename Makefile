# Compilers
NVCC = nvcc
CCBIN = gcc-10
CXX = g++

# Flags
NVCCFLAGS = -O2 -g -std=c++17 -Xcompiler "-Wall -Wextra -Werror" -lineinfo -I/usr/include/SDL2 -Iimgui -Iimgui/backends
CXXFLAGS = -O2 -g -Wall -Wextra -Werror -I/usr/include/SDL2 -Iimgui -Iimgui/backends

# Linker flags
LDFLAGS = -lstdc++ -lm -lSDL2 -lGL

# Sources
SRC = $(wildcard src/*.cu)
IMGUI_SRC = \
    imgui/imgui.cpp \
    imgui/imgui_demo.cpp \
    imgui/imgui_draw.cpp \
    imgui/imgui_tables.cpp \
    imgui/imgui_widgets.cpp \
    imgui/backends/imgui_impl_sdl2.cpp \
    imgui/backends/imgui_impl_opengl3.cpp

# Object files
OBJ = $(SRC:.cu=.o)
IMGUI_OBJ = $(IMGUI_SRC:.cpp=.o)

# Target
TARGET = raytracer
OUTPUT_NAME = output

# Default target
all: $(TARGET)

# Compile CUDA files
%.o: %.cu
	$(NVCC) -ccbin $(CCBIN) $(NVCCFLAGS) -c $< -o $@

# Compile C++ (ImGui) files
%.o: %.cpp
	$(CXX) $(CXXFLAGS) -c $< -o $@

# Link all objects
$(TARGET): $(OBJ) $(IMGUI_OBJ)
	$(NVCC) -ccbin $(CCBIN) $^ -o $@ $(LDFLAGS)

# Run targets
run: all
	./$(TARGET)

run-image: all
	./$(TARGET) --image
	convert $(OUTPUT_NAME).ppm $(OUTPUT_NAME).png

run-video: all
	./$(TARGET) --video
	convert -delay 5 -loop 0 frame_*.ppm camera_rotation.gif

# Clean
clean:
	rm -f $(TARGET) $(OUTPUT_NAME).* frame_* *.o **/*.o imgui.ini
