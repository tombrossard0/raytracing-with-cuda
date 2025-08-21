# Compilers
NVCC = nvcc
CCBIN = gcc-10
CXX = g++

# Flags
NVCCFLAGS = -O2 -g -std=c++17 -Xcompiler "-Wall -Wextra -Werror -Wno-error=cpp" -lineinfo -I/usr/include/SDL2 -Iimgui -Iimgui/backends
CXXFLAGS = -O2 -g -Wall -Wextra -Wpedantic -Werror -I/usr/include/SDL2 -Iimgui -Iimgui/backends

# Linker flags
LDFLAGS = -lstdc++ -lm -lSDL2 -lGL -lassimp

# Sources
SRC_CU    = $(wildcard src/*.cu src/shaders/*.cu)
SRC_CPP   = $(wildcard src/*.cpp)
IMGUI_SRC = \
    imgui/imgui.cpp \
    imgui/imgui_demo.cpp \
    imgui/imgui_draw.cpp \
    imgui/imgui_tables.cpp \
    imgui/imgui_widgets.cpp \
    imgui/backends/imgui_impl_sdl2.cpp \
    imgui/backends/imgui_impl_opengl3.cpp

BUILD_DIR = build

# Object files
OBJ_CU    = $(patsubst src/%.cu,$(BUILD_DIR)/%.o,$(SRC_CU))
OBJ_CPP   = $(patsubst src/%.cpp,$(BUILD_DIR)/%.o,$(SRC_CPP))
IMGUI_OBJ = $(patsubst %.cpp,$(BUILD_DIR)/%.o,$(IMGUI_SRC))

# Target
TARGET = $(BUILD_DIR)/raytracer
OUTPUT_NAME = $(BUILD_DIR)/output

.PHONY: all clean run run-image run-video

# Default target
all: $(TARGET)

# Rule to ensure build folder exists
$(BUILD_DIR):
	mkdir -p $(BUILD_DIR) $(BUILD_DIR)/imgui $(BUILD_DIR)/imgui/backends

# Compile CUDA files
$(BUILD_DIR)/%.o: src/%.cu | $(BUILD_DIR)
	$(NVCC) -ccbin $(CCBIN) $(NVCCFLAGS) -c $< -o $@

# Compile C++ files
$(BUILD_DIR)/%.o: src/%.cpp | $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) -c $< -o $@

# Compile IMGUI files
$(BUILD_DIR)/%.o: %.cpp | $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) -c $< -o $@

# Link all objects
$(TARGET): $(OBJ_CU) $(OBJ_CPP) $(IMGUI_OBJ)
	$(NVCC) -ccbin $(CCBIN) $^ -o $@ $(LDFLAGS)

# Run targets
run: all
	./$(TARGET)

run-image: all
	./$(TARGET) --image --output $(OUTPUT_NAME).ppm
	convert $(OUTPUT_NAME).ppm $(OUTPUT_NAME).png

run-video: all
	./$(TARGET) --video
	convert -delay 5 -loop 0 build/frame_*.ppm build/camera_rotation.gif

dry-run: all
	./$(TARGET) --dry-run

# Clean
clean:
	rm -rf $(BUILD_DIR) $(OUTPUT_NAME).* frame_*
