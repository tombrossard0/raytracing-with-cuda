# Compilers
NVCC = nvcc
CCBIN = gcc-10
CXX = g++

# Flags
NVCCFLAGS = -O2 -g -std=c++17 -rdc=true \
            -Xcompiler "-Wall -Wextra -Werror -Wno-error=cpp" -lineinfo \
            -I/usr/include/SDL2 -Iimgui -Iimgui/backends
CXXFLAGS  = -O2 -g -Wall -Wextra -Wpedantic -Werror \
            -I/usr/include/SDL2 -Iimgui -Iimgui/backends

# Linker flags
LDFLAGS = -lstdc++ -lm -lSDL2 -lGL

# Sources
SRC_CU_ROOT    = $(wildcard src/*.cu)
SRC_CU_SHADERS = $(wildcard src/shaders/*.cu)
SRC_CU         = $(SRC_CU_ROOT) $(SRC_CU_SHADERS)
SRC_CPP        = $(wildcard src/*.cpp)
IMGUI_SRC = \
    imgui/imgui.cpp \
    imgui/imgui_demo.cpp \
    imgui/imgui_draw.cpp \
    imgui/imgui_tables.cpp \
    imgui/imgui_widgets.cpp \
    imgui/backends/imgui_impl_sdl2.cpp \
    imgui/backends/imgui_impl_opengl3.cpp

BUILD_DIR = build

# Objects (mirror subdirs so patterns work)
OBJ_CU_ROOT    = $(patsubst src/%.cu,$(BUILD_DIR)/%.o,$(SRC_CU_ROOT))
OBJ_CU_SHADERS = $(patsubst src/shaders/%.cu,$(BUILD_DIR)/shaders/%.o,$(SRC_CU_SHADERS))
OBJ_CU         = $(OBJ_CU_ROOT) $(OBJ_CU_SHADERS)
OBJ_CPP        = $(patsubst src/%.cpp,$(BUILD_DIR)/%.o,$(SRC_CPP))
IMGUI_OBJ      = $(patsubst %.cpp,$(BUILD_DIR)/%.o,$(IMGUI_SRC))

TARGET = $(BUILD_DIR)/raytracer
OUTPUT_NAME = $(BUILD_DIR)/output

.PHONY: all clean run run-image run-video dry-run

all: $(TARGET)

# Ensure build dirs exist
$(BUILD_DIR):
	mkdir -p $(BUILD_DIR) $(BUILD_DIR)/imgui $(BUILD_DIR)/imgui/backends $(BUILD_DIR)/shaders

# CUDA compile rules (root and shaders)
$(BUILD_DIR)/%.o: src/%.cu | $(BUILD_DIR)
	$(NVCC) -ccbin $(CCBIN) $(NVCCFLAGS) -c $< -o $@

$(BUILD_DIR)/shaders/%.o: src/shaders/%.cu | $(BUILD_DIR)
	$(NVCC) -ccbin $(CCBIN) $(NVCCFLAGS) -c $< -o $@

# C++ sources
$(BUILD_DIR)/%.o: src/%.cpp | $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) -c $< -o $@

# ImGui sources
$(BUILD_DIR)/%.o: %.cpp | $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) -c $< -o $@

# Link (nvcc performs device link because of -rdc=true)
$(TARGET): $(OBJ_CU) $(OBJ_CPP) $(IMGUI_OBJ)
	$(NVCC) -ccbin $(CCBIN) $^ -o $@ $(LDFLAGS)

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

clean:
	rm -rf $(BUILD_DIR) $(OUTPUT_NAME).* frame_*
