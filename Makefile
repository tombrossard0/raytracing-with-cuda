NVCC = nvcc
NVCCFLAGS = -O2 -g -std=c++17 -Xcompiler "-Wall -Wextra -Werror" -lineinfo

CCBIN = gcc
CFLAGS = -O2 -g -Wall -Wextra -Werror

LDFLAGS =

SRC = $(wildcard src/*.cu)
TARGET = raytracer
OUTPUT_NAME = output

all: $(TARGET)

$(TARGET): $(SRC)
	$(NVCC) -ccbin $(CCBIN) $(NVCCFLAGS) $^ -o $@ $(LDFLAGS)

run: all
	./raytracer
	convert $(OUTPUT_NAME).ppm $(OUTPUT_NAME).png

run-video: all
	./raytracer --video
	convert -delay 5 -loop 0 frame_*.ppm camera_rotation.gif

clean:
	rm -f $(TARGET) $(OUTPUT_NAME).* frame_*
