# CUDA Raytracer

This project is a simple GPU-based raytracer using CUDA, C++, SDL2, ImGui and
OpenGL (OpenGL is only used to support ImGui). It supports rendering single
images, generating frame sequences from user inputs in realtime and creating
animated GIFs.

![screenshot](images/screenshot.png)

## Requirements

- **CUDA Toolkit** (for `nvcc`)
- **C++17 compiler** (`g++` recommended, this is the default used by `Make`)
- **SDL2 development libraries**
- **ImageMagik** (for converting PPM frames to PNG/GIF, you can changed it as you wish)
- **Make**

## Build instructions

Clone the repository and build the project using `make`:

```sh
git clone --recurse-submodules git@github.com:tombrossard0/raytracing-with-cuda.git
cd raytracing-with-cuda
make
```

This will compile all sources and place the executable in `build/raytracer`.

## Running

### Render an image

```sh
make run-image
```

This command will generate the `build/output.ppm` file and converts it to
`build/output.png` with **ImageMakik**.

### Render a GIF

```sh
make run-video
```

This command will generate all `build/frame_XXX.ppm` files and converts them to
`build/camera_rotation.gif` with **ImageMakik**.

### Run the interactive Application

```sh
make run
```

Opens a window with **SDL2** and **ImGui** controls, e.g, to adjust camera,
raytracing settings and add/remove objects from the scene.

### Cleaning Build Artifacts

```sh
make clean
```

Removes `build/` folder, PPM/PNG outputs, and generated frames.

### Controls

- Adjust **Max Bounces** and **Number of Rays per Pixel** via ImGui sliders.

## Notes

- Ensure your GPU supports **CUDA**.
- Large numbers of rays per pixel or bounces may significantly increase render time.
