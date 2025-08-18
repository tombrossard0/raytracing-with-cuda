#include <SDL2/SDL.h>
#include <SDL2/SDL_opengl.h>
#include <fstream>
#include <iomanip> // for std::setw, std::setfill
#include <iostream>
#include <sstream>

#include "camera.cuh"
#include "engine.cuh"
#include "ppm.hpp"
#include "scene.hpp"
#include "sphere.cuh"
#include "vec3.cuh"

#include "render.hpp"

#include "imgui.h"
#include "imgui_impl_opengl3.h"
#include "imgui_impl_sdl2.h"

enum RUN_MODE {
    REALTIME,
    PPM,
    GIF,
};

void run_realtime(Scene &scene) {
    Engine engine{1920, 1080};

    // --- Run Scene ---
    scene.texture = engine.createTexture(scene.width, scene.height);

    while (engine.running) {
        // --- Timing ---
        Uint64 currentTime = SDL_GetTicks64();
        float deltaTime = (currentTime - engine.lastFrameTime) / 1000.0f; // seconds
        engine.lastFrameTime = currentTime;

        // --- Input ---
        const Uint8 *keystate = SDL_GetKeyboardState(NULL);
        scene.processInputs(keystate, deltaTime, engine.running, &engine.event, engine.mouse);

        // --- CUDA render ---
        scene.renderFrame();

        // --- Upload framebuffer to OpenGL texture ---
        engine.uploadFbToTexture(scene);
        engine.clearScreen();

        // --- ImGui frame ---
        engine.renderImGui(&scene);

        SDL_GL_SwapWindow(engine.window);

        // --- FPS ---
        engine.computeFPS(currentTime);
    }
}

int main(int argc, char **argv) {
    // Default params
    RUN_MODE mode = RUN_MODE::REALTIME;
    int nFrames = 60;
    float totalAngle = 360.0f;
    std::string output = "output.ppm";
    int width = 800, height = 800;

    // Parse args
    for (int i = 1; i < argc; i++) {
        std::string arg = argv[i];
        if (arg == "--video") {
            mode = RUN_MODE::GIF;
        } else if (arg == "--image") {
            mode = RUN_MODE::PPM;
        } else if (arg == "--frames" && i + 1 < argc) {
            nFrames = std::atoi(argv[++i]);
        } else if (arg == "--angle" && i + 1 < argc) {
            totalAngle = std::atof(argv[++i]);
        } else if (arg == "--output" && i + 1 < argc) {
            output = argv[++i];
        } else if (arg == "--res" && i + 1 < argc) {
            std::string res = argv[++i];
            size_t xPos = res.find('x');
            if (xPos != std::string::npos) {
                width = std::atoi(res.substr(0, xPos).c_str());
                height = std::atoi(res.substr(xPos + 1).c_str());
            } else {
                std::cerr << "Invalid resolution format. Use WIDTHxHEIGHT (e.g., 1920x1080)\n";
                return 1;
            }
        } else if (arg == "--help") {
            std::cout << "Usage: " << argv[0] << " [--image | --video] [options]\n"
                      << "Options:\n"
                      << "  --realtime            Render in realtime (default)\n"
                      << "  --image               Render single image\n"
                      << "  --video               Render multiple frames\n"
                      << "  --frames <n>          Number of frames for video (default 60)\n"
                      << "  --angle <deg>         Total rotation angle (default 360)\n"
                      << "  --output <file>       Output file for static image (default output.ppm)\n"
                      << "  --res <WxH>           Resolution, e.g. 1920x1080 (default 800x800)\n"
                      << "  --help                Show this help message\n";
            return 0;
        }
    }

    Scene scene(width, height);

    switch (mode) {
    case RUN_MODE::REALTIME:
        run_realtime(scene);
        break;
    case RUN_MODE::GIF:
        std::cout << "Rendering video: " << nFrames << " frames, " << totalAngle << " degrees rotation, "
                  << width << "x" << height << "\n";
        scene.renderGIF(nFrames, totalAngle);
        break;
    case RUN_MODE::PPM:
        std::cout << "Rendering static image: " << output << " (" << width << "x" << height << ")\n";
        scene.renderPPM(output);
        break;
    }

    return 0;
}
