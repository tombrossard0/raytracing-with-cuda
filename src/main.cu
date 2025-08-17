#include <iomanip> // for std::setw, std::setfill
#include <iostream>
#include <fstream>
#include <sstream>
#include <SDL2/SDL.h>
#include <SDL2/SDL_opengl.h>

#include "vec3.h"
#include "sphere.h"
#include "camera.h"
#include "ppm.h"
#include "scene.h"

#include "render.h"

#include "imgui.h"
#include "imgui_impl_sdl2.h"
#include "imgui_impl_opengl3.h"

// ---------- Main ----------

enum MODE {
    REALTIME,
    PPM,
    GIF,
};

int main(int argc, char** argv) {
    // Default params
    MODE mode = MODE::REALTIME;
    int nFrames = 60;
    float totalAngle = 360.0f;
    std::string output = "output.ppm";
    int width = 800, height = 800;

    // Parse args
    for (int i = 1; i < argc; i++) {
        std::string arg = argv[i];
        if (arg == "--video") {
            mode = MODE::GIF;
        } else if (arg == "--image") {
            mode = MODE::PPM;
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
        case MODE::REALTIME:
            return scene.renderSDL2();
        case MODE::GIF:
            std::cout << "Rendering video: " << nFrames << " frames, "
                    << totalAngle << " degrees rotation, "
                    << width << "x" << height << "\n";
            scene.renderGIF(nFrames, totalAngle);
            break;
        case MODE::PPM:
            std::cout << "Rendering static image: " << output
                    << " (" << width << "x" << height << ")\n";
            scene.renderPPM(output);
            break;
    }

    return 0;
}
