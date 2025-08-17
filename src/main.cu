#include <iomanip> // for std::setw, std::setfill
#include <iostream>
#include <fstream>
#include <sstream>
#include <SDL2/SDL.h>

#include "vec3.h"
#include "sphere.h"
#include "camera.h"

#include "imgui.h"
#include "imgui_impl_sdl2.h"
#include "imgui_impl_opengl3.h"

void render(Vec3 *fb, int width, int height, Sphere *sphere, int nSpheres, Camera *cam);

// ---------- Utility functions ----------

void savePPM(const std::string &filename, Vec3 *fb, int width, int height) {
    std::ofstream ofs(filename, std::ios::binary);
    ofs << "P6\n" << width << " " << height << "\n255\n";
    for (int i = 0; i < width * height; i++) {
        unsigned char r = static_cast<unsigned char>(255.99f * fb[i].x);
        unsigned char g = static_cast<unsigned char>(255.99f * fb[i].y);
        unsigned char b = static_cast<unsigned char>(255.99f * fb[i].z);
        ofs.write(reinterpret_cast<char*>(&r), 1);
        ofs.write(reinterpret_cast<char*>(&g), 1);
        ofs.write(reinterpret_cast<char*>(&b), 1);
    }
}

class Scene {
public:
    int width, height;
    Vec3 *fb;
    Sphere *spheres;
    int nSpheres;
    Vec3 center;
    float radius; // Distance from center of the scene
    float angleDeg;
    float yawDeg;
    float pitchDeg;
    float minRadius;
    float maxRadius;


    Scene(int w, int h)
        : width(w), height(h), fb(nullptr), spheres(nullptr),
          nSpheres(0), center(0,0,-3), radius(5.0f) , yawDeg(0.0f),
          pitchDeg(0.0f), minRadius(1.0f), maxRadius(20.0)
    {
        size_t fb_size = width * height * sizeof(Vec3);
        cudaMallocManaged(&fb, fb_size);

        nSpheres = 3;
        cudaMallocManaged(&spheres, nSpheres * sizeof(Sphere));
        spheres[0] = Sphere(center + Vec3(0.5f, 0, 0), .5f, Vec3(1, 0, 0));
        spheres[1] = Sphere(center + Vec3(0, 1, -0.5), .3f, Vec3(0, 1, 0));
        spheres[2] = Sphere(center + Vec3(0.3, -1, -0.5), .4f, Vec3(0, 0, 1));
    }

    ~Scene() {
        if (fb) cudaFree(fb);
        if (spheres) cudaFree(spheres);
    }

    Camera makeCamera() {
        // Convert to radians
        float yawRad = yawDeg * M_PI / 180.0f;
        float pitchRad = pitchDeg * M_PI / 180.0f;

        // Spherical coordinates around center
        float x = center.x + radius * cosf(pitchRad) * cosf(yawRad);
        float y = center.y + radius * sinf(pitchRad);
        float z = center.z + radius * cosf(pitchRad) * sinf(yawRad);

        Vec3 camPos(x, y, z);
        Vec3 forward = (center - camPos).normalize();

        return Camera(
            camPos,
            forward,
            Vec3(0, 1, 0), // world up
            90.0f,          // fov
            float(width) / float(height)
        );
    }

    void renderFrame() {
        Camera cam = makeCamera();
        render(fb, width, height, spheres, nSpheres, &cam);
    }

    int renderSDL2() {
        if (SDL_Init(SDL_INIT_VIDEO) < 0) {
            std::cerr << "Failed to init SDL: " << SDL_GetError() << std::endl;
            return 1;
        }

        SDL_Window* window = SDL_CreateWindow("CUDA Raytracer",
        SDL_WINDOWPOS_CENTERED, SDL_WINDOWPOS_CENTERED,
        width, height, SDL_WINDOW_SHOWN);

        if (!window) {
            std::cerr << "Failed to create SDL window: " << SDL_GetError() << std::endl;
            SDL_Quit();
            return 1;
        }

        SDL_Renderer* renderer = SDL_CreateRenderer(window, -1, SDL_RENDERER_ACCELERATED);
        SDL_Texture* texture = SDL_CreateTexture(renderer,
            SDL_PIXELFORMAT_RGB24, SDL_TEXTUREACCESS_STREAMING, width, height);

        Scene scene(width, height);

        bool running = true;
        SDL_Event event;

        // --- Mouse look variables ---
        bool mouseLook = false;
        int lastMouseX = 0, lastMouseY = 0;
        float sensitivity = 0.2f; // adjust rotation speed

        // FPS tracking
        Uint32 lastTime = SDL_GetTicks();
        int frameCount = 0;
        float fps = 0.0f;

        while (running) {
            // --- Input ---
            while (SDL_PollEvent(&event)) {
                if (event.type == SDL_QUIT) running = false;
                if (event.type == SDL_MOUSEBUTTONDOWN) {
                    if (event.button.button == SDL_BUTTON_LEFT) {
                        mouseLook = true;
                        SDL_GetMouseState(&lastMouseX, &lastMouseY);
                        SDL_SetRelativeMouseMode(SDL_TRUE); // hide cursor and capture mouse
                    }
                }
                if (event.type == SDL_MOUSEBUTTONUP) {
                    if (event.button.button == SDL_BUTTON_LEFT) {
                        mouseLook = false;
                        SDL_SetRelativeMouseMode(SDL_FALSE); // release cursor
                    }
                }
                if (event.type == SDL_MOUSEMOTION && mouseLook) {
                    int dx = event.motion.xrel; // relative motion
                    int dy = -event.motion.yrel;

                    // update angles
                    scene.yawDeg += dx * sensitivity;    // horizontal rotation
                    scene.pitchDeg += dy * sensitivity;  // vertical rotation

                    if (scene.pitchDeg > 89.0f) scene.pitchDeg = 89.0f;
                    if (scene.pitchDeg < -89.0f) scene.pitchDeg = -89.0f;
                }
                if (event.type == SDL_MOUSEWHEEL) {
                    scene.radius -= event.wheel.y * 0.5f; // zoom factor
                    if (scene.radius < scene.minRadius) scene.radius = scene.minRadius;
                    if (scene.radius > scene.maxRadius) scene.radius = scene.maxRadius;
                }
                if (event.type == SDL_KEYDOWN) {
                    if (event.key.keysym.sym == SDLK_ESCAPE) running = false;
                }
            }

            // --- Render with CUDA ---
            scene.renderFrame();

            // --- Copy CUDA framebuffer to SDL texture ---
            void* pixels;
            int pitch;
            SDL_LockTexture(texture, nullptr, &pixels, &pitch);

            unsigned char* dst = (unsigned char*)pixels;
            for (int y = 0; y < height; y++) {
                unsigned char* row = dst + y * pitch;
                for (int x = 0; x < width; x++) {
                    Vec3 color = scene.fb[y * width + x];
                    row[x*3 + 0] = static_cast<unsigned char>(255.99f * fminf(color.x, 1.0f));
                    row[x*3 + 1] = static_cast<unsigned char>(255.99f * fminf(color.y, 1.0f));
                    row[x*3 + 2] = static_cast<unsigned char>(255.99f * fminf(color.z, 1.0f));
                }
            }

            SDL_UnlockTexture(texture);

            // --- Present ---
            SDL_RenderClear(renderer);
            SDL_RenderCopy(renderer, texture, nullptr, nullptr);
            SDL_RenderPresent(renderer);

            // --- FPS calculation ---
            frameCount++;
            Uint32 currentTime = SDL_GetTicks();
            Uint32 elapsed = currentTime - lastTime;
            if (elapsed >= 1000) { // update every second
                fps = frameCount * 1000.0f / elapsed;
                frameCount = 0;
                lastTime = currentTime;

                std::string title = "CUDA Raytracer - FPS: " + std::to_string((int)fps);
                SDL_SetWindowTitle(window, title.c_str());
            }
        }

        SDL_DestroyTexture(texture);
        SDL_DestroyRenderer(renderer);
        SDL_DestroyWindow(window);
        SDL_Quit();

        return 0;
    }

    void renderPPMFrame(float angleDeg, const std::string &filename) {
        this->angleDeg = angleDeg;
        Camera cam = makeCamera();
        render(fb, width, height, spheres, nSpheres, &cam);
        savePPM(filename, fb, width, height);
    }

    void renderPPM(const std::string &filename = "output.ppm") {
        renderPPMFrame(0.0f, filename);
        std::cout << "Static render saved to " << filename << std::endl;
    }

    void renderGIF(int nFrames, float totalAngle) {
        for (int i = 0; i < nFrames; i++) {
            float angle = (totalAngle / nFrames) * i;
            std::ostringstream filename;
            filename << "frame_" << std::setw(3) << std::setfill('0') << i << ".ppm";
            renderPPMFrame(angle, filename.str());
            std::cout << "Saved " << filename.str() << std::endl;
        }
        std::cout << "Video render complete!" << std::endl;
    }
};

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