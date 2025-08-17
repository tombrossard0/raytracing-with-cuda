#include <iomanip> // for std::setw, std::setfill
#include <iostream>
#include <fstream>
#include <sstream>
#include <SDL2/SDL.h>
#include <SDL2/SDL_opengl.h>

#include "vec3.h"
#include "sphere.h"
#include "camera.h"

#include "imgui.h"
#include "imgui_impl_sdl2.h"
#include "imgui_impl_opengl3.h"

const int MAX_SPHERES = 64;

void render(Vec3 *fb, int width, int height, Sphere *sphere, int nSpheres, Camera *cam);

// ---------- Utility functions ----------

void savePPM(const std::string &filename, Vec3 *fb, int width, int height) {
    auto toByte = [](float x) {
        // gamma correction
        return static_cast<unsigned char>(255 * powf(fminf(fmaxf(x, 0.0f), 1.0f), 1.0f/2.2f));
    };

    std::ofstream ofs(filename, std::ios::binary);
    ofs << "P6\n" << width << " " << height << "\n255\n";
    for (int i = 0; i < width * height; i++) {
        unsigned char r = toByte(fb[i].x);
        unsigned char g = toByte(fb[i].y);
        unsigned char b = toByte(fb[i].z);

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

        cudaMallocManaged(&spheres, MAX_SPHERES * sizeof(Sphere));
        nSpheres = 4;
        spheres[0] = Sphere(center + Vec3(-4.418, -5.648, -3), 5, Vec3(1, 1, 1));
        spheres[0].material.emissionColour = Vec3(1, 1, 1);
        spheres[0].material.emissionStrength = 1;

        spheres[1] = Sphere(center + Vec3(0.92, 0, -3), .3f, Vec3(0, 1, 0));
        spheres[2] = Sphere(center + Vec3(2.23, 1.05, -6.13), .4f, Vec3(0, 0, 1));
        spheres[3] = Sphere(center + Vec3(1.59, 5.28, -3.850), 5, Vec3(1, 0, 0));
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
            float(width) / float(height),
            1
        );
    }

    void renderFrame() {
        Camera cam = makeCamera();
        render(fb, width, height, spheres, nSpheres, &cam);
    }

    void renderGUI(GLuint &tex) {
        ImGui_ImplSDL2_NewFrame();
        ImGui_ImplOpenGL3_NewFrame();
        ImGui::NewFrame();

        ImGui::Begin("Render Window");
        ImGui::Image((void*)(intptr_t)tex, ImVec2(width, height));
        ImGui::End();

        ImGui::Begin("Camera Controls");
        ImGui::SliderFloat("Radius", &radius, minRadius, maxRadius);
        ImGui::SliderFloat("Yaw", &yawDeg, -180.0f, 180.0f);
        ImGui::SliderFloat("Pitch", &pitchDeg, -89.0f, 89.0f);
        ImGui::End();

        ImGui::Begin("Spheres");
        if (ImGui::Button("Add Sphere")) {
            if (nSpheres < MAX_SPHERES) {
                spheres[nSpheres++] = Sphere(Vec3(), 1);
            }
        }

        for (int i = 0; i < nSpheres; i++) {
            std::string nodeLabel = "Sphere " + std::to_string(i);
            if (ImGui::CollapsingHeader(nodeLabel.c_str())) {
                ImGui::DragFloat3(("Position##" + std::to_string(i)).c_str(), &spheres[i].center.x, 0.01f);
                ImGui::DragFloat(("Radius##" + std::to_string(i)).c_str(), &spheres[i].radius, 0.01f, 0.1f, 50.0f);
                ImGui::ColorEdit3(("Color##" + std::to_string(i)).c_str(), &spheres[i].material.colour.x);
                ImGui::ColorEdit3(("Emission color##" + std::to_string(i)).c_str(), &spheres[i].material.emissionColour.x);
                ImGui::DragFloat(("EMission strength##" + std::to_string(i)).c_str(), &spheres[i].material.emissionStrength, 0.0f, 0.1f, 1.0f);

                if (ImGui::Button(("Remove##" + std::to_string(i)).c_str())) {
                    for (int j = i; j < nSpheres - 1; j++) spheres[j] = spheres[j + 1];
                    nSpheres--;
                }
            }
        }
        ImGui::End();

        ImGui::Begin("Screenshots");
        if (ImGui::Button("save PPM")) {
            renderPPM();
        }
        ImGui::End();
    }

    int renderSDL2() {
        if (SDL_Init(SDL_INIT_VIDEO) < 0) {
            std::cerr << "Failed to init SDL: " << SDL_GetError() << std::endl;
            return 1;
        }

        SDL_Window* window = SDL_CreateWindow(
            "CUDA Raytracer",
            SDL_WINDOWPOS_CENTERED, SDL_WINDOWPOS_CENTERED,
            1920, 1080,
            SDL_WINDOW_OPENGL | SDL_WINDOW_RESIZABLE
        );

        if (!window) {
            std::cerr << "Failed to create SDL window: " << SDL_GetError() << std::endl;
            SDL_Quit();
            return 1;
        }

        SDL_GLContext gl_context = SDL_GL_CreateContext(window);
        SDL_GL_MakeCurrent(window, gl_context);
        SDL_GL_SetSwapInterval(1); // vsync

        // --- OpenGL texture for CUDA framebuffer ---
        GLuint tex;
        glGenTextures(1, &tex);
        glBindTexture(GL_TEXTURE_2D, tex);
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB32F, width, height, 0, GL_RGB, GL_FLOAT, nullptr);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
        glBindTexture(GL_TEXTURE_2D, 0);

        // --- Init ImGui ---
        IMGUI_CHECKVERSION();
        ImGui::CreateContext();
        ImGuiIO& io = ImGui::GetIO(); (void)io;
        ImGui::StyleColorsDark();
        ImGui_ImplSDL2_InitForOpenGL(window, gl_context);
        ImGui_ImplOpenGL3_Init("#version 330");

        bool running = true;
        SDL_Event event;

        bool mouseLook = false;
        int lastMouseX = 0, lastMouseY = 0;
        float sensitivity = 0.2f;

        Uint32 lastTime = SDL_GetTicks();
        int frameCount = 0;
        float fps = 0.0f;

        while (running) {
            // --- Input ---
            while (SDL_PollEvent(&event)) {
                ImGui_ImplSDL2_ProcessEvent(&event);
                if (event.type == SDL_QUIT) running = false;

                if (event.type == SDL_MOUSEBUTTONDOWN && event.button.button == SDL_BUTTON_RIGHT) {
                    mouseLook = true;
                    SDL_GetMouseState(&lastMouseX, &lastMouseY);
                    SDL_SetRelativeMouseMode(SDL_TRUE);
                }
                if (event.type == SDL_MOUSEBUTTONUP && event.button.button == SDL_BUTTON_RIGHT) {
                    mouseLook = false;
                    SDL_SetRelativeMouseMode(SDL_FALSE);
                }
                if (event.type == SDL_MOUSEMOTION && mouseLook) {
                    int dx = event.motion.xrel;
                    int dy = -event.motion.yrel;
                    yawDeg += dx * sensitivity;
                    pitchDeg += dy * sensitivity;
                    if (pitchDeg > 89.0f) pitchDeg = 89.0f;
                    if (pitchDeg < -89.0f) pitchDeg = -89.0f;
                }
                if (event.type == SDL_MOUSEWHEEL) {
                    radius -= event.wheel.y * 0.5f;
                    if (radius < minRadius) radius = minRadius;
                    if (radius > maxRadius) radius = maxRadius;
                }
                if (event.type == SDL_KEYDOWN && event.key.keysym.sym == SDLK_ESCAPE) {
                    running = false;
                }
            }

            // --- CUDA render ---
            renderFrame();

            // --- Upload framebuffer to OpenGL texture ---
            glBindTexture(GL_TEXTURE_2D, tex);
            glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, width, height, GL_RGB, GL_FLOAT, fb);
            glBindTexture(GL_TEXTURE_2D, 0);

            // --- Clear screen ---
            glClearColor(0, 0, 0, 1);
            glClear(GL_COLOR_BUFFER_BIT);

            // --- ImGui frame ---
            renderGUI(tex);

            ImGui::Render();
            ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());

            SDL_GL_SwapWindow(window);

            // --- FPS ---
            frameCount++;
            Uint32 currentTime = SDL_GetTicks();
            Uint32 elapsed = currentTime - lastTime;
            if (elapsed >= 1000) {
                fps = frameCount * 1000.0f / elapsed;
                frameCount = 0;
                lastTime = currentTime;
                std::string title = "CUDA Raytracer - FPS: " + std::to_string((int)fps);
                SDL_SetWindowTitle(window, title.c_str());
            }
        }

        // --- Cleanup ---
        ImGui_ImplOpenGL3_Shutdown();
        ImGui_ImplSDL2_Shutdown();
        ImGui::DestroyContext();

        glDeleteTextures(1, &tex);
        SDL_GL_DeleteContext(gl_context);
        SDL_DestroyWindow(window);
        SDL_Quit();

        return 0;
    }

    void renderPPMFrame(const std::string &filename) {
        Camera cam = makeCamera();
        render(fb, width, height, spheres, nSpheres, &cam);
        savePPM(filename, fb, width, height);
    }

    void renderPPM(const std::string &filename = "output.ppm") {
        renderPPMFrame(filename);
        std::cout << "Static render saved to " << filename << std::endl;
    }

    void renderGIF(int nFrames, float totalAngle) {
        for (int i = 0; i < nFrames; i++) {
            yawDeg = (totalAngle / nFrames) * i;
            std::ostringstream filename;
            filename << "frame_" << std::setw(3) << std::setfill('0') << i << ".ppm";
            renderPPMFrame(filename.str());
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