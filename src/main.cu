#include <iomanip> // for std::setw, std::setfill
#include <iostream>
#include <fstream>
#include <sstream>

#include "vec3.h"
#include "sphere.h"
#include "camera.h"

void render(Vec3 *fb, int width, int height, Sphere *sphere, int nSpheres, Camera *cam);

struct SceneParams {
    int width;
    int height;

    SceneParams() : width(800), height(800) {}
};

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

Camera makeCamera(const Vec3 &center, float radius, float angleDeg, int width, int height) {
    float rad = angleDeg * M_PI / 180.0f;

    Vec3 camPos(
        center.x + radius * cosf(rad),
        1.0f,
        center.z + radius * sinf(rad)
    );

    Vec3 forward = (center - camPos).normalize();
    return Camera(
        camPos,
        forward,
        Vec3(0, -1, 0), // up
        90.0f,
        float(width) / float(height)
    );
}

class Scene {
public:
    int width, height;
    Vec3 *fb;
    Sphere *spheres;
    int nSpheres;
    Vec3 center;
    float radius;

    Scene(int w, int h)
        : width(w), height(h), fb(nullptr), spheres(nullptr),
          nSpheres(0), center(0,0,-3), radius(5.0f) 
    {
        size_t fb_size = width * height * sizeof(Vec3);
        cudaMallocManaged(&fb, fb_size);

        // Allocate spheres
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

    void renderFrame(float angleDeg, const std::string &filename) {
        Camera cam = makeCamera(center, radius, angleDeg, width, height);
        render(fb, width, height, spheres, nSpheres, &cam);
        savePPM(filename, fb, width, height);
    }

    void renderStatic(const std::string &filename = "output.ppm") {
        renderFrame(0.0f, filename);
        std::cout << "Static render saved to " << filename << std::endl;
    }

    void renderVideo(int nFrames, float totalAngle) {
        for (int i = 0; i < nFrames; i++) {
            float angle = (totalAngle / nFrames) * i;
            std::ostringstream filename;
            filename << "frame_" << std::setw(3) << std::setfill('0') << i << ".ppm";
            renderFrame(angle, filename.str());
            std::cout << "Saved " << filename.str() << std::endl;
        }
        std::cout << "Video render complete!" << std::endl;
    }
};

// ---------- Main ----------

int main(int argc, char** argv) {
    // Default params
    bool videoMode = false;
    int nFrames = 60;
    float totalAngle = 360.0f;
    std::string output = "output.ppm";
    int width = 800, height = 800;

    // Parse args
    for (int i = 1; i < argc; i++) {
        std::string arg = argv[i];
        if (arg == "--video") {
            videoMode = true;
        } else if (arg == "--image") {
            videoMode = false;
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
                      << "  --image               Render single image (default)\n"
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

    if (videoMode) {
        std::cout << "Rendering video: " << nFrames << " frames, "
                  << totalAngle << " degrees rotation, "
                  << width << "x" << height << "\n";
        scene.renderVideo(nFrames, totalAngle);
    } else {
        std::cout << "Rendering static image: " << output
                  << " (" << width << "x" << height << ")\n";
        scene.renderStatic(output);
    }

    return 0;
}