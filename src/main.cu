#include <iomanip> // for std::setw, std::setfill
#include <iostream>
#include <fstream>

#include "vec3.h"

#include "sphere.h"
#include "camera.h"

void render(Vec3 *fb, int width, int height, Sphere *sphere, int nSpheres, Camera *cam);

struct SceneParams {
    int width;
    int height;

    SceneParams() : width(800), height(800) {}
};

void static_image(SceneParams sceneParams);
void video(SceneParams sceneParams);

int main() {
    SceneParams sceneParams{};

    // static_image(sceneParams);
    video(sceneParams);

    return 0;
}

void static_image(SceneParams sceneParams) {
    const int width = sceneParams.width;
    const int height = sceneParams.height;

    Vec3 *fb;
    size_t fb_size = width * height * sizeof(Vec3);
    cudaMallocManaged(&fb, fb_size);

    Vec3 center(0, 0, -3);
    float radius = 5.0f; // distance from center
    float angle = 0;
    float rad = angle * M_PI / 180.0f;

    Vec3 camPos(
        center.x + radius * cosf(rad),
        1.0f, // keep height constant
        center.z + radius * sinf(rad)
    );

    Vec3 forward = (center - camPos).normalize();
    Camera cam(
        camPos, // position
        forward, // forward
        Vec3(0, -1, 0), // up
        90.0f, // fov
        float(width) / float(height) // aspect
    );

    int nSpheres = 2;
    Sphere* spheres;
    cudaMallocManaged(&spheres, nSpheres * sizeof(Sphere));
    spheres[0] = Sphere(center + Vec3(0.5f, 0, 0), .5f, Vec3(1, 0, 0));
    spheres[1] = Sphere(center + Vec3(0, 1, -0.5), .3f, Vec3(0, 1, 0));

    render(fb, width, height, spheres, nSpheres, &cam);

    // Save PPM
    std::ofstream ofs("output.ppm");
    ofs << "P3\n" << width << " " << height << "\n255\n";
    for (int i=0; i < width * height; i++) {
        int r = static_cast<int>(255.99 * fb[i].x);
        int g = static_cast<int>(255.99 * fb[i].y);
        int b = static_cast<int>(255.99 * fb[i].z);
        ofs << r << " " << g << " " << b << "\n";
    }
    ofs.close();

    cudaFree(fb);
    std::cout << "Render finished, output.ppm created!" << std::endl;
}

void video(SceneParams sceneParams) {
    const int width = sceneParams.width;
    const int height = sceneParams.height;

    const int nFrames = 60; // number of frames for full 260° rotation
    const float rotationAngle = 360.0f; // degrees
    Vec3 center(0, 0, -3); // look-at point

    Vec3 *fb;
    size_t fb_size = width * height * sizeof(Vec3);
    cudaMallocManaged(&fb, fb_size);

    float radius = 5.0f; // distance from center

    for (int i = 0; i < nFrames; i++) {
        float angle = (rotationAngle / nFrames) * i;
        float rad = angle * M_PI / 180.0f;

        // camera position on a circle around center
        Vec3 camPos(
            center.x + radius * cosf(rad),
            1.0f, // keep height constant
            center.z + radius * sinf(rad)
        );

        Vec3 forward = (center - camPos).normalize();
        Camera cam(
            camPos, // position
            forward, // forward
            Vec3(0, -1, 0), // up
            90.0f, // fov
            float(width) / float(height) // aspect
        );

        int nSpheres = 2;
        Sphere* spheres;
        cudaMallocManaged(&spheres, nSpheres * sizeof(Sphere));
        spheres[0] = Sphere(center + Vec3(0.5f, 0, 0), .5f, Vec3(1, 0, 0));
        spheres[1] = Sphere(center + Vec3(0, 1, -0.5), .3f, Vec3(0, 1, 0));

        render(fb, width, height, spheres, nSpheres, &cam);

        std::ostringstream filename;
        filename << "frame_" << std::setw(3) << std::setfill('0') << i << ".ppm";
        std::ofstream ofs(filename.str());
        ofs << "P3\n" << width << " " << height << "\n255\n";
        for (int j=0; j < width*height; j++) {
            int r = static_cast<int>(255.99 * fb[j].x);
            int g = static_cast<int>(255.99 * fb[j].y);
            int b = static_cast<int>(255.99 * fb[j].z);
            ofs << r << " " << g << " " << b << "\n";
        }
        ofs.close();
        std::cout << "Saved " << filename.str() << std::endl;
    }

    cudaFree(fb);
    std::cout << "Render finished!" << std::endl;
}