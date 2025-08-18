#include <iomanip> // for std::setw, std::setfill
#include <iostream>

#include "imgui.h"
#include "imgui_impl_opengl3.h"
#include "imgui_impl_sdl2.h"
#include "ppm.hpp"
#include "render.hpp"
#include "scene.hpp"

#include <cuda_runtime_api.h>
#include <iomanip>
#include <iostream>
#include <sstream>

Scene::Scene(int w, int h)
    : width(w), height(h), fb(nullptr), spheres(nullptr), nSpheres(0), center(0, 0, -3), radius(5.0f),
      yawDeg(0.0f), pitchDeg(0.0f), minRadius(1.0f), maxRadius(20.0), texture(0) {
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

Scene::~Scene() {
    if (texture) { glDeleteTextures(1, &texture); }

    cudaDeviceSynchronize(); // ensure all kernels are finished
    if (fb) { cudaFree(fb); };
    if (spheres) { cudaFree(spheres); };
}

Camera Scene::makeCamera() {
    // Convert to radians
    float yawRad = yawDeg * M_PI / 180.0f;
    float pitchRad = pitchDeg * M_PI / 180.0f;

    // Spherical coordinates around center
    float x = center.x + radius * cosf(pitchRad) * cosf(yawRad);
    float y = center.y + radius * sinf(pitchRad);
    float z = center.z + radius * cosf(pitchRad) * sinf(yawRad);

    Vec3 camPos(x, y, z);
    Vec3 forward = (center - camPos).normalize();

    return Camera(camPos, forward, Vec3(0, 1, 0), // world up
                  90.0f,                          // fov
                  float(width) / float(height));
}

void Scene::renderFrame() {
    Camera cam = makeCamera();
    render(fb, width, height, spheres, nSpheres, &cam);
}

void Scene::renderGUI(GLuint &tex) {
    ImGui::Begin("Render Scene");
    ImGui::Image((void *)(intptr_t)tex, ImVec2(width, height));
    if (ImGui::IsItemClicked()) focus = true;
    bool hovered = ImGui::IsItemHovered();
    ImGui::End();

    if (!hovered && ImGui::IsMouseClicked(ImGuiMouseButton_Left)) focus = false;

    ImGui::Begin("Camera Controls");
    ImGui::SliderFloat("Radius", &radius, minRadius, maxRadius);
    ImGui::SliderFloat("Yaw", &yawDeg, -180.0f, 180.0f);
    ImGui::SliderFloat("Pitch", &pitchDeg, -89.0f, 89.0f);
    ImGui::End();

    ImGui::Begin("Spheres");
    if (ImGui::Button("Add Sphere")) {
        if (nSpheres < MAX_SPHERES) { spheres[nSpheres++] = Sphere(Vec3(), 1); }
    }

    for (int i = 0; i < nSpheres; i++) {
        std::string nodeLabel = "Sphere " + std::to_string(i);
        if (ImGui::CollapsingHeader(nodeLabel.c_str())) {
            ImGui::DragFloat3(("Position##" + std::to_string(i)).c_str(), &spheres[i].center.x, 0.01f);
            ImGui::DragFloat(("Radius##" + std::to_string(i)).c_str(), &spheres[i].radius, 0.01f, 0.1f,
                             50.0f);
            ImGui::ColorEdit3(("Color##" + std::to_string(i)).c_str(), &spheres[i].material.colour.x);
            ImGui::ColorEdit3(("Emission color##" + std::to_string(i)).c_str(),
                              &spheres[i].material.emissionColour.x);
            ImGui::DragFloat(("EMission strength##" + std::to_string(i)).c_str(),
                             &spheres[i].material.emissionStrength, 0.0f, 0.1f, 1.0f);

            if (ImGui::Button(("Remove##" + std::to_string(i)).c_str())) {
                for (int j = i; j < nSpheres - 1; j++) spheres[j] = spheres[j + 1];
                nSpheres--;
            }
        }
    }
    ImGui::End();

    ImGui::Begin("Screenshots");
    if (ImGui::Button("save PPM")) { renderPPM(); }
    ImGui::End();
}

void Scene::renderPPMFrame(const std::string &filename) {
    Camera cam = makeCamera();
    render(fb, width, height, spheres, nSpheres, &cam);
    savePPM(filename, fb, width, height);
}

void Scene::renderPPM(const std::string &filename) {
    renderPPMFrame(filename);
    std::cout << "Static render saved to " << filename << std::endl;
}

void Scene::renderGIF(int nFrames, float totalAngle) {
    for (int i = 0; i < nFrames; i++) {
        yawDeg = (totalAngle / nFrames) * i;
        std::ostringstream filename;
        filename << "frame_" << std::setw(3) << std::setfill('0') << i << ".ppm";
        renderPPMFrame(filename.str());
        std::cout << "Saved " << filename.str() << std::endl;
    }
    std::cout << "Video render complete!" << std::endl;
}
