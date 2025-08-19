#include <iomanip> // for std::setw, std::setfill
#include <iostream>

#include "imgui.h"
#include "imgui_impl_opengl3.h"
#include "imgui_impl_sdl2.h"
#include "ppm.hpp"
#include "render.cuh"
#include "scene.hpp"

#include <cuda_runtime_api.h>
#include <iomanip>
#include <iostream>
#include <sstream>

Scene::Scene(int w, int h)
    : width(w), height(h), fb(nullptr), spheres(nullptr), nSpheres(0), radius(15.0f), yawDeg(64.0f),
      pitchDeg(-16.0f), minRadius(1.0f), maxRadius(20.0), texture(0) {
    makeCamera();

    size_t fb_size = width * height * sizeof(Vec3);
    cudaMallocManaged(&fb, fb_size);

    cudaMallocManaged(&spheres, MAX_SPHERES * sizeof(Sphere));
    nSpheres = 6;
    spheres[0] = Sphere(cam->center + Vec3(-17.218, -13.568, -3.990), 11.07, Vec3(1, 1, 1));
    spheres[0].material.emissionColour = Vec3(1, 1, 1);
    spheres[0].material.emissionStrength = 1;

    spheres[1] = Sphere(cam->center + Vec3(0.92, -0.71, -3), .73f, Vec3(0, 1, 0));
    spheres[2] = Sphere(cam->center + Vec3(2.23, -0.81, -6.13), .88f, Vec3(0, 0, 1));
    spheres[3] = Sphere(cam->center + Vec3(1.59, 23.14, -3.850), 23.05, Vec3(1, 1, 1));
    spheres[4] = Sphere(cam->center + Vec3(0.16, -1.52, -1.07), 1, Vec3(1, 0, 0));
    spheres[5] = Sphere(cam->center + Vec3(-2.3, -0.8, -2.69), 1, Vec3(1, 1, 0.45));
}

Scene::~Scene() {
    if (texture) { glDeleteTextures(1, &texture); }

    cudaDeviceSynchronize(); // ensure all kernels are finished
    if (fb) { cudaFree(fb); };
    if (spheres) { cudaFree(spheres); };
    if (cam) { cudaFree(cam); }
}

void Scene::makeCamera() {
    cudaMallocManaged(&cam, sizeof(Camera));

    cam->maxBounces = 10;
    cam->numberOfRayPerPixel = 10;

    cam->center = Vec3(0, 0, 0);

    cam->updateCameraPosition(yawDeg, pitchDeg, radius);

    cam->up = Vec3(0, 1, 0);
    cam->fov = 90.0f;
    cam->aspect = float(width) / float(height);
}

void Scene::renderFrame(int i, int j) {
    cam->updateCameraPosition(yawDeg, pitchDeg, radius);
    render(i, j);
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
    ImGui::DragFloat3("Center", &cam->center.x, 0.01f);
    ImGui::DragInt("Max Bounces", &cam->maxBounces, 1, 0, 1000);
    ImGui::DragInt("Number of ray per pixel", &cam->numberOfRayPerPixel, 1, 0, 1000);
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
    cam->numberOfRayPerPixel = 1000;
    render(0, 0);
    savePPM(filename, fb, width, height);
}

void Scene::renderPPM(const std::string &filename) {
    renderPPMFrame(filename);
    std::cout << "Static render saved to " << filename << std::endl;
}

void Scene::renderGIF(int nFrames, float totalAngle) {
    for (int i = 0; i < nFrames; i++) {
        yawDeg = (totalAngle / nFrames) * i;
        cam->updateCameraPosition(yawDeg, pitchDeg, radius);
        std::ostringstream filename;
        filename << "build/frame_" << std::setw(3) << std::setfill('0') << i << ".ppm";
        renderPPMFrame(filename.str());
        std::cout << "Saved " << filename.str() << std::endl;
    }
    std::cout << "Video render complete!" << std::endl;
}

void Scene::render(int numRenderedFramesA, int numRenderedFramesB) {
    dim3 threads(16, 16);
    dim3 blocks((width + 15) / 16, (height + 15) / 16);

    // Note: the kernel runs on the GPU, which cannot directly access host
    // memory unless we use managed memory or cudaMemcpy
    SceneProperties sceneProperties{
        fb, width, height, spheres, nSpheres, cam, numRenderedFramesA, numRenderedFramesB};

    render_scene<<<blocks, threads>>>(sceneProperties);

    cudaDeviceSynchronize();
}
