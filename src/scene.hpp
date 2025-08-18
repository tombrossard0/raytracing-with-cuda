#pragma once

#include "camera.cuh"
#include "mouse.hpp"
#include "sphere.cuh"
#include "vec3.cuh"

#include <SDL2/SDL.h>
#include <SDL2/SDL_opengl.h>
#include <string>

class Scene {
  public:
    int width, height;
    Vec3 *fb;
    Sphere *spheres;
    int nSpheres;
    Vec3 center;
    float radius;
    float angleDeg;
    float yawDeg;
    float pitchDeg;
    float minRadius;
    float maxRadius;

    GLuint texture; // Only sets in realtime engine

    Scene(int w, int h);
    ~Scene();

    Camera makeCamera();
    void renderFrame();
    void renderGUI(GLuint &tex);
    void processInputs(const Uint8 *keystate, float deltaTime, bool &running, SDL_Event *event, Mouse &mouse);

    void renderPPMFrame(const std::string &filename);
    void renderPPM(const std::string &filename = "output.ppm");
    void renderGIF(int nFrames, float totalAngle);
};
