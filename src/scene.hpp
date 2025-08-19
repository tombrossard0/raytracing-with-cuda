#pragma once

#include "camera.cuh"
#include "sphere.cuh"
#include "vec3.cuh"

#include <SDL2/SDL.h>
#include <SDL2/SDL_opengl.h>
#include <string>

struct SceneProperties {
    Vec3 *fb;
    int width, height;
    Sphere *spheres;
    int nSpheres;
    Camera *cam;
};

class Scene {
  public:
    int width, height;
    Vec3 *fb;
    Sphere *spheres;
    int nSpheres;
    float radius;
    float angleDeg;
    float yawDeg;
    float pitchDeg;
    float minRadius;
    float maxRadius;

    bool focus;

    Camera *cam;

    GLuint texture; // Only sets in realtime engine

    Scene(int w, int h);
    ~Scene();

    void makeCamera();
    void renderFrame();
    void renderGUI(GLuint &tex);

    void renderPPMFrame(const std::string &filename);
    void renderPPM(const std::string &filename = "output.ppm");
    void renderGIF(int nFrames, float totalAngle);

    void render();
};
