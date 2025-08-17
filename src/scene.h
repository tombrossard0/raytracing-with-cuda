#pragma once

#include <SDL2/SDL.h>
#include <SDL2/SDL_opengl.h>
#include "vec3.h"
#include "sphere.h"
#include "camera.h"
#include <string>

class Scene
{
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

    Scene(int w, int h);
    ~Scene();

    Camera makeCamera();
    void renderFrame();
    void renderGUI(GLuint &tex);
    void processKeyboard(const Uint8 *keystate, float deltaTime);
    int renderSDL2();

    void renderPPMFrame(const std::string &filename);
    void renderPPM(const std::string &filename = "output.ppm");
    void renderGIF(int nFrames, float totalAngle);
};
