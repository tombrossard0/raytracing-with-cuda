#pragma once

#include "camera.cuh"
#include "entity.cuh"
#include "vec3.cuh"

#include <SDL2/SDL.h>
#include <SDL2/SDL_opengl.h>
#include <string>

#include "InputManager.hpp"

struct SceneProperties {
    Vec3 *fb;
    int width, height;
    Entity *entities;
    int nEntities;
    Camera *cam;

    int numRenderedFramesA;
    int numRenderedFramesB;
};

class Scene {
  public:
    // Display
    int width, height;
    Vec3 *fb;

    // Objects in the scene
    Entity *entities;
    int nEntities;

    // ImGui setting
    bool focus;

    Camera *cam;

    GLuint texture; // Only sets in realtime engine

    Scene(int w, int h);
    ~Scene();

    void makeCamera();

    void processInputs(InputManager inputManager, MouseState mouseState, float deltaTime);

    void renderFrame(int i, int j);
    void renderGUI(GLuint &tex);

    void renderPPMFrame(const std::string &filename);
    void renderPPM(const std::string &filename = "output.ppm");
    void renderGIF(int nFrames, float totalAngle);

    void render(int numRenderedFramesA, int numRenderedFramesB);
};
