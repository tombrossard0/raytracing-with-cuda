#include "imgui.h"
#include "imgui_impl_opengl3.h"
#include "imgui_impl_sdl2.h"

#include <SDL2/SDL.h>
#include <SDL2/SDL_opengl.h>
#include <SDL2/SDL_video.h>
#include <iostream>

#include "mouse.hpp"
#include "scene.hpp"

class Engine {
  public:
    int window_width;
    int window_height;
    SDL_Window *window;
    SDL_GLContext sdl_gl_context;

    SDL_Event event;
    Mouse mouse;

    bool running;

    Uint64 currentTime;
    float deltaTime;
    Uint64 lastFrameTime;
    Uint64 lastFPSTime;
    int frameCount;
    float fps;

    Engine(int w, int h);
    ~Engine();

    void updateTime();

    GLuint createTexture(int w, int h);
    void uploadFbToTexture(Scene &scene);
    void clearScreen();
    void renderImGui(Scene *scene);

    void computeFPS();

  private:
    void initWindow();
    void initContext();
    void initImGUI();
};
