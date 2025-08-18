#include "imgui.h"
#include "imgui_impl_opengl3.h"
#include "imgui_impl_sdl2.h"

#include <SDL2/SDL.h>
#include <SDL2/SDL_opengl.h>
#include <SDL2/SDL_video.h>
#include <iostream>

class Engine {
  public:
    int window_width;
    int window_height;
    SDL_Window *window;
    SDL_GLContext sdl_gl_context;

    Engine(int w, int h);
    ~Engine();

  private:
    void initWindow();
    void initContext();
};
